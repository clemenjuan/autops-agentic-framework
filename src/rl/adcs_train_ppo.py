"""PPO training for the ADCS attitude-control task (Ray RLlib).

Trains a policy on EventSatEnv: 7 continuous actions (4 reaction-wheel torques,
3 magnetorquer dipoles) against the 11D attitude-error observation.

Requires ray[rllib] and wandb, both in the `rl` extra:

    uv sync --extra dev --extra rl

Which mission is trained, and how it is configured, comes from the command
line -- see `parse_args` for the full list:

    # the default mission, full-length run
    uv run python -m src.rl.adcs_train_ppo

    # a different mission, with two of its config fields changed
    uv run python -m src.rl.adcs_train_ppo --mission target_track \
        --mission-override num_targets=3 --mission-override tolerance_deg=2.0

    # continue the last run of that mission
    uv run python -m src.rl.adcs_train_ppo --mission target_track --resume

Checkpoints are per mission, under data/trained_models/adcs_ppo/<mission>/,
because a policy trained on one mission restored into another is wrong
rather than an error.
"""
from __future__ import annotations

import os
import argparse
import ast
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Sequence, Optional

import numpy as np

try:
    import ray
    from ray.tune.registry import register_env
    from ray.rllib.algorithms.ppo import PPOConfig
except ImportError as exc:  # pragma: no cover - dependency is optional
    raise ImportError(
        "ray[rllib] is required for ADCS PPO training but is not installed. "
        "Install it with: uv sync --extra dev --extra rl"
    ) from exc

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - logging is optional
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore

from src.environment.orbital.adcs.adcs_gymnasium_wrapper import EventSatEnv
from src.environment.orbital.adcs.configs import AdcsEnvConfig, MissionConfig
from src.environment.orbital.adcs.eventsat import (
    DEFAULT_MISSION,
    MISSION_CONFIGS,
    actuators,
    env as EVENTSAT_ENV_CONFIG,
)
from src.mission.registry import MISSION_TYPES

# =============================================================================
# Constants and paths
# =============================================================================

# Repo root is three levels up: rl -> src -> root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# One id, used both to register the env with Ray and to request it in PPOConfig.
# These drifting apart is silent: RLlib just fails to find the env.
ENV_ID = "gymnasium_env/adcs_sim-v0"

# Default seed, overridable with --seed.
SEED = 42

# Result keys that are not scalar metrics and would only bloat the W&B run.
EXCLUDED_RESULT_KEYS = {"config", "hist_stats"}

# W&B is opt-out via the environment, so a run can be made without touching the
# code and without an account: WANDB_MODE=offline or WANDB_MODE=disabled.
USE_WANDB = WANDB_AVAILABLE and os.environ.get("WANDB_MODE", "").lower() != "disabled"

# Checkpoints go to the git-ignored data/ tree (see .gitignore: data/trained_models/*/).
# Writing them next to the source, e.g. adcs/checkpoints/, is NOT git-ignored and
# would end up committed.
CHECKPOINT_ROOT = REPO_ROOT / "data" / "trained_models" / "adcs_ppo"


def checkpoint_dir(mission: str) -> Path:
    """Where `mission`'s policy is read from and written to.

    Per mission rather than one shared directory: the checkpoints are not
    interchangeable. A policy trained on one mission restored into another
    is wrong rather than an error. A single directory means
    training the second mission overwrites the first.
    """
    return CHECKPOINT_ROOT / mission


# =============================================================================
# Reward bound and PPO hyperparameters
# =============================================================================

def estimate_return_bound(config) -> float:
    """Rough bound on |episode return|, used to size vf_clip_param.

    PPO clips value targets to +/- vf_clip_param, so it has to cover the actual
    return range: too small silently truncates the value function, too large
    wastes the clipping. Deriving it from the weights means it tracks any
    reward change instead of being a constant that goes stale.

    This is a *lower* bound on the true range: slew_rate_boundary_penalty is
    quadratic in the rate excess and therefore unbounded above, so a tumbling
    episode can exceed it. The 1.5x margin below absorbs some of that.
    """
    weights = config.reward_weights
    n_wheels = len(actuators.reaction_wheels)

    def weight(name: str) -> float:
        return abs(float(weights.get(name) or 0.0))

    # Worst dense step: full pointing error plus every wheel at saturation.
    worst_step = weight("pointing_error_penalty") + weight("rw_saturation_reward") * n_wheels
    # Best dense step: sitting inside the boundary layer.
    best_step = weight("boundary_layer_reward")

    events = (
        weight("target_cleared_bonus") * config.mission.num_targets
        + weight("mission_done_bonus")
    )
    return 1.5 * (max(worst_step, best_step) * config.mission.max_steps + events)


def ppo_hyperparams(env_config) -> Dict[str, Any]:
    """
    Single source of truth for the PPO hyperparameters: these are unpacked into
    PPOConfig.training() *and* logged as the W&B run config, so what is recorded
    can never drift from what was actually trained with.
    """
    return dict(
        lr=3e-4,
        num_epochs=10,            # 30 (RLlib's default) collapsed the policy's std early on
        # gamma, lambda and the timestep dt -- really the controller update rate --
        # are the main parameters for how much change can be observed before an
        # update occurs, and thus how much gradient can be identified.
        gamma=0.99,
        lambda_=0.98,
        vf_loss_coeff=1.0,  # Might be unneeded since the policy and
                                # value function network are not set to share layers
        vf_clip_param=estimate_return_bound(env_config),
        entropy_coeff=0.005,      # keeps exploration alive through the early slew-learning phase
        clip_param=0.2,
        # Must stay well above max_steps: batches smaller than a few episodes leave
        # iterations that finished no episode at all, and episode_return_mean comes
        # back NaN.
        train_batch_size_per_learner=16000,
        minibatch_size=1024,      # RLlib's default of 128 against a 16k batch is both slow and noisy
    )


# =============================================================================
# RLlib API-stack shims
# =============================================================================
# RLlib renamed these between the old and new API stacks. Preferring the new
# name and falling back keeps the script working across both rather than
# failing at the first save, an hour into a run.

def build_algo(config):
    builder = getattr(config, "build_algo", None) or config.build
    return builder()


def save_checkpoint(algo, path: Path) -> None:
    saver = getattr(algo, "save_to_path", None) or algo.save
    saver(str(path))


def restore_checkpoint(algo, path: Path) -> None:
    restorer = getattr(algo, "restore_from_path", None) or algo.restore
    restorer(str(path))


# =============================================================================
# Metrics and W&B logging
# =============================================================================

def flatten_metrics(result, parent_key="", sep="/"):
    """Flattens RLlib's nested result dict into a flat {name: float} mapping.

    Logging this wholesale means every metric RLlib reports -- policy loss, KL,
    entropy, gradient norms, sampler timings, per-component episode stats -- ends
    up in W&B automatically, rather than having to be predicted in advance and
    added to a hand-picked list.

    Non-finite values are dropped rather than logged, because W&B renders a NaN
    as a gap in the chart that looks identical to a metric that was never
    reported. The count is returned so the caller can log that instead.
    """
    metrics: Dict[str, float] = {}
    dropped = 0
    for key, value in result.items():
        if key in EXCLUDED_RESULT_KEYS:
            continue
        name = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            nested, nested_dropped = flatten_metrics(value, name, sep)
            metrics.update(nested)
            dropped += nested_dropped
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float, np.integer, np.floating)):
            if np.isfinite(value):
                metrics[name] = float(value)
            else:
                dropped += 1
    return metrics, dropped


def _get_metric(result, *keys):
    """Looks a metric up at the result's top level, then under `env_runners`,
    trying each alias in turn (RLlib's key names differ between API stacks)."""
    for key in keys:
        value = result.get(key)
        if value is not None:
            return float(value)
    env_runner = result.get("env_runners")
    if isinstance(env_runner, dict):
        for key in keys:
            value = env_runner.get(key)
            if value is not None:
                return float(value)
    return float('NaN')


def get_mean_reward(result):
    return _get_metric(result, "episode_reward_mean", "episode_return_mean")


def get_mean_episode_len(result):
    """Mean episode length, which here reads as time-to-completion.

    There is no rate-based termination (see EventSatEnv.step), so the only way
    out of an episode before max_steps is mission_done. Short episodes are
    therefore *good*: the policy cleared its targets early. A mean pinned at
    max_steps means the opposite -- the targets are never being cleared -- and
    is the signal to watch alongside the mean reward.

    Note this inverts if a tumble truncation is ever added: an early exit would
    then be ambiguous between success and a policy spinning up to escape the
    pointing penalty, and the two would have to be told apart by the
    termination reason rather than by length alone.
    """
    return _get_metric(result, "episode_len_mean", "episode_length_mean")


def _loggable(value: Any) -> Any:
    """Coerce a config value into something W&B can serialise.

    Mission configs carry numpy: the pointing vectors are ndarrays and any
    angle built with np.deg2rad is a np.float64, neither of which is JSON
    serialisable. Only the mission fields are passed through here -- the env
    fields and the reward weights are plain scalars by construction.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def wandb_config(env_config, hyperparams: Dict[str, Any], **extra) -> Dict[str, Any]:
    """Run config for W&B, built from the same objects the run is configured
    with rather than a hand-maintained copy.

    Args:
        env_config: The config the run trains against.
        hyperparams: What was passed to PPOConfig.training(). Required rather
            than optional: a run logged without them looks complete and is
            not reproducible.
        **extra: Run parameters that live outside both, e.g. num_env_runners.
    """
    # Popped rather than read: both are nested, and left in env_fields they
    # would log as one opaque blob each instead of as sweepable scalars.
    env_fields = asdict(env_config)
    weights = env_fields.pop("reward_weights", {}) or {}
    mission = env_fields.pop("mission", {}) or {}
    return {
        **hyperparams,
        # Off the config rather than passed in: it is the seed the run was
        # actually built with, so the two cannot disagree.
        "seed": env_config.seed,
        **{f"env/{key}": value for key, value in env_fields.items()},
        # sigma is a property rather than a field, so asdict never sees it.
        "env/sigma": env_config.mission.sigma,
        **{f"mission/{key}": _loggable(value) for key, value in mission.items()},
        **{f"reward_weight/{key}": value for key, value in weights.items()},
        **extra,
    }

# =============================================================================
# Command line
# =============================================================================

def parse_mission_overrides(
    pairs: Sequence[str], mission_cfg: MissionConfig
) -> Dict[str, Any]:
    """Turn ``--mission-override KEY=VALUE`` strings into a field dict.

    Args:
        pairs: Raw ``KEY=VALUE`` strings, as argparse collected them.
        mission_cfg: The config they will be applied to. Read for its field
            names and their current types, so a typo is caught here rather
            than surfacing as a TypeError from dataclasses.replace.

    Returns:
        Field name -> coerced value, ready for ``dataclasses.replace``.

    Raises:
        ValueError: On a malformed pair, an unknown field, or an attempt to
            override ``type``.
    """
    fields = mission_cfg.__dataclass_fields__
    overrides: Dict[str, Any] = {}

    for pair in pairs:
        key, sep, raw = pair.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(
                f"malformed --mission-override {pair!r}; expected KEY=VALUE"
            )

        # `type` selects the mission class and the config class has to match
        # it. Overriding it hands the registry a config of the wrong class --
        # which builds the wrong mission, or dies on a field the other
        # mission's config does not have.
        if key == "type":
            raise ValueError(
                "--mission-override cannot change 'type'; use --mission to "
                f"pick a mission (one of {sorted(MISSION_TYPES)})"
            )
        if key not in fields:
            raise ValueError(
                f"unknown mission field {key!r}; "
                f"{type(mission_cfg).__name__} accepts {sorted(fields)}"
            )

        # literal_eval rather than eval: it reads ints, floats, bools and
        # lists and nothing else. A bare word is not a Python literal, so it
        # falls back to the string it already is.
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw

        # Match the field's current type where it is not a plain scalar. The
        # pointing vectors are ndarrays, and a list would work by accident in
        # some places and not others.
        current = getattr(mission_cfg, key)
        if isinstance(current, np.ndarray):
            value = np.asarray(value, dtype=float)
        elif isinstance(current, float) and isinstance(value, int):
            value = float(value)

        overrides[key] = value

    return overrides


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI: which mission, how it is configured, and how long to train.

    Args:
        argv: Argument list. None reads sys.argv, so the CLI behaves
            normally; passing a list lets a test drive the parser without a
            subprocess.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="adcs_train_ppo",
        description="Train an ADCS attitude-control policy with PPO.",
    )
    parser.add_argument(
        "--mission",
        # Read from the registry, so a mission added there is offered by the
        # CLI without a second edit.
        choices=sorted(MISSION_TYPES),
        default=DEFAULT_MISSION,
        help="which mission to train on (default: %(default)s)",
    )
    parser.add_argument(
        "--mission-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one field of the mission config, e.g. num_targets=3. "
             "Repeatable. The mission `type` cannot be overridden: it is what "
             "selects the mission class, and the config class has to match it.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="env, policy and RLlib seed (default: %(default)s)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_500_000,
        help="total env steps to train for (default: %(default)s)",
    )
    parser.add_argument(
        "--env-runners",
        type=int,
        default=6,
        help="parallel rollout workers; 0 samples on the driver "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="iterations between checkpoints; the final policy is always "
             "saved (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        # Spelled out rather than %(default)s: the default is None, and what
        # is worth showing is the path it resolves to.
        help=f"where to read and write checkpoints "
             f"(default: {CHECKPOINT_ROOT}/<mission>)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the checkpoint directory instead of from scratch",
    )

    args = parser.parse_args(argv)

    # `type` cannot express "positive". parser.error gives the usage message
    # and exit code 2 that a CLI should, rather than a traceback.
    if args.timesteps < 1:
        parser.error("--timesteps must be >= 1")
    if args.env_runners < 0:
        parser.error("--env-runners must be >= 0")
    if args.checkpoint_interval < 1:
        parser.error("--checkpoint-interval must be >= 1")

    #Check if the mission config overwrites are valid
    try:
        args.mission_override = parse_mission_overrides(
            args.mission_override, MISSION_CONFIGS[args.mission]
        )
    except ValueError as exc:
        parser.error(str(exc))

    return args



# =============================================================================
# Entry point
# =============================================================================

def build_env_config(args: argparse.Namespace) -> AdcsEnvConfig:
    """The env config this run trains against.

    The mission named by ``--mission`` with any ``--mission-override`` fields
    applied, carried on the EventSat env config at the run's seed. The mission
    *type* is baked in here rather than passed through RLlib's env_config,
    because `with_overrides` rebuilds a nested dataclass with
    `replace(current, **value)` and so can only change a mission's fields,
    never its class.

    Args:
        args: Parsed command line. `mission_override` is already a dict by
            this point -- `parse_args` validates and converts it.

    Returns:
        The config handed to the env creator and to `ppo_hyperparams`.
    """
    mission_cfg = replace(MISSION_CONFIGS[args.mission], **args.mission_override)
    return replace(EVENTSAT_ENV_CONFIG, mission=mission_cfg, seed=args.seed)


def train(
    algo,
    num_iterations: int,
    checkpoints: Path,
    checkpoint_interval: int,
) -> None:
    """Run the training loop, logging and checkpointing as it goes.

    Separate from `main` so that the try/except/finally around it reads as
    what it is -- save on interrupt, always stop the algo -- rather than
    wrapping thirty lines of loop body.

    Args:
        algo: The built RLlib algorithm, already restored if resuming.
        num_iterations: Training iterations to run.
        checkpoints: Directory to save into.
        checkpoint_interval: Iterations between saves. The final policy is
            always saved, whether or not it lands on the interval.
    """
    print("Starting training...")
    for i in range(num_iterations):
        result = algo.train()
        mean_reward = get_mean_reward(result)
        mean_episode_len = get_mean_episode_len(result)
        # algo.iteration reflects the true global iteration count (correct
        # even after resuming from a checkpoint), unlike the loop index i.
        iteration = algo.iteration
        print(f"Iteration: {iteration}, Mean Reward: {mean_reward:.3f}, Mean Episode Length: {mean_episode_len:.1f}")

        # Everything RLlib reported this iteration, plus the few derived
        # values the loop itself owns.
        metrics, dropped = flatten_metrics(result)
        metrics.update({
            "iteration": iteration,
            "mean_reward": mean_reward,
            "metrics_dropped_non_finite": dropped,
        })
        if USE_WANDB:
            wandb.log(metrics)

        if (i + 1) % checkpoint_interval == 0:
            save_checkpoint(algo, checkpoints)
            print(f"  Checkpoint saved at iteration {iteration}")

    # Always checkpoint the final policy, even if it doesn't land on the interval
    save_checkpoint(algo, checkpoints)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv=argv)

    # Ray. No runtime_env: the rollout workers are processes on this machine
    # and inherit the driver's working directory, so they import src.*
    # straight from the repo.
    #
    # TODO(cluster): a multi-node run needs the code shipped, i.e. a
    # runtime_env working_dir -- and that reintroduces a bug, so do not simply
    # add it back. Ray omits from that copy everything any .gitignore in the
    # tree names, which includes orekit-data.zip; propagator.py resolves the
    # archive relative to its own __file__, so on a worker it looks inside the
    # shipped copy and finds nothing. Orekit then fails to initialise -- as a
    # log warning, not an exception -- and the env dies much later in
    # configure(). Our archive is also patched (it carries an IGRF.COF the
    # stock distribution omits), so the fix is not "install the data package":
    # see the decision brief on how the archive should reach a node.
    ray.init(ignore_reinit_error=True)

    selected = build_env_config(args)

    def make_env(env_config):
        return EventSatEnv(config=selected.with_overrides(env_config))

    register_env(ENV_ID, make_env)

    # PPO configuration
    hyperparams = ppo_hyperparams(selected)
    batch_size = hyperparams["train_batch_size_per_learner"]
    num_iterations = max(1, args.timesteps // batch_size)
    checkpoints = args.checkpoint_dir or checkpoint_dir(args.mission)

    config = (
        PPOConfig()
        .environment(ENV_ID)
        .env_runners(num_env_runners=args.env_runners)
        .training(**hyperparams)
        # RLlib's default of 100 averages episode_return_mean over the last 100
        # episodes, which at a few dozen episodes per iteration lags real
        # progress by several iterations and smears early learning away.
        .reporting(metrics_num_episodes_for_smoothing=20)
        .debugging(seed=args.seed)
    )

    print(f"Env config: {selected}")
    print(f"vf_clip_param: {hyperparams['vf_clip_param']:.0f} (from the reward weights)")
    print(f"Training for {num_iterations} iterations x {batch_size} steps = {num_iterations * batch_size:,} env steps")

    # W&B logging (run `wandb login` once, or set WANDB_MODE=offline/disabled)
    if USE_WANDB:
        wandb.init(
            project="CubeSat-ADCS-ReactionWheels",
            config=wandb_config(
                selected,
                hyperparams=hyperparams,
                num_env_runners=args.env_runners,
                num_iterations=num_iterations,
                total_timesteps=num_iterations * batch_size,
            ),
        )
    else:
        print("W&B logging disabled (not installed, or WANDB_MODE=disabled).")

    # Build and train
    algo = None
    checkpoints.mkdir(parents=True, exist_ok=True)

    try:
        algo = build_algo(config)

        if args.resume:
            if any(checkpoints.iterdir()):
                restore_checkpoint(algo, checkpoints)
                print(f"Resumed training from checkpoint at {checkpoints}")
            else:
                print(f"No checkpoint found at {checkpoints}, starting from scratch.")

        train(algo, num_iterations, checkpoints, args.checkpoint_interval)

    except (KeyboardInterrupt, Exception):
        # Best-effort: the algo may have failed during build, and the save
        # itself can raise -- neither should mask the original exception.
        if algo is not None:
            print("Training interrupted, saving checkpoint before exiting...")
            try:
                save_checkpoint(algo, checkpoints)
            except Exception as save_error:
                print(f"Warning: could not save checkpoint: {save_error}")
        raise
    finally:
        # Cleanup
        if algo is not None:
            try:
                algo.stop()
            except Exception as e:
                print(f"Warning during algo.stop(): {e}")

        if USE_WANDB:
            wandb.finish()
        ray.shutdown()


if __name__ == "__main__":
    main()
