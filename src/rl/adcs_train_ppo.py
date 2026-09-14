"""PPO training for the ADCS attitude-control task (Ray RLlib).

Trains a policy on EventSatEnv: 7 continuous actions (4 reaction-wheel torques,
3 magnetorquer dipoles) against the 11D attitude-error observation.

Requires ray[rllib] and wandb, which are NOT declared in pyproject.toml yet:

    uv pip install "ray[rllib]" wandb

Run with:

    uv run python -m src.rl.adcs_train_ppo
"""
from __future__ import annotations

import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import ray
    from ray.tune.registry import register_env
    from ray.rllib.algorithms.ppo import PPOConfig
except ImportError as exc:  # pragma: no cover - dependency is optional
    raise ImportError(
        "ray[rllib] is required for ADCS PPO training but is not installed. "
        'Install it with: uv pip install "ray[rllib]" wandb'
    ) from exc

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - logging is optional
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore

from src.environment.orbital.adcs.adcs_gymnasium_wrapper import EventSatEnv
from src.environment.orbital.adcs.eventsat import actuators
from src.environment.orbital.adcs.eventsat import env as EVENTSAT_ENV_CONFIG

# Repo root is three levels up: rl -> src -> root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# One id, used both to register the env with Ray and to request it in PPOConfig.
# These drifting apart is silent: RLlib just fails to find the env.
ENV_ID = "gymnasium_env/adcs_sim-v0"

SEED = 42
ENV_CONFIG = replace(EVENTSAT_ENV_CONFIG, seed=SEED)


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


# Single source of truth for the PPO hyperparameters: these are unpacked into
# PPOConfig.training() *and* logged as the W&B run config, so what is recorded
# can never drift from what was actually trained with.
PPO_HYPERPARAMS = dict(
    lr=3e-4,
    num_epochs=10,            # 30 (RLlib's default) collapsed the policy's std early on
    # gamma, lambda and the timestep dt -- really the controller update rate --
    # are the main parameters for how much change can be observed before an
    # update occurs, and thus how much gradient can be identified.
    gamma=0.99,
    lambda_=0.98,
    vf_loss_coeff=1.0,  # Might be unneeded since the policy and
                        # value function network are not set to share layers
    vf_clip_param=estimate_return_bound(ENV_CONFIG),
    entropy_coeff=0.005,      # keeps exploration alive through the early slew-learning phase
    clip_param=0.2,
    # Must stay well above max_steps: batches smaller than a few episodes leave
    # iterations that finished no episode at all, and episode_return_mean comes
    # back NaN.
    train_batch_size_per_learner=16000,
    minibatch_size=1024,      # RLlib's default of 128 against a 16k batch is both slow and noisy
)

# Result keys that are not scalar metrics and would only bloat the W&B run.
EXCLUDED_RESULT_KEYS = {"config", "hist_stats"}

# Checkpoints go to the git-ignored data/ tree (see .gitignore: data/trained_models/*/).
# Writing them next to the source, e.g. adcs/checkpoints/, is NOT git-ignored and
# would end up committed.
CHECKPOINT_DIR = REPO_ROOT / "data" / "trained_models" / "adcs_ppo"

# W&B is opt-out via the environment, so a run can be made without touching the
# code and without an account: WANDB_MODE=offline or WANDB_MODE=disabled.
USE_WANDB = WANDB_AVAILABLE and os.environ.get("WANDB_MODE", "").lower() != "disabled"


# 1. Register the custom environment with Ray
def env_creator(env_config):
    """Build one EventSatEnv for an RLlib rollout worker.

    RLlib builds an env per rollout worker, each in its own process, so what it
    is handed is this recipe rather than an env object.

    ENV_CONFIG is closed over rather than routed through RLlib's env_config:
    putting it there means serialising it with `asdict`, which is recursive and
    flattens `mission` into a plain dict. env_config then carries only what a
    sweep varies per trial, merged on top.
    """
    return EventSatEnv(config=ENV_CONFIG.with_overrides(env_config))


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


def wandb_config(env_config, **extra) -> Dict[str, Any]:
    """Run config for W&B, built from the same objects the run is configured
    with rather than a hand-maintained copy."""
    # Popped rather than read: both are nested, and left in env_fields they
    # would log as one opaque blob each instead of as sweepable scalars.
    env_fields = asdict(env_config)
    weights = env_fields.pop("reward_weights", {}) or {}
    mission = env_fields.pop("mission", {}) or {}
    return {
        **PPO_HYPERPARAMS,
        "seed": SEED,
        **{f"env/{key}": value for key, value in env_fields.items()},
        # sigma is a property rather than a field, so asdict never sees it.
        "env/sigma": env_config.mission.sigma,
        **{f"mission/{key}": value for key, value in mission.items()},
        **{f"reward_weight/{key}": value for key, value in weights.items()},
        **extra,
    }


def main():
    register_env(ENV_ID, env_creator)

    # 2. Initialize Ray. No runtime_env: the rollout workers are processes on
    # this machine and inherit the driver's working directory, so they import
    # src.* straight from the repo.
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

    # 3. Configure PPO
    total_timesteps = 1_500_000
    batch_size = PPO_HYPERPARAMS["train_batch_size_per_learner"]
    num_iterations = max(1, total_timesteps // batch_size)
    checkpoint_interval = 10  # save a checkpoint every N iterations, plus always at the end
    num_env_runners = 6
    resume_from_checkpoint = False  # Set to True to continue from the last checkpoint instead of from scratch

    config = (
        PPOConfig()
        .environment(ENV_ID)
        .env_runners(num_env_runners=num_env_runners)
        .training(**PPO_HYPERPARAMS)
        # RLlib's default of 100 averages episode_return_mean over the last 100
        # episodes, which at a few dozen episodes per iteration lags real
        # progress by several iterations and smears early learning away.
        .reporting(metrics_num_episodes_for_smoothing=20)
        .debugging(seed=SEED)
    )

    print(f"Env config: {ENV_CONFIG}")
    print(f"vf_clip_param: {PPO_HYPERPARAMS['vf_clip_param']:.0f} (from the reward weights)")
    print(f"Training for {num_iterations} iterations x {batch_size} steps = {num_iterations * batch_size:,} env steps")

    # 4. W&B logging (run `wandb login` once, or set WANDB_MODE=offline/disabled)
    if USE_WANDB:
        wandb.init(
            project="CubeSat-ADCS-ReactionWheels",
            config=wandb_config(
                ENV_CONFIG,
                num_env_runners=num_env_runners,
                num_iterations=num_iterations,
                total_timesteps=num_iterations * batch_size,
            ),
        )
    else:
        print("W&B logging disabled (not installed, or WANDB_MODE=disabled).")

    # 5. Build and Train
    algo = None
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        algo = build_algo(config)

        if resume_from_checkpoint:
            if any(CHECKPOINT_DIR.iterdir()):
                restore_checkpoint(algo, CHECKPOINT_DIR)
                print(f"Resumed training from checkpoint at {CHECKPOINT_DIR}")
            else:
                print(f"No checkpoint found at {CHECKPOINT_DIR}, starting from scratch.")

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
                save_checkpoint(algo, CHECKPOINT_DIR)
                print(f"  Checkpoint saved at iteration {iteration}")

        # Always checkpoint the final policy, even if it doesn't land on the interval
        save_checkpoint(algo, CHECKPOINT_DIR)

    except (KeyboardInterrupt, Exception):
        # Best-effort: the algo may have failed during build, and the save
        # itself can raise -- neither should mask the original exception.
        if algo is not None:
            print("Training interrupted, saving checkpoint before exiting...")
            try:
                save_checkpoint(algo, CHECKPOINT_DIR)
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
