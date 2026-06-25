# Research Tracker

Last updated: 2026-06-24

Active AUTOPS-side tracker for current research work. This is intentionally kept
in `docs/` because it records live scientific direction, not historical archive
material.

## Active Threads

| Thread | AUTOPS owns | External handoff |
|---|---|---|
| EventSat O-benchmark | Simulator, 32-cell architecture matrix, configs, metrics, board. | None; this repo is canonical. |
| SSA scaling | Scenario simulator, AO configs, ISL/coverage metrics. | Active multi-agent extension. |
| World-model scheduling | EventSat trace export, AUTOPS evaluation configs, metrics, board cells. | `space-world-models` owns LeWM/Dreamer training, probes, CEM planner artifacts, Jetson export. |

## Current State

- `data/world_model/` is generated export data and is ignored by Git.
- World-model entries should appear as EventSat board cells, not only as a side table.
- Random shooting is out of scope for the current paper plan.
- Normal LLM board runs do not automatically produce world-model traces.

## Trace Export

Symbolic traces can be exported without waiting for `llm-a`. Include LLM/HLLM
rollouts only if the training dataset must represent those behaviours.

Canonical initial export, inheriting each config's `num_episodes` and `max_steps`:

```bash
uv run python scripts/export_eventsat_world_model_traces.py \
  configs/experiments/eventsat_sas_ao_symb.yaml \
  configs/experiments/eventsat_sas_ag_symb.yaml \
  configs/experiments/eventsat_sas_ah_symb_symb.yaml \
  --out data/world_model/eventsat_autops_v1
```

Current EventSat configs use `num_episodes: 5` and `max_steps: 10080`. If the
final board campaign uses a different paired-seed budget, pass it explicitly, for
example `--episodes 100 --steps 10080`.

Expected outputs:

- `data/world_model/eventsat_autops_v1/eventsat_world_model_v1.npz`
- `data/world_model/eventsat_autops_v1/eventsat_world_model_v1.metadata.json`
- `data/world_model/eventsat_autops_v1/runs/*/trace/episode_*.npz`

Smoke export:

```bash
uv run python scripts/export_eventsat_world_model_traces.py \
  configs/experiments/eventsat_sas_ao_symb.yaml \
  --episodes 2 \
  --steps 64 \
  --out data/world_model/eventsat_autops_smoke
```

## Dataset Contract

The stacked AUTOPS world-model dataset contains:

- `obs[E,T,25]`
- `action[E,T,7]`
- `state[E,T,S]`
- `reward[E,T]`
- `mode[E,T]`
- `resolved_mode[E,T]`
- `forced_mode[E,T]`
- `episode_seed[E]`
- `episode_id[E]`

The 7D action vector is one-hot over the seven EventSat operational modes.
Thermal and pointing targets are not invented; v1 uses AUTOPS-native state fields.

## Handoff To `space-world-models`

Use repo-relative placeholders instead of personal absolute paths:

```bash
AUTOPS_ROOT=/path/to/autops-agentic-framework
SWM_ROOT=/path/to/space-world-models
DATASET="$AUTOPS_ROOT/data/world_model/eventsat_autops_v1/eventsat_world_model_v1.npz"
```

Train LeWM with AUTOPS action dimension 7:

```bash
cd "$SWM_ROOT"
WANDB_MODE=online \
WANDB_PROJECT=space-world-models \
WANDB_RUN_NAME=eventsat-autops-action7-lewm-full \
.venv/bin/python -m swm_eventsat.experiments.train_world_model \
  --config-name train_autops
```

Local smoke training:

```bash
cd "$SWM_ROOT"
.venv/bin/python -m swm_eventsat.experiments.train_world_model \
  data.path="$DATASET" \
  model.action_encoder.input_dim=7 \
  model.action_encoder.smoothed_dim=7 \
  trainer.max_epochs=1 \
  wandb.enabled=false
```

Probe smoke check before real frozen LeWM latents are available:

```bash
cd "$SWM_ROOT"
.venv/bin/python -m swm_eventsat.experiments.train_autops_probes \
  --dataset "$DATASET" \
  --out outputs/eventsat_autops_probe_smoke.npz
```

After the LeWM checkpoint is trained, export latents, train latent probes, then
package the planner artifact:

```bash
cd "$SWM_ROOT"
.venv/bin/python -m swm_eventsat.experiments.export_autops_latents \
  --dataset "$DATASET" \
  --checkpoint /path/to/lewm.ckpt \
  --out outputs/eventsat_autops_latents.npz

.venv/bin/python -m swm_eventsat.experiments.train_autops_probes \
  --dataset "$DATASET" \
  --latents outputs/eventsat_autops_latents.npz \
  --out outputs/eventsat_autops_probe_latent.npz

.venv/bin/python -m swm_eventsat.experiments.write_planner_artifact \
  --dataset "$DATASET" \
  --checkpoint /path/to/lewm.ckpt \
  --probe outputs/eventsat_autops_probe_latent.npz \
  --out outputs/eventsat_autops_lewm/planner_artifact.json
```

## AUTOPS Evaluation

World-model configs:

- `configs/experiments/world_model/eventsat_sas_ao_lewm-cem.yaml`
- `configs/experiments/world_model/eventsat_sas_ao_dreamerv3.yaml`

Once real artifacts exist and the config paths point at them:

```bash
uv run --extra rl autops run configs/experiments/world_model/eventsat_sas_ao_lewm-cem.yaml
uv run autops run configs/experiments/world_model/eventsat_sas_ao_dreamerv3.yaml
uv run python scripts/refresh_board.py
```

## Still Needed

- Real LeWM `PlannerArtifact` load path and artifact validation in AUTOPS.
- DreamerV3 policy artifact load path once training/export exists.
- Final mission-mode weights for science/safe/downlink utility.
- Full-scale traces from representative symbolic, LLM/HLLM, stress, and learned-controller rollouts.
- Paired-seed final evaluation and board refresh.
- Jetson Orin Nano latency, memory, and rollout-throughput benchmark values.
