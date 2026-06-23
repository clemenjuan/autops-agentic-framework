# Research Tracker

Last updated: 2026-06-23

This is the single AUTOPS-side tracker for the current paper work. Keep it short: status, handoffs, commands, and decisions that must stay aligned with `space-world-models`.

## Active Threads

| Thread | AUTOPS owns | External owner / handoff |
|---|---|---|
| EventSat O-benchmark | Simulator, 32-cell architecture matrix, configs, metrics, board. | None; this repo is canonical. |
| Flamingo scaling | Scenario simulator, scaling configs, board scripts. | Future multi-agent extensions. |
| World-model scheduling | EventSat trace export, canonical evaluation configs, metrics, board cells. | `space-world-models` owns LeWM/Dreamer training, probes, CEM planner artifacts, Jetson export. |

## Current State

- LLM/agentic EventSat runs may still be active under `logs/`; do not edit running LLM configs or LLM/agentic representation code mid-campaign.
- `data/world_model/` is generated export data and is ignored by Git.
- `notebooks/` is local scratch and is no longer tracked.
- World-model entries are intended to appear as board grid cells, not as a separate side table only.
- Random shooting is out of scope for the current paper plan.

## Trace Export

Symbolic traces can be exported without waiting for `llm-a`. Wait for LLM/HLLM only if the training dataset must include those behaviours. Normal LLM board runs do not automatically produce world-model traces.

Canonical initial export, inheriting each config's `num_episodes` and `max_steps`:

```bash
uv run python scripts/export_eventsat_world_model_traces.py \
  configs/experiments/eventsat_sas_ao_symb.yaml \
  configs/experiments/eventsat_sas_ag_symb.yaml \
  configs/experiments/eventsat_sas_ah_symb_symb.yaml \
  --out data/world_model/eventsat_autops_v1
```

Current generated EventSat configs use `num_episodes: 5` and `max_steps: 10080`. If the final board campaign uses a different paired-seed budget, pass the same values explicitly, for example `--episodes 100 --steps 10080`.

Expected outputs:

- `data/world_model/eventsat_autops_v1/eventsat_world_model_v1.npz`
- `data/world_model/eventsat_autops_v1/eventsat_world_model_v1.metadata.json`
- Per-run traces under `data/world_model/eventsat_autops_v1/runs/*/trace/`

Smoke export only:

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
- `action[E,T,11]`
- `state[E,T,S]`
- `reward[E,T]`
- `mode[E,T]`
- `resolved_mode[E,T]`
- `forced_mode[E,T]`
- `episode_seed[E]`
- `episode_id[E]`

The 11D action vector is 7 mode one-hot values plus 2 data-priority values plus 2 pipeline-routing values. Thermal and pointing are not invented targets; v1 uses AUTOPS-native state fields only.

## Handoff To `space-world-models`

After export, train LeWM with AUTOPS action dimension 11 and W&B enabled:

```bash
cd ~/space-world-models
WANDB_MODE=online \
WANDB_PROJECT=space-world-models \
WANDB_RUN_NAME=eventsat-autops-action11-lewm-full \
.venv/bin/python -m swm_eventsat.experiments.train_world_model \
  --config-name train_autops
```

Local smoke training:

```bash
cd ~/space-world-models
.venv/bin/python -m swm_eventsat.experiments.train_world_model \
  data.path=/home/clemente/autops-agentic-framework/data/world_model/eventsat_autops_v1/eventsat_world_model_v1.npz \
  model.action_encoder.input_dim=11 \
  model.action_encoder.smoothed_dim=11 \
  trainer.max_epochs=1 \
  wandb.enabled=false
```

Probe smoke check, before real frozen LeWM latents are available:

```bash
cd ~/space-world-models
.venv/bin/python -m swm_eventsat.experiments.train_autops_probes \
  --dataset /home/clemente/autops-agentic-framework/data/world_model/eventsat_autops_v1/eventsat_world_model_v1.npz \
  --out outputs/eventsat_autops_probe_smoke.npz
```

After the LeWM checkpoint is trained, export latents, train latent probes, then package the planner artifact:

```bash
cd ~/space-world-models
.venv/bin/python -m swm_eventsat.experiments.export_autops_latents \
  --dataset /home/clemente/autops-agentic-framework/data/world_model/eventsat_autops_v1/eventsat_world_model_v1.npz \
  --checkpoint /path/to/lewm.ckpt \
  --out outputs/eventsat_autops_latents.npz

.venv/bin/python -m swm_eventsat.experiments.train_autops_probes \
  --dataset /home/clemente/autops-agentic-framework/data/world_model/eventsat_autops_v1/eventsat_world_model_v1.npz \
  --latents outputs/eventsat_autops_latents.npz \
  --out outputs/eventsat_autops_probe_latent.npz

.venv/bin/python -m swm_eventsat.experiments.write_planner_artifact \
  --dataset /home/clemente/autops-agentic-framework/data/world_model/eventsat_autops_v1/eventsat_world_model_v1.npz \
  --checkpoint /path/to/lewm.ckpt \
  --probe outputs/eventsat_autops_probe_latent.npz \
  --out outputs/eventsat_autops_lewm/planner_artifact.json
```

## AUTOPS Evaluation

World-model configs:

- `configs/experiments/world_model/eventsat_sas_ao_lewm-cem.yaml`
- `configs/experiments/world_model/eventsat_sas_ao_dreamerv3.yaml`

Once real artifacts exist:

```bash
uv run --extra rl autops run configs/experiments/world_model/eventsat_sas_ao_lewm-cem.yaml
uv run autops run configs/experiments/world_model/eventsat_sas_ao_dreamerv3.yaml
uv run python scripts/refresh_board.py
```

## Still Needed

- Real LeWM PlannerArtifact load path and artifact validation in AUTOPS.
- DreamerV3 policy artifact load path once training/export exists.
- Final mission-mode weights for science/safe/downlink utility.
- Full-scale traces from representative symbolic, LLM/HLLM, stress, and later learned-controller rollouts.
- Paired-seed final evaluation and board refresh.
- Jetson Orin Nano latency, memory, and rollout-throughput benchmark values.
