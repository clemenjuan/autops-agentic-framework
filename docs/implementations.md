# Implementation Registry

Persistent record of every implemented component in the morphological matrix,
its paper basis, and key design decisions. Grows as new components are added.

> **Terminology / alignment note.** The canonical framing is the O framework
> ([`morphological_matrix.md`](morphological_matrix.md)): an architecture is organisation ×
> representation (cognitive **substrate** × **action space**) × operational paradigm. The seven
> representations are `symb · rl · hrl · llm-s · llm-a · hllm-s · hllm-a`; run names follow
> `eventsat_sas_<paradigm>_<rep>` (ah: `_<onboard>_<ground>`). The `decision_procedure` and
> `behaviour` modules and config fields are held at defaults (not framework components). **NB —
> work in progress:** configs now declare the 7-cell token in `representation` (the loader
> normalises it to the internal substrate + `action_space`, so the `@register` class names below —
> `rule_based_eventsat`, `subsymbolic_eventsat`, `llm_eventsat`, `agentic_eventsat`,
> `*_scheduler_eventsat` — are unchanged). Dual-core AH with *independent* onboard/ground representations (the 21
> `ah_<onboard>_<ground>` pairs) is **implemented and runnable**, and the LLM **ground** cells
> are all real now: single-shot (`llm_scheduler_eventsat` hllm-s / `llm_single_scheduler_eventsat`
> llm-s) and **agentic** (`agentic_scheduler_eventsat` hllm-a / `llm_agentic_scheduler_eventsat`
> llm-a). Still pending: the real `hrl` core (RL + symbolic, documented placeholder in
> `placeholders.py`) and the PPO-trained `subsymbolic_scheduler` learned-scheduling line.
> The component descriptions below document the **current code** — map them to the
> framework via `morphological_matrix.md` §2.

---

## Agent Organizations

Formal definition: an agent system **S = (A, E, C, Ω)** where A = agents, E = environment, C = communication topology, Ω = orchestration policy (Kim et al. 2025 [FVFQ73RF]).

**Empirical prediction for all Organization experiments** (Kim et al. 2025, 180 configs): satellite mode selection is sequential constraint satisfaction → centralized org predicted to outperform distributed. Capability saturation (β̂=−0.404) means multi-agent overhead negates gains once single-agent baseline > ~45%.

Full taxonomy: Kim et al. (2025) [FVFQ73RF] "Towards a Science of Scaling Agent Systems".

### SingleAgentSystem (SAS)

- **File**: `src/core/organization/single_agent_system.py`
- **Paper basis**: Kim et al. (2025) [FVFQ73RF] Single-Agent System (SAS) — |A|=1, single reasoning locus, C undefined, Ω direct, complexity O(k).
- **Structure**: One central agent receives full constellation observation and selects actions for all satellites. No inter-agent communication. Zero coordination overhead.
- **Key property**: Maximum context integration (unified memory stream, full prior-history access). Upper bound for context-quality; lower bound for parallelism.
- **Configs**: the EventSat·SAS matrix is **32 experiments** — conventional 1 + ag 7 + ao 3 + ah 21 (morphological_matrix.md §4); `decision_procedure` is held fixed, not a multiplied axis.

### CentralizedMAS

- **File**: `src/core/organization/centralized_mas.py`
- **Paper basis**: Kim et al. (2025) [FVFQ73RF] Centralized MAS — orchestrator routes to sub-agents, C = {(a_orch, aᵢ) : ∀i}, Ω = hierarchical, complexity O(rnk). Also: ECSS-E-ST-70-11C autonomy levels (mission management layer vs onboard autonomy layer).
- **Structure**: A = {mission_manager, sat_agent_0}; C = star (manager→local, unidirectional); Ω = hierarchical (local agent action is used; manager action stored as directive for next step).
- **EventSat design decisions**:
  - `distribute_observation`: Both agents receive the full environment observation (single satellite, no meaningful state partitioning). `sat_agent_0` additionally receives the manager's previous-step action as a `messages` entry (directive context).
  - `collect_actions`: Manager action stored as `_last_manager_directive`; local agent action returned as environment action. Fallback to manager action if no local agent output.
  - Manager directive carries over step-to-step via `_last_manager_directive`; reset to `None` in `initialize()`.
  - Latency: `ExperimentRunner` accumulates manager + local agent latencies as the total step latency (sequential execution, both contribute to decision overhead).
- **Scope**: the MAS organisations (cmas/imas/dmas/hmas) belong to the **future multi-satellite scenario** and are not exercised by the EventSat benchmark (morphological_matrix.md §1). The code below is the single-satellite wiring kept for that future work; no EventSat configs use it.

### DecentralizedMAS — Implemented (Flamingo N≥3)

- **File**: `src/core/organization/decentralized_mas.py`
- **Paper basis**: Kim et al. (2025) [FVFQ73RF] Decentralized MAS — all-to-all peer exchange, C = {(aᵢ, aⱼ) : ∀i,j, i≠j}, Ω = consensus, complexity O(dnk).
- **Structure**: One peer agent per satellite, no manager. All-to-all exchange: every peer shares what it sees, so each ends up with the same global information.
- **Flamingo design decisions**:
  - `distribute_observation`: every peer receives the full observation (the decentralized counterpart of SAS's global view), plus the other peers' previous-step proposals as `messages` (the all-to-all channel C).
  - `collect_actions`: peers running the shared deterministic protocol on identical information converge on the same deconflicted plan; the **consensus** (plurality, ties by agent index) is returned. So DMAS deconflicts like SAS/CMAS and — unlike IMAS — wastes nothing.
  - `get_metrics`: surfaces the coordination cost — `coordination_messages = n·(n-1)` per round (6 at N=3) and `consensus_rounds`. The runner threads this into the Flamingo metrics, so the cost side of the axis is measured.
  - **Outcome vs cost**: with the capable global `rule_based_flamingo`, DMAS matches SAS/CMAS mission utility (validated: utility 660, duplicate rate 0 under the contended scenario) while paying a strictly higher message cost. A decentralized org only loses *outcome* when consensus fails (Kim et al. 17.2× error amplification), which a single deterministic round does not trigger.
- **Status**: Runnable at N≥3 (`configs/experiments/flamingo_dmas_ag_symb.yaml`), all-to-all topology. Ring/mesh/visibility-limited topology ablations are future work. Degenerate at N=1.

### IndependentMAS — Implemented (Flamingo N≥3)

- **File**: `src/core/organization/independent_mas.py`
- **Paper basis**: Kim et al. (2025) [FVFQ73RF] Independent MAS — C = ∅, no inter-agent coordination.
- **Structure**: A = {sat_agent_0 … sat_agent_{n−1}}, one agent per satellite; C = ∅; Ω = independent (no consensus, no manager).
- **Flamingo design decisions**:
  - `distribute_observation`: agent `sat_agent_i` is mapped by index to the i-th satellite and receives a **local view** containing only that satellite's state and only that satellite's visible tasks — it cannot see what the others see or intend.
  - `collect_actions`: per-satellite actions are merged **verbatim, without deconfliction**, so independent agents that pick the same RSO reach the environment as duplicate observations (the coordination cost the organisation axis measures).
  - Contention is supplied by the scenario, not the org: `configs/scenarios/flamingo.yaml` sets `satellite_phase_shift: 0` so the constellation shares visibility windows and the agents must compete. Validated: under that scenario SAS/CMAS keep duplicate rate at 0 while IMAS wastes ≈⅔ of attempts and loses utility/coverage.
- **Status**: Runnable at N≥3 (`configs/experiments/flamingo_imas_ag_symb.yaml`). Degenerate at N=1 (equivalent to SAS, no coordination overhead).

### HybridMAS — Implemented (Flamingo N≥3)

- **File**: `src/core/organization/hybrid_mas.py`
- **Paper basis**: Kim et al. (2025) [FVFQ73RF] Hybrid MAS — heterogeneous mixed topology combining star + all-to-all + independent sub-topologies.
- **Structure**: the constellation is partitioned into clusters; each cluster has a head agent (`cluster_agent_i`). Coordination happens *within* a cluster, none *across* clusters — C = heterogeneous, Ω = hybrid.
- **Flamingo design decisions**:
  - `distribute_observation`: each cluster head receives a view of only its own cluster's satellites and their visible tasks; running `rule_based_flamingo` on it deconflicts that cluster (a mini-SAS).
  - `collect_actions`: per-cluster assignments are merged **without cross-cluster deconfliction**, so satellites in different clusters can still collide on the same RSO.
  - **Tunable midpoint**: `num_clusters` (default 2) spans the whole organisation axis — `1` ≡ SAS (one cluster sees all → 0 duplicates), `n` ≡ IMAS (singletons → maximal duplicates), in between partial coordination. `get_metrics` reports the localised cost `coordination_messages = Σ c_i·(c_i-1)` (= `n·(n-1)` at one cluster, `0` at singletons). Explicit `clusters` partitions are also accepted.
  - Validated at N=3 (default 2 clusters): utility 607 ± 78, duplicate rate 0.376, coordination 2 — strictly between SAS/CMAS/DMAS (716 ± 87, dup 0) and IMAS (404 ± 57, dup 0.667).
- **Status**: Runnable at N≥3 (`configs/experiments/flamingo_hmas_ag_symb.yaml`). Visibility-/capability-based clustering is future work.

---

## Decision Loops

### SDA (Sense-Decide-Act) — Phase 2 baseline

- **File**: `src/core/decision_procedure/sda_loop.py`
- **Paper basis**: Classic reactive agent pattern (sense-decide-act cycle)
- **Structure**: Single-pass — encode observation via representation, select action, return.
  No iteration, reflection, or memory interaction.
- **Memory**: Ignored. `process()` returns `(action, memory)` with memory passed through unchanged.
- **Metrics**: `decision_latency_s`, `total_decisions`, `has_rationale`
- **Significance**: Simplest possible decision loop — lower bound for decision overhead
  and the baseline for all loop comparisons.


---

## Representations

> The registered name below (the "Registered as" field) is **resolved** at runtime from
> `representation × action_space × operations_paradigm` (`ExperimentConfig.resolved_representation_type`);
> configs no longer set `representation_config.type` except as an explicit override (e.g. `_algobase`).

### Rule-Based EventSat — Phase 2 baseline

- **File**: `src/eventsat/symbolic.py`
- **Registered as**: `rule_based_eventsat`
- **Paper basis**: Hand-designed priority rule chain (domain engineering)
- **Structure**: Priority rules across categories (R1-R7 + default).
  `encode_observation()` extracts flat state dict from environment.
  `select_action(context)` evaluates rules in priority order, returns first match.
- **Rationale**: Provides a human-readable rationale indicating which rule fired
  for the explainability metric.
- **Operations paradigm**: Paired with autonomous onboard / hybrid symbolic cells.

### Schedule-Based EventSat — Phase 2 baseline

- **File**: `src/eventsat/schedule_symbolic.py`
- **Registered as**: `schedule_based_eventsat`
- **Paper basis**: Traditional ground operations scheduling — greedy cyclic
  battery-aware planner with power model from PDR Table 6.2.
- **Structure**: During ground passes, generates time-tagged command sequences
  via greedy planning. Between passes, commands play back from schedule.
  Telemetry-first sequencing: downlink HK -> fresh data -> generate schedule.
- **Operations paradigm**: Paired with autonomous ground / hybrid ground-planner slots.

### Conventional Schedule EventSat — Phase 3 (human-realistic)

- **File**: `src/eventsat/conventional.py`
- **Registered as**: `conventional_schedule_eventsat`
- **Paper basis**:
  - Sellmaier, Uhlig & Schmidhuber (2022), "Spacecraft Operations" [SGJTLF4D] —
    ground segment planning workflows, ops timelines.
  - ECSS-E-ST-70C Ground Systems and Operations (2008) [CIYT2V68] —
    planning margin standards, shift structures, commanding timelines.
  - Endsley (1995), "Toward a Theory of Situation Awareness in Dynamic Systems",
    Human Factors 37(1):32-64. [46MUS93H] — SA Level-3 projection loss model.
  - Castano et al. (2022), "Operations for Autonomous Spacecraft" [2IJJ7ILS] —
    human vs autonomous ops contrast.
- **Structure**: Subclasses `ScheduleBasedEventSat`, overrides `_generate_schedule()`
  to introduce five human cognitive constraint parameters:
  - `conservative_margin` (default 1.3): Multiply charge block durations by this
    factor. Humans use worst-case power estimates from the ops handbook (20–40%
    margin is standard practice per ECSS-E-ST-70C §4.4).
  - `planning_horizon_discount` (default 0.85): Plan only 85% of the inter-pass
    gap. Human teams cannot reliably plan the full gap — cognitive load and shift
    handover limits cause them to leave ~10–20% as a charging buffer.
  - `max_observations_per_gap` (default 2): Cap observation blocks per schedule.
    Each observation creates a compress→detect→send cascade that is complex to
    reason about manually. Real LEO mission teams typically schedule 1–3 obs/gap.
  - `shift_handover_probability` (default 0.10): With this probability per
    planning cycle, a shift handover occurred. The incoming team has degraded
    context (Endsley 1995 SA Level-3 projection loss) and applies a safety margin.
  - `shift_handover_soc_penalty` (default 0.10): Add this to `min_soc_for_operations`
    when a handover is detected (e.g., 0.40 → 0.50). Models conservative
    threshold inflation under degraded situational awareness.
- **Stochasticity**: Shift handover is seeded via `seed(value)` for reproducibility.
- **Key context**: The schedule generated here will be uploaded one pass later
  (per `ConventionalGround` paradigm). The conservative parameters account for
  the uncertainty of planning ahead based on stale telemetry.
- **Operations paradigm**: Paired with `conventional_ground`.

### Placeholder Schedulers — ground-paradigm stand-ins (non-symbolic AG/CG cells)

- **File**: `src/eventsat/placeholders.py`
- **Registered as**: `subsymbolic_scheduler_eventsat` (the **only remaining
  placeholder** scheduler). The LLM ground schedulers it once stood beside are now
  real: `llm_scheduler_eventsat` / `llm_single_scheduler_eventsat` (single-shot, see
  below) and `agentic_scheduler_eventsat` / `llm_agentic_scheduler_eventsat`
  (agentic, see *Agentic EventSat Scheduler*).
- **Why they exist**: The ground paradigms (`autonomous_ground`,
  `conventional_ground`) drive between-pass behavior from a `schedule` the
  representation emits during a pass. Only the symbolic planners
  (`schedule_based_eventsat`, `conventional_schedule_eventsat`) emit one; the
  per-step controllers `subsymbolic_eventsat` / `llm_eventsat` / `agentic_eventsat`
  do not. Pairing those with a ground paradigm produced a **degenerate** run — the
  satellite charged every inter-pass step and the representation barely influenced
  behavior (confirmed: the entire inter-pass gap fell back to `default_mode`). For
  CG it was worse: `process_action` forces `communication` during passes, so all
  non-symbolic CG cells collapsed to an identical trivial trajectory.
- **What it does (PLACEHOLDER)**: `subsymbolic_scheduler_eventsat` subclasses
  `ScheduleBasedEventSat` and emits a real schedule via the **symbolic greedy
  planner** — NOT the RL policy. `is_placeholder = True` is surfaced in
  `results["experiment_statistics"]["metadata"]["representation_is_placeholder"]`
  so analysis can exclude it from headline comparisons.
- **Extension point (future research, "P3 — learned scheduling")**: replace the
  remaining RL placeholder with a **PPO-trained schedule producer** — the deferred
  RL-vs-LLM scheduling comparison. (The LLM and agentic schedule producers are
  already built.)
- **Guard**: `config_loader` **errors** if a ground paradigm is paired with a
  non-schedule-producing representation type, so the degenerate cell cannot be
  recreated silently. Note: the subsymbolic ground cell no longer requires `torch`
  (it delegates to the symbolic planner).

### Subsymbolic EventSat — Phase 4b (RL learned)

- **File**: `src/eventsat/rl.py`
- **Registered as**: `subsymbolic_eventsat`
- **Paradigm**: Subsymbolic (deep RL policy; Brooks 1991, Colelough & Regli 2025)
- **Paper basis**:
  - Oliver et al. EUCASS 2025 [8KDZ5Z53] — Dec-POMDP formulation, PPO [256,256] tanh,
    negative reward baseline (more stable), gamma=0.966–0.98, 30 epochs, 50μs Jetson inference
  - Hamilton et al. 2025 [GWQ3LK6H] — Observation space design ablation: task-relevant
    sensors (orbital position, timing lookahead) drastically reduce sample complexity;
    10-seed significance threshold; PPO clip=0.3, 30 SGD epochs
  - BSK-RL Stephenson & Schaub [ACUQK9VV] — Gymnasium wrapper pattern; Eclipse and
    OpportunityProperties lookahead; semi-MDP variable-duration actions; modular obs
  - Wang et al. 2022 [RRFQ6WCN] — Resource state (battery, memory) + visibility windows;
    encoder-decoder for task scheduling
- **Observation space (25D)**:
  - Group 1 (4D) — Resource fill fractions: battery_soc, obc_fill, jetson_raw_fill, jetson_compressed_fill
  - Group 2 (6D) — Orbital phase & timing: sin/cos(orbital_phase), time_to_eclipse, time_to_pass, remaining_pass_duration, episode_progress
  - Group 3 (3D) — Binary environment flags: in_sunlight, ground_pass_active, health_nominal
  - Group 4 (5D) — Pipeline state: uncompressed_obs, compression_progress, undetected_obs, detection_progress, downlink_utilization
  - Group 5 (7D) — Current mode one-hot
- **Action space**: `MultiDiscrete([7, 2, 2])`:
  - Sub-action 0: Primary operational mode (7 modes)
  - Sub-action 1: data_priority {0=normal, 1=urgent} → 1.5x downlink in comms mode
  - Sub-action 2: pipeline_routing {0=compress_first, 1=detect_first} → redirects between compression and detection pipelines
- **Architecture**: ActorCritic — shared trunk 25→256→256 (Tanh, orthogonal init) → 3 actor heads + 1 critic head; ~70K parameters
- **Training**: PPO (Schulman et al. 2017) with GAE-λ (λ=0.95), factored joint log-prob over MultiDiscrete heads
- **Hyperparameters** (Oliver et al. EUCASS 2025): lr=1e-4→1e-5, gamma=0.97, clip=0.3, 30 SGD epochs, batch=4096, minibatch=256
- **Symbolic grounding** (same constraints as LLMEventSat):
  - Anomaly → forced safe (cannot be overridden)
  - SoC < 0.20 → forced charging
  - Communication without active pass → forced charging
- **Mock mode**: `rl_mock: true` uses `RandomPolicy` — no torch, for CI
- **reason()**: Returns per-head action probabilities as structured explanation steps
- **update()**: Delegates to PPOTrainer (called from experiment_runner post-episode in learned mode)
- **Orthogonality**: Works with the fixed SDA decision driver and all 4 ops paradigms (ao/ah/ag/conventional)
- **Training script**: `scripts/train_subsymbolic.py`
- **Gymnasium wrapper**: `src/eventsat/gymnasium_wrapper.py` (EventSatGymnasium)
- **Supporting modules**: `src/core/behaviour/rollout_buffer.py` (RolloutBuffer + GAE), `src/core/behaviour/training_pipeline.py` (PPOTrainer)
- **Architecture note**: Current MLP baseline; RNN (LSTM/GRU) is a known improvement direction for partial observability — subject to optimization by Giulio Vaccari (exchange PhD)
- **Configs** (rl cell): `eventsat_sas_ao_rl.yaml`, `eventsat_sas_ag_rl.yaml`, `eventsat_sas_ah_rl_rl.yaml`

### LLM EventSat — Phase 4a (hybrid)

- **File**: `src/eventsat/llm.py`
- **Registered as**: `llm_eventsat`
- **Substrate / action space**: Hybrid (subsymbolic LLM + symbolic safety constraints), **reactive** action space (single-shot encode→call→select)
- **Paper basis**:
  - Rodriguez-Fernandez et al. (2024), "Language Models are Spacecraft Operators" [WC5WU34U]
    — LLM prompt design for satellite operations (§3.2 state formatting).
  - Li (2025), "AI Agents for Satellite Operations" [UAA3GIVK]
    — LLM agent architecture for satellite ops.
- **Structure**:
  - `encode_observation()`: Same feature extraction as rule-based (for comparability).
  - `select_action(context)`: Formats state into structured prompt → LLM call → JSON
    parse → symbolic grounding validates mode → retry on invalid → **fails the episode**
    if no valid mode (substrate-integrity invariant, `ec1b83b`; no symbolic substitution).
  - `reason()`: LLM-based structured reasoning for explanations/debugging.
  - `get_rationale()`: LLM's natural language rationale.
- **Symbolic grounding checks**:
  - Mode must be one of 7 valid EventSat modes.
  - Anomaly → forced safe mode (no LLM override).
  - Communication requires active ground pass.
  - SoC < 0.20 → forced charging (hard safety limit).
- **LLM infrastructure** (`src/core/llm_client.py`):
  - Dual provider: TUM Ollama (primary, via `OLLAMA_HOST`) + OpenAI API (fallback).
  - File-based response cache (`data/llm_cache/`) keyed on prompt hash.
  - Mock mode (`llm_mock: true`) for CI — no live LLM calls.
  - Configurable model, temperature, provider via YAML.
- **Orthogonality**: Works with the fixed SDA decision driver and the ops paradigms
  (ah/ag/conventional). Unlike symbolic which needed 3 separate representation types per ops
  paradigm, the LLM representation handles all ops contexts through its prompt.
- **Diagnostics**: `llm_api_calls`, `llm_cache_hit_rate`, `llm_total_latency_s`,
  `llm_mean_call_latency_s`, `llm_tokens_prompt`, `llm_tokens_completion`,
  and `llm_schedule_entries`. These are cost/reproducibility telemetry, not part
  of the M-01...M-14 scientific metric registry; cache-hit reruns correctly report
  zero live-call latency and zero new token counts.
- **Operations paradigm**: All (autonomous_hybrid, autonomous_ground, conventional_ground).
- **Learned-emergence variant** (Phase 5):
  - `prompt_optimized` (`_hyre_lep_*`): loads offline-optimised system prompt. `FixedMemory`
    invariant preserved. **Note**: `writable_coala` does NOT apply here — emergent·memory is
    gated by the *agentic* action space (writing is an action), which the reactive single-shot
    LLM lacks (see [morphological_matrix.md](morphological_matrix.md)).
- **Cell**: `hllm-s` (hybrid LLM + symbolic, single-shot). **Config**: `eventsat_sas_ag_hllm-s` (ground planner via `llm_scheduler_eventsat`). The AH-ground hllm-s and a prompt-optimized variant are pending the dual-core + 32-config increment.

### Agentic EventSat — Phase 4c (agentic hybrid)

- **File**: `src/eventsat/agentic.py`
- **Registered as**: `agentic_eventsat`
- **Substrate / action space**: Hybrid (LLM agent + symbolic tools + grounding constraints), **agentic** action space (tool-call loop + structured memory). Same substrate as `llm_eventsat`; differs only in action space.
- **Supporting modules**:
  - `src/eventsat/agentic_tools.py` — 6 domain tools (pure functions on state/memory)
  - `src/eventsat/agentic_prompts.py` — system prompt, planning/reflect/reasoning templates
- **Paper basis**:
  - Sumers et al. (2024), "Cognitive Architectures for Language Agents" [CoALA] —
    4-memory architecture (working, episodic, semantic, procedural), action decomposition
    into internal (reasoning, retrieval) and external (tool use, grounding) actions.
  - Sapkota et al. (2026) — agentic satellite operations.
  - Li (2025), "AI Agents for Satellite Operations" [UAA3GIVK] — tool-augmented AI agents.
  - Rodriguez-Fernandez et al. (2024) [WC5WU34U] — LLM prompt design (shared with Phase 4a).
- **Architecture**: Multi-step Plan-Tool-Reflect-Decide loop within `select_action()`:
  1. PLAN: LLM analyzes state and selects a tool to query.
  2. TOOL: Domain tool executes (pure Python, no LLM), returns structured result.
  3. REFLECT: LLM incorporates tool result, optionally calls another tool or decides.
  4. DECIDE: LLM selects final mode after sufficient information gathering. If the
     tool budget exhausts (or the loop stalls) without a decision, one **forced
     decision-only call** closes the cycle (`format_forced_decision_prompt`, no tool
     option — bounded-loop answer extraction). First live
     122B run showed 66 % of steps riding the budget to exhaustion without it.
     If even that yields no valid mode, the episode **fails** (substrate-integrity
     invariant — agentic sibling of `ec1b83b`; no symbolic substitution).
  5. GROUND: Same symbolic safety constraints as llm_eventsat.
- **Max LLM calls per decision**: `max_agentic_steps` (default **5**, configurable in YAML)
  + at most 1 forced-decide call. Raised from 3 on 2026-06-12 (`6866ab5`): the first
  live 122B run rode the 3-call budget to exhaustion on 66 % of steps without deciding.
- **Domain tools** (6):
  - `check_battery`: SoC, charging rate, feasible modes.
  - `check_ground_pass`: Pass active, time to next, remaining duration, OBC data.
  - `check_data_pipeline`: Pipeline status, bottleneck identification.
  - `check_constraints`: Pre-validate proposed mode against hard/soft constraints.
  - `recall_history`: Query episodic memory (recent modes, battery trend).
  - `evaluate_plan`: Heuristic utility and risk assessment for proposed mode.
- **CoALA memory mapping** (FixedMemory not modified):
  - Working: `DecisionContext.state` + tool results accumulated in-loop.
  - Episodic: `FixedMemory.history` + `task_history` (via `recall_history` tool).
  - Semantic: Domain rules hardcoded in `AGENTIC_SYSTEM_PROMPT`.
  - Procedural: Tool definitions in `TOOL_SCHEMAS`.
- **Key comparison**: vs `llm_eventsat` (single-shot LLM) — does multi-step agentic
  reasoning with tool use improve satellite operations decisions?
- **Symbolic grounding**: Same as llm_eventsat (anomaly→safe, SoC<0.20→charging,
  no-pass→no-comms).
- **Mock mode**: `llm_mock: true` short-circuits to symbolic fallback (0 LLM calls).
- **Decision interaction**: Plan-Tool-Reflect-Decide is the representation's internal
  reasoning protocol (CoALA §3.2 "internal actions") behind the fixed SDA driver.
  Optional `DecisionContext.enrichments` are serialized into the prompt when present,
  but current benchmark configs do not vary the outer decision loop.
- **Inference gating**: For AG/CG ops paradigms, `should_allow_inference()` returns
  `False` between passes. The experiment runner skips the agentic loop entirely; schedule
  playback handles actions. This avoids wasted LLM API calls on stale inter-pass data
  (Rossi et al. 2023 [5EG3E3BP]).
- **Simulation assumption**: The 122B-parameter Qwen model represents upper-bound
  reasoning quality, not a deployable onboard configuration. AH configs model ideal
  onboard reasoning; AG/CG configs model ground-based inference.
- **Agentic-loop cost**: the internal loop is cost-controlled via
  `max_agentic_steps` and forced final answer extraction.
- **Metrics**: All LLM client metrics + `agentic_total_tool_calls`,
  `agentic_avg_steps_per_decision`, `agentic_grounding_overrides`, per-tool histogram.
- **Learned-emergence variants** (Phase 5):
  - `hand_designed` (default): fixed `AGENTIC_SYSTEM_PROMPT` + `FixedMemory`.
  - `prompt_optimized` (`_hyag_lep_*`): loads offline-optimised system prompt from
    `data/trained_prompts/<experiment_id>/prompt.txt` (written by `PromptOptimizer`).
    `FixedMemory` invariant preserved.
  - `writable_coala` (`_hyag_lec_*`): swaps `FixedMemory` for `WritableMemory`;
    injects two writable memory tools; expands system prompt with CoALA memory
    instructions. **Fairness trade-off**: compared against `_hyag_hd_` baseline only.
- **Cell**: `hllm-a` (hybrid LLM + symbolic, agentic), as a **per-step** core. **Config**: `eventsat_sas_ag_hllm-a` (and the AH ground-planner) resolve `hllm-a` to the *agentic schedule producer* `agentic_scheduler_eventsat`, not this per-step core — see *Agentic EventSat Scheduler* below. The writable_coala online-learning variant (mechanism, not a separate class) for the scheduler is pending.

### Agentic EventSat Scheduler — Phase 4.e (agentic ground planner: hllm-a / llm-a)

- **File**: `src/eventsat/agentic_scheduler.py`
- **Registered as**: `agentic_scheduler_eventsat` (hllm-a) and
  `llm_agentic_scheduler_eventsat` (llm-a) — **replacing the former placeholders**.
- **Substrate / action space**: the **agentic** schedule producer — the agentic
  analogue of the single-shot `llm_scheduler_eventsat`, and the schedule-producing
  sibling of the per-step `agentic_eventsat`. At each ground contact, instead of one
  LLM call, it runs a CoALA Plan-Tool-Reflect-Decide loop (reusing the six domain
  tools in `agentic_tools.py`) whose **terminal DECIDE step emits a whole-pass
  schedule** (`[[mode, steps], …]`) executed autonomously until the next contact.
- **Why it was needed**: AG/AH **ground** slots need a whole-pass schedule producer,
  so the real per-step `agentic_eventsat` (a single-mode controller) was never a
  scoring core in the matrix; the `hllm-a`/`llm-a` ground cells resolved to symbolic
  greedy *placeholders* (`is_placeholder=True`). This module makes them real, closing
  the deferred "Phase 4.e" gap. Resolution is unchanged — same `@register` names — so
  no config/loader change was required; only the classes behind the names changed.
- **Design (reuse)**: subclasses `LLMSchedulerEventSat` (the hllm-s single-shot core),
  inheriting the per-pass control flow (stale-HK-first → plan-once-on-fresh-telemetry),
  `encode_observation`, schedule validation, and the symbolic safety shield; it
  overrides only schedule *generation* (`_generate_schedule_llm` → the agentic loop)
  and adds agentic metrics. New prompts live in `agentic_prompts.py`
  (`AGENTIC_SCHEDULE_SYSTEM_PROMPT`, `format_schedule_planning_prompt`,
  `format_schedule_tool_result_prompt`, `format_forced_schedule_prompt`); the JSON
  parser is shared via `agentic_eventsat.parse_agentic_json`.
- **hllm-a vs llm-a — symbolic grounding toggle (the *only* difference)**: exactly as
  `hllm-s` vs `llm-s`. `AgenticSchedulerEventSat` (hllm-a) keeps `_symbolic_grounding=True`
  — the CoALA tools **and** the symbolic safety/format shield on the emitted schedule
  (drop communication, clamp/pad to the gap, veto operational blocks in a critical
  battery/storage state → charging). `LLMAgenticSchedulerEventSat` (llm-a) sets
  `_symbolic_grounding=False` — same loop, tools and prompt, but the schedule is taken
  as the model produced it (safety lives in the prompt; the env enforces hard limits at
  execution). The CoALA tools are information-gathering *external actions* (CoALA §3),
  part of the agentic action space, not the symbolic substrate — so they are kept for
  both cells, and the comparison isolates **only** the symbolic safety layer (user
  decision 2026-06-22).
- **Substrate integrity**: if the loop yields no valid schedule after `MAX_RETRIES`,
  the episode **fails** (`RuntimeError`) — no symbolic substitution (consistent with
  `llm_scheduler_eventsat` and the per-step `agentic_eventsat`).
- **Decision extraction**: accepts both the protocol form
  `{"decision": {"schedule": …}}` and the flattened `{"schedule": …}` that reasoning
  models (and the mock client) emit.
- **Metrics**: LLM client metrics + `llm_schedule_entries` (inherited) +
  `agentic_total_tool_calls`, `agentic_avg_steps_per_decision`, per-tool histogram.
- **`max_agentic_steps`**: CoALA loop budget per schedule decision (default **5**,
  configurable in YAML), matching the per-step agentic core.
- **Decisive prompt — bounding runaway reasoning (2026-06-23)**: qwen3.6:35b is a
  *thinking* model, and the original open-ended `AGENTIC_SCHEDULE_SYSTEM_PROMPT`
  ("you may call tools multiple times … verify assumptions rather than guessing") drove
  it to deliberate the entire Plan-Tool-Reflect-Decide protocol in its **internal
  `<thinking>` trace** before emitting any JSON — an unbounded spiral that blew the
  client's 300 s wall-clock timeout and never produced a schedule (the *single-shot*
  prompt, being direct, terminates in ~29.5 s). Disabling thinking (`think:false`) was
  tried and rejected: it stops the spiral but makes the model emit **malformed JSON**
  (the thinking phase is what gets the structure right) — every cell then tripped the
  integrity violation. The fix keeps the known-good config (think on, streaming) and
  instead makes the prompt **decisive**: it now instructs *"1-2 tool calls is usually
  enough; keep INTERNAL reasoning concise (a few sentences); do not simulate many
  scenarios — think briefly, then act."* This bounds the internal deliberation while
  **preserving genuine tool use** (the agentic action space). Verified live (think on,
  streaming): the PLAN call terminates in **8-33 s** with valid schedule JSON and
  `tool_call` still emitted, `done_reason=stop` — vs >300 s runaway before. The
  `llm-s`↔`llm-a` contrast remains the action space (one call vs the tool-using loop),
  not the inference config.
- **Papers**: Sumers et al. (2024) [CoALA] — tool use, action decomposition;
  Yao et al. (2023) — bounded agent loop with forced answer extraction;
  Rodriguez-Fernandez et al. (2024) — schedule-prompt design for sat ops.
- **Tests**: `tests/test_agentic_scheduler.py` (registration, schedule contract,
  grounding toggle, decision extraction, substrate integrity) — mocked, no live LLM.

---

## Operations Paradigms

Four paradigms (ladder): **autonomous_onboard** (onboard only) → **autonomous_hybrid** (onboard +
ground) → **autonomous_ground** (ground only) → **conventional_ground** (human ground). A
Jetson-based onboard core (subsymbolic/hybrid onboard, AO/AH) sets `env.onboard_compute_active=True`
via `config.onboard_uses_jetson`, adding the ~7 W Jetson-on draw (`power.onboard_compute_w`) to
non-Jetson-compute modes (not compress/detect/send); symbolic onboard (OBC rules) and ground
paradigms carry no overhead.

### Autonomous Onboard — onboard-only primitive

- **File**: `src/core/operations/autonomous_onboard.py`
- **Structure**: Pass-through observation (full real-time state), acts every step, no ground plan /
  no schedule — a single per-step onboard core, closed-loop. `has_onboard_autonomy()=True`.
- **Resolves to**: the onboard core (symbolic→`rule_based_eventsat`, subsymbolic→`subsymbolic_eventsat`).
  Hybrid+AO excluded (no standalone onboard LLM).
- **Configs**: `eventsat_sas_ao_{symb,rl,hrl}` and related matrix cells.
- **Key property**: pure onboard autonomy with the Jetson power overhead; the AO↔AH contrast isolates
  the value of adding a ground plan.

### Autonomous Hybrid — dual-slot (onboard + ground planner)

- **File**: `src/core/operations/autonomous_hybrid.py`
- **Two cores**: the runner runs the **onboard** loop (`resolved_onboard_type`) every step on full
  real-time state, and a **ground-planner** loop (`resolved_ground_planner_type`, the same artifact
  AG uses) at ground passes on the stale view → `set_uplinked_plan`.
- **Arbitration** (`process_action`): during a pass → `communication` (downlink telemetry + uplink
  plan, matching AG so the ground planner is identical across AH/AG); between passes → follow the
  uplinked plan unless the onboard mode is a safety mode (`onboard_override_modes`, default
  `{charging, safe}`) differing from the plan → **override** (counted). No/exhausted plan → onboard.
  `onboard_authoritative=True` makes onboard always win (ablation knob).
- **Metrics**: `onboard_overrides`, `onboard_override_rate` (per-episode, in `paradigm_metrics`).
- **`has_onboard_autonomy()=True`** (Jetson overhead). **Anomaly recovery**: onboard FDIR.
- **Note**: the AH **onboard** slot follows the configured substrate (`a3768bf`,
  per the O framework (morphological_matrix.md)): `hybrid·reactive → llm_eventsat`, `hybrid·agentic → agentic_eventsat`,
  `subsymbolic → subsymbolic_eventsat`. (Before `a3768bf`, `resolved_onboard_type` silently
  substituted the RL policy for hybrid AH cells, so no LLM ever ran in hyre/hyag AH — fixed.)
  The AH **ground-planner** slot uses the real LLM schedule producers — single-shot
  `llm_scheduler_eventsat` (hllm-s) / `llm_single_scheduler_eventsat` (llm-s) and **agentic**
  `agentic_scheduler_eventsat` (hllm-a) / `llm_agentic_scheduler_eventsat` (llm-a). Only the RL
  schedule producer (`subsymbolic_scheduler_eventsat`) and the `_lep_`/`_lec_` *ground* learned
  variants remain pending. The onboard LLM/agentic core itself is also live.

### Autonomous Ground — Phase 3 (renamed from ConventionalGround)

- **File**: `src/core/operations/autonomous_ground.py`
- **Paper basis**:
  - Sellmaier, Uhlig & Schmidhuber (2022), "Spacecraft Operations" [SGJTLF4D]
  - Castano et al. (2022), "Operations for Autonomous Spacecraft" [2IJJ7ILS]
- **Structure**: Filtered observation (stale between passes — only downlinked
  telemetry visible), actions only during ground passes, schedule upload during
  pass, schedule playback between passes.
- **Key property**: Algorithmic scheduler generates optimal schedules **instantly**
  during ground passes. No planning latency, no cognitive constraints. Represents
  the ideal algorithmic ground system — the upper bound for ground-based ops.
- **Contrast with ConventionalGround**: The algorithmic scheduler has zero planning
  delay (schedule used in the same gap it was planned for), vs. the one-pass delay
  that human teams incur in conventional ground operations.
- **Anomaly recovery**: Requires active ground pass to resume operations.
- **Naming convention**: `ag` (autonomous ground) in experiment config IDs.

### Conventional Ground — Phase 3 (human-realistic)

- **File**: `src/core/operations/conventional_ground.py`
- **Paper basis**:
  - Sellmaier, Uhlig & Schmidhuber (2022), "Spacecraft Operations" [SGJTLF4D] —
    ground operations planning cycle, pass-based commanding.
  - ECSS-E-ST-70C Ground Systems and Operations (2008) [CIYT2V68] —
    commanding timelines, planning procedures.
  - Castano et al. (2022), "Operations for Autonomous Spacecraft" [2IJJ7ILS] —
    human vs autonomous ops contrast; planning latency effects.
  - Endsley (1995), "Situation Awareness in Dynamic Systems" [46MUS93H] —
    SA Level-3 degradation model for shift handovers.
- **Real planning workflow** (Sellmaier et al. 2022, ECSS-E-ST-70C):
  - Pass N: Downlink telemetry. Upload schedule S(N-1) planned after previous pass.
  - Between passes N and N+1: Ground team analyses pass-N telemetry and plans S(N).
    This takes the full inter-pass gap (hours for LEO missions).
  - Pass N+1: Upload S(N). Downlink fresh telemetry. Start planning S(N+1).
- **Key consequence — ONE-PASS DELAY**: The schedule executing between passes N and
  N+1 was planned based on telemetry from pass N-1 (two states ago). This is a
  fundamental constraint of conventional ground operations, not a tunable parameter.
- **Three-buffer model** (link-gated uplink, 2026-06-12):
  - `_active_schedule`: Currently being executed by the satellite (uplinked at last pass).
  - `_planned_schedule`: Prepared by the representation after latest telemetry.
  - `_upload_candidate`: Staged at pass start; promoted to `_active_schedule` only
    when the comm link is actually established (`update_ground_knowledge`, i.e. the
    runner's resolved-communication signal — same gate as telemetry downlink). A pass
    shorter than ADCS settling transfers nothing; the candidate returns to
    `_planned_schedule` for the next contact.
- **Planning horizon**: `estimated_gap_steps` = the **following gap** (next-pass end →
  subsequent-pass start) from the environment pass table — the window this schedule
  will actually cover given the one-pass delay. AG/AH-ground get the **next** gap
  (current-pass end → next-pass start). Pass prediction is deterministic FDS
  capability (Sellmaier et al. 2022 §16.4); the previous one-orbit constant (93)
  capped every ground schedule inside 92–764-step real gaps.
- **Cold start**: At the first pass, no prior schedule exists — satellite stays in
  `default_mode` until the second pass provides the first uploadable schedule.
- **pass_upload_done**: Ensures only the first schedule during a multi-step pass is
  stored (humans plan once per pass, not once per step).
- **Anomaly recovery**: Requires active ground pass to resume operations.
- **Naming convention**: `cg` (conventional ground) in experiment config IDs.

---

## Memory

### FixedMemory — All variants (default)

- **File**: `src/core/memory/fixed_memory.py`
- **Used by**: All hand-designed variants and all non-CoALA learned variants.
- **Structure**: Sliding-window history (default depth 100), task queue, resource state,
  constellation state. Fully read-only from the agent's perspective — no write API.
- **Fairness invariant**: All variants that use `FixedMemory` are on equal footing;
  memory cannot be a confound in cross-architecture comparisons.

### WritableMemory — `_lec_` variants only (CoALA learning)

- **File**: `src/core/memory/writable_memory.py`
- **Used by**: Only `behaviour_config.mechanism = "writable_coala"` configs.
- **Wiring (source of truth)**: `ExperimentRunner._create_memory()` constructs the
  `WritableMemory` (for `writable_coala`) or `FixedMemory` (everything else) and the
  decision loops inject it into `DecisionContext.memory`. The representation always uses
  `context.memory`; its own internal `_resolve_memory()` instance is only a fallback for
  unit tests that call `select_action()` directly with no runner. (Earlier the runner
  always injected `FixedMemory`, which silently downgraded `_lec_` to the fixed-memory
  baseline — every write hit the `not hasattr(memory, "write_semantic_rule")` guard in
  `agentic_tools.py` and no-oped. Fixed; regression-tested in `test_orchestration.py`.)
- **Paper basis**: Sumers et al. (2024) [CoALA] §3 — four-memory architecture; semantic and
  episodic stores as the primary learning mechanism for language agents.
- **Extends**: `FixedMemory` (inherits all working/task/resource slots).
- **Writable stores**:
  - **Semantic store** (`_semantic_store`): append-only list of domain rules. Written via
    `write_semantic_rule(rule_text, condition, action, provenance)`. Persists across all
    episodes and the entire run. Read via `recall_semantic(query)`.
  - **Episodic store** (`_episodic_store`): ring-buffer (default 50) of episode trajectory
    summaries. Written via `write_episodic_entry(summary, outcome, episode_id)`. Persists
    across episodes within a run. Read via `recall_episodic(query, last_n)`.
- **Persistence**: JSON file at `memory_config.memory_path`; `save(path)` / `load(path)`.
  Auto-loads on init if the file exists. Enables knowledge accumulation across runs on the
  same server.
- **Reset semantics**: `reset()` clears working/task memory only (inherited from
  FixedMemory). Writable stores deliberately NOT cleared — they accumulate across episodes.
  Use `clear_learned_state()` for a full wipe.
- **Fairness note**: `_lec_` variants intentionally deviate from the FixedMemory invariant.
  These configs are compared against `_hyag_hd_` baselines only, not against symbolic or
  LLM variants. The deviation is documented here and in CLAUDE.md.

---

## Emergence (Behaviour overlay)

Maps to the **Behaviour** overlay ([morphological_matrix.md](morphological_matrix.md)):
`ppo`/`prompt_optimized` = emergent·policy (gated by substrate); `writable_coala` = emergent·memory
(gated by the agentic action space). Mechanism is derived from Behaviour × substrate, not chosen freely.

### PPO Training — `_le_` subsymbolic variants

- **File**: `src/core/behaviour/training_pipeline.py` (PPOTrainer)
- **Mechanism**: `behaviour_config.mechanism = "ppo"`
- **Command**: `uv run autops train configs/experiments/eventsat_sas_rl_ah.yaml`
- **Output**: `data/trained_models/<experiment_id>/policy.pt`

### PromptOptimizer — `_lep_` LLM/agentic variants

- **File**: `src/core/behaviour/prompt_optimizer.py`
- **Mechanism**: `behaviour_config.mechanism = "prompt_optimized"`
- **Paper basis**: Khattab et al. (2023) [DSPy] — programmatic prompt optimization; bootstrap
  few-shot and MIPRO as reference algorithms. Implemented as a minimal in-house bootstrap
  optimizer (no DSPy runtime dependency).
- **Algorithm**:
  1. Load step records from a hand-designed baseline results dir (`steps.json` / `steps.jsonl`).
  2. Select high-utility examples (utility ≥ 0.6, or top-N if insufficient).
  3. Generate `num_candidates` few-shot-augmented system prompt candidates.
  4. Score candidates on 20% held-out split (mock-mode: proxy score; live: LLM accuracy).
  5. Write best prompt to `data/trained_prompts/<experiment_id>/prompt.txt` + `metadata.json`.
- **Command**: `uv run autops train configs/experiments/eventsat_sas_llm_ah.yaml`
- **Source dir**: auto-derived from experiment_id (`_lep_` → `_hd_`), or explicit via
  `--source-dir`.
- **Runtime**: `LLMEventSat` and `AgenticEventSat` load the prompt at `__init__`; fall back
  to the default system prompt with a warning if the file is missing.
- **Why no DSPy**: Keeps the dependency graph minimal. The bootstrap-fewshot approach is
  sufficient for the experimental goals; DSPy's full optimizer suite is overkill at this
  stage and would add a heavy dependency.

### WritableCoALA — `_lec_` agentic variants (online learning)

- **Mechanism**: `behaviour_config.mechanism = "writable_coala"`
- **Pre-training**: None. Memory accretion happens online at run-time.
- **Command**: `uv run autops train configs/experiments/eventsat_sas_agentic_ah.yaml`
  (prints guidance; no artifact written)
- **Runtime flow**: `AgenticEventSat.__init__` detects `writable_coala` and injects the
  `memory_write_rule` + `memory_write_episode` tools into the tool schema and CoALA memory
  instructions into the system prompt. The `WritableMemory` itself is built by
  `ExperimentRunner._create_memory()` and supplied via `DecisionContext.memory` (see the
  WritableMemory "Wiring" note above), so the LLM's write-tool calls reach a real writable
  store during the Plan-Tool-Reflect-Decide loop.
- **Why `agentic_eventsat` only**: emergent·memory is gated by the **agentic action space** —
  writing to a store is itself an internal action (CoALA's *learning* action), which needs the
  tool-call loop. `llm_eventsat` (reactive, single-shot) has no action with which to issue a
  write. The gate is action space, not substrate.

---

## Metrics

- **File**: src/eventsat/metrics.py
- **Canonical EventSat metrics measured**: M-01 mission utility, M-02 mean AoI,
  M-03 peak AoI, M-04 autonomous recovery efficiency, M-05 safety-override rate,
  M-06 resource efficiency, M-07 decision latency, M-08 explainability coverage,
  M-09 cross-episode robustness CV, M-11 downlink efficiency, M-12 value of
  information, M-13 constraint-violation rate, and M-14 commanding effort.
  M-10 scale efficiency remains reserved for the future multi-satellite scenario.
- **VoI implementation**: EventSat tracks raw-equivalent captured value and
  raw-equivalent delivered value through the compression/OBC/downlink pipeline;
  value_of_information = downlink_raw_equivalent_mb / total_raw_captured_mb.
- **AoI implementation**: mean_aoi_s and peak_aoi_s are computed from
  per-step fresh downlink delivery; each step_downlinked_mb > 0 resets age.
- **Resource-efficiency implementation**: resource_efficiency = utility / total_energy_consumed_wh; this is utility per Wh consumed, not normalized by an episode energy budget.

- **Recovery implementation**: robustness_mean_recovery_steps counts from
  anomaly onset until the anomaly is cleared and the spacecraft is no longer in
  safe mode; incomplete recoveries are horizon-censored.
- **Constraint and command ledgers**: constraint_violation_rate counts
  environment-clamped invalid commands, excluding anomaly-forced safe mode;
  commanding_effort counts requested-mode changes plus weighted manual
  interventions per mission-day.
- **Cross-episode**: robustness_cv, the coefficient of variation of utility.
- **AH-specific** (paradigm metrics, per episode): onboard_overrides and
  onboard_override_rate, the between-pass steps where the onboard core overrode
  the uplinked plan.

---

## Layer Mapping (Bhati 2026)

Bhati (2026) [Z5TF79HY] proposes a six-layer reference architecture for **agentic
software engineering** systems. The autops framework is positioned as a parallel
reference architecture in a **sibling domain** (autonomous satellite operations).
The mapping below tags each component already documented above with its layer in
Bhati's stack — it is illustrative, not a structural adoption. See
[morphological_matrix.md](morphological_matrix.md)
for the framing.

| Component | File / module | Bhati layer | Existing paper basis | Cross-domain note |
| --- | --- | --- | --- | --- |
| SingleAgentSystem (SAS) | `src/core/organization/single_agent_system.py` | **L4** Orchestration | Kim et al. 2025 | Single cognitive locus — analogue of single-agent loops (e.g.\ Claude Code) |
| CentralizedMAS | `src/core/organization/centralized_mas.py` | **L4** Orchestration | Kim et al. 2025 | Role-specialized analogue of MetaGPT / ChatDev orchestrators |
| IndependentMAS (IMAS) | `src/core/organization/independent_mas.py` | **L4** Orchestration | Kim et al. 2025 | Per-satellite agents, C = ∅; runnable on Flamingo N≥3 |
| DecentralizedMAS (DMAS) | `src/core/organization/decentralized_mas.py` | **L4** Orchestration | Kim et al. 2025 | Peer all-to-all consensus, C = full; runnable on Flamingo N≥3 |
| HybridMAS (HMAS) | `src/core/organization/hybrid_mas.py` | **L4** Orchestration | Kim et al. 2025 | Clustered (coordinate within, independent across); tunable SAS↔IMAS midpoint; runnable on Flamingo N≥3 |
| SDA loop | `src/core/decision_procedure/` (SDA) | **L1** Reasoning | classical control loop | Baseline reactive scaffolding |
| Rule-Based / Schedule-Based EventSat | `src/eventsat/symbolic.py`, `src/eventsat/schedule_symbolic.py` | **L1** (reasoning interface only) | hand-designed (Brooks 1991 reactive) | **No L0 substrate** — pure symbolic |
| Conventional Schedule EventSat | `src/eventsat/conventional.py` | **L1** | Sellmaier et al. 2022 | Human-realistic ground baseline; no L0 |
| Subsymbolic EventSat | `src/eventsat/rl.py` | **L0** (policy net) + **L1** | Wang et al. 2022 (DRL) | L0 is the RL policy network rather than an LLM |
| LLM EventSat | `src/eventsat/llm.py`, `src/core/llm_client.py` | **L0** (LLM) + **L1** | Rodriguez-Fernandez et al. 2024 | L0 substrate: Ollama / OpenAI backend |
| Agentic EventSat | `src/eventsat/agentic.py` | **L0** (LLM) + **L1** (CoALA) | Sumers et al. 2024 (CoALA) | Direct sibling of Bhati's L1 cognitive scaffolding |
| FixedMemory | `src/core/memory/fixed_memory.py` | **L1** Memory | fairness invariant | Read-only short/long-term state |
| WritableMemory | `src/core/memory/writable_memory.py` | **L1** Memory | Sumers et al. 2024 (CoALA §3) | Writable semantic + episodic stores — closest analogue of Bhati L1 "memory files" |
| BehaviourController | `src/core/behaviour/controller.py` | **L1** Self-reflection / learning controller | `@register` factory | Selects hand-designed vs learned variant |
| PPOTrainer | `src/core/behaviour/training_pipeline.py` | **L1** (learned reasoning) | PPO (Schulman et al. 2017) | RL-based learning loop |
| PromptOptimizer | `src/core/behaviour/prompt_optimizer.py` | **L1** (self-improvement) | DSPy / TextGrad family | Sibling of Bhati L1 self-critique |
| WritableCoALA | `_lec_` configs | **L1** (online learning) | Sumers et al. 2024 | Online memory write — closest match to Bhati's "memory files" |
| Scenario actions/tools | `src/eventsat/agentic_tools.py` and scenario action dictionaries | **L2** Agent–Computer Interface | (no external paper basis) | YAML-serializable, stateless action definitions exposed to the cognitive layer |
| Satellite environment | `src/core/satellite_env.py`, `src/eventsat/`, `src/flamingo/`, `src/orbital/` | **L3** Tools & Environment | Orekit; mission constraints | Analogue of filesystem + test runners; deterministic physics layer |
| Autonomous Hybrid | `src/core/operations/autonomous_hybrid.py` | **L5** Governance & Safety | Rossi et al. 2023 | Onboard FDIR; no ground gate; closest to "auto" autonomy level |
| Autonomous Ground | `src/core/operations/autonomous_ground.py` | **L5** Governance & Safety | Sellmaier et al. 2022 | Ground-pass gating — analogue of human approval at high-impact actions |
| Conventional Ground | `src/core/operations/conventional_ground.py` | **L5** Governance & Safety | ECSS standards | Human-realistic ground operator; analogue of traditional SDLC supervision |
| Env-enforced safe mode | `src/eventsat/env.py` | **L5** (cross-cutting) | mission-safety invariant | Feedback flows downward (Bhati Fig. 3): safety overrides cross all representations |

**The L0 gap.** Pure-symbolic variants (`symb`) have no L0 substrate in Bhati's sense
— there is no foundation model. They sit at L1 directly. This asymmetry is
load-bearing for the fair-comparison invariant: holding L2–L5 fixed across the
matrix, the variation in L0 (none / RL policy / LLM) isolates the cognitive-paradigm
effect (Brooks 1991; Colelough & Regli 2025). The asymmetry is explicit in
[`morphological_matrix.md`](morphological_matrix.md).

---

## Cross-Cutting Design Decisions

### Decision Procedure × Representation Interaction Model

The benchmark uses a fixed SDA driver. SDA produces a `DecisionContext` with the
encoded state and passes it to the selected representation's `select_action()`.
Architectural variation now lives in the representation, organization, and
operations-paradigm layers; agentic planning/tool use is internal to the
representation, not an alternate outer decision loop.

### Operations Paradigm × Inference Location

The operations paradigm controls three aspects of the decision pipeline
(Sellmaier et al. 2022 [SGJTLF4D] §16.4; Rossi et al. 2023 [5EG3E3BP]):

1. **Observation filtering**: What data the agent sees (real-time vs stale from last
   downlink).
2. **Inference timing**: When representation inference runs. Ground operations prepare
   plans **between passes** using last-received telemetry (Rossi et al. 2023: tactical
   planning cycle between contacts; Sellmaier et al. 2022: offline preparation between
   passes). The pass window is for telemetry downlink and plan uplink, not computation.
3. **Action gating**: When actions are executed (every step vs schedule playback).

| Paradigm | Observation | Ground Computation | Action Execution | Literature Basis |
|----------|-------------|--------------------|-----------------|------------------|
| **AH** | Onboard: real-time. Ground: stale between passes, fresh during pass | Between passes (ground); onboard rules/DNN every step | Every step (onboard immediate) | Dual onboard/ground per Rossi et al. 2023 [5EG3E3BP] |
| **AG** | Stale between passes; fresh during pass | Between passes using last-received telemetry | Schedule playback between passes | Rossi et al. 2023 [5EG3E3BP]: "tactical level planning... incorporating data collected in prior downlinks" |
| **CG** | Same as AG | Between passes, one-pass delay (plan from pass N uploaded at pass N+1) | Delayed schedule (Sellmaier et al. 2022 [SGJTLF4D]) | ECSS-E-ST-70C [CIYT2V68]: commanding timelines |

**Inference location for LLM/agentic representations**:

- **AH**: In the EventSat scenario, LLM inference runs exclusively on ground
  infrastructure — no LLM is deployed onboard, only small DNNs are feasible for
  onboard processing (Oliver et al. 2025 [8KDZ5Z53]: 50μs Jetson inference for RL).
  In the AH paradigm, ground prepares detailed analysis and plans between passes
  using last-received telemetry, which are uplinked during the next contact. Between
  passes, onboard autonomy (rules, small DNNs) executes with real-time state and can
  override ground plans for fault detection or opportunistic events. The 122B-parameter
  Qwen model represents ground-side reasoning capability, not an onboard deployment.
- **AG/CG**: LLM inference runs on ground infrastructure (cf. Rossi et al. 2023:
  "Prediction Engine" on Kubernetes cloud). Ground prepares plans between passes using
  last-received telemetry. The satellite executes the pre-uploaded schedule between
  passes. `should_allow_inference()` gates this in the experiment runner.

**Known simulation simplification**: The current AH implementation gives the LLM
real-time state every step (`filter_observation` returns full state). This is an
optimistic simplification — in reality, the ground LLM would only see stale telemetry
between passes. A future improvement would add dual observation paths (onboard=real-time,
ground=stale) to the AH paradigm.

**Latency assumption**: LLM call latency (~13s for qwen3.5:122b) is NOT modeled as
communication delay. For AG/CG, ground compute time ≪ inter-pass interval (~90 min
for LEO). Whether to also compute during passes using freshly downlinked telemetry
is a separate research question on optimal uplink timing.

### Representation × Operations Paradigm Pairing

> **Representation vs Operations Paradigm**: The representation dimension defines the
> *methods* used for decision-making (rules, DNN, LLM, or any hybrid combination —
> that is what makes it "hybrid"). The operations paradigm defines *where and when*
> those methods execute. Ground operations (AH-ground, AG, CG) prepare plans **between
> passes** using last-received telemetry (Rossi et al. 2023 [5EG3E3BP], Sellmaier et al.
> 2022 [SGJTLF4D] §16.4), uplinked during the next contact window. In AH, onboard
> autonomy (rules, small DNNs) additionally operates with real-time state every step
> and can override ground plans on fault detection or opportunistic events.

| Repr Type | AH (onboard) | AH (ground) | AG | CG |
|-----------|--------------|-------------|----|----|
| rule_based | Rules every step | Rules between passes → uplinked plan | Rules between passes → schedule | Rules between passes → delayed schedule |
| schedule_based | N/A | N/A | Greedy planner between passes → schedule | N/A |
| conventional_schedule | N/A | N/A | N/A | Human-modeled planner between passes → delayed schedule |
| llm_eventsat (hllm-s) | LLM every step (a3768bf; Jetson-class) | LLM between passes → uplinked plan | LLM between passes → schedule | LLM between passes → delayed schedule |
| agentic_eventsat (hllm-a) | Agentic loop every step (a3768bf; Jetson-class) | Agentic between passes → uplinked plan | Agentic between passes → schedule | Agentic between passes → delayed schedule |
| subsymbolic | DNN every step | DNN between passes → uplinked plan | DNN between passes → schedule | DNN between passes → delayed schedule |

> **Ground columns** (AH-ground / AG / CG) are produced by the **`*_scheduler_eventsat`
> schedule producers**, not the per-step cores: `schedule_based_eventsat` (symb),
> `llm_scheduler_eventsat` / `llm_single_scheduler_eventsat` (hllm-s / llm-s),
> `agentic_scheduler_eventsat` / `llm_agentic_scheduler_eventsat` (hllm-a / llm-a),
> `subsymbolic_scheduler_eventsat` (RL placeholder). The per-step `llm_eventsat` /
> `agentic_eventsat` run only in the AH **onboard** slot. The cell parens above name
> the framework cell, not the concrete ground class.

| Repr Type | Typical Inference Time | Onboard Feasible? |
|-----------|----------------------|-------------------|
| rule_based | μs | Yes |
| schedule_based / conventional_schedule | ms–minutes | Ground only (planning) |
| llm_eventsat | ~13s/call | Ground only (LLM) |
| agentic_eventsat | ~13–40s/decision | Ground only (LLM) |
| subsymbolic | 50μs (Oliver et al. 2025 [8KDZ5Z53]) | Yes (small DNN) |

### Observability and Debugging

- **LLM cache** (`data/llm_cache/<model>/`): Each LLM call is cached with
  `{model, temperature, system_prompt, user_prompt, response, timestamp}`.
  Prompts are stored alongside responses for full debugging.
- **Decision trace** (`data/results/<exp_id>/decisions_ep<N>.jsonl`): When
  `log_level: DEBUG`, a JSONL file records per-step decisions:
  `{step, mode, rationale, inference, latency_s}`.
- **Experiment log** (`data/results/<exp_id>/experiment.log`): Full timestamped
  log including anomaly injection/clearance, component initialization, and
  (at DEBUG level) LLM prompt summaries and responses.

---

## Experiment Configurations

Config IDs follow `eventsat_sas_<paradigm>_<rep>` (morphological_matrix.md §5):
`paradigm` ∈ {conventional, ag, ao, ah}; `rep` ∈ the 7 cells
{symb, rl, hrl, llm-s, llm-a, hllm-s, hllm-a}; `ah` names both cores onboard-first
(`eventsat_sas_ah_<onboard>_<ground>`). The full EventSat·SAS matrix is **32
experiments** (conventional 1 + ag 7 + ao 3 + ah 21). `decision_procedure` and
`behaviour` are held fixed (not framework components).

**Shipped so far** — the framework-valid cells with runnable cores (symb · rl ·
hllm-s · hllm-a):

- `eventsat_sas_conventional_symb`
- `eventsat_sas_ag_{symb, rl, hllm-s, hllm-a}`
- `eventsat_sas_ao_{symb, rl}`
- `eventsat_sas_ah_symb_symb`, `eventsat_sas_ah_rl_rl` (single-rep AH)
- `eventsat_sas_ah_rl_symb`, `eventsat_sas_ah_symb_hllm-a` (dual-core AH examples)

**Dual-core AH** is supported: a config names two cores via nested `onboard:` /
`ground:` blocks, each with its own `representation` + `representation_config`
(onboard ∈ {symb, rl, hrl}, ground ∈ the 7 cells). Resolution keys off the
per-core substrate (`ExperimentConfig.resolved_onboard_type` /
`resolved_ground_planner_type`); the runner feeds each core its own config. Omitting
both blocks keeps the single-`representation` behaviour (AO/AG/CG, single-rep AH).

**Pending** (later increments): generating all 21 `ah_<onboard>_<ground>` pairs
(config generator); the `hrl` / `llm-s` / `llm-a` real cores (currently documented
placeholders, `placeholders.py`); and the learned ground LLM schedulers.
Learned behaviour is wired via `behaviour_config` (`ppo` for RL, `writable_coala`
for agentic online learning), not separate cells.

### Comparison axes

- **Ops paradigm ladder** (same rep): conventional vs AG vs AO vs AH → human-ground vs autonomous-ground vs onboard-only vs onboard+ground
- **Value of a ground plan** (same onboard core): AO vs AH → does adding the ground plan help beyond pure onboard
- **Onboard-override effect** (same rep, shared ground planner): AH vs AG → what the onboard per-step override buys (`onboard_overrides`)
- **Human vs algorithmic** (symbolic ground): conventional vs AG → effect of cognitive constraints (Endsley 1995 SA degradation)
- **Representation comparison** (same paradigm): symb vs rl vs hllm-s vs hllm-a → cognitive-substrate effectiveness
- **Single-shot vs agentic** (LLM cells): hllm-s vs hllm-a → does multi-step tool use improve decisions?
- **Hand-designed vs learned**: `ppo` (RL) / `writable_coala` (agentic online learning) vs their fixed-baseline siblings
- **Fixed vs writable memory** (agentic only): hllm-a fixed-memory vs `writable_coala` → does CoALA memory accretion improve decisions across episodes?
