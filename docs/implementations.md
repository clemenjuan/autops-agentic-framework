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
> `*_scheduler_eventsat` — are unchanged). Still pending: real `hrl`/`llm-a` cores (currently
> documented placeholders, `placeholder_cells.py`), the learned ground LLM schedulers, and
> dual-core AH with *independent* onboard/ground representations (the 21 `ah_<onboard>_<ground>`
> pairs). The component descriptions below document the **current code** — map them to the
> framework via `morphological_matrix.md` §2.

---

## Agent Organizations

Formal definition: an agent system **S = (A, E, C, Ω)** where A = agents, E = environment, C = communication topology, Ω = orchestration policy (Kim et al. 2025 [FVFQ73RF]).

**Empirical prediction for all Organization experiments** (Kim et al. 2025, 180 configs): satellite mode selection is sequential constraint satisfaction → centralized org predicted to outperform distributed. Capability saturation (β̂=−0.404) means multi-agent overhead negates gains once single-agent baseline > ~45%.

Full taxonomy: Kim et al. (2025) [FVFQ73RF] "Towards a Science of Scaling Agent Systems".

### SingleAgentSystem (SAS)

- **File**: `src/agent_organization/single_agent_system.py`
- **Paper basis**: Kim et al. (2025) [FVFQ73RF] Single-Agent System (SAS) — |A|=1, single reasoning locus, C undefined, Ω direct, complexity O(k).
- **Structure**: One central agent receives full constellation observation and selects actions for all satellites. No inter-agent communication. Zero coordination overhead.
- **Key property**: Maximum context integration (unified memory stream, full prior-history access). Upper bound for context-quality; lower bound for parallelism.
- **Configs**: the EventSat·SAS matrix is **32 experiments** — conventional 1 + ag 7 + ao 3 + ah 21 (morphological_matrix.md §4); `decision_procedure` is held fixed, not a multiplied axis.

### CentralizedMAS

- **File**: `src/agent_organization/centralized_mas.py`
- **Paper basis**: Kim et al. (2025) [FVFQ73RF] Centralized MAS — orchestrator routes to sub-agents, C = {(a_orch, aᵢ) : ∀i}, Ω = hierarchical, complexity O(rnk). Also: ECSS-E-ST-70-11C autonomy levels (mission management layer vs onboard autonomy layer).
- **Structure**: A = {mission_manager, sat_agent_0}; C = star (manager→local, unidirectional); Ω = hierarchical (local agent action is used; manager action stored as directive for next step).
- **EventSat design decisions**:
  - `distribute_observation`: Both agents receive the full environment observation (single satellite, no meaningful state partitioning). `sat_agent_0` additionally receives the manager's previous-step action as a `messages` entry (directive context).
  - `collect_actions`: Manager action stored as `_last_manager_directive`; local agent action returned as environment action. Fallback to manager action if no local agent output.
  - Manager directive carries over step-to-step via `_last_manager_directive`; reset to `None` in `initialize()`.
  - Latency: `ExperimentRunner` accumulates manager + local agent latencies as the total step latency (sequential execution, both contribute to decision overhead).
- **Scope**: the MAS organisations (cmas/imas/dmas/hmas) belong to the **future multi-satellite scenario** and are not exercised by the EventSat benchmark (morphological_matrix.md §1). The code below is the single-satellite wiring kept for that future work; no EventSat configs use it.

### DecentralizedMAS — Implemented (Flamingo N≥3)

- **File**: `src/agent_organization/decentralized_mas.py`
- **Paper basis**: Kim et al. (2025) [FVFQ73RF] Decentralized MAS — all-to-all peer exchange, C = {(aᵢ, aⱼ) : ∀i,j, i≠j}, Ω = consensus, complexity O(dnk).
- **Structure**: One peer agent per satellite, no manager. All-to-all exchange: every peer shares what it sees, so each ends up with the same global information.
- **Flamingo design decisions**:
  - `distribute_observation`: every peer receives the full observation (the decentralized counterpart of SAS's global view), plus the other peers' previous-step proposals as `messages` (the all-to-all channel C).
  - `collect_actions`: peers running the shared deterministic protocol on identical information converge on the same deconflicted plan; the **consensus** (plurality, ties by agent index) is returned. So DMAS deconflicts like SAS/CMAS and — unlike IMAS — wastes nothing.
  - `get_metrics`: surfaces the coordination cost — `coordination_messages = n·(n-1)` per round (6 at N=3) and `consensus_rounds`. The runner threads this into the Flamingo metrics, so the cost side of the axis is measured.
  - **Outcome vs cost**: with the capable global `rule_based_flamingo`, DMAS matches SAS/CMAS mission utility (validated: utility 660, duplicate rate 0 under the contended scenario) while paying a strictly higher message cost. A decentralized org only loses *outcome* when consensus fails (Kim et al. 17.2× error amplification), which a single deterministic round does not trigger.
- **Status**: Runnable at N≥3 (`configs/experiments/flamingo_dmas_ag_symb.yaml`), all-to-all topology. Ring/mesh/visibility-limited topology ablations are future work. Degenerate at N=1.

### IndependentMAS — Implemented (Flamingo N≥3)

- **File**: `src/agent_organization/independent_mas.py`
- **Paper basis**: Kim et al. (2025) [FVFQ73RF] Independent MAS — C = ∅, no inter-agent coordination.
- **Structure**: A = {sat_agent_0 … sat_agent_{n−1}}, one agent per satellite; C = ∅; Ω = independent (no consensus, no manager).
- **Flamingo design decisions**:
  - `distribute_observation`: agent `sat_agent_i` is mapped by index to the i-th satellite and receives a **local view** containing only that satellite's state and only that satellite's visible tasks — it cannot see what the others see or intend.
  - `collect_actions`: per-satellite actions are merged **verbatim, without deconfliction**, so independent agents that pick the same RSO reach the environment as duplicate observations (the coordination cost the organisation axis measures).
  - Contention is supplied by the scenario, not the org: `configs/scenarios/flamingo.yaml` sets `satellite_phase_shift: 0` so the constellation shares visibility windows and the agents must compete. Validated: under that scenario SAS/CMAS keep duplicate rate at 0 while IMAS wastes ≈⅔ of attempts and loses utility/coverage.
- **Status**: Runnable at N≥3 (`configs/experiments/flamingo_imas_ag_symb.yaml`). Degenerate at N=1 (equivalent to SAS, no coordination overhead).

### HybridMAS — Placeholder (deferred to N≥3)

- **File**: `src/agent_organization/hybrid_mas.py`
- **Paper basis**: Kim et al. (2025) [FVFQ73RF] Hybrid MAS — heterogeneous mixed topology combining star + all-to-all + independent sub-topologies.
- **Status**: Stub (`NotImplementedError`). Reserved for complex multi-cluster constellations.

---

## Decision Loops

### SDA (Sense-Decide-Act) — Phase 2 baseline

- **File**: `src/decision_procedure/sda_loop.py`
- **Paper basis**: Classic reactive agent pattern (sense-decide-act cycle)
- **Structure**: Single-pass — encode observation via representation, select action, return.
  No iteration, reflection, or memory interaction.
- **Memory**: Ignored. `process()` returns `(action, memory)` with memory passed through unchanged.
- **Metrics**: `decision_latency_s`, `total_decisions`, `has_rationale`
- **Significance**: Simplest possible decision loop — lower bound for decision overhead
  and the baseline for all loop comparisons.

### OODA (Observe-Orient-Decide-Act) — Phase 3

- **File**: `src/decision_procedure/ooda_loop.py`
- **Paper basis**:
  - Miller, Hasbrouck & Udrea (2021), "Development of Human-Machine Collaborative Systems
    Using OODA Loops", ASCEND 2021. DOI: 10.2514/6.2021-4092
  - Hartmann et al. (2024), "METIS: An AI Assistant Enabling Autonomous Spacecraft
    Operations", IEEE Aerospace Conference.
  - Richards (2020), "Boyd's OODA Loop", Necesse 5(1):142-165.
- **Structure**: Four-phase loop with feedback (Boyd, 1987):
  1. **Observe**: Encode observation + classify operational situation into regimes
     (cf. METIS Monitoring Agent's 7 telemetry categories).
  2. **Orient**: Core differentiator — simplified Case-Based Reasoning
     (Retrieve→Reuse→Revise→Retain, cf. METIS Reasoning Agent) + trend analysis +
     urgency scoring. Synthesizes Boyd's "cultural traditions" (mission rules),
     "genetic heritage" (physics constraints), "new information" (current telemetry),
     and "previous experience" (memory history) through analysis & synthesis.
  3. **Decide**: Pass orient-enriched state to representation's `select_action()`.
  4. **Act**: Execute action + store situation+action+outcome in memory for future
     CBR retrieval. Set attention guidance for next Observe (Boyd's implicit
     guidance & control feedback loop).
- **Memory**: Actively reads `memory.query("history")` in Orient; writes constellation
  state and orient assessment back. Enables within-episode learning via CBR Retain.
- **Feedback loops** (Boyd):
  - Orient → Observe: attention guidance stored in memory for next cycle
  - Orient → Act: urgency bypass when situation is critical
- **Metrics**: `decision_latency_s` (total, gamma-distributed TPM per Miller et al.),
  `observe_latency_s`, `orient_latency_s`, `decide_latency_s`,
  `orient_iterations`, `orient_urgency`, `orient_cases_retrieved`,
  `total_decisions`, `has_rationale`
- **DecisionContext**: Orient enrichments are passed to representations via
  `DecisionContext.enrichments` (not merged into state dict). This decouples
  the loop's situation assessment from the raw observation.

### ReAct (Reason-Act-Observe) — Phase 3

- **File**: `src/decision_procedure/react_loop.py`
- **Paper basis**:
  - Yao et al. (2023), "ReAct: Synergizing Reasoning and Acting in Language Models",
    ICLR 2023. [R8AVHEAP / 6ATE8J8S]
  - Li (2025), "AI Agents for Satellite Operations", arXiv. [UAA3GIVK]
- **Structure**: Iterative Thought-Action-Observation cycle (Yao et al. §3):
  1. **Thought**: `representation.reason(state, memory)` produces a structured
     reasoning trace explaining the decision factors at play. For symbolic
     representations this is a rule evaluation trace; for future LLM representations
     it is chain-of-thought text.
  2. **Action**: `representation.select_action(context)` proposes an action.
     The reasoning trace and any prior violations are passed in `enrichments`
     so the representation can revise its decision.
  3. **Observation**: Grounding checks validate the proposed action against
     operational constraints (battery feasibility, pass-window timing). If
     violations are found they are fed back into the next iteration. This is
     Yao et al.'s key insight: environmental feedback in the reasoning loop.
- **Termination**: Stops when the action passes all checks (converged=True) or
  `max_iterations` is reached. On non-convergence, falls back to `charging`.
- **Key differences from other loops**:
  - vs SDA: Adds explicit reasoning trace and iterative refinement.
  - vs OODA: OODA has a fixed 4-phase structure with Orient enriching a single
    decision; ReAct iterates until constraints are satisfied (convergence-based).
- **Grounding checks** (configurable, default: both enabled):
  - `battery_feasibility`: Energy-intensive modes (observe, compress, detect,
    send, communicate) require SoC ≥ 0.30. Fallback to charging prevents
    battery-depleting actions when the satellite cannot sustain them.
  - `pass_window_timing`: `communication` mode requires `ground_pass_active=True`.
    Prevents uplink/downlink commands from being issued between passes.
- **reason() method**: Optional method on `Representation` base (default no-op,
  backward compatible). Symbolic representations override to return structured
  reasoning traces. Future LLM representations will return chain-of-thought.
- **Memory**: Passed through unchanged (like SDA). Future LLM representations
  may update memory with reasoning traces.
- **Metrics**: `decision_latency_s`, `reasoning_depth` (total thought steps),
  `iterations` (cycles to convergence), `grounding_violations` (total violations),
  `converged` (1.0 = passed grounding, 0.0 = fallback), `has_rationale`,
  `total_decisions`
- **DecisionContext enrichments** for ReAct:
  - `reasoning_trace`: List of thought step dicts from all iterations
  - `iteration`: Current iteration number (0-indexed)
  - `grounding_violations`: List of violation dicts from previous iterations
- **Note on naming**: Deployed frontier systems (Claude, GPT-4o, Gemini tools API)
  standardize on ReAct-style reason-act-observe cycles in their orchestration layers.
  CoALA (Sumers et al. 2024) is a higher-level architecture blueprint that subsumes
  ReAct; it is implemented as the hybrid-**agentic** action-space flavor
  (`agentic_eventsat`) — an action-space property, not a separate decision procedure
  or representation substrate.

---

## DecisionContext Interface — Phase 3

- **File**: `src/decision_procedure/context.py`
- **Purpose**: Structured wrapper between decision loops and representations.
  Every loop produces a `DecisionContext`; every representation consumes one.
- **Fields**:
  - `state: Dict[str, Any]` — raw encoded observation (loop-agnostic)
  - `loop_type: str` — producing loop identifier ("sda", "ooda", "react", ...)
  - `memory: Any` — agent memory reference
  - `enrichments: Dict[str, Any]` — loop-specific data (orient assessment, reasoning trace, LLM prompts, tensors)
  - `loop_metadata: Dict[str, Any]` — operational metadata (latency, iterations)
- **Design rationale**: Decouples loop enrichments from raw state so representations
  can opt into loop-specific data without breaking the base interface. Enables the
  morphological matrix to produce different decisions when the decision loop varies.
- **SDA**: `DecisionContext(state=encoded, loop_type="sda", enrichments={})`
- **OODA**: `enrichments={situation_class, urgency, battery_trend, battery_trending_down,
  data_pressure, competing_priorities, similar_case, attention_guidance, anomaly_is_new,
  sunlight_transition, entered_eclipse}`
- **ReAct**: `enrichments={reasoning_trace, iteration, grounding_violations}`
- **Future LLM/RL**: `enrichments={prompt, reasoning_steps, tensor_obs, policy_logits, ...}`

---

## Representations

> The registered name below (the "Registered as" field) is **resolved** at runtime from
> `representation × action_space × operations_paradigm` (`ExperimentConfig.resolved_representation_type`);
> configs no longer set `representation_config.type` except as an explicit override (e.g. `_algobase`).

### Rule-Based EventSat — Phase 2 baseline + Phase 3 OODA-aware + ReAct-capable

- **File**: `src/representation/rule_based_eventsat.py`
- **Registered as**: `rule_based_eventsat`
- **Paper basis**: Hand-designed priority rule chain (domain engineering)
- **Structure**: Priority rules across categories (R1–R7 + default).
  `encode_observation()` extracts flat state dict from environment.
  `select_action(context)` evaluates rules in priority order, returns first match.
- **OODA-aware modifications** (active when `context.loop_type == "ooda"`):
  - **R2e-OODA**: Eclipse preparation — charge at SoC < 0.60 when Orient detects
    sun→eclipse transition. Boyd's "genetic heritage" (physics constraint awareness).
  - **R2-OODA**: Proactive charging — charge at SoC < 0.45 when Orient detects
    `battery_trending_down` + `urgency > 0`. Orient's trend analysis catches
    decline before SDA's fixed 0.35 threshold.
  - **R3-OODA**: Urgency-based pass prioritization — communicate during pass when
    `urgency > 0.5` even without OBC data (pre-emptive HK downlink/command uplink).
  - **R5-OODA**: Observation batching — when Orient confirms battery stable/rising,
    no imminent pass, and low urgency, defer compression to allow 2 observations
    before compressing. Reduces ADCS mode-switching overhead vs SDA's strict 1:1
    observe→compress interleave. Boyd's "analysis & synthesis" enables more
    efficient pipeline utilization.
  - **R6-OODA**: Orient-confident observation — observe at SoC > 0.50 (vs SDA's
    0.60) when Orient's trend analysis confirms battery is stable/rising.
  - **SDA fallback**: If `loop_type != "ooda"` or enrichments empty, behaves
    identically to Phase 2 baseline.
- **reason() override** (for ReAct): Evaluates all decision factors (battery,
  health, pass, pipeline backlog, pipeline pressure) and returns a structured
  trace with which rules would match and why. The ReAct loop uses this trace
  as the Thought step and includes it in the next Action's enrichments.
- **Rationale**: Always provides human-readable rationale indicating which rule
  fired and whether OODA enrichments influenced the decision (explainability metric).
- **Operations paradigm**: Paired with `autonomous_hybrid`.

### Schedule-Based EventSat — Phase 2 baseline + Phase 3 OODA-aware + ReAct-capable

- **File**: `src/representation/schedule_based_eventsat.py`
- **Registered as**: `schedule_based_eventsat`
- **Paper basis**: Traditional ground operations scheduling — greedy cyclic
  battery-aware planner with power model from PDR Table 6.2.
- **Structure**: During ground passes, generates time-tagged command sequences
  via greedy planning. Between passes, commands play back from schedule.
  Telemetry-first sequencing: downlink HK → fresh data → generate schedule.
- **OODA-aware modifications** (active when `context.loop_type == "ooda"`):
  - **Urgency-aware reserve**: When `urgency > 0.6`, charge reserve fraction is
    halved (min 6%) to front-load productive operations.
- **reason() override** (for ReAct): Returns a schedule planning intent summary
  (pass state, battery SoC, gap length, pipeline backlog, downlink opportunity).
  Used as the Thought step in the ReAct cycle.
- **Operations paradigm**: Paired with `autonomous_ground`.

### Conventional Schedule EventSat — Phase 3 (human-realistic)

- **File**: `src/representation/conventional_schedule_eventsat.py`
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

- **File**: `src/representation/placeholder_schedulers.py`
- **Registered as**: `subsymbolic_scheduler_eventsat`, `llm_scheduler_eventsat`,
  `agentic_scheduler_eventsat`
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
- **What they do (PLACEHOLDER)**: Each subclasses `ScheduleBasedEventSat` and emits
  a real schedule via the **symbolic greedy planner** — NOT the family's actual
  policy. `is_placeholder = True` is surfaced in
  `results["experiment_statistics"]["metadata"]["representation_is_placeholder"]`
  so analysis can exclude these cells from headline comparisons.
- **Extension point (future research, "P3 — learned scheduling")**: replace each
  placeholder with a real schedule producer — a PPO-trained scheduler
  (`subsymbolic_scheduler_eventsat`, the deferred Phase 4.e), an LLM-generated
  schedule (`llm_scheduler_eventsat`), and a tool-using agentic planner
  (`agentic_scheduler_eventsat`) — to make the RL-vs-LLM scheduling comparison.
- **Guard**: `config_loader` now **errors** if a ground paradigm is paired with a
  non-schedule-producing representation type, so the degenerate cell cannot be
  recreated silently. The non-symbolic ground cells use these placeholder types.
  Note: the subsymbolic ground cells no longer require `torch`
  (they delegate to the symbolic planner).

### Subsymbolic EventSat — Phase 4b (RL learned)

- **File**: `src/representation/subsymbolic_eventsat.py`
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
- **reason()**: Returns per-head action probabilities as structured Think steps for ReAct loop
- **update()**: Delegates to PPOTrainer (called from experiment_runner post-episode in learned mode)
- **Orthogonality**: Works with all 3 decision procedures (SDA/OODA/ReAct, held fixed) and all 4 ops paradigms (ao/ah/ag/conventional)
- **Training script**: `scripts/train_subsymbolic.py`
- **Gymnasium wrapper**: `src/environment/gymnasium_wrapper.py` (EventSatGymnasium)
- **Supporting modules**: `src/behaviour/rollout_buffer.py` (RolloutBuffer + GAE), `src/behaviour/training_pipeline.py` (PPOTrainer)
- **Architecture note**: Current MLP baseline; RNN (LSTM/GRU) is a known improvement direction for partial observability — subject to optimization by Giulio Vaccari (exchange PhD)
- **Configs** (rl cell): `eventsat_sas_ao_rl.yaml`, `eventsat_sas_ag_rl.yaml`, `eventsat_sas_ah_rl_rl.yaml`

### LLM EventSat — Phase 4a (hybrid)

- **File**: `src/representation/llm_eventsat.py`
- **Registered as**: `llm_eventsat`
- **Substrate / action space**: Hybrid (subsymbolic LLM + symbolic safety constraints), **reactive** action space (single-shot encode→call→select)
- **Paper basis**:
  - Rodriguez-Fernandez et al. (2024), "Language Models are Spacecraft Operators" [WC5WU34U]
    — LLM prompt design for satellite operations (§3.2 state formatting).
  - Li (2025), "AI Agents for Satellite Operations" [UAA3GIVK]
    — ReAct LLM agent architecture for satellite ops.
- **Structure**:
  - `encode_observation()`: Same feature extraction as rule-based (for comparability).
  - `select_action(context)`: Formats state into structured prompt → LLM call → JSON
    parse → symbolic grounding validates mode → retry on invalid → **fails the episode**
    if no valid mode (substrate-integrity invariant, `ec1b83b`; no symbolic substitution).
  - `reason()`: LLM-based structured reasoning for ReAct thought step.
  - `get_rationale()`: LLM's natural language rationale.
- **Symbolic grounding checks**:
  - Mode must be one of 7 valid EventSat modes.
  - Anomaly → forced safe mode (no LLM override).
  - Communication requires active ground pass.
  - SoC < 0.20 → forced charging (hard safety limit).
- **LLM infrastructure** (`src/representation/llm_client.py`):
  - Dual provider: TUM Ollama (primary, via `OLLAMA_HOST`) + OpenAI API (fallback).
  - File-based response cache (`data/llm_cache/`) keyed on prompt hash.
  - Mock mode (`llm_mock: true`) for CI — no live LLM calls.
  - Configurable model, temperature, provider via YAML.
- **Orthogonality**: Works with all 3 decision procedures (SDA/OODA/ReAct, held fixed) and the ops paradigms
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

- **File**: `src/representation/agentic_eventsat.py`
- **Registered as**: `agentic_eventsat`
- **Substrate / action space**: Hybrid (LLM agent + symbolic tools + grounding constraints), **agentic** action space (tool-call loop + structured memory). Same substrate as `llm_eventsat`; differs only in action space.
- **Supporting modules**:
  - `src/representation/agentic_tools.py` — 6 domain tools (pure functions on state/memory)
  - `src/representation/agentic_prompts.py` — system prompt, planning/reflect/reasoning templates
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
     option — bounded-loop answer extraction per ReAct, Yao et al. 2023). First live
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
- **Loop interaction**: Plan-Tool-Reflect-Decide is the representation's internal
  reasoning protocol (CoALA §3.2 "internal actions"), orthogonal to the outer decision
  loop. OODA/ReAct enrichments ARE visible in the planning prompt (`format_planning_prompt`
  includes a SITUATION ASSESSMENT block when enrichments are present), but the code does
  not branch on `context.loop_type`. The LLM implicitly adapts to richer context. See
  "Cross-Cutting Design Decisions" for the rationale.
- **Inference gating**: For AG/CG ops paradigms, `should_allow_inference()` returns
  `False` between passes. The experiment runner skips the agentic loop entirely; schedule
  playback handles actions. This avoids wasted LLM API calls on stale inter-pass data
  (Rossi et al. 2023 [5EG3E3BP]).
- **Simulation assumption**: The 122B-parameter Qwen model represents upper-bound
  reasoning quality, not a deployable onboard configuration. AH configs model ideal
  onboard reasoning; AG/CG configs model ground-based inference.
- **ReAct amplification**: ReAct outer loop (max 3 iterations) × agentic inner loop
  (max 5 steps + 1 forced-decide call) = up to 18 LLM calls per decision step.
  Cost-controlled via `max_agentic_steps` (lower it for ReAct experiments).
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
- **Cell**: `hllm-a` (hybrid LLM + symbolic, agentic). **Config**: `eventsat_sas_ag_hllm-a`. The writable_coala online-learning variant (mechanism, not a separate class) and AH-ground hllm-a are pending.

---

## Operations Paradigms

Four paradigms (ladder): **autonomous_onboard** (onboard only) → **autonomous_hybrid** (onboard +
ground) → **autonomous_ground** (ground only) → **conventional_ground** (human ground). A
Jetson-based onboard core (subsymbolic/hybrid onboard, AO/AH) sets `env.onboard_compute_active=True`
via `config.onboard_uses_jetson`, adding the ~7 W Jetson-on draw (`power.onboard_compute_w`) to
non-Jetson-compute modes (not compress/detect/send); symbolic onboard (OBC rules) and ground
paradigms carry no overhead.

### Autonomous Onboard — onboard-only primitive

- **File**: `src/operations/autonomous_onboard.py`
- **Structure**: Pass-through observation (full real-time state), acts every step, no ground plan /
  no schedule — a single per-step onboard core, closed-loop. `has_onboard_autonomy()=True`.
- **Resolves to**: the onboard core (symbolic→`rule_based_eventsat`, subsymbolic→`subsymbolic_eventsat`).
  Hybrid+AO excluded (no standalone onboard LLM).
- **Configs**: `eventsat_sas_{sda,ooda,react}_{symb_hd,subm_le}_ao` (6 SAS).
- **Key property**: pure onboard autonomy with the Jetson power overhead; the AO↔AH contrast isolates
  the value of adding a ground plan.

### Autonomous Hybrid — dual-slot (onboard + ground planner)

- **File**: `src/operations/autonomous_hybrid.py`
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
  The AH **ground-planner** slot is still the `*_scheduler` placeholder, so the LLM/agentic
  *ground* planner (and `_lep_`/`_lec_` ground variants) remain pending the real planner contract
  (Phase 4.e); the onboard LLM/agentic core itself is live.

### Autonomous Ground — Phase 3 (renamed from ConventionalGround)

- **File**: `src/operations/autonomous_ground.py`
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

- **File**: `src/operations/conventional_ground.py`
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

- **File**: `src/memory/fixed_memory.py`
- **Used by**: All hand-designed variants and all non-CoALA learned variants.
- **Structure**: Sliding-window history (default depth 100), task queue, resource state,
  constellation state. Fully read-only from the agent's perspective — no write API.
- **Fairness invariant**: All variants that use `FixedMemory` are on equal footing;
  memory cannot be a confound in cross-architecture comparisons.

### WritableMemory — `_lec_` variants only (CoALA learning)

- **File**: `src/memory/writable_memory.py`
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

- **File**: `src/behaviour/training_pipeline.py` (PPOTrainer)
- **Mechanism**: `behaviour_config.mechanism = "ppo"`
- **Command**: `uv run autops train configs/experiments/eventsat_sas_rl_ah.yaml`
- **Output**: `data/trained_models/<experiment_id>/policy.pt`

### PromptOptimizer — `_lep_` LLM/agentic variants

- **File**: `src/behaviour/prompt_optimizer.py`
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

- **File**: src/orchestration/eventsat_metrics.py
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
- **OODA-specific** (Phase 3): mean_orient_latency_s, mean_orient_iterations,
  mean_orient_urgency.
- **ReAct-specific** (Phase 3): reasoning_depth, iterations, grounding_violations,
  converged; aggregated over episodes for comparison.
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
| SingleAgentSystem (SAS) | `src/agent_organization/single_agent_system.py` | **L4** Orchestration | Kim et al. 2025 | Single cognitive locus — analogue of single-agent loops (e.g.\ Claude Code) |
| CentralizedMAS | `src/agent_organization/centralized_mas.py` | **L4** Orchestration | Kim et al. 2025 | Role-specialized analogue of MetaGPT / ChatDev orchestrators |
| IndependentMAS (IMAS) | `src/agent_organization/independent_mas.py` | **L4** Orchestration | Kim et al. 2025 | Per-satellite agents, C = ∅; runnable on Flamingo N≥3 |
| DecentralizedMAS (DMAS) | `src/agent_organization/decentralized_mas.py` | **L4** Orchestration | Kim et al. 2025 | Peer all-to-all consensus, C = full; runnable on Flamingo N≥3 |
| HybridMAS | `src/agent_organization/hybrid_mas.py` | **L4** Orchestration | Kim et al. 2025 | Deferred to later Flamingo increments |
| SDA loop | `src/decision_procedure/` (SDA) | **L1** Reasoning | classical control loop | Baseline reactive scaffolding |
| OODA loop | `src/decision_procedure/` (OODA) | **L1** Reasoning | Miller / Hartmann / Richards | Orient-stage deliberation |
| ReAct loop | `src/decision_procedure/` (ReAct) | **L1** Reasoning + self-reflection | Yao et al. 2023 | Direct analogue of Bhati L1 self-reflection mechanism |
| Rule-Based / Schedule-Based EventSat | `src/representation/...rule_based_eventsat`, `schedule_based_eventsat` | **L1** (reasoning interface only) | hand-designed (Brooks 1991 reactive) | **No L0 substrate** — pure symbolic |
| Conventional Schedule EventSat | `src/representation/...conventional_schedule_eventsat` | **L1** | Sellmaier et al. 2022 | Human-realistic ground baseline; no L0 |
| Subsymbolic EventSat | `src/representation/...subsymbolic_eventsat` | **L0** (policy net) + **L1** | Wang et al. 2022 (DRL) | L0 is the RL policy network rather than an LLM |
| LLM EventSat | `src/representation/...llm_eventsat`, `llm_client.py` | **L0** (LLM) + **L1** | Rodriguez-Fernandez et al. 2024 | L0 substrate: Ollama / OpenAI backend |
| Agentic EventSat | `src/representation/...agentic_eventsat` | **L0** (LLM) + **L1** (CoALA) | Sumers et al. 2024 (CoALA) | Direct sibling of Bhati's L1 cognitive scaffolding |
| FixedMemory | `src/memory/fixed_memory.py` | **L1** Memory | fairness invariant | Read-only short/long-term state |
| WritableMemory | `src/memory/writable_memory.py` | **L1** Memory | Sumers et al. 2024 (CoALA §3) | Writable semantic + episodic stores — closest analogue of Bhati L1 "memory files" |
| BehaviourController | `src/behaviour/controller.py` | **L1** Self-reflection / learning controller | `@register` factory | Selects hand-designed vs learned variant |
| PPOTrainer | `src/behaviour/training_pipeline.py` | **L1** (learned reasoning) | PPO (Schulman et al. 2017) | RL-based learning loop |
| PromptOptimizer | `src/behaviour/prompt_optimizer.py` | **L1** (self-improvement) | DSPy / TextGrad family | Sibling of Bhati L1 self-critique |
| WritableCoALA | `_lec_` configs | **L1** (online learning) | Sumers et al. 2024 | Online memory write — closest match to Bhati's "memory files" |
| Tools (BaseTool + scenario actions) | `src/tools/` | **L2** Agent–Computer Interface | (no external paper basis) | YAML-serializable, stateless action definitions exposed to the cognitive layer |
| Satellite environment | `src/environment/`, `src/environment/orbital/`, `src/environment/scenarios/` | **L3** Tools & Environment | Orekit; mission constraints | Analogue of filesystem + test runners; deterministic physics layer |
| Autonomous Hybrid | `src/operations/autonomous_hybrid.py` | **L5** Governance & Safety | Rossi et al. 2023 | Onboard FDIR; no ground gate; closest to "auto" autonomy level |
| Autonomous Ground | `src/operations/autonomous_ground.py` | **L5** Governance & Safety | Sellmaier et al. 2022 | Ground-pass gating — analogue of human approval at high-impact actions |
| Conventional Ground | `src/operations/conventional_ground.py` | **L5** Governance & Safety | ECSS standards | Human-realistic ground operator; analogue of traditional SDLC supervision |
| Env-enforced safe mode | `src/environment/scenarios/eventsat_env.py` | **L5** (cross-cutting) | mission-safety invariant | Feedback flows downward (Bhati Fig. 3): safety overrides cross all representations |

**The L0 gap.** Pure-symbolic variants (`symb`) have no L0 substrate in Bhati's sense
— there is no foundation model. They sit at L1 directly. This asymmetry is
load-bearing for the fair-comparison invariant: holding L2–L5 fixed across the
matrix, the variation in L0 (none / RL policy / LLM) isolates the cognitive-paradigm
effect (Brooks 1991; Colelough & Regli 2025). The asymmetry is explicit in
[`morphological_matrix.md`](morphological_matrix.md).

---

## Cross-Cutting Design Decisions

### Decision Procedure × Representation Interaction Model

All decision loops produce a `DecisionContext` containing `state`, `enrichments`, and
`loop_metadata`. Representations consume this via their `select_action()` method. The
interaction model differs by representation paradigm:

- **Symbolic (rule-based, schedule-based)**: Explicit `if loop_type == "ooda"` branches
  activate loop-specific rules (e.g., 6 OODA-aware rules in `rule_based_eventsat`:
  R2e-OODA eclipse preparation, R2-OODA proactive charging, R3-OODA urgency-based pass,
  R5-OODA observation batching, R6-OODA orient-confident observation). The
  representation's behavior is deterministically different depending on which loop
  calls it.
- **LLM/Agentic (hybrid)**: Enrichments are serialized into prompt text. The LLM
  receives richer context from OODA (situation class, urgency, trends, CBR matches) or
  ReAct (reasoning trace, grounding violations) but no code branches on `loop_type`.
  The LLM implicitly adapts its reasoning to the available context.
- **Subsymbolic (RL)**: Enrichments are not used (the policy network operates on the
  fixed 25D observation vector). Loop variation only affects the temporal calling
  pattern (ReAct may call `select_action()` multiple times per step).

**Why agentic doesn't branch on loop_type**: The Plan-Tool-Reflect-Decide protocol is
a *representation-internal reasoning protocol* — what CoALA (Sumers et al. 2024 §3.2)
calls "internal actions" (reasoning + retrieval). The outer decision loop determines
*context richness*, not the representation's internal control flow:

- SDA: Minimal context (raw state only) → agentic loop plans with state alone
- OODA: Rich context (situation, urgency, trends) → agentic sees these in planning
  prompt via `format_planning_prompt()` SITUATION ASSESSMENT block
- ReAct: Iterative context (reasoning trace accumulates, violations feed back) →
  agentic loop benefits from prior reasoning across ReAct iterations

The scientific comparison axis is: *does the same representation produce better
decisions when given richer loop context?* And conversely: *does structured
multi-step reasoning (agentic) benefit less from richer loop context because it
already gathers similar information via tool use?*

**What changes across the matrix is the prompt, not the reasoning architecture.**
The decision loop controls context richness (enrichments); the representation
controls how that context is consumed (single-shot vs multi-step tool-use).

**LLM call counts per decision step:**

| | SDA | OODA | ReAct (max 3 iterations) |
|---|---|---|---|
| `llm_eventsat` | 1 LLM call | 1 LLM call (richer prompt) | Up to 3 LLM calls |
| `agentic_eventsat` | Up to 3 LLM calls | Up to 3 LLM calls (richer initial plan) | Up to 9 LLM calls (3 ReAct × 3 agentic) |

Note: `llm_eventsat` uses a single-shot prompt (state → one LLM call → mode decision).
`agentic_eventsat` uses a CoALA-style Plan-Tool-Reflect-Decide loop with domain tools
(check battery, pipeline, constraints, etc.) — multiple LLM calls per decision. Both
share the same symbolic grounding layer and LLM backend.

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
placeholders, `placeholder_cells.py`); and the learned ground LLM schedulers.
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
