# Implementation Registry

Persistent record of every implemented component in the morphological matrix,
its paper basis, and key design decisions. Grows as new components are added.

> **Terminology / alignment note.** The canonical conceptual framing is the M × O × T spec
> ([`decision_matrix.md` §3](decision_matrix.md)): O is, per active core, **substrate**
> (symbolic / subsymbolic{RL, LLM} / neurosymbolic) × **action space** (reactive / agentic);
> *learning is folded in* (offline per core + the agentic online-learning action) and *decision
> procedure is held fixed* — neither is a peer axis. **This registry documents the current code**,
> whose tokens are unchanged pending the planned code rename: `src/decision_procedure/`, the
> `behaviour` / `behaviour_config` field (`src/behaviour/`), and `llm_eventsat` (= subsymbolic·LLM,
> legacy token `hyre`) / `agentic_eventsat` (= hybrid·agentic, legacy token `hyag`). So the
> "Decision Loops" and "Emergence" headings below keep their **implemented** names; map them to the
> spec via the crosswalk in `decision_matrix.md` §3.

---

## Agent Organizations

Formal definition: an agent system **S = (A, E, C, Ω)** where A = agents, E = environment, C = communication topology, Ω = orchestration policy (Kim et al. 2025 [FVFQ73RF]).

**Empirical prediction for all Organization experiments** (Kim et al. 2025, 180 configs): satellite mode selection is sequential constraint satisfaction → centralized org predicted to outperform distributed. Capability saturation (β̂=−0.404) means multi-agent overhead negates gains once single-agent baseline > ~45%.

Full taxonomy: Kim et al. (2025) [FVFQ73RF] "Towards a Science of Scaling Agent Systems".

### SingleAgentSystem (SAS) — Phase 2

- **File**: `src/agent_organization/single_agent_system.py`
- **Paper basis**: Kim et al. (2025) [FVFQ73RF] Single-Agent System (SAS) — |A|=1, single reasoning locus, C undefined, Ω direct, complexity O(k).
- **Structure**: One central agent receives full constellation observation and selects actions for all satellites. No inter-agent communication. Zero coordination overhead.
- **Key property**: Maximum context integration (unified memory stream, full prior-history access). Upper bound for context-quality; lower bound for parallelism.
- **Configs**: 36 `eventsat_sas_*` configs — 3 loops × 4 representations × 3 ops paradigms.

### CentralizedMAS — Phase 3 (EventSat single-satellite)

- **File**: `src/agent_organization/centralized_mas.py`
- **Paper basis**: Kim et al. (2025) [FVFQ73RF] Centralized MAS — orchestrator routes to sub-agents, C = {(a_orch, aᵢ) : ∀i}, Ω = hierarchical, complexity O(rnk). Also: ECSS-E-ST-70-11C autonomy levels (mission management layer vs onboard autonomy layer).
- **Structure**: A = {mission_manager, sat_agent_0}; C = star (manager→local, unidirectional); Ω = hierarchical (local agent action is used; manager action stored as directive for next step).
- **EventSat design decisions**:
  - `distribute_observation`: Both agents receive the full environment observation (single satellite, no meaningful state partitioning). `sat_agent_0` additionally receives the manager's previous-step action as a `messages` entry (directive context).
  - `collect_actions`: Manager action stored as `_last_manager_directive`; local agent action returned as environment action. Fallback to manager action if no local agent output.
  - Manager directive carries over step-to-step via `_last_manager_directive`; reset to `None` in `initialize()`.
  - Latency: `ExperimentRunner` accumulates manager + local agent latencies as the total step latency (sequential execution, both contribute to decision overhead).
- **Configs**: 12 `eventsat_cmas_*_ah.yaml` configs — 3 loops × 4 representations × 1 ops (AH only; CG/AG degenerate at single-satellite scale as ground already acts as the strategic layer).

### DecentralizedMAS — Placeholder (deferred to N≥3)

- **File**: `src/agent_organization/decentralized_mas.py`
- **Paper basis**: Kim et al. (2025) [FVFQ73RF] Decentralized MAS — all-to-all peer exchange, C = {(aᵢ, aⱼ) : ∀i,j, i≠j}, Ω = consensus, complexity O(dnk).
- **Structure**: Each satellite has its own agent; agents communicate peer-to-peer. Consensus formation through debate rounds. Enables parallel exploration but incurs coordination tax and information fragmentation.
- **Risk**: Independent error amplification (17.2× per Kim et al.) if consensus fails. Suited for parallelisable tasks, predicted to underperform on sequential satellite scheduling.
- **Status**: Stub (`NotImplementedError`). Deferred to constellation scenarios (N≥3); peer-to-peer coordination is degenerate at N=1.

### IndependentMAS — Placeholder (deferred to N≥3)

- **File**: `src/agent_organization/independent_mas.py`
- **Paper basis**: Kim et al. (2025) [FVFQ73RF] Independent MAS — C = ∅, no inter-agent coordination.
- **Status**: Stub (`NotImplementedError`). Meaningful only with subsystem-level agents (ADCS/payload/comms) or N≥3 satellites.

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
  recreated silently. The 36 non-symbolic ground configs were rewired to these
  placeholder types. Note: the subsymbolic ground cells no longer require `torch`
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
- **Orthogonality**: Works with all 3 loops (SDA/OODA/ReAct) and all 4 ops paradigms (AO/AH/AG/CG)
- **Training script**: `scripts/train_subsymbolic.py`
- **Gymnasium wrapper**: `src/environment/gymnasium_wrapper.py` (EventSatGymnasium)
- **Supporting modules**: `src/behaviour/rollout_buffer.py` (RolloutBuffer + GAE), `src/behaviour/training_pipeline.py` (PPOTrainer)
- **Architecture note**: Current MLP baseline; RNN (LSTM/GRU) is a known improvement direction for partial observability — subject to optimization by Giulio Vaccari (exchange PhD)
- **Configs**: 9 YAML files `eventsat_sas_{sda,ooda,react}_subm_le_{ah,ag,cg}.yaml`

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
- **Orthogonality**: Works with all 3 loops (SDA/OODA/ReAct) and all 4 ops paradigms
  (AH/AG/CG). Unlike symbolic which needed 3 separate representation types per ops
  paradigm, the LLM representation handles all ops contexts through its prompt.
- **Metrics**: `llm_api_calls`, `llm_cache_hit_rate`, `llm_total_latency_s`,
  `llm_tokens_prompt`, `llm_tokens_completion`, `llm_grounding_overrides`.
- **Operations paradigm**: All (autonomous_hybrid, autonomous_ground, conventional_ground).
- **Learned-emergence variant** (Phase 5):
  - `prompt_optimized` (`_hyre_lep_*`): loads offline-optimised system prompt. `FixedMemory`
    invariant preserved. **Note**: `writable_coala` does NOT apply here — emergent·memory is
    gated by the *agentic* action space (writing is an action), which the reactive single-shot
    LLM lacks (see [decision_matrix §3.2/§3.4](decision_matrix.md#32-behaviour-overlay)).
- **Configs**: 12 SAS + 3 CMAS = 15 hand-designed `*_hyre_hd_*`; 12 `*_hyre_lep_*`

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
- **Max LLM calls per decision**: `max_agentic_steps` (default 3, configurable in YAML)
  + at most 1 forced-decide call.
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
  (max 3 steps) = up to 9 LLM calls per decision step. Cost-controlled via
  `max_agentic_steps` config (recommend 2 for ReAct experiments).
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
- **Configs**: 12 SAS + 3 CMAS = 15 hand-designed `*_hyag_hd_*`; 12 `*_hyag_lep_*`; 12 `*_hyag_lec_*`

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
- **Note**: hybrid AH onboard is the subsymbolic per-step core; its ground planner is currently the
  `*_scheduler` placeholder, so LLM-learned mechanisms (`_lep_`/`_lec_`) are inert under AH until the
  real LLM/agentic planners land (Phase 4.e).

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
- **Two-buffer model**:
  - `_active_schedule`: Currently being executed by the satellite (uploaded at last pass).
  - `_planned_schedule`: Prepared by the representation after latest telemetry.
    Promoted to `_active_schedule` at the START of the next pass.
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

Maps to the **Behaviour** overlay ([decision_matrix §3.2](decision_matrix.md#32-behaviour-overlay)):
`ppo`/`prompt_optimized` = emergent·policy (gated by substrate); `writable_coala` = emergent·memory
(gated by the agentic action space). Mechanism is derived from Behaviour × substrate, not chosen freely.

### PPO Training — `_le_` subsymbolic variants

- **File**: `src/behaviour/training_pipeline.py` (PPOTrainer)
- **Mechanism**: `behaviour_config.mechanism = "ppo"`
- **Command**: `uv run autops train configs/experiments/eventsat_sas_sda_subm_le_ah.yaml`
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
- **Command**: `uv run autops train configs/experiments/eventsat_sas_sda_hyre_lep_ah.yaml`
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
- **Command**: `uv run autops train configs/experiments/eventsat_sas_sda_hyag_lec_ah.yaml`
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

- **File**: `src/orchestration/eventsat_metrics.py`
- **7 research metrics**: utility, data_downlink_efficiency, mean_latency_s,
  robustness_mean_recovery_steps, resource_efficiency, operator_load,
  explainability_score
- **Cross-episode**: robustness_cv (coefficient of variation of utility)
- **OODA-specific** (Phase 3): mean_orient_latency_s, mean_orient_iterations,
  mean_orient_urgency
- **ReAct-specific** (Phase 3): reasoning_depth, iterations, grounding_violations,
  converged (per-step); aggregated over episodes for comparison
- **AH-specific** (paradigm metrics, per episode): `onboard_overrides`, `onboard_override_rate` —
  between-pass steps where the onboard core overrode the uplinked plan (surfaced in each episode's
  `paradigm_metrics`).

---

## Layer Mapping (Bhati 2026)

Bhati (2026) [Z5TF79HY] proposes a six-layer reference architecture for **agentic
software engineering** systems. The autops framework is positioned as a parallel
reference architecture in a **sibling domain** (autonomous satellite operations).
The mapping below tags each component already documented above with its layer in
Bhati's stack — it is illustrative, not a structural adoption. See
[`decision_matrix.md` §2.1](decision_matrix.md#21-parallel-reference-architecture-bhati-2026)
for the framing.

| Component | File / module | Bhati layer | Existing paper basis | Cross-domain note |
| --- | --- | --- | --- | --- |
| SingleAgentSystem (SAS) | `src/agent_organization/single_agent_system.py` | **L4** Orchestration | Kim et al. 2025 | Single cognitive locus — analogue of single-agent loops (e.g.\ Claude Code) |
| CentralizedMAS | `src/agent_organization/centralized_mas.py` | **L4** Orchestration | Kim et al. 2025 | Role-specialized analogue of MetaGPT / ChatDev orchestrators |
| Decentralized/Independent/Hybrid MAS | `src/agent_organization/{decentralized,independent,hybrid}_mas.py` | **L4** Orchestration | Kim et al. 2025 | Deferred to Flamingo N≥3 |
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
effect (Brooks 1991; Colelough & Regli 2025) that RQ1 targets. The asymmetry is
explicit in §2.1 of decision_matrix.

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
| llm_eventsat | LLM every step (a3768bf; Jetson-class, R-COMPUTE1) | LLM between passes → uplinked plan | LLM between passes → schedule | LLM between passes → delayed schedule |
| agentic_eventsat | Agentic loop every step (a3768bf; Jetson-class, R-COMPUTE2) | Agentic between passes → uplinked plan | Agentic between passes → schedule | Agentic between passes → delayed schedule |
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

Config IDs follow: `eventsat_<org>_<loop>_<repr>_<emrg>_<ops>` where
`org` ∈ {sas, cmas}, `loop` ∈ {sda, ooda, react}, `repr` ∈ {symb, hyre, subm, hyag},
`emrg` ∈ {hd, le, lep, lec}, `ops` ∈ {ao, ah, ag, cg}.

### Symbolic (Phases 2–3) — SAS only

9 SAS configs (`eventsat_sas_{sda,ooda,react}_symb_hd_{ah,ag,cg}`); no CMAS symbolic configs.

### LLM Hybrid hand-designed (Phase 4a)

12 SAS (`eventsat_sas_{sda,ooda,react}_hyre_hd_{ah,ag,cg}`) + 3 CMAS
(`eventsat_cmas_{sda,ooda,react}_hyre_hd_ah`).

### LLM Hybrid prompt-optimized (Phase 5)

12 SAS (`eventsat_sas_{sda,ooda,react}_hyre_lep_{ah,ag,cg}`) + 3 CMAS
(`eventsat_cmas_{sda,ooda,react}_hyre_lep_ah`). Mechanism: `prompt_optimized`.

### Subsymbolic RL (Phase 4b)

9 SAS `*_subm_le_{ah,ag,cg}` + 3 CMAS `*_subm_le_ah`. Mechanism: `ppo`.

### Agentic Hybrid hand-designed (Phase 4c)

12 SAS (`eventsat_sas_{sda,ooda,react}_hyag_hd_{ah,ag,cg}`) + 3 CMAS
(`eventsat_cmas_{sda,ooda,react}_hyag_hd_ah`).

### Agentic Hybrid prompt-optimized (Phase 5)

12 SAS (`eventsat_sas_{sda,ooda,react}_hyag_lep_{ah,ag,cg}`) + 3 CMAS
(`eventsat_cmas_{sda,ooda,react}_hyag_lep_ah`). Mechanism: `prompt_optimized`.

### Agentic Hybrid writable-CoALA (Phase 5)

12 SAS (`eventsat_sas_{sda,ooda,react}_hyag_lec_{ah,ag,cg}`) + 3 CMAS
(`eventsat_cmas_{sda,ooda,react}_hyag_lec_ah`). Mechanism: `writable_coala`.

### Autonomous Onboard (Stage 1 — onboard-only)

6 SAS `eventsat_sas_{sda,ooda,react}_{symb_hd,subm_le}_ao` — onboard core only (no ground plan).

**Total**: 91 experiment configs + 1 template.

### Comparison axes

- **Loop comparison** (same repr + ops): SDA vs OODA vs ReAct → decision quality vs latency
- **Ops paradigm ladder** (same loop + repr): AO vs AH vs AG vs CG → onboard-only vs onboard+ground vs ground-only vs human-ground
- **Onboard-override effect** (same repr, shared ground planner): AH vs AG → what the onboard per-step override buys (`onboard_overrides`)
- **Value of a ground plan** (same onboard core): AO vs AH → does adding the ground plan help beyond pure onboard
- **Human vs algorithmic** (same loop, AH excluded): AG vs CG → effect of cognitive constraints (Endsley 1995 SA degradation)
- **Representation comparison** (same loop + ops): symbolic vs LLM vs agentic vs RL → cognitive paradigm effectiveness
- **Single-shot vs agentic** (same loop + ops, AH only): `hyre_hd` vs `hyag_hd` → does multi-step reasoning with tools improve decisions?
- **Hand-designed vs learned** (same repr + loop + ops): `_hd_` vs `_lep_` → does offline prompt optimization improve LLM/agentic decisions?
- **Fixed vs writable memory** (agentic AH only): `_hyag_hd_` vs `_hyag_lec_` → does CoALA-style memory accretion improve decisions across episodes?
- **Ground ops baseline**: CG with conventional schedule + SDA loop → human lower bound
