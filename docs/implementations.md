# Implementation Registry

Persistent record of every implemented component in the morphological matrix,
its paper basis, and key design decisions. Grows as new components are added.

---

## Decision Loops

### SDA (Sense-Decide-Act) — Phase 2 baseline

- **File**: `src/decision_loop/sda_loop.py`
- **Paper basis**: Classic reactive agent pattern (sense-decide-act cycle)
- **Structure**: Single-pass — encode observation via representation, select action, return.
  No iteration, reflection, or memory interaction.
- **Memory**: Ignored. `process()` returns `(action, memory)` with memory passed through unchanged.
- **Metrics**: `decision_latency_s`, `total_decisions`, `has_rationale`
- **Significance**: Simplest possible decision loop — lower bound for decision overhead
  and the baseline for all loop comparisons.

### OODA (Observe-Orient-Decide-Act) — Phase 3

- **File**: `src/decision_loop/ooda_loop.py`
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

- **File**: `src/decision_loop/react_loop.py`
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
  CoALA (Sumers et al. 2024) is a higher-level framework that subsumes ReAct; it will
  be implemented as a separate, distinct loop in a future phase.

---

## DecisionContext Interface — Phase 3

- **File**: `src/decision_loop/context.py`
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

---

## Operations Paradigms

### Autonomous Hybrid — Phase 2 baseline

- **File**: `src/operations/autonomous_hybrid.py`
- **Structure**: Passthrough observation (full real-time state), unrestricted
  action timing (every timestep), immediate execution.
- **Anomaly recovery**: Onboard FDIR — agent clears safe mode once countdown expires.
- **Key property**: Zero information delay, zero planning latency. Upper bound for
  operations paradigm performance.

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

---

## Experiment Configurations

| Config ID | Org | Loop | Repr | Emergence | Ops | Phase |
|-----------|-----|------|------|-----------|-----|-------|
| `eventsat_cen_sda_symb_hd_ah` | Centralized | SDA | Rule-based | Hand-designed | Autonomous Hybrid | 2 |
| `eventsat_cen_sda_symb_hd_ag` | Centralized | SDA | Schedule-based | Hand-designed | Autonomous Ground | 3 |
| `eventsat_cen_sda_symb_hd_cg` | Centralized | SDA | Conventional Schedule | Hand-designed | Conventional Ground | 3 |
| `eventsat_cen_ooda_symb_hd_ah` | Centralized | OODA | Rule-based | Hand-designed | Autonomous Hybrid | 3 |
| `eventsat_cen_ooda_symb_hd_ag` | Centralized | OODA | Schedule-based | Hand-designed | Autonomous Ground | 3 |
| `eventsat_cen_ooda_symb_hd_cg` | Centralized | OODA | Conventional Schedule | Hand-designed | Conventional Ground | 3 |
| `eventsat_cen_react_symb_hd_ah` | Centralized | ReAct | Rule-based | Hand-designed | Autonomous Hybrid | 3 |
| `eventsat_cen_react_symb_hd_ag` | Centralized | ReAct | Schedule-based | Hand-designed | Autonomous Ground | 3 |
| `eventsat_cen_react_symb_hd_cg` | Centralized | ReAct | Conventional Schedule | Hand-designed | Conventional Ground | 3 |

### Comparison axes

- **Loop comparison** (same repr + ops): SDA vs OODA vs ReAct → decision quality vs latency
- **Ops paradigm comparison** (same loop + repr): AH vs AG vs CG → cost of planning delay
- **Human vs algorithmic** (same loop, AH excluded): AG vs CG → effect of cognitive constraints
- **Ground ops baseline**: CG with conventional schedule + SDA loop → human lower bound
