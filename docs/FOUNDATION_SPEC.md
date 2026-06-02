# Implementation Foundation for PhD Experimental Framework

**Project:** Custom Modular Architecture for Cognitive Satellite Constellation Autonomy Research

**Researcher:** Clemente J. Juan Oliver, TUM Chair of Spacecraft Systems

**Repository:** autops-agentic-framework

**Date:** June 1, 2026 (updated — morphological matrix restructuring, implemented in code + configs)

> **Scope of this document** (per the doc map in `CLAUDE.md`): research questions, the
> morphological matrix structure, the cognitive-paradigm taxonomy, and the phase roadmap.
> Component registry and paper basis live in `implementations.md`; metric definitions in
> `metrics.md`; scenario specs in `scenarios.md`; the architecture diagram, data flow, and
> directory layout in `architecture.md`; EventSat physics in `CLAUDE.md`. This spec points to
> those rather than restating them.

***

## 1. Project Overview

### Objective

Build a modular experimental framework to systematically compare cognitive architectures for
autonomous satellite constellation management, evaluated under identical conditions. The
framework spans a morphological matrix whose structural axes are **Organization × Representation
× Decision Procedure × Operations Paradigm**, with a **Behaviour** overlay (hand-designed vs
emergent) over the cognitive modules. Constellation scale (1 → 100+ satellites) is the cross-cutting
RQ3 dimension. The full axis structure is defined in §3.

### Research Questions

**Fundamental Research Question:** How do cognitive architecture and agent organization choices
shape the performance trade-space of autonomous constellation management, and how does this
trade-space evolve as constellation scale and structural complexity grow?

Scalability is framed as a **2D space**: *constellation size* (1 → 500+ satellites) × *structural
complexity* (centralized → distributed, with super-linear effort scaling). The following
interconnected sub-questions operationalize the fundamental RQ:

**RQ1 — Cognitive Architecture**
- How do **decision procedures** (SDA, OODA, ReAct), **knowledge representations** (symbolic,
  subsymbolic, hybrid), and **degree of emergence** affect key performance metrics (utility,
  latency, robustness, resource efficiency, operator load, and explainability)?
- Can Pareto frontiers between competing objectives (e.g., utility vs. resource efficiency vs.
  operator interventions) be characterised for different cognitive architecture configurations?
- Which cognitive architecture patterns offer the most favourable trade-offs for which
  operational scenarios?
- The *emergence × explainability* trade-off connects to Bhati's (2026) open problem on technical
  debt under sustained agent contribution: higher emergence buys capability at the cost of
  legibility, which in the satellite-ops setting maps to operator-load and explainability metrics.

**RQ2 — Agent Organization**
- How do different agent organizations (SAS, centralized/decentralized/independent/hybrid MAS)
  affect the performance/robustness trade-off under identical cognitive components?
- Are certain cognitive architectures better matched to certain organizations (e.g., emergent
  subsymbolic agents in distributed constellations vs. hybrid agents in hierarchical setups)?
- How does the choice of architecture family determine the type and degree of explainability
  available to human operators — and how does this interact with mission safety requirements?
- **Testable hypothesis (Kim et al. 2025 [FVFQ73RF]):** Satellite mode selection is sequential
  constraint satisfaction → centralized org should outperform distributed. Capability saturation
  predicts multi-agent overhead negates gains once single-agent baselines exceed ~45%.

**RQ3 — Scale & Complexity**
- How do different cognitive and agent architectures degrade or adapt as constellation size, task
  load, and constraint density grow (e.g., from 5 to 500 satellites)?
- How does structural complexity — increasing from centralized towards distributed topologies,
  with super-linear effort scaling — interact with the performance trade-offs of different
  architecture families?
- Do different architectures exhibit fundamentally different composability trade-offs as scale
  grows (e.g., integrating heterogeneous cognitive components without emergent negative side
  effects)?
- Can scaling laws be derived jointly over constellation size and structural complexity and
  converted into architecture-selection heuristics for a target mission profile?

### Key Design Principles

1. **Two-tier matrix, not flat orthogonality.** Structural axes describe what the agent *is*; the
   Behaviour overlay describes which module was *learned vs specified*. The axes are not all
   mutually independent — action-space richness only varies under the hybrid substrate, and
   Behaviour is an overlay over the cognitive modules rather than a peer axis (see §3).
2. **Modularity**: Components can be swapped without affecting others.
3. **Reproducibility**: Configuration-driven experiments with seed control.
4. **Fair comparison**: Same environment, memory, and metrics for all variants (memory invariant,
   §3).
5. **Scientific rigor**: Every implementation cites a specific paper.

***

## 2. System Architecture

![AUTOPS overall system architecture diagram](images/autops-overall-system-architecture.svg)

The data-flow and component-interaction view is maintained in `architecture.md`; the directory
layout lives there as well.

### 2.1 Parallel Reference Architecture (Bhati 2026)

Bhati (2026) [Z5TF79HY] proposes a **six-layer reference architecture** (L0 Foundation Model →
L5 Governance & Safety) for agentic software-engineering systems, and contrasts a traditional
SDLC with an **A-SDLC** in which an orchestrator coordinates specialized sub-agents under human
supervision.

The autops morphological matrix is positioned as a **parallel reference architecture in a
sibling domain** (autonomous satellite operations). It is *not* a structural adoption of Bhati's
stack — the matrix remains canonical — but the cross-domain mapping makes the autops choices
legible to the broader agentic-AI literature and surfaces one substantive asymmetry: **no
foundation model exists beneath symbolic variants (the L0 gap)**, itself a thesis-relevant
observation about the cognitive-paradigm axis (Brooks 1991; Colelough & Regli 2025). The ops-paradigm
spectrum (CG → AG → AH) operationalizes an analogous autonomy gradient to A-SDLC but is grounded
in Rossi et al. (2023), Sellmaier et al. (2022), and ECSS standards rather than restructured around it.

The per-layer mapping table is maintained in
[`implementations.md` → Layer Mapping (Bhati 2026)](implementations.md#layer-mapping-bhati-2026)
and the layered system view in
[`architecture.md` → Layered View](architecture.md#layered-view-parallel-to-the-matrix).

***

## 3. Morphological Matrix Structure

> The code and the 84 EventSat configs implement this structure directly (config fields
> `decision_procedure` / `behaviour` / `behaviour_config`, `representation_config.action_space`,
> filename tokens `symb`/`subm`/`hyre`/`hyag`). The migration was a 1:1 *re-labeling* of the existing
> config set, not a re-enumeration.

The matrix has **two tiers**. The structural tier describes the blueprint — what the agent is. The
Behaviour overlay describes which cognitive module's competence was learned rather than specified.
This separation is grounded in CoALA (Sumers et al. 2024, TMLR [`7X8SRMIG`]), which decomposes any
agent into three modules — **memory, action space, decision procedure** — a decomposition inherited
from classical cognitive architectures (Soar; Newell; Laird 2022) and therefore **substrate-general**,
applying to the symbolic and RL cells, not only language agents. CoALA's finer internal-action
taxonomy (reasoning / retrieval / learning) is the LLM-specific specialization that applies only
when the core is a language model.

### 3.1 Structural axes

| Axis | Values | Notes |
|---|---|---|
| **Organization** | SAS · centralized_mas · decentralized_mas · independent_mas · hybrid_mas | Only SAS + CMAS instantiated; the three MAS topologies deferred to N≥3 (Flamingo). Taxonomy: Kim et al. 2025 (§4). |
| **Representation** | symbolic · subsymbolic · hybrid | **Substrate only.** Kautz 2022; Garcez & Lamb 2023; Colelough & Regli 2025. |
| ↳ Action space (hybrid only) | reactive \| agentic | A **hybrid-only flavor**, not a separate axis — it has no degrees of freedom under symbolic/subsymbolic (those are always reactive; an agentic tool-call loop requires an LLM core). This is where the `hyre` (reactive) and `hyag` (agentic) hybrid configs differ. |
| **Decision Procedure** | SDA · OODA · ReAct | Code field: `decision_procedure`. |
| **Operations Paradigm** | AO · AH · AG · CG | Defines which *autonomy slots* are active (§3.3). |

The four representation filename tokens (`symb`, `subm`, `hyre`, `hyag`) map to **three substrates
with hybrid split by action space**: `hyre` = hybrid-reactive, `hyag` = hybrid-agentic. Representation
denotes *only* substrate.

The concrete implementation class is **resolved** from `representation × action_space (hybrid only)
× operations_paradigm` — `operations_paradigm` selects the per-step controller (AH) vs the
schedule-producer (AG/CG), and `action_space` selects reactive vs agentic. Configs therefore carry
no `representation_config.type`; it remains only as an optional explicit override (e.g. the
`_algobase` cell: symbolic CG forced to the *algorithmic* `schedule_based_eventsat` instead of the
human `conventional_schedule_eventsat`). See `ExperimentConfig.resolved_representation_type`.

### 3.2 Behaviour overlay

Behaviour is **not a flat algorithm enum** and **not a peer axis** — it asks, of the cognitive
modules, *which one was learned*. It is **orthogonal to substrate** ("is the policy/memory learned?"
can be asked of any agent), but each mechanism is **gated by a different structural capability**:
policy-learning is gated by the **substrate** (a learnable policy), while memory-learning is gated
by the **action space** (writing to a store is itself an internal action — CoALA's *learning*
action — so it needs the agentic action-space repertoire, not the reactive single-shot one). The
substrate then decides *what* gets written.

| Behaviour | Module learned | Gated by | Mechanism | Filename suffix |
|---|---|---|---|---|
| hand-designed | — | — | — | `_hd_` |
| emergent · policy | the policy / reasoning competence | substrate | **ppo** (subsymbolic) \| **prompt_optimized** (hybrid — works for reactive *and* agentic) | `_le_` / `_lep_` |
| emergent · memory | declarative (semantic + episodic) memory | action space (agentic) | **writable_coala** | `_lec_` |

So `prompt_optimized` is action-space-agnostic (hence both `hyre_lep` and `hyag_lep`), whereas
`writable_coala` is tied to the agentic action space — *not* to the hybrid substrate. A
symbolic-agentic agent could in principle write to memory too (cf. Soar); it simply isn't
instantiated here.

**Memory invariant.** All variants use `FixedMemory` for fair comparison. The *only* thing that
flips memory to `WritableMemory` is `emergent · memory` (writable_coala) — so the fairness-invariant
exception is the visible side-effect of that one Behaviour state, not a separate axis. `_lec_` is
compared against its `_hyag_hd_` counterpart (same representation, loop, and paradigm). See the
Memory invariant note in `CLAUDE.md`.

### 3.3 Operations Paradigm as autonomy slots

The paradigm decides *which autonomy slots are active*; each slot is a **distinct core**. Two slots:

- **ground planner** (long-horizon, synthesizes the whole-pass uplinked schedule) — active in **AH and AG**
- **onboard** (closed-loop, per-step) — active in **AO and AH**

Four-paradigm ladder, each comparison sharing a core:

| paradigm | onboard slot | ground slot | shares |
|---|---|---|---|
| **AO** autonomous_onboard | ✓ | — | onboard with AH (AO↔AH = does a ground plan help?) |
| **AH** autonomous_hybrid | ✓ | ✓ | both |
| **AG** autonomous_ground | — | ✓ | ground planner with AH (AH↔AG = onboard-override effect) |
| **CG** conventional_ground | — | ✓ (human, one-pass delay) | — |

**Onboard power cost.** A *Jetson-based* onboard core (subsymbolic / hybrid onboard, in AO and AH)
keeps the Jetson powered every step to run per-step inference — a continuous draw
(`power.onboard_compute_w`, ~ Jetson-on payload modes) on top of per-mode consumption, modelled via
`config.onboard_uses_jetson` → `env.onboard_compute_active`. **Symbolic** onboard rules run on the
OBC (sub-watt) → no overhead; AG/CG decide on the ground → no overhead. This makes Jetson-based
onboard autonomy a real energy/responsiveness trade-off.

**Design decision (distinct cores; shared ground planner).** The two slots are *different* trained
cores, not one core in two modes:

- The **ground planner** is a long-horizon planner that emits a full-pass schedule — a subsymbolic
  full-pass RL planner, or an LLM / agentic planner. **The same ground-planner artifact is used in
  both AH and AG** (it is shared across those two paradigms, not retrained per paradigm).
- The **onboard** core is the per-step RL policy (optionally wrapped with symbolic safety rules,
  e.g. SoC < 20 % ⇒ charge), active only in AH, and it can override the uplinked plan.

A single config still carries **one `representation` value** describing the whole system: a
*subsymbolic* AH config is subsymbolic in **both** slots (full-pass RL planner + per-step RL
onboard); a *hybrid* AH config pairs an **LLM/agentic ground planner** with a **subsymbolic
(+rules) onboard** — the "hybrid" label is exactly that ground-LLM + onboard-RL/symbolic mix.

Because the ground planner **and the simulations** are held identical across AH and AG, **AH-vs-AG
isolates the real effect of the onboard per-step override** — it is the only moving part. (If the
ground planner differed between AH and AG, the two paradigms would mostly reproduce each other or
confound the override effect with a planner difference.)

### 3.4 Exclusion rules (locked)

Not every theoretical cell is scientifically meaningful. The following are excluded by principle:

| Excluded cell | Reason |
|---|---|
| `symb` + any emergent | Symbolic reps are deterministic rule chains — no learnable parameters. Symbolic appears only with hand-designed. |
| subsymbolic + hand-designed | A PPO policy is learned by construction; a hand-designed subsymbolic agent is incoherent here. Subsymbolic appears only with emergent·policy (ppo). |
| reactive (any substrate) + emergent·memory | Memory-writing is an internal action; a reactive single-shot agent has no action to issue it. emergent·memory requires the **agentic action space** — gated by action space, not by substrate. (In this framework the only agentic action space is hybrid-agentic, so `_lec_` appears there.) |
| subsymbolic + prompt_optimized / writable_coala | PPO has no prompt or declarative store to accrete into. |
| hybrid + `AO` (autonomous_onboard) | AO is onboard-only; a hybrid has no standalone onboard core (its LLM is a ground component). AO applies to symbolic & subsymbolic only. |
| `cmas × {AG, CG}` | CMAS coordination is degenerate at N=1 when ground is the strategic layer. CMAS appears only with AH. |
| decentralized / independent / hybrid MAS (any) | Require N≥3 satellites; deferred to Flamingo. |

**Comparison scope.** Because each substrate has a *different* learning mechanism, emergence is
compared **within a representation family, not across** (e.g. `hyag_hd` vs `hyag_lep` vs `hyag_lec`;
`hyre_hd` vs `hyre_lep`). Across-family comparisons hold Behaviour at the
"representation-appropriate default" (e.g. `symb_hd` vs `hyre_hd` vs `hyag_hd` vs `subm_le`). There
is no single "learned-everywhere" axis because no unified mechanism exists across substrates.

**Loop × representation orthogonality note.** With deterministic symbolic reps, SDA vs OODA vs
ReAct produce identical decisions (the rules do not iterate); the cells are retained as configs
(validator *warns*, does not error) but collapse behaviourally. With hybrid/subsymbolic reps the
decision procedure becomes load-bearing (LLMs generate different traces; RL has stochastic
exploration).

### 3.5 Feasible combinations (SAS, EventSat)

Per representation-state, the available Behaviours (locked rules) and meaningful decision
procedures:

| Representation state | Behaviours | Decision Proc. (meaningful) | × Ops | enumerated | distinct |
|---|---|---|---|---|---|
| symbolic | hd (1) | SDA only (others collapse) | 3 | 9 | 3 |
| subsymbolic | ppo (1) | SDA only | 3 | 9 | 3 |
| hybrid-reactive | hd, lep (2) | SDA/OODA/ReAct (3) | 3 | 18 | 18 |
| hybrid-agentic | hd, lep, lec (3) | SDA/OODA/ReAct (3) | 3 | 27 | 27 |

**≈ 51 behaviourally distinct SAS architectures** (63 if the deterministic × non-SDA cells are
kept as nominal configs, as they currently are). Plus **12 CMAS** (AH-only). Multiply by Organization
once the MAS topologies come online at N≥3.

*Open items flagged for supervisor review:* (a) ReAct ⇄ agentic coupling — `hybrid-reactive + ReAct`
may collapse into agentic on inspection; (b) distinct-core ground planner (§3.3); (c) whether
`emergent·policy` is one state or two (is PPO learning the same module as prompt-opt, or
action-selection vs reasoning?).

***

## 4. Core Component Specifications

Abstract base classes are defined before implementation. This section gives each component's
**purpose and scientific grounding**; method signatures and concrete implementations are in
`implementations.md` and the source tree.

### 4.1 Satellite Environment — `src/environment/satellite_env.py`

Unified, scenario-agnostic environment (`reset`/`step`/`get_observation`/`get_metrics`). Specific
task types, rewards, and constraints live in scenario subclasses (EventSat physics in `CLAUDE.md`).

### 4.2 Agent Organization — `src/agent_organization/base.py`

Abstract coordination pattern: distributes observations to agents and aggregates their actions.

**Formal definition** (Kim et al. 2025 [FVFQ73RF], "Towards a Science of Scaling Agent Systems"):
an agent system is **S = (A, E, C, Ω)** where **A = {a₁…aₙ}** (each aᵢ = (Φᵢ reasoning policy, Aᵢ
action space, Mᵢ memory, πᵢ decision function)), **E** the shared environment, **C** the
communication topology, **Ω** the orchestration policy. |A|=1 → SAS; |A|>1 → MAS.

| Topology (Kim et al.) | config value | suffix | C | Ω | Complexity |
|---|---|---|---|---|---|
| Single-Agent | `sas` | `sas` | — | direct | O(k) |
| Centralized MAS | `centralized_mas` | `cmas` | orchestrator → sub-agents | hierarchical | O(rnk) |
| Decentralized MAS | `decentralized_mas` | `dmas` | all-to-all peer | consensus | O(dnk) |
| Independent MAS | `independent_mas` | `imas` | agent → aggregator | synthesis_only | O(nk) |
| Hybrid MAS | `hybrid_mas` | `hmas` | star + peer | hierarchical + lateral | O(rnk + pn) |

(k = reasoning iterations, r = orchestration rounds, d = debate rounds, n = agents.) The config
value is what `config_loader.py` accepts; the suffix is the experiment-filename abbreviation.
`dmas`/`imas`/`hmas` are registered but raise `NotImplementedError` — deferred to N≥3.

**Empirical scaling effects** (Kim et al., 180-config controlled study) — the basis for RQ2
hypotheses:

1. **Capability saturation** (β̂ = −0.404, p<0.001): once the single-agent baseline exceeds ~45%
   accuracy, adding agents yields diminishing/negative returns.
2. **Topology-dependent error amplification**: independent agents amplify errors 17.2×; centralized
   coordination contains this to 4.4× via validation bottlenecks → centralized org should be safer
   for anomaly handling.
3. **Task-type dependency**: coordination benefits are task-contingent — parallelisable tasks
   benefit from centralization (+80.8%), dynamic exploration from decentralization (+9.2%), but
   **sequential constraint satisfaction (planning) degraded under *every* multi-agent variant
   (−39% to −70%)**. Satellite mode selection (charge → observe → compress → detect → send →
   communicate) is sequential constraint satisfaction → **prediction: centralized org outperforms
   distributed for EventSat** (testable RQ2 hypothesis).
4. **Intelligence–coordination alignment**: higher-capability representations need the *right*
   topology to benefit → the **Organization × Representation interaction is a first-class RQ2
   question**.

A held-out architecture-selection rule reached 87% accuracy keyed on task properties
(decomposability, tool complexity, sequential depth), not agent count — autops aims to derive
analogous heuristics from its own matrix.

### 4.3 Decision Procedure — `src/decision_procedure/base.py`

Abstract temporal control flow (`process(observation, memory) → (action, memory)`). Implementations
follow their source papers strictly:

- **SDA**: linear reactive single-pass sense-decide-act, no iteration.
- **OODA**: fixed four-phase Observe-Orient-Decide-Act with situation classification, case-based
  reasoning, urgency scoring, and feedback (Boyd; Miller/Hartmann).
- **ReAct**: iterative Thought-Action-Observation cycle (Yao et al. 2023), via
  `representation.reason()`, with grounding validation before each action; converges or falls back
  to charging. ReAct's iteration is *endogenous* (the representation decides when to stop), in
  contrast to OODA's *exogenous* fixed phases — an asymmetry relevant to §3.5's open ReAct⇄agentic
  question.

**CoALA is not a decision procedure.** CoALA (Sumers et al. 2024) is the substrate-general
architecture blueprint of §3 (memory + action space + decision procedure). In this framework the
agentic pattern it inspires is the **hybrid-agentic action-space flavor** (the
`agentic_eventsat` representation type), not a decision procedure and not a separate substrate.

### 4.4 Representation — `src/representation/base.py`

How knowledge and decisions are represented (`encode_observation`, `select_action`, optional
`reason`/`update`). **Cognitive paradigms** (Brooks 1991; Colelough & Regli 2025; Navarro 2025):

| Substrate | Where knowledge lives | Reasoning | Implementations |
|---|---|---|---|
| **Symbolic** | Explicit rules, ontologies, world models | Deductive, constraint-based | `rule_based_eventsat`, `schedule_based_eventsat`, `conventional_schedule_eventsat` |
| **Subsymbolic** | Network weights, embeddings | Statistical, distributed | `subsymbolic_eventsat` |
| **Hybrid** | Both (Kahneman System 1/2) | Fast intuition + deliberate reasoning | `llm_eventsat` (reactive), `agentic_eventsat` (agentic) |

**Key rule.** A bare LLM is subsymbolic (implicit distributed representations); an LLM combined with
tools, structured memory, or symbolic constraints becomes hybrid (Colelough & Regli 2025; Kahneman
System 1/2). Within hybrid, the **action space** distinguishes reactive (single-shot
encode→constrained-call→select, `llm_eventsat`) from agentic (tool-call loop + structured memory,
`agentic_eventsat`). The agentic pattern is an *action-space flavor of the hybrid paradigm*, not a
new paradigm — see §3.1.

### 4.5 Memory — `src/memory/`

`FixedMemory` (default, all variants): unified store of current state, sliding-window history, task
queue/completion, and resource budgets — fixed across experiments for fair comparison.
`WritableMemory` (writable_coala only): adds writable semantic + episodic stores (CoALA §3). See the
Memory invariant in §3.2 and `CLAUDE.md`.

### 4.6 Behaviour Controller — `src/behaviour/controller.py`

Factory selecting hand-designed vs emergent representations and wiring the derived mechanism
(`get_representation(...)`). Realizes the Behaviour overlay of §3.2; the `@register("name")`
decorator auto-registers representation implementations.

### 4.7 Operations Paradigm — `src/operations/base.py`

The deployment/autonomy envelope sitting between organization and environment: filters observations
(`filter_observation`), gates actions (`can_act`, `process_action`), and updates ground knowledge on
downlink (`update_ground_knowledge`). Defines the autonomy slots of §3.3.

- **AutonomousOnboard**: onboard-only — a single per-step core acts every step on full real-time
  state, closed-loop, no ground plan. `has_onboard_autonomy()=True` (Jetson power overhead). The
  onboard-only primitive of the ladder (§3.3).
- **AutonomousHybrid** (dual-slot): the onboard per-step core acts every step on full real-time
  state; the ground planner (same artifact as AG) refreshes the uplinked plan at passes from stale
  telemetry; between passes the satellite follows the plan unless the onboard core overrides on a
  safety mode (counted as `onboard_overrides`). During contact AH communicates (matching AG). Real
  full-pass RL / LLM planners are Phase 4.e (placeholders today). (Rossi et al. 2023; Sellmaier et al.
  2022 §16.4.)
- **AutonomousGround**: ground sees only downlinked (stale) telemetry; an algorithmic scheduler
  prepares optimal schedules between passes, uplinked next contact; satellite executes with zero
  onboard autonomy. The algorithmic ideal — no planning delay.
- **ConventionalGround**: same information constraints as AG plus a **one-pass planning delay**
  (schedule planned after pass N uploaded at pass N+1), modelling a real flight-dynamics cycle.
  Paired with `ConventionalScheduleEventSat` (conservative margins, limited horizon, shift
  handovers; Endsley 1995). Two-buffer design (`_active_schedule` + `_planned_schedule`).

`GroundKnowledge` (dataclass) captures what operators know — SoC, stored data, mode, health,
staleness — updated only during passes.

### 4.8 Experiment Orchestration — `src/orchestration/experiment_runner.py`

Configuration-driven execution: load YAML, instantiate all components, run episodes with metrics
collection, save results with full provenance, support batch runs. No hardcoded experimental
choices — everything is configurable via YAML.

***

## 5. Configuration System

Experiments are fully specified by YAML; the canonical template with every field documented is
`configs/experiments/template.yaml`, and the Pydantic v2 schema + cross-field validators are in
`src/orchestration/config_loader.py`. Validation on load checks required fields, valid morphological
choices, the §3.4 exclusion rules, constellation-size limits, and scenario completeness. (The
validator *warns* rather than errors on degenerate loop × deterministic-representation cells.)

***

## 6. Metrics Framework

Eight metrics are collected; full definitions, formulas, rationale, and the pre-registered analysis
plan live in `metrics.md`. In brief: **utility** (weighted observation + downlink achievement, minus
anomaly rate), **latency** (wall-clock per decision cycle), **robustness** (CV of utility across the
launch-lottery episodes), **resource efficiency** (utility per Wh), **operator load** (fraction of
steps needing a safety override), **scale & complexity** (degradation across size × topology),
**data downlink efficiency** (downlinked / max achievable), **explainability** (fraction of steps
with a rationale string). The `MetricsCollector` ABC (`src/orchestration/metrics_collector.py`)
defines `collect_step_metrics` / `aggregate_episode_metrics` / `compute_statistics`.

***

## 7. Implementation Phases & Roadmap

### Phase 1 — Foundation
Abstract base classes, configuration system, orchestrator skeleton, scenario-agnostic environment
base, test framework.

### Phase 2 — First complete path
EventSat scenario chosen and implemented; SAS; one decision loop and one representation;
end-to-end execution producing valid metrics.

### Phase 3 — Morphological expansion
Decision procedures OODA (Boyd/Miller/Hartmann) and ReAct (Yao et al. 2023); three-tier ops
paradigm (AutonomousHybrid, AutonomousGround [algorithmic], ConventionalGround [human-realistic,
one-pass delay; Sellmaier et al. 2022, ECSS-E-ST-70C]); `ScheduleBasedEventSat` and
`ConventionalScheduleEventSat` (Endsley 1995); `Representation.reason()` for the ReAct Thought step.

### Phase 4 — Hybrid & subsymbolic representations + learned emergence
- **4a** LLM hybrid (reactive) — `llm_eventsat` (Rodriguez-Fernandez et al. 2024; Li 2025).
- **4b** Subsymbolic/RL — `subsymbolic_eventsat` + PPO pipeline, Gymnasium wrapper, 25D obs,
  MultiDiscrete actions (Oliver et al. EUCASS 2025; Hamilton et al. 2025; BSK-RL).
- **4c** Agentic hybrid — `agentic_eventsat`, CoALA-style multi-step reasoning with 6 domain tools
  (Sumers et al. 2024; Sapkota et al. 2026; Li 2025).
- **4d** Inference gating by ops paradigm — AG/CG skip LLM inference between passes (Rossi et al.
  2023; Sellmaier et al. 2022); SHA-keyed LLM prompt cache; decision-trace JSONL in DEBUG.

### Phase 5 — Kim et al. taxonomy + learned emergence for LLM representations
- Organization taxonomy rename to Kim et al. 2025 (SAS / CentralizedMAS instantiated; Decentralized
  / Independent / Hybrid MAS as N≥3 stubs).
- `behaviour_config.mechanism` with cross-field validators (`ppo`→subsymbolic;
  `prompt_optimized`→hybrid; `writable_coala`→hybrid + agentic).
- `WritableMemory` (CoALA §3) and CoALA memory-write tools (`memory_write_rule`,
  `memory_write_episode`) injected only for writable_coala.
- `PromptOptimizer` (bootstrap few-shot, Khattab et al. 2023 [DSPy]; no DSPy runtime dep) and the
  `autops train` CLI dispatch (PPO / prompt-opt / writable-coala guidance).
- **84 EventSat configs** total (48 hand-designed + 36 learned), 36 SAS + 12 CMAS at the structural
  level.

### Phase 4.e — Real ground planners (planned)
**The AH dual-slot mechanism is built** (`AutonomousHybrid` runs onboard + ground-planner cores with
plan-default/triggered override; runner wires both; §4.7). What remains is replacing the
symbolic-planner placeholders (`*_scheduler_eventsat`, `is_placeholder=True`) with the real
distinct-core ground planners of §3.3, all emitting `[(mode, num_steps), …]`:

- **full-pass RL planner** (subsymbolic) — a *distinct* artifact from the onboard per-step
  `subsymbolic_eventsat`: own training objective + sequence/schedule action space (not the per-step
  policy rolled out). Open: policy architecture, duration representation, reward shaping, training
  harness. (Relates to Giulio's separate RLlib branch.)
- **LLM / agentic planners** — `llm_scheduler_eventsat` / `agentic_scheduler_eventsat` as real
  schedule generators (single-shot vs tool-using), not symbolic stand-ins. This also re-activates the
  learned mechanisms for hybrid AH (`_lep_`/`_lec_` would optimize / accrete on the LLM ground
  planner), which are inert while the placeholders stand in.

### Next steps
Phases 1–5 are complete. The active roadmap lives in `implementations.md` (component registry,
per-phase test counts) and the first-round LLM experiment plan (`_hd_` baselines → `_lep_` →
`_lec_`). Remaining instantiation: the Flamingo N≥3 constellation (activates the deferred MAS
topologies) and the large-scale (100+) RQ3 study.

***

## 8. Operational Scenarios

Three scenarios provide the RQ3 progression from single-satellite to large constellation across the
2D scalability space. Full task/constraint/metric specifications are in `scenarios.md`.

1. **EventSat** (1 sat, minimal complexity) — TUM's own satellite; high-fidelity internal mission
   data. The baseline where the cognitive-architecture comparison begins.
2. **Vyoma Flamingo** (up to 12 sats, medium complexity) — SSA constellation with AUTOPS partner
   Vyoma; activates multi-agent organization comparison and the deferred MAS topologies.
3. **Large-scale constellation** (100+ sats, high complexity) — fully distributed, heterogeneous;
   literature-based modelling. The high-scale endpoint for scaling laws and composability limits.

***

## 9. Development Standards

Python 3.11+, `uv` for dependency management (extras `dev` / `orbital` / `llm` / `rl`), pytest with
coverage, Google-style docstrings, type hints on public APIs, black/isort/ruff (line length 100),
Pydantic v2 for config validation. Commands, venv handling, and test invocation are in `CLAUDE.md`;
the dependency manifest is `pyproject.toml`. Step-by-step guidance for adding components is in
`implementation_guide.md`. Trunk-based development with conventional commits; tests stay green per
commit; reproducibility (same config + seed → identical metrics) is a validation requirement.

***

## 10. References

**Cognitive paradigm taxonomy**
- Brooks (1991), *Intelligence Without Representation*, Artificial Intelligence 47(1).
- Colelough & Regli (2025), *Neuro-Symbolic AI in 2024: A Systematic Review*.
- Kautz (2022), *The Third AI Summer*; Garcez & Lamb (2023), *Neurosymbolic AI: The 3rd Wave*.
- Navarro (2025), *Enhancing Cognitive Functions in LLMs Towards AGI*.

**Cognitive architectures & decision procedures**
- Sumers et al. (2024), *Cognitive Architectures for Language Agents* (CoALA), TMLR [`7X8SRMIG`].
- Yao et al. (2023), *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR.
- Laird (2022), *The Soar Cognitive Architecture*; Newell (1990), *Unified Theories of Cognition*.

**Representations for satellite ops**
- Rodriguez-Fernandez et al. (2024), *Language Models are Spacecraft Operators*.
- Li (2025), *Developing AI Agents for Satellite Operations*.
- Wang et al. (2022), *DRL-based Autonomous Mission Planning for AEOSs*.
- Khattab et al. (2023), *DSPy* (prompt optimization).

**Agent organization & scaling**
- Kim et al. (2025), *Towards a Science of Scaling Agent Systems* [FVFQ73RF] — S=(A,E,C,Ω) topology
  taxonomy, 180-config study, scaling effects, 87%-accuracy selection rule.
- Sapkota et al. (2026), *AI Agents vs. Agentic AI: A Conceptual Taxonomy*.
- Masterman et al. (2024), *The Landscape of Emerging AI Agent Architectures*.

**Operations paradigm**
- Rossi et al. (2023); Sellmaier et al. (2022); ECSS-E-ST-70C; Endsley (1995).

**Parallel domain**
- Bhati (2026), *Agentic AI in the Software Development Lifecycle*, arXiv:2604.26275 [Z5TF79HY] —
  six-layer reference architecture and A-SDLC; positioned in §2.1.

See the Zotero "Autonomous Operations" library for full references; every implementation cites its
source paper.

***

**Researcher:** Clemente J. Juan Oliver · TUM Chair of Spacecraft Systems · clemente.juan@tum.de

**END OF FOUNDATION SPECIFICATION** — architectural foundation only; concrete implementations are
registered in `implementations.md`.
