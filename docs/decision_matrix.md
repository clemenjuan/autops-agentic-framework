# AUTOPS — Decision/Tradespace Matrix (M × O × T)
**Project:** autops-agentic-framework · branch `migrate/matrix-restructure`
**June 2026 · the single canonical spec — design-forward.** This document defines the *target* system: axes, validity rules, enumeration, metrics, tests, validity argument, and analysis protocol. Implementation follows this spec (reuse-first policy, §7); implementation status is deliberately **not** tracked here. Metric definitions (M-01…M-14) live in §5.2 — `docs/metrics.md` has been retired into this file. Standards verified against source (CCSDS 520.0-G-3 §3.4 Table 3-1; ECSS-E-ST-70-11C Rev.1 §5.7 — execution E1–E4, data-management D1–D2, FDIR §5.7.5).

---

## 1. Structure

The decision tool is a three-axis matrix **M × O × T**:

| Axis | What it encodes | Input / output role |
|------|----------------|---------------------|
| **M** — Satellite System | Five sub-dimensions characterising the mission | *Input*: architect fills in SSP profile |
| **O** — Operations System | Candidate cognitive architecture from the autops morphological space | *Rows*: options being compared |
| **T** — Tests | Standardised experiments grounded in CCSDS 520 MOFs | *Columns*: evidence cells |

The autops-agentic-framework is the **test engine** — not the central contribution. It generates the empirical scores that populate M × O × T. No scores are pre-filled; this document defines the measurement protocol.

### 1.1 Contribution and scope

The contribution is the **method**, not any single mission. It is a **multi-fidelity tradespace method** for selecting cognitive architectures for autonomous satellite operations:

- **M** — a mission-agnostic Satellite-System Profile (§2).
- **O** — an architecture axis on two orthogonal, literature-grounded dimensions, applied **per operational core**: cognitive **substrate** (symbolic / subsymbolic{RL, LLM} / hybrid; Brooks 1991; Kautz 2022; Garcez & Lamb 2023) × **action space** (reactive / agentic; CoALA, Sumers et al. 2024). See §3.
- **T** — profile-parametric tests grounded in CCSDS 520 (§5), so a score measured once maps, by construction, onto other profiles.
- **Coverage** — the valid M×O space (≈3.6·10⁵ cells, §3.4) cannot be evaluated exhaustively; it is covered by a **multi-fidelity surrogate** (§4; Forrester et al. 2008; Peherstorfer et al. 2018) calibrated by high-fidelity runs at **anchor missions**.

The scientific question O answers is **which space-oriented agent architectures fit which missions and tasks** — including how agentic architectures *scale* where feasible and across operations paradigms (CG / AG / AO / AH). This is the exposé's **RQ1** (cognitive architecture × paradigm → performance trade-space) and **RQ3** (scaling heuristics → *architecture-selection guidelines for a target mission profile*). ⚠️ The exposé predates this M×O×T framing and needs a reconciliation pass; it is a draft, not ground truth. **EventSat** (an event-camera CubeSat) is the first high-fidelity anchor: the setting where the reasoning-load-bearing cells (VoI triage, multi-objective trade-offs, predictive FDIR, tool-use) are measured directly rather than projected. Flamingo (multi-satellite) and a large-scale constellation study are the further anchors. *EventSat is where the method is measured, not what the method is about.*

### 1.2 Research questions

The full RQ set lives in the exposé (`papers/expose/rqs.tex`): **RQ1** cognitive architecture (substrate × action × paradigm → performance trade-space), **RQ2** agent organisation (SAS vs MAS; architecture–organisation matching; explainability × safety), **RQ3** scale & complexity (degradation 1→~100 sats, extrapolation with quantified uncertainty, scaling heuristics → architecture-selection guidelines).

### 1.3 Design principles

1. **Orthogonal, per-core axes — not flat enumeration.** O is (substrate × action) per active core (§3); coverage is multi-fidelity (§4). No behaviour or decision-procedure peer-axis.
2. **Modularity** — components swap without affecting others.
3. **Reproducibility** — configuration-driven, seed-controlled (same config + seed → identical metrics).
4. **Fair comparison** — identical environment, memory, and metrics across variants (the memory invariant; §3).
5. **Scientific rigour** — every design element cites a specific paper or standard; counts are regenerated from rules, never asserted.

---

## 2. Axis 1 — Satellite System (M)

M is a five-dimensional profile vector. Every mission maps to a **Satellite System Profile (SSP)** code `[A][B][C][D][E]`.

### SS-A — Payload / Mission Function

Mission **function only** — what the satellite is *for* — kept **orthogonal to the other M sub-dimensions**: sensor modality is a *sub-mode* (not a class), orbit/latency → SS-D, ISL → SS-E, constellation scale → SS-C, platform/power → SS-B. Grounding: mission-type → autonomy-driver mapping (Amini et al. 2020); mission-function taxonomy (Boden & Larson 2006; Sellmaier et al. 2022).

| Code | Mission function | Sub-modes | Key autonomy driver |
|------|-----------------|-----------|---------------------|
| A1 | Earth Observation | optical · SAR · hyperspectral · **event-driven** | Reactive tasking, onboard data triage (VoI/AoI), quality gating |
| A2 | Communications | bent-pipe · regenerative · store-and-forward | Contact scheduling; autonomous routing/handover *when* ISL present (→ SS-E) |
| A3 | Navigation / PNT | — | Onboard orbit determination, clock & ephemeris integrity, signal-integrity FDIR |
| A4 | Science | altimetry · gravimetry · heliophysics · formation flying | Calibration automation, formation maintenance, onboard science targeting |
| A5 | Space Situational Awareness | — | Reactive sensor retasking, conjunction assessment |
| A6 | Planetary & Exploration | orbiter · lander · rover | Onboard GNC, AEGIS-class autonomous science targeting (deep-space latency → SS-D) |
| A7 | Technology Demonstration | — | Autonomy-frontier; relaxed risk posture, all architectures under test |

EventSat is **A1, event-driven EO** — the event camera is a sensor modality, not a mission class.

### SS-B — Engineering Tier

Limits the **onboard** representation substrate via the platform power/compute budget. The binding rules are R-COMPUTE1/2 (§3.2); the ground core is never SS-B-constrained.

| Code | Class | Mass | Power budget | Compute class |
|------|-------|------|-------------|---------------|
| B1 | CubeSat | <24 kg | 1–20 W total | OBC to Jetson Orin Nano (≤15 W) |
| B2 | SmallSat | 12–150 kg | 10–200 W | Jetson + FPGA coprocessor |
| B3 | MedBus | 150–1000 kg | 200 W–2 kW | Radiation-tolerant GPU / space AI ASIC |
| B4 | LargePlt | >1000 kg | 2–30 kW | Redundant compute complex |

### SS-C — Constellation Scale

| Code | N | Autonomy rationale |
|------|---|--------------------|
| C1 | 1 | Driven by latency or cost, not scale |
| C2 | 2–10 | Threshold for heterogeneous coordination automation |
| C3 | 10–100 | Routine scheduling and FDIR must be fully automated |
| C4 | 100–1000 | No per-satellite operator attention feasible (Nag et al. 2020) |
| C5 | >1000 | Fleet-level policy only |

CG and AG remain valid comparison baselines at all scales; they are operationally insufficient at C3–C5 but are retained as reference cells.

### SS-D — Communication Latency Regime

Autonomy is physically required when \(\tau_{comm} > \tau_{dyn}\) for a given function (Amini et al. 2020).

| Code | Regime | Typical RTT | Hard autonomy floor |
|------|--------|------------|---------------------|
| D1 | LEO near-real-time | 37–214 ms | AOCS and sub-second FDIR always onboard |
| D2 | GEO | ~250–600 ms | All FDIR onboard mandatory |
| D3 | Lunar | ~2.56 s | EDL and proximity ops fully autonomous |
| D4 | Mars | 6–44 min | All tactical functions autonomous |
| D5 | Deep Space | >1 hr | Ground provides strategic goals only |

Key SSP variables derived from SS-D (used in test thresholds):

| Symbol | Definition |
|--------|-----------|
| \(T_{orb}\) | Orbital period |
| \(B_{dl}\) | Daily downlink budget: \(C_{link} \times \bar{T}_{pass} \times N_{passes/day}\) |
| \(T_{pass}\) | Ground pass duration range |
| \(\lambda_{anomaly}\) | Anomaly rate (per orbit, mission-defined) |

### SS-E — ISL Topology

| Code | Description |
|------|------------|
| E0 | No ISL |
| E1 | Intra-plane ISL only |
| E2 | ISL planned / under validation |
| E3 | Full-mesh ISL |

### 2.1 M-level validity rules

Two kinds of rule, kept distinct: **hard gates** remove combinations that are structurally impossible; **soft flags** mark combinations with no mission-realistic counterpart today — they stay in the tradespace (future missions may populate them) but receive no high-fidelity sampling budget (§4).

| Rule | Kind | Constraint | Rationale |
|------|------|-----------|-----------|
| R-ISL | **hard** | SS-E > E0 requires SS-C ≥ C2 | No inter-satellite link on a single satellite |
| R-LAT | **hard** | \(\tau_{comm} > \tau_{dyn}(\text{fn})\) ⇒ that function must be autonomous | Amini et al. 2020 (latency primacy; constrains O per function, not SSP validity) |
| R-GEO | soft | D2 (GEO): SS-C ≤ C3 | GEO arc capacity physically limits large constellations |
| R-D3/D4 | soft | D3/D4: SS-C ≤ C3 | Documented lunar/Mars assets are small constellations (Mars Relay Network ~5 assets; Lunar Gateway multi-element; Walker-constellation studies exist at concept level) |
| R-D5 | soft | D5 (>1 hr RTT): SS-C ≤ C2 | No documented deep-space constellation beyond single relay spacecraft |

**Valid SSPs: 2,380** (of 5·4·5·7·4 = 2,800 unconstrained; R-ISL removes 420). Counts are regenerated from the rule set by the enumeration script (§7), never hand-frozen.

### 2.2 Reference SSPs (15 profiles)

| SSP | Description | A | B | C | D | E |
|-----|-------------|---|---|---|---|---|
| 01 | GEO flagship telecomms | A2 | B4 | C1 | D2 | E0 |
| 02 | LEO comms, no ISL, medium fleet | A2 | B2 | C3 | D1 | E0 |
| 03 | LEO broadband mega-constellation | A2 | B2 | C5 | D1 | E3 |
| 04 | Event-camera CubeSat — event-driven EO (**primary anchor**) | A1 | B1 | C1 | D1 | E0 |
| 05 | Agile optical single satellite | A1 | B2 | C1 | D1 | E0 |
| 06 | Agile optical medium constellation | A1 | B2 | C3 | D1 | E0 |
| 07 | SAR medium constellation | A1 | B2 | C3 | D1 | E0 |
| 08 | Hyperspectral science | A1 | B3 | C2 | D1 | E0 |
| 09 | SSA small constellation (**planned anchor**) | A5 | B2 | C2 | D1 | E2 |
| 10 | SSA large constellation | A5 | B2 | C3 | D1 | E3 |
| 11 | Earth science formation flying | A4 | B3 | C2 | D1 | E1 |
| 12 | Mars orbiter | A6 | B3 | C1 | D4 | E0 |
| 13 | Lunar lander/rover | A6 | B3 | C1 | D3 | E0 |
| 14 | Mega-constellation (large scale, **planned**) | TBD | B3 | C5 | D1 | E3 |
| 15 | Autonomy tech demo (OPS-SAT class) | A7 | B1 | C2 | D1 | E0 |

---

## 3. Axis 2 — Operations System (O)

### 3.1 Morphological Space

An operations-system candidate is a point in a small, **mission-agnostic** categorical space — that mission-independence is exactly what lets the *same* O axis tile across every mission in M×O×T. None of the axis values name a scenario.

**Axis A — Organisation** (Kim et al. 2025: an agent system is **S = (A, E, C, Ω)** — agents · environment · communication · orchestration)

| Token | Label |
|-------|-------|
| `sas` | Single-agent system |
| `cmas` | Centralised MAS |
| `dmas` | Decentralised MAS |
| `imas` | Independent MAS |
| `hmas` | Hierarchical MAS |

Kim et al.'s 180-config study sets the RQ2 priors: **capability saturation** (multi-agent gains vanish once a single agent clears ~45%), **topology-dependent error amplification** (independent agents 17.2× vs centralized 4.4×), and **task-type dependency** — *sequential constraint satisfaction degrades under every MAS variant*. Satellite mode selection *is* sequential constraint satisfaction → **prediction: centralized (CMAS) ≥ distributed** (testable RQ2 hypothesis).

**Axis B — Representation (cognitive substrate × action space)** — two **orthogonal** sub-dimensions (Brooks 1991; Kautz 2022; Garcez & Lamb 2023; Colelough & Regli 2025).

**B.1 — Substrate:**

| Substrate · core | What the cognition is | Reasoning |
|---|---|---|
| **Symbolic** | Explicit rules / models / ontologies | Deductive, constraint-based |
| **Subsymbolic · RL** | Learned connectionist policy (network weights) | Statistical, distributed |
| **Subsymbolic · LLM** | A language model as the cognitive core | Statistical, distributed |
| **Hybrid** | A *genuine combination* of symbolic + subsymbolic | System-1 neural + System-2 symbolic structure (Kahneman; Kautz Type 2/3) |

A bare LLM — even one whose output is masked by a validity check — is **subsymbolic · LLM**: the cognition is purely neural, the symbolic part is I/O guarding. It becomes **hybrid** only when symbolic structure enters the *reasoning loop*: an LLM invoking symbolic tools, an RL policy wrapped by symbolic safety rules, or a symbolic planner feeding a neural executor.

**B.2 — Action space** (orthogonal to substrate):

| Action space | What it is | Valid for |
|---|---|---|
| **reactive** | Single-shot stimulus → (constrained) response | all substrates |
| **agentic** | A tool / internal-action repertoire run under a control loop — CoALA's *reasoning · retrieval · learning* internal actions plus external actions (Sumers et al. 2024) | LLM-bearing only: **subsymbolic · LLM**, **hybrid** |

Symbolic and subsymbolic·RL are reactive by nature (rules fire once; an RL policy is one forward pass per step). Agentic requires a core that can *issue internal actions* — an LLM. The two agentic cells differ by what the loop touches: **subsymbolic·LLM·agentic** is a pure-neural reasoning loop (no symbolic tools); add symbolic tools or a structured writable store and the same loop becomes **hybrid·agentic**.

This yields **6 per-core cells**: symbolic·re · RL·re · LLM·re · LLM·ag · hybrid·re · hybrid·ag.

The mapping from these six cells to concrete representation classes and config tokens is **implementation-level and lives in `implementations.md`** (the component registry); the legacy tokens are scheduled for renaming during the engine adaptation, and this spec speaks only the cell names. Two cells are expressible but not yet instantiated: *subsymbolic·LLM·agentic* (pure reasoning loop, no symbolic tools) and *hybrid·reactive* (e.g. an RL policy wrapped by symbolic safety rules).

**Axis C — Operations paradigm** (defines which *autonomy slots* are active)

Two slots: a **ground planner** (long-horizon, emits the uplinked whole-pass schedule; active in AH and AG) and an **onboard** core (closed-loop, per-step; active in AO and AH).

| Token | Label | Onboard slot | Ground slot | Key isolation |
|-------|-------|-------------|------------|---------------|
| `ao` | Autonomous Onboard | ✓ per-step | — | shares onboard with AH → AO↔AH: does a ground plan help? |
| `ah` | Autonomous Hybrid | ✓ per-step | ✓ planner | both slots active |
| `ag` | Autonomous Ground | — | ✓ planner | shares planner with AH → AH↔AG: onboard per-step override effect |
| `cg` | Conventional Ground | — | ✓ (human, one-pass delay) | operational baseline |

The ground planner is the **same artifact** in AH and AG, so AH↔AG isolates the onboard override as the only moving part.

**CG semantics.** CG is the conventional-ground *process* — whole-pass schedules released by a human flight controller, one-pass delay — and the substrate names the ground *decision-support core* drafting what the human releases. This is how AI-augmented mission control (Palladino et al. 2026) enters the tradespace, but only in its **decision-policy dimension**: the framework scores what the released schedule achieves, not the human factors (workload, trust, HCI), which sit in the deliberately unexercised Manual Operations MOF (§5.1). Within the engine a CG·LLM cell is behaviourally an AG·LLM cell plus the human-release delay — so the hypothesis that CG cells collapse toward their reactive/algorithmic baseline is *testable, not assumed*: if the V4 forward check (§5.5) finds no substrate variance within the CG column, it is pruned to the algorithmic baseline and the question is answered empirically.

**Substrate × action applies per *active core*, not per system.** CG / AG / AO have one active core → one (substrate, action) coordinate. **AH has two** → an AH architecture is a **pair** ⟨onboard | ground⟩. "Hybrid" therefore arises two ways: **within one core** (a neurosymbolic core) or **across the two AH slots** (e.g. symbolic/RL onboard + LLM/agentic ground). **AH pairing is the dominant contributor to O-space size** (§3.4).

**No separate behaviour axis — learning is folded in (CoALA-faithful).** In CoALA, *learning* is one of the agent's internal actions, not an architectural axis. It splits into two things already in the model: **offline adaptation** — a property of how a core was *produced* (symbolic = never trained; subsymbolic·RL = trained by construction; LLM = zero-shot *or* prompt-optimised); **online learning** — CoALA's *learning* internal action (writing to a semantic/episodic store), available **iff agentic**. Prior comparisons survive re-expressed: zero-shot vs prompt-optimised LLM (offline adaptation); and an agentic core whose internal actions include learning — writes to a structured semantic/episodic store — against the same core without that action (online learning).

**Decision procedure (SDA / OODA / ReAct) is also not a tradespace axis** — held at the default (`sda`). For deterministic substrates the three collapse to identical decisions; for LLM cells the difference is an internal control-flow detail that does not interact with mission selection.

**O-cell descriptor:** (Organisation · per-core Substrate × Action-space · Paradigm).

### 3.2 Validity rules (O and O×M)

Validity follows from the coherence of the orthogonal axes plus explicit gates. As at M-level, **hard** gates exclude; **soft** flags only steer sampling.

**Structural (substrate × action):**

| Substrate · core | reactive | agentic |
|---|---|---|
| Symbolic | ✓ | — |
| Subsymbolic · RL | ✓ | — |
| Subsymbolic · LLM | ✓ | ✓ |
| Hybrid | ✓ | ✓ |

Agentic is invalid for symbolic and subsymbolic·RL: neither core can issue the internal actions (reasoning / retrieval / memory-write) an agentic loop is made of.

**Organisation gates:**

| Rule | Kind | Constraint | Rationale |
|------|------|-----------|-----------|
| R-ORG1 | **hard** | `cmas` is valid **iff paradigm = AH**, at SS-C ≥ C2. At C1, CMAS·AH is *identified with* SAS·AH: the orchestrator Ω with a single managed agent reduces to the AH ground planner that SAS·AH already has — one cell, counted once. For CG/AG/AO there is no coordination structure → `cmas` excluded. | Kim et al. 2025 S=(A,E,C,Ω); degeneracy at N=1 |
| R-ORG2 | **hard** | `dmas` · `imas` · `hmas` require SS-C ≥ C3 | Distributed coordination is meaningful only at fleet scale (Flamingo+) |
| R-ORG3 | **hard** | `dmas` requires SS-E ≥ E1 (peer coordination needs an inter-satellite channel); `imas` needs none (no coordination by definition); `hmas` may route hierarchy via ground → soft flag at E0 | Decentralised consensus without ISL is structurally impossible |

**Compute gates (onboard slot only; the ground core is unconstrained):**

| Rule | Kind | Constraint | Rationale |
|------|------|-----------|-----------|
| R-COMPUTE1 | **hard** | Onboard **LLM** core (subsymbolic·LLM or hybrid·LLM, in AO/AH onboard slot) requires SS-B ≥ B2 | A B1 CubeSat (1–20 W *total*) cannot sustain continuous LLM inference at the decision cadence on top of payload + bus loads. **Locked assumption, falsifiable** — supersedes the earlier "feasible at B1+ with quantised models" note: quantised 7–8B inference on Orin Nano is physically demonstrable but not operationally sustainable as a per-step flight core at B1 power budgets. |
| R-COMPUTE2 | **hard** | Onboard **agentic** loop (multi-call tool loop per decision) requires SS-B ≥ B3 | Agentic loops multiply inference cost per decision |

R-COMPUTE1/2 gate the **onboard slot of the architecture against the SSP**, not the test engine: onboard-LLM cells *simulated* in the EventSat scenario remain valid evidence — they inform the B2 neighbourhood of the surrogate (e.g. SSP-05, agile optical B2/C1), while SSP-04 (B1) scores its LLM-bearing cells only where the LLM sits on the ground (CG/AG ground core, AH ground slot). Onboard RL at B1 is valid (Jetson-class inference, single forward pass).

**Conservative LLM-bearing convention:** hybrid·reactive can be instantiated without an LLM (RL + symbolic rules). For fidelity budgeting (§4) all hybrid cells are counted **LLM-bearing** unless a concrete instantiation is LLM-free; the enumeration stays conservative.

### 3.3 ECSS / SAL Autonomy Mapping (informative, self-contained)

- **SAL 0–5** (Baker & Phillips 2025): 0 Teleoperated · 1 Supervised · 2 Conditional · 3 Delegated · 4 Supervised-autonomy · 5 Full. SAL carries the **decision-sophistication** gradient.
- **ECSS-E-ST-70-11C Rev.1 (2024)**, three function categories (§5.7), verified against source — a cell carries an (E, D, F) triple:
  - **E — execution** (Table 5-1, E1–E4 verbatim): E1 ground real-time control · E2 pre-planned ops via on-board scheduler · E3 adaptive/event-based ops · E4 goal-oriented on-board re-planning.
  - **D — mission *data management*** (Table 5-2, D1–D2 verbatim): D1 on-board storage of *essential* data through a ground outage · D2 on-board storage of *all* mission data (space-segment independence from ground).
  - **F — on-board fault management (FDIR)** (§5.7.5): requirements-based — Rev.1 does **not** number F-levels. "F2" below is operational shorthand for autonomous recovery without escalating to a pass; "F1" = recovery only via ground hand-over (§5.7.5.c makes ground the *highest* escalation instance).
- **Structural floor:** operational "F2" is reachable only by paradigms with an onboard slot (AO/AH); AG/CG sit at "F1" by construction.

| Cell (paradigm · substrate) | SAL | E — execution | D — data management | F — FDIR (operational) |
|---|---|---|---|---|
| CG · Symbolic | 1 Supervised | E1 ground-triggered | D1 essential-data buffering | F1 ground-recovered |
| AG · Symbolic | 2 Conditional | E2 stored sequence | D1 | F1 |
| AG · Subsymbolic (RL / LLM) | 2 | E2 | D1 | F1 |
| AG · Hybrid | 2–3 | E2 | D1–D2 | F1 |
| AO · Symbolic | 2–3 Delegated | E2–E3 event-driven | D2 onboard data independence | F2 autonomous recovery |
| AO · Subsymbolic · RL | 3 | E3 | D2 | F2 |
| AO · Hybrid · reactive (RL + rules) | 3 | E3 | D2 | F2 |
| AH · Symbolic | 2–3 | E2–E3 | D1–D2 | F2 |
| AH · Subsymbolic · RL | 3 | E3 | D2 | F2 |
| AH · Subsymbolic · LLM (agentic) | 3–4 Supervised-auton. | E3–E4 | D2 | F2 |
| AH · Hybrid · agentic | 4–5 Full | E3–E4 goal-directed | D2 | F2 (+ predictive FDIR as a research extension, not an ECSS level) |

### 3.4 Enumeration (regenerated from the rules, never asserted)

With 6 per-core cells, CG/AG/AO carrying one core and AH carrying an ⟨onboard | ground⟩ pair, and onboard cells gated by SS-B — ob(B1)=2, ob(B2)=4, ob(B3)=ob(B4)=6 — the O-space per organisation:

| Org | Unique O configs (B3+) | LLM-bearing | RL-bearing | Scope |
|-----|----------------------|-------------|------------|-------|
| sas | 54 | 44 | 14 | always valid |
| cmas | 36 | 32 | 11 | AH only, SS-C ≥ C2 (identified with SAS·AH at C1) |
| dmas | 54 | 44 | 14 | SS-C ≥ C3 ∧ SS-E ≥ E1 (R-ORG3) |
| imas | 54 | 44 | 14 | SS-C ≥ C3 |
| hmas | 54 | 44 | 14 | SS-C ≥ C3 (soft flag at E0: hierarchy routed via ground) |
| **Total** | **252** | **208** | **67** | |

LLM- and RL-bearing **overlap**: a mixed AH pair ⟨RL onboard | LLM ground⟩ (or vice versa) carries both cores and needs **both fidelity ladders** (§4.2) — the columns do not sum to the total.

Per-SSP O-cell counts vary by (SS-B, SS-C), the two gating dimensions:

| SSP | Profile | O-cells | LLM-bearing | RL-bearing |
|-----|---------|---------|-----|---------|
| 04 EventSat | A1/B1/C1/D1/E0 | **26** | 16 | 10 |
| 15 TechDemo | A7/B1/C2/D1/E0 | **38** | 24 | 17 |
| 05 Agile 1-sat | A1/B2/C1/D1/E0 | **40** | 30 | 12 |
| 01 GEO telecomms | A2/B4/C1/D2/E0 | **54** | 44 | 14 |
| 12 Mars orbiter | A6/B3/C1/D4/E0 | **54** | 44 | 14 |
| 09 SSA small const. | A5/B2/C2/D1/E2 | **64** | 50 | 21 |
| 11 Formation flying | A4/B3/C2/D1/E1 | **90** | 76 | 25 |
| 06 Agile med. const. | A1/B2/C3/D1/E0 | **144** | 110 | 45 |
| 03 Mega-constellation | A2/B2/C5/D1/E3 | **184** | 140 | 57 |
| 10 SSA large const. | A5/B2/C3/D1/E3 | **184** | 140 | 57 |

EventSat (B1/C1) has the **smallest O-space (26 cells)** — favourable for the primary high-fidelity anchor. The C1→C3 jump is large: the distributed organisations unlock with their full AH cross-products (dmas only where ISL exists, per R-ORG3).

**Full M×O size** (2,380 SSPs × per-SSP O-cells), verified by enumeration:

| Rule set | Total M×O cells | LLM-bearing | RL-bearing | symbolic-only |
|---|---|---|---|---|
| **adopted:** {R-ISL, R-ORG1/2/3, R-COMPUTE1/2} | **364,980** | **286,020** | **108,780** | 30,240 |
| without R-ORG3 (comparison only) | 383,250 | 300,090 | 114,030 | 31,920 |

(60,060 mixed AH pairs are counted in both the LLM- and RL-bearing columns — they need both ladders; total = LLM + RL − mixed + symbolic-only.)

These numbers exist to size the coverage problem (§4), not to be run: the LLM-bearing **and RL-bearing** blocks are both surrogate-covered, each on its own fidelity ladder (§4.2); only the symbolic cells are directly computable at negligible cost.

---

## 4. Multi-fidelity evaluation over M × O

### 4.1 Response regimes

The score surface over M×O is **not smooth**: it contains discontinuities *by design* — regime boundaries, not noise. Each regime gets a different treatment:

| Regime | Cells | Treatment |
|--------|-------|-----------|
| **Deterministic-cheap** | symbolic·reactive | Run everywhere needed at full fidelity — evaluation is negligible; the one regime that needs no ladder |
| **Trained-deterministic** | subsymbolic·RL | Surrogate-covered, on the **RL fidelity ladder** (§4.2): the cost driver is *training* (hours per (M,O) point), so the rungs cheapen training — transfer, truncated budget — not inference. "Non-LLM = free" is false for RL. |
| **LLM-stochastic** | subsymbolic·LLM + hybrid, all paradigms | Surrogate-covered, on the **LLM fidelity ladder** (§4.2): the cost driver is inference |
| **Structural** | CG/AG on onboard-required tests; gated cells (R-COMPUTE, R-ORG) | Hard-coded zeros / ceilings / N/A per §5.3 — **excluded from the surrogate entirely** |

Examples: CG/AG score 0 on AU.02 (no onboard slot — structural zero); symbolic scores 1.0 on AN.01 (control ceiling); PL.07 is N/A for symbolic (no learned policy); onboard-LLM cells at B1 are gated out (R-COMPUTE1), not scored low.

### 4.2 Fidelity ladders — parallel per family, plus tier 0

A **rung** is one step of a fidelity ladder: a concrete *evaluator* of a cell at a given cost/accuracy point. Ladders are **parallel per substrate family** — each family's rungs cheapen *its own* cost driver (RL: training GPU-hours, offline; LLM: inference, online), and rungs are never comparable across families. The surrogate's acquisition chooses **which cell, on which ladder, at which rung** (multi-fidelity in the general sense of Peherstorfer et al. 2018: multiple information sources of differing cost and accuracy).

**Tier 0 — the surrogate posterior — is not a rung.** It is the fusion layer's *estimate with quantified uncertainty*, and it is how most of the ≈3.6·10⁵ cells are covered. "Predicted" (tier 0) and "measured cheaply" (an LF rung) are different epistemic categories, kept distinct throughout — a GP is never trained on its own outputs.

| Family | LF rungs *(indicative — tuned per anchor at implementation time, not locked)* | HF |
|---|---|---|
| Symbolic | none needed — single cheap fidelity | full run |
| Subsymbolic · RL | **R1** anchor-policy transfer (zero-shot at the target cell, no training) · **R2** truncated training — checkpoints saved at fixed fractions of the convergence budget (successive-halving/Hyperband tradition, Li et al. 2018) | **R3** full training + full evaluation |
| LLM-bearing | **L1** small/quantised model + completion cache | **L2** full model (`qwen3.5:122b`) at anchors |
| AH mixed pairs | rung = ⟨onboard rung, ground rung⟩; cheapest meaningful: R1 × L1 | R3 × L2 |

- **Cross-cutting fidelity knob:** the statistical budget (episodes × steps × seeds) is orthogonal to the evaluator and applies to every family, including symbolic.
- **Bias directions differ — the acquisition must know it.** R1/R2 *underestimate* the cell's achievable score by construction (the policy is not converged for the target cell), so the LF-RL bias direction is known a priori; L1's bias direction is unknown a priori. Co-kriging's \(\rho\)-scaling absorbs systematic bias, but cell-ranking decisions may legitimately trust LF-RL data differently from LF-LLM data — stated here, not hidden.
- A fixed-response *mock* is **not** a rung — it has zero correlation with the real response; `llm_mock: true` is CI/smoke-testing only and never scores a cell. Orekit propagation is sub-second and stays on at every rung.

*(This section supersedes the earlier lock "the fidelity ladder lives only in LLM-bearing cells" — unlocked 2026-06-11.)*

### 4.3 Two loops, kept strictly separate

1. **Outer loop — the surrogate** decides *which (M, O) point to evaluate, on which ladder, at which rung*: GP posterior + active-learning acquisition (EI / entropy search). Acquisition respects the soft flags (§2.1): no HF budget on mission-unrealistic combinations (e.g. D5×C4+). Many AH pairs are near-redundant; the posterior tightens quickly and deprioritises them.
2. **Inner loop — per-evaluation cost reduction, family-specific.** Cheapens *one* evaluation at its chosen rung:
   - **LLM — FrugalGPT** (Chen et al. 2023): completion cache (`data/llm_cache/`), model approximation/distillation, cascade + reliability scoring. The accept-or-escalate scorer is the per-call analogue of the surrogate's trust-or-sample rule.
   - **RL — checkpoint machinery**: training checkpoints saved at fixed fractions of the convergence budget (these *are* the R2 rungs; joint with the RLlib collaboration), a **checkpoint library** for warm-starting neighbouring cells instead of training from scratch (the RL analogue of the completion cache), and early stopping on learning-curve extrapolation (successive halving). The R1→R2→R3 escalation under an accept-or-escalate rule mirrors the FrugalGPT cascade exactly.

Conflating the loops was the historical error; they compose at different granularities.

### 4.4 Surrogate design

A single global stationary GP is **incorrect** — it assumes a continuous input space and a smooth response, and the surface has neither (§4.1). Required structure:

1. **Partitioned / treed GPs** (Gramacy & Lee 2008): partition M×O by regime; fit local GPs only within smooth LLM-stochastic sub-regions; structural cells live outside the GP. Forrester et al. (2008) co-kriging supplies the multi-fidelity *concept*, not a prescription for global application.
2. **Kernel over M** — a product of per-sub-dimension kernels: SS-C, SS-D **ordinal** (the orderings C1<…<C5, D1<…<D5 are meaningful); SS-B, SS-E **categorical/Hamming**; SS-A **group-similarity** (A1 EO ↔ A5 SSA closer, autonomy-wise, than A1 ↔ A6 Planetary).
3. **Kernel over O** — fully categorical; candidate encodings: Hamming distance on the (organisation, substrate, action, paradigm) tuple; a learned embedding into \(\mathbb{R}^d\) + RBF; for AH pairs, the sum of onboard-slot and ground-slot kernel distances.
4. **Noise term mandatory:** \(f_H(x) = \rho\, f_L(x) + \delta(x) + \varepsilon\), \(\varepsilon\sim\mathcal N(0,\sigma^2)\) — the noise source differs per family: LLM outputs are stochastic at fixed input; RL scores vary across training seeds. Both \(\sigma^2\) are estimated from repeated anchor-cell runs (inference repeats / seed repeats respectively).

### 4.5 Open empirical questions (methodology validation)

| # | Question | Method | RQ |
|---|----------|--------|----|
| 1 | Cross-SSP score correlation: does the EventSat anchor predict other SSPs in the same SS-A class? | Compare HF scores across 2–3 anchor SSPs | RQ3 |
| 2 | AH onboard dominance: does the onboard core dominate AH scores, collapsing the effective O-space? | Ablation: fix onboard, vary ground substrate | RQ2 |
| 3 | LF–HF correlation \(\rho\): is the small model a reliable proxy for `qwen3.5:122b`? | Repeated runs on anchor cells at both rungs | methodology |
| 4 | Scale inflection: at which SS-C do dmas/imas/hmas overtake sas/cmas? | Surrogate extrapolation from the Flamingo anchor | RQ3 |
| 5 | Surrogate family: global vs regional GPs; which FrugalGPT techniques enter the L1 rung at what accuracy cost? | Pilot comparison | methodology |
| 6 | LF–HF correlation for RL: is the anchor-transferred policy (R1/R2) a reliable proxy for full training (R3) — and how is the transfer penalty structured across the M-slice? | Both rungs at anchor cells (the RL sibling of Q3) | methodology + RQ3 |

---

## 5. Axis 3 — Tests (T)

All thresholds are functions of SSP variables (\(T_{orb}\), \(B_{dl}\), \(\lambda_{anomaly}\), \(T_{pass}\), …) — never mission-specific constants — so the T axis applies unchanged to **every valid M×O cell**; what varies per cell is applicability (§5.3), not the instrument.

### 5.0 Metric · test · benchmark — three levels of one instrument

The three words name *different levels of the instrument*, not synonyms (JCGM 200 / VIM: *measurand* · *measurement procedure* · *measurement result*; benchmark tradition: Gray 1993; Huppler 2009):

| Level | What it is | Defines | Lives in |
|---|---|---|---|
| **Metric** (M-01…M-14) | A *quantity definition* for a construct: formula + normalisation + validity conditions — the VIM *measurand* | **what** is quantified | §5.2 |
| **Test** (MC.01…AN.01) | A *standardised measurement procedure*: SSP-parametric stimulus + applicability + primary metric + threshold + scoring rule | **under which conditions** the number is produced, and **which O-cell contrast** it resolves | §5.4 |
| **Benchmark** | The *fixed, versioned suite*: tests × metrics × scenarios × seeds × applicability states × aggregation (§6) × analysis protocol (§5.6) | **comparability** — across O-cells, SSPs, time, and users of the method | this document |

The relations are many-to-many — one metric serves several tests (M-01 under PL.01, PL.02, AU.06 stimuli); one test may read several metrics (PL.03 co-reports M-06, M-05, M-08) — and validity attaches **per level**: a formula can fail to measure its construct (metric-level), a procedure can read a valid metric under a stimulus it cannot see (test-level), and reproducibility/fairness are properties of the whole protocol (benchmark-level). Each level can fail independently, which is why each is validated separately (§5.5).

**Benchmark desiderata, mapped** (Gray 1993: *relevant · portable · scalable · simple*; Huppler 2009: *relevant · repeatable · fair · verifiable · economical*): **portable** = SSP-parametric thresholds; **scalable** = the SS-C axis + multi-sat tests; **repeatable** = config + seed determinism + the pre-registered protocol (§5.6); **fair** = identical environment, memory invariant, paired launch-lottery seeds; **verifiable** = the coverage matrices (§5.5); **economical** = the multi-fidelity ladder (§4); **relevant/simple** = discrimination-density weighting + the sparse cell-state convention (§5.3). There is no field-standard autonomous-operations benchmark to inherit — building one that satisfies this checklist *is* the contribution (§1.1).

### 5.1 Why these tests — provenance and coverage

**Function space (closed, verified).** The taxonomy is anchored on **Table 3-1 "Mission Operations Functions"** of CCSDS 520.0-G-3 (Issue 3, Dec 2010, §3.4), which lists **exactly 10 functions**, verbatim:

> Monitoring and Control · Manual Operations · Automation · Planning · Software Management · Flight Dynamics · Time Management · Location · Analysis · Data Product Management

**6 are exercised** — those an autonomous-operations architecture can plausibly differ on:

| Exercised MOF | Code | Why it discriminates architectures |
|---|---|---|
| Monitoring and Control | **MC** | anomaly detection / observability / command verification |
| Automation | **AU** | FDIR recovery, schedule execution, lights-out endurance |
| Planning | **PL** | schedule generation, replanning, multi-objective and multi-agent allocation |
| Flight Dynamics | **FD** | onboard OD/propagation, collision avoidance |
| Data Product Management | **DP** | AoI/VoI-driven data triage and buffer management |
| Analysis | **AN** | trending, anomaly reporting, explainability |

The other 4 are deliberately not exercised (a scoping decision, stated): **Manual Operations** is the *absence* of autonomy, captured by the CG baseline; **Software Management** and **Time Management** are infrastructure functions orthogonal to the architecture comparison; **Location** is partially folded into FD.01. **FDIR is not a top-level MOF**: fault detection sits under Monitoring & Control, autonomous recovery under Automation — which is why the FDIR tests are split across MC and AU.

**Stressor space (closed by declaration — an assumption, §5.5).** Every test is a *function × stressor* pair. Six stressor classes, grounded in the ECSS-E-ST-70-11C contingency framing (faults, resource management, ground-link dependence) and the operational-contingency literature:

S1 nominal load · S2 fault/anomaly · S3 resource scarcity & degradation · S4 comms/link denial · S5 time-critical transient · S6 scale & coordination

**Coverage table — the test catalogue is the populated function × stressor matrix:**

| | S1 nominal | S2 fault | S3 resource | S4 comms denial | S5 transient | S6 scale |
|---|---|---|---|---|---|---|
| **MC** | MC.01 | →AU.01/02 ⁽ᵃ⁾ | →PL.03 ⁽ᵃ⁾ | MC.02 · MC.03 | MC.04 | **gap** ⁽ᶜ⁾ |
| **AU** | AU.04 | AU.01 · AU.02 | AU.03 | AU.06 | AU.05 | →PL.04–07 ⁽ᵃ⁾ |
| **PL** | PL.01 | PL.02 | PL.03 | PL.06 | →AU.05 ⁽ᵃ⁾ | PL.04 · PL.05 · PL.07 |
| **FD** | — ⁽ᵇ⁾ | — ⁽ᵇ⁾ | — ⁽ᵇ⁾ | FD.01 | FD.02 | — ⁽ᵇ⁾ |
| **DP** | DP.01 | — ⁽ᵇ⁾ | DP.02 · DP.03 | →MC.02/AU.06 ⁽ᵃ⁾ | — ⁽ᵇ⁾ | — ⁽ᵇ⁾ |
| **AN** | **gap** ⁽ᶜ⁾ | AN.01 | — ⁽ᵇ⁾ | — ⁽ᵇ⁾ | — ⁽ᵇ⁾ | — ⁽ᵇ⁾ |

Empty cells carry one of three justifications — **(a) covered by a sibling-MOF test** (the CCSDS split puts fault *detection* under MC but recovery under AU; data-management-through-outage is exercised by MC.02/AU.06); **(b) void by design** — no architecture contrast exists in the cell (nominal orbit determination is identical across O-cells; FD under resource stress adds nothing FD.01/02 don't measure); **(c) open gap, named**: *MC × S6* (fleet-level monitoring) and *AU × S6* (fleet-level automation beyond allocation) are deferred to the multi-sat anchors; *AN × S1* (nominal trend analysis & reporting) is a candidate test pending a discriminating stimulus. Missing tests are found mechanically from this table, not by intuition.

**Why the count is uneven (7 Planning, 1 Analysis).** Weighting is by *discrimination density*, not equal allocation per MOF: Planning is where substrate and agentic reasoning are most load-bearing (RQ1/RQ3); Analysis collapses to one interpretability axis (M-08). §6 aggregates *per MOF group*, so the imbalance does not inflate Planning's contribution.

### 5.2 The metrics — definitions, hierarchy, rationale

#### Objectives hierarchy (why these fourteen)

The metric set is structured as a value hierarchy (Keeney & Raiffa 1976/1993: attribute sets must be *complete, operational, decomposable, non-redundant, minimal*): fundamental objectives on top, metrics as their attributes. This is what makes "why these metrics?" answerable — every metric is the attribute of a named objective, and every objective an operator of a satellite system recognisably holds:

| Fundamental objective | Metrics | Note |
|---|---|---|
| Mission value | M-01 | per SS-A function class (table below) |
| Information delivery | M-02/M-03 (timeliness) · M-11 (throughput) · M-12 (value) | known near-collinearity — pre-registered correlation check (§5.6) |
| Fault resilience | M-04 (in-episode recovery) · M-09 (cross-episode consistency) | two distinct facets, kept separate |
| Autonomy risk / burden | M-05 (system interventions) · M-13 (command safety) | adjacent constructs by design — the V3 correlation check applies to this pair first |
| Resource efficiency | M-06 | |
| Operations cost | M-14 | ground-segment effort; reads the MC.03 command ledger |
| Responsiveness | M-07 | |
| Interpretability | M-08 | |
| Scalability | M-10 | RQ3 axis; degenerate at C1 |

#### Normalisation policy

- **(A) Absolute, SSP-referenced** — raw value divided by an SSP-parametric reference (quarter-orbit, inter-pass interval, …), clamped to \([0,1]\). Preferred: scores comparable across missions by construction.
- **(R) Relative, min–max within the M-slice** — when no principled absolute reference exists (M-06). Valid only *within* an SSP slice; never compared across SSPs (standard MCDA practice; same normalisation as Kim et al. 2025).
- Intrinsic fractions (M-05, M-08, M-11, M-12) need no further normalisation.

#### Definitions

| ID | Metric | Symbol | Raw formula | Norm. | →[0,1] score | Primary citation | Required signals |
|----|--------|--------|-------------|:-----:|--------------|------------------|------------------|
| **M-01** | Mission Utility | \(U\) | \(U=\sum_j w_j\,\phi_j\), \(\sum w_j=1\), \(\phi_j\in[0,1]\) per SS-A | A | \(U\) (clamp) | CCSDS 520 §4; Boden & Larson 2006 | per-objective achievement ratios |
| **M-02** | Mean Age of Information | \(\bar\Delta\) | \(\bar\Delta=\frac1T\int_0^T\Delta(t)\,dt\) | A | \(\max(0,1-\bar\Delta/\Delta_{ref})\), \(\Delta_{ref}\)=mean inter-downlink interval | Kaul et al. 2012; Yates et al. 2021 | per-product age, clock at observation capture |
| **M-03** | Peak Age of Information | \(A_{peak}\) | \(\max_i(T_i+Y_i)\) | A | \(\max(0,1-A_{peak}/A_{ref})\), \(A_{ref}\)=max inter-pass gap | Costa et al. 2016 | same field as M-02 |
| **M-04** | Autonomous Recovery Efficiency | \(R_{FDIR}\) | \(\bar s_{rec}\) = steps from anomaly onset to (anomaly cleared ∧ nominal mode) | A | \(\max(0,1-\bar s_{rec}/(T_{orb}/4\Delta t))\) | ECSS Rev.1 §5.7.5; Gallon et al. 2024 | persistent anomaly-active state + mode |
| **M-05** | Safety-Override Rate (operator proxy) | \(OIR\) | \(N_{override}/N_{steps}\) | — | \(1-OIR\) | ECSS Rev.1 §5.7.3; Sellmaier et al. 2022 | requested vs resolved action per step |
| **M-06** | Resource Efficiency | \(\eta_R\) | \(U/\hat E\), \(\hat E=E_{consumed}/E_{budget}\) (SS-B) | R | min–max within M-slice | Boden & Larson 2006 | per-step energy + budget |
| **M-07** | Decision Latency | \(L_{dec}\) | mean wall-clock per decision cycle | A | \(\max(0,1-L_{dec}/L_{ref})\), \(L_{ref}=\min(\Delta t,\tau_{dyn})\) | Gallon et al. 2024; Amini et al. 2020 | per-cycle wall-clock |
| **M-08** | Explainability Coverage | \(\xi\) | \(N_{rationale}/N_{decisions}\) | — | \(\xi\) | ECSS Rev.1 §5.7.2 + §5.7.5(d–e); Palladino et al. 2026 | rationale per decision (+ quality score at H) |
| **M-09** | Robustness | \(CV_U\) | \(\sigma_U/\mu_U\) across MC episodes | A | \(1-\min(1,CV_U)\); **N/A if \(\mu_U\le\varepsilon\)** | Kim et al. 2025; Forrester et al. 2008 | per-episode utility |
| **M-10** | Scale Efficiency | \(\eta_{scale}(N)\) | \(U(N)/N\) normalised to \(U(1)\) | — | \(\eta_{scale}\in(0,1]\) | Kim et al. 2025; Nag et al. 2020 | multi-sat utility |
| **M-11** | Channel / Downlink Efficiency | \(\eta_{dl}\) | delivered / max-achievable through the SSP's primary channel | — | \(\eta_{dl}\) | CCSDS 520 Data Product Mgmt | delivered + max-achievable volume |
| **M-12** | Value-of-Information fraction | \(\xi_{VoI}\) | \(\sum_{delivered}V_j/\sum_{all}V_j\) | — | \(\xi_{VoI}\) | Howard 1966; Yates et al. 2021; Boden & Larson 2006 | per-product value weight \(V_j\) |
| **M-13** | Constraint-Violation Rate | \(p_{viol}\) | \(N_{viol}/N_{decisions}\); stochastic cores co-report \(pass^k\) (all \(k\) repeated trials violation-free) | — | \(1-p_{viol}\) | Yao et al. 2024 (τ-bench); Kim et al. 2025 (error amplification) | proposed action + declared-constraint evaluation per step; repeated-trial harness |
| **M-14** | Commanding Effort | \(E_{cmd}\) | \((N_{cmd} + w_m N_{manual})\) per mission-day | R | min–max within M-slice | Nag et al. 2020; Castano et al. 2022; Boden & Larson 2006 | per-command ledger (shared with MC.03); manual-intervention count |

#### Per-metric notes (where the table needs unpacking)

**M-01 — operationalised per SS-A function class.** The *form* is fixed (\(U=\sum_j w_j\phi_j\)); the sub-objectives are chosen per mission function so no single mission type is imposed:

| SS-A | Sub-objectives \(\phi_j\) (each normalised to its SSP target) |
|---|---|
| A1 EO | observation-time ratio; downlink ratio; (event-driven adds detection-recall) |
| A2 Comms | offered-throughput ratio; link availability; (store-and-forward: delivery ratio) |
| A3 Nav/PNT | ephemeris/clock-integrity uptime; signal-availability ratio |
| A4 Science | calibrated-acquisition ratio; (formation modes add formation-keeping accuracy) |
| A5 SSA | RSO detection ratio; timeliness \((1-\bar\Delta/\Delta_{ref})\) — couples to M-02 |
| A6 Planetary | mission-phase objectives achieved (GNC/targeting KPI), mission-defined |
| A7 TechDemo | mission-defined KPI (autonomy-frontier metric under test) |

**M-02/M-03 — the age clock starts at observation capture.** Age runs from the moment the generating observation is taken at the sensor, so age-at-delivery includes the full data pipeline (compression, detection, internal transfer, downlink wait) by construction. End-to-end / system latency is thereby **folded into AoI** rather than added as a separate metric — the AoI literature defines age from generation at the source (Yates et al. 2021), and a separate system-latency metric would double-count M-02. Per-stage dwell times are diagnostic telemetry, not a core metric.

**M-04 vs M-09 — two robustness facets.** M-04 is *within-episode fault recovery* (steps from anomaly onset until the anomaly is cleared **and** a nominal mission mode resumes); M-09 is *cross-episode consistency* (CV of utility over the launch-lottery Monte-Carlo). M-04 is **structurally zero for CG/AG** (no onboard slot → recovery waits for a pass, exceeding a quarter-orbit by construction) — the primary FDIR discriminator, grounded in ECSS Rev.1 §5.7.5.1 ("recover the on-board functions … such that mission operations can continue", verbatim) with ground as the *highest* escalation instance (§5.7.5.c).

**M-05 — what an override is.** \(N_{override}\) counts environment-enforced corrections of the architecture's requested action (safety floors, anomaly-forced safe, invalid requests). It measures *command quality* — how often the architecture proposes actions the safety layer must veto. For CG/AG it doubles as an operator-action proxy; in AO no operator exists, hence the name. It is **not** a miss-detection signal — a silently dropped uplink command never triggers an override (MC.03 carries its own indicator \(d_{cmd}\)).

**M-06 — why relative normalisation.** \(U/\hat E\) has no principled absolute ceiling, so it is min–max-normalised across the O-cells of one SSP slice. This is where the onboard responsiveness-vs-power trade surfaces: continuously-powered onboard compute (Jetson-class cores in AO/AH) raises \(\hat E\) against symbolic-on-OBC or ground paradigms.

**M-07 — caveat.** Wall-clock latency is a property of the test-harness host, not the flight computer — reported as a *relative substrate ordering* (symbolic ≪ RL ≪ LLM), never an absolute flight figure; the AI-FDIR <1 s requirement (Gallon et al. 2024) and SS-B feasibility are the standards-side counterparts.

**M-09 — scoring guard.** \(CV_U\) is undefined for \(\mu_U\le0\); a degenerate cell must be reported N/A, never given a default that scores as "perfectly robust".

**M-12 — distinct from AoI.** AoI is freshness of *any* update; M-12 is the fraction of total information *value* delivered under a bandwidth constraint, each product carrying \(V_j\) (e.g. detection confidence × target priority). Anchor: decision-theoretic Value of Information (Howard 1966); the value/semantics-of-information line (Yates et al. 2021) and data-prioritisation rationale (Boden & Larson 2006) carry it into the satellite context. This is the metric the headline VoI-triage test (DP.02) discriminates on.

**M-13 vs M-05 — two safety facets, kept distinct.** M-05 counts *applied* environment interventions (burden on the safety layer, including anomaly-forced steps the agent did not cause); M-13 counts *agent-proposed* constraint violations before enforcement — command safety at the source. For stochastic cores it co-reports \(pass^k\) — the probability that all \(k\) repeated identical trials stay violation-free (τ-bench): a core that is safe on average but unsafe on the tail is exactly what \(pass^k\) exposes, and the tail is what flight acceptance cares about. The M-05/M-13 pair is adjacent by design and is the first target of the V3 correlation check. Cross-cutting: measured in every test, no dedicated stimulus.

**M-14 — the economic argument for autonomy, measured.** Ground-segment effort (uplinked commands + weighted manual interventions per mission-day) is the cost axis on which CG/AG and AO/AH differ most (Nag et al. 2020 operator scaling; Castano et al. 2022). Cross-cutting like M-06/M-09; relative normalisation (no principled absolute ceiling); its signal is the same per-command ledger MC.03 requires.

#### Test-local indicators (instruments, not cross-cutting dimensions)

| Indicator | Symbol | Definition | Used by |
|---|---|---|---|
| Safe-mode correctness | \(p_{safemode}\) | fraction of anomalies → correct safe entry (=1.0 by construction; gate) | AU.01 |
| Buffer-overflow rate | \(r_{ovf}\) | data-loss (or at-cap) steps / \(N_{steps}\) over the storage pools | DP.03 |
| Data readiness at AOS | \(\rho\) | buffered-data ratio at acquisition of signal vs pass capacity | DP.01 helper |
| Command-loss detection | \(d_{cmd}\) | fraction of silently-dropped uplink commands detected & compensated within \(k\) steps | MC.03 |
| Preemptive avoidance | \(p_{avoid}\) | fraction of degradation episodes with no environment-forced intervention | AU.03 |
| Consensus conflict rate | \(p_{conflict}\) | fraction of allocation conflicts under partial ISL | PL.06 |
| Transfer ratio | \(TR\) | \(U\) at unseen \(N\) / \(U\) at trained \(N\) | PL.07 |

### 5.3 Test applicability and cell states (sparsity by design)

A decision matrix is **meant to be sparse**: each test exists to resolve a *specific* O-cell contrast. A test that returned the same value for all cells would carry zero selection information. Every (test × O-cell) entry is one of four states:

| State | Meaning | Example |
|---|---|---|
| **scored** | the targeted contrast is live in this cell | VoI triage (DP.02) across LLM-vs-symbolic data cells |
| **structural-zero** | the cell fails *by construction* — informative as a lower bound | AU.02 on CG/AG: no onboard slot → recovery waits for a pass |
| **control-ceiling** | the cell trivially maxes the test *by construction* — a control | AN.01 on symbolic: the rule fired *is* the rationale, \(\xi\equiv1.0\) |
| **N/A** | the mechanism under test does not exist in the cell, or SS-C/SS-E preclude it | PL.07 on symbolic (no learned policy); multi-agent tests on a C1 SSP |

Inert cells are intended: PL.07 separates the *learned* multi-agent cells (RL vs LLM) where over-fitting to the trained \(N\) is real; symbolic is its trivial-transfer control. AU.01 is a non-discriminating **correctness gate** (every cell must score 1.0; a miss flags an environment bug, not an architecture). Each test names its target contrast in a *Discriminates* line.

### 5.4 Test catalogue (grouped by MOF)

Each test carries: SSP-parametric **stimulus** · **metric / threshold / score** · **fidelity** (§4.2: **L** = an LF rung suffices to discriminate (L1 / R1 / R2) · **H** = needs an HF rung at an anchor (L2 / R3) · **L→H** screen-then-confirm) · **Discriminates** (the targeted O-cell contrast) · **applicability** (SS-C/SS-E) · **Requires** (the environment/collector capabilities the test engine must provide — consolidated in §7). Numeric defaults (e.g. \(p_{drop}=0.10\), \(CV_{max}=0.15\)) are engineering defaults subject to the pre-registered sensitivity sweep (§5.6), not literature constants.

#### MC — Monitoring and Control

**MC.01 — Decision-cycle latency**
- *Stimulus:* nominal operations; wall-clock per monitor→decide→command cycle.
- *Metric:* M-07; pass \(L_{dec}\le L_{ref}\), \(L_{ref}=\min(\Delta t,\tau_{dyn})\); \(s=\max(0,1-L_{dec}/L_{ref})\).
- *Fidelity:* L. *Discriminates:* substrate compute cost — not paradigm (M-07 caveat applies). *Applic.:* C1+.
- *Requires:* per-cycle wall-clock capture.

**MC.02 — Telemetry continuity during contact gap**
- *Stimulus:* ground outage \(T_{blackout}=k\,T_{orb}\) (default \(k=2\)); one anomaly injected inside the gap.
- *Metric:* M-04 + M-01 (windowed); pass = all gap anomalies cleared onboard; \(s=(1-\text{uncleared rate})\cdot\min(1,\text{obs ratio}_{blackout})\).
- *Fidelity:* L→H (paradigm split at L; AH ground-planner pre-staging quality at H). *Discriminates:* onboard-vs-ground; AH planner substrate at H. *Applic.:* C1+.
- *Requires:* ground-outage window injection; windowed metric evaluation.

**MC.03 — Command execution verification under degraded link**
- *Stimulus:* uplink-command model; silently drop fraction \(p_{drop}\) (default 0.10) of uplinked commands (no NACK).
- *Metric:* test-local \(d_{cmd}\) — dropped commands *detected and compensated* (re-issue, self-report, or compensating action) within \(k\) steps (default \(k=T_{orb}/4\Delta t\)); time-to-detection co-reported. Pass \(d_{cmd}\ge\theta\); \(s=d_{cmd}\).
- *Fidelity:* L→H (presence of onboard execution monitoring at L; self-diagnosis quality at H). *Discriminates:* onboard execution monitoring (AO/AH) vs ground verification at the next pass (AG/CG = structural delay floor). *Applic.:* C1+.
- *Requires:* uplink-command model + silent-drop injection + per-command execution ledger (issued / executed / detected-missing). M-05 is deliberately *not* the metric — an override never fires on a command that silently vanished.

**MC.04 — Detection-triggered payload retasking**
- *Stimulus:* inject high-value transient events; measure end-to-end pipeline completion (observe→process→deliver) per trigger.
- *Metric:* test-local pipeline completion rate; pass \(\ge\rho_{min}\); \(s=\) completion fraction.
- *Fidelity:* L→H (reactive trigger at L; agentic re-prioritisation quality at H). *Discriminates:* reactive-vs-agentic retasking + onboard-vs-ground. *Applic.:* C1+.
- *Requires:* transient-event injection with value weights \(V_j\) (shared with DP.02); per-product pipeline tracking (per-trigger completion is undefined over aggregate-only storage pools).

#### AU — Automation

**AU.01 — Safe-mode entry (correctness gate)**
- *Stimulus:* anomaly at \(\lambda_{anomaly}\); safe mode is environment-enforced.
- *Metric:* \(p_{safemode}\); pass \(=1.0\); \(s=p_{safemode}\).
- *Fidelity:* L. *Discriminates:* nothing — gate by design (sub-1.0 flags an environment bug). *Applic.:* all SSPs.
- *Requires:* anomaly injection; per-step mode + anomaly state.

**AU.02 — Autonomous recovery without ground pass** — **primary FDIR discriminator**
- *Stimulus:* anomaly injected during an inter-pass interval (no contact within \(T_{orb}\)); the fault carries a **recovery task** — a diagnostic/mitigation procedure whose execution quality determines recovery time (recovery must depend on agent actions, not on a fixed countdown, or the H-rung has nothing to measure).
- *Metric:* M-04; pass \(\bar s_{rec}\le T_{orb}/4\Delta t\); \(s=\max(0,1-\bar s_{rec}/(T_{orb}/4\Delta t))\).
- *Fidelity:* L→H (onboard-vs-ground at L; recovery-plan quality across AH substrates at H). *Discriminates:* onboard (AO/AH) vs ground — **CG/AG = structural-zero** (ECSS "F2" floor). *Applic.:* C1+.
- *Requires:* gap-timed (scheduled) anomaly injection; recovery-task fault model; persistent anomaly-state signal for M-04.

**AU.03 — Predictive FDIR via resource-trend monitoring**
- *Stimulus:* slow, *mitigable* resource degradation over \(N_{cyc}\) cycles (e.g. battery charge-rate −1 %/cycle); preemptive load-shedding can compensate; unmitigated, it ends in an environment-forced safe entry.
- *Metric:* test-local \(p_{avoid}\) — degradation episodes with **no** environment-forced intervention; pass \(p_{avoid}\ge\theta\); \(s=p_{avoid}\). M-09 co-reported (consistency under degradation), not primary — CV measures consistency, not prediction.
- *Fidelity:* H (trend-reasoning requires a live LLM). *Discriminates:* substrate reasoning depth — symbolic threshold rules fire at the limit (floor) vs LLM/agentic trend extrapolation acting early (ceiling). *Applic.:* C1+.
- *Requires:* mitigable resource-degradation mechanism (degradation parameter + load-shedding effect on the trend); forced-intervention flag.

**AU.04 — Activity scheduling: contact-window utilisation**
- *Stimulus:* nominal episode; delivered / max-achievable over the SSP's passes.
- *Metric:* M-11; pass \(\eta_{dl}\ge\eta_{min}\); \(s=\eta_{dl}\).
- *Fidelity:* L→H (rule-fill at L; planner quality at H). *Discriminates:* schedule-fill quality across planning substrates. *Applic.:* C1+.
- *Requires:* delivered + max-achievable volume signals.

**AU.05 — Preemptive rescheduling on priority event**
- *Stimulus:* high-priority target injected mid-schedule; does the cell preempt the running plan?
- *Metric:* M-01 restricted to high-priority observation ratio \(obs_{hp}\); pass \(\ge\theta\); \(s=obs_{hp}\).
- *Fidelity:* L→H. *Discriminates:* reactive-vs-agentic + planner responsiveness. *Applic.:* C1+.
- *Requires:* priority-event injection + priority-segmented utility tracking.

**AU.06 — Lights-out endurance**
- *Stimulus:* extended blackout \(k\,T_{orb}\) (\(k\ge4\)), no ground contact.
- *Metric:* M-01, M-09; pass \(U_{blackout}/U_{nom}\ge\theta\); \(s=\min(1,U_{blackout}/U_{nom})\).
- *Fidelity:* L→H. *Discriminates:* onboard autonomy — ground cells degrade across the blackout (structural-zero in the limit). *Applic.:* C1+.
- *Requires:* extended-outage window (generalises MC.02).

#### PL — Planning

**PL.01 — Daily schedule optimality**
- *Stimulus:* nominal episode; utility vs theoretical max from orbit geometry + downlink windows.
- *Metric:* M-01; pass \(U\ge U_{min}\); \(s=\min(1,U/U_{max,theoretical})\).
- *Fidelity:* L→H. *Discriminates:* planning substrate + paradigm. *Applic.:* C1+.
- *Requires:* \(U_{max,theoretical}\) computed from the orbital context.

**PL.02 — Reactive replanning after anomaly**
- *Stimulus:* anomaly at \(t\sim U[T_{orb}/4,\,T_{orb}/2]\); recovered utility vs a no-anomaly paired baseline (same seed).
- *Metric:* M-01 ratio \(U_{post}/U_{baseline}\); pass \(\ge0.70\); \(s=\min(1,(U_{post}/U_{baseline})/0.70)\).
- *Fidelity:* L→H. *Discriminates:* replanning quality — agentic vs reactive vs none. *Applic.:* C1+.
- *Requires:* scheduled (deterministically timed) anomaly injection; paired same-seed baseline runs; segmented utility.

**PL.03 — Multi-objective trade-off (power vs utility)**
- *Stimulus:* conflicting-resource scenario (payload duty cycle stressing the power budget).
- *Metric:* M-06 (+ M-05, M-08 co-reported); pass \(\eta_R\ge0.8\,\eta_{R,max}\) ∧ \(OIR\le\theta\); \(s=(\eta_R/\eta_{R,max})\cdot\max(0,1-OIR/\theta)\).
- *Fidelity:* H (quantified trade-off reasoning is the load-bearing capability). *Discriminates:* reasoning depth — LLM/agentic resolve *and explain* (M-08) the trade-off vs symbolic fixed thresholds. *Applic.:* C1+.
- *Requires:* resource-conflict scenario parameterisation.

**PL.04 — Multi-satellite pass deconfliction**
- *Stimulus:* \(N\) satellites contend for shared ground passes.
- *Metric:* fleet-mean \(\overline{\eta_{dl}}\); pass \(\ge\theta\). *Fidelity:* L→H. *Discriminates:* organisation (SAS = N/A; CMAS vs distributed). *Applic.:* **C2+**.
- *Requires:* multi-satellite scenario (Flamingo).

**PL.05 — Distributed task allocation (constellation)**
- *Stimulus:* fleet-wide task pool allocated under ISL.
- *Metric:* M-10; pass \(\eta_{scale}(N)\ge\theta\). *Fidelity:* L→H. *Discriminates:* organisation + scale. *Applic.:* **C2+, E2+**.
- *Requires:* multi-satellite scenario + ISL model.

**PL.06 — Consensus under partial ISL failure**
- *Stimulus:* drop a fraction of ISL links mid-episode.
- *Metric:* test-local \(p_{conflict}\); pass \(\le\theta\); \(s=\max(0,1-p_{conflict}/\theta)\). *Fidelity:* L→H. *Discriminates:* distributed-coordination robustness. *Applic.:* **C2+, E2+**.
- *Requires:* ISL-failure injection.

**PL.07 — MARL policy generalisation to new \(N\)**
- *Stimulus:* train at \(N_{train}\), evaluate at unseen \(N_{test}\).
- *Metric:* test-local \(TR=U(N_{test})/U(N_{train})\); pass \(\ge\theta\); \(s=\min(1,TR)\). *Fidelity:* H. *Discriminates:* **learned** multi-agent cells (RL vs LLM) — over-fitting to trained \(N\); symbolic = N/A (trivial-transfer control \(TR\equiv1\)). *Applic.:* **C4+, E3+**.
- *Requires:* cross-\(N\) evaluation harness.

#### FD — Flight Dynamics

**FD.01 — Onboard orbit propagation under GPS denial**
- *Stimulus:* deny GPS; onboard ephemeris drift / next-AOS prediction error \(\Delta t_{AOS}\).
- *Metric:* M-07-class timing error; pass \(\Delta t_{AOS}\le\theta\); \(s=\max(0,1-\Delta t_{AOS}/\theta)\).
- *Fidelity:* L (physics, not LLM). *Discriminates:* cells carrying an onboard propagator vs those without (subsumes MOF *Location*). *Applic.:* C1+.
- *Requires:* GPS-denial mechanism + onboard propagation model.

**FD.02 — Collision avoidance under short warning**
- *Stimulus:* conjunction with short warning \(T_{warn}\).
- *Metric:* binary success \(\times(1-OIR)\); pass = manoeuvre before TCA.
- *Fidelity:* L. *Discriminates:* onboard reaction (AO/AH) vs ground — ground = structural-zero for short \(T_{warn}\). *Applic.:* C1+.
- *Requires:* conjunction injection.

#### DP — Data Product Management

**DP.01 — AoI-optimal telemetry scheduling**
- *Stimulus:* nominal episode; freshness of delivered products.
- *Metric:* M-02 (report M-03); pass \(\bar\Delta\le\Delta_{ref}\); \(s=\max(0,1-\bar\Delta/\Delta_{ref})\).
- *Fidelity:* L→H. *Discriminates:* scheduling-for-freshness across planning substrates. *Applic.:* C1+.
- *Requires:* per-product age tracking, clock at observation capture (M-02/M-03 note — end-to-end latency folds in here).

**DP.02 — VoI-based downlink triage** — **headline H-tier test**
- *Stimulus:* products with heterogeneous value \(V_j\) under a bandwidth limit; delivered value fraction.
- *Metric:* M-12; pass \(\xi_{VoI}\ge\theta\); \(s=\xi_{VoI}\).
- *Fidelity:* H — the LLM reasoning *is* the prioritisation policy. *Discriminates:* reasoning-as-policy — LLM/agentic triage vs symbolic priority rules vs FIFO. *Applic.:* C1+.
- *Requires:* per-product value weights (heterogeneous-value product generation).

**DP.03 — Storage overflow prevention**
- *Stimulus:* observation rate able to exceed buffer caps.
- *Metric:* test-local \(r_{ovf}\); pass \(\le\theta\); \(s=\max(0,1-r_{ovf}/\theta)\).
- *Fidelity:* L→H. *Discriminates:* backpressure/planning foresight across substrates. *Applic.:* C1+.
- *Requires:* buffer caps + explicit data-loss accounting on overflow (silent clamping makes loss invisible).

#### AN — Analysis

**AN.01 — Anomaly report completeness / explainability**
- *Stimulus:* anomaly events; fraction with an operator-readable, causally-complete rationale.
- *Metric:* M-08; pass \(\xi\ge\theta\); \(s=\xi\).
- *Fidelity:* L→H (rationale *presence* at L; rationale *quality* at H). *Discriminates:* substrate explainability — symbolic = control-ceiling (\(\xi\equiv1.0\)), subsymbolic·RL ≈ floor (post-hoc only), LLM/agentic generate narrative rationale. *Applic.:* C1+.
- *Requires:* rationale capture per decision; rationale-quality scorer for the H-rung (rubric or LLM-judge).

### 5.5 Why this instrument is correct — and where it is incomplete (assumptions)

"Complete" is unprovable for any benchmark; claiming it is the known failure mode of benchmark construction (Raji et al. 2021). The claim made here is **four named validity properties**, each with its method, distributed over the instrument levels of §5.0 — plus an explicit assumptions register. There is no field-standard autonomous-satellite-operations benchmark to validate against (existing suites are narrow: anomaly-detection sets, orbital-game environments, vision sets), so the instrument is grounded by *covering standards-defined spaces* and made falsifiable by the checks below.

**V1 — Content validity (test level).** The catalogue spans a closed *function* space (CCSDS Table 3-1, §5.1) × a declared *stressor* space (S1–S6, §5.1); the coverage table makes every empty cell justified or a named gap. *Method:* maintain the table; every catalogue change re-derives it.

**V2 — Construct validity (metric level)** (Cronbach & Meehl 1955). A metric is admissible only with a verified causal chain *environment mechanism → emitted signal → collector → formula*. *Method:* the metric↔signal audit, re-run on every environment/collector change; the *Required signals* column of §5.2 is its checklist. The procedure has already caught real construct failures in the predecessor implementation (a recovery metric reading an injection pulse; an operator metric counting safety vetoes) — it is the standing guard, not a one-off.

**V3 — Non-redundancy & minimality (metric-set level)** (Keeney & Raiffa: complete · operational · decomposable · non-redundant · minimal). The hierarchy in §5.2 is the structure; the *pre-registered correlation check* (§5.6) is the empirical teeth: metric pairs collinear across the whole O-space are collapsed or explicitly justified. Known tensions: the information-delivery cluster (M-02/03, M-11, M-12) and the M-05/M-13 pair.

**V4 — Discriminative sufficiency (benchmark level).** *Forward:* per test, between-cell variance decomposition across O per SSP — a test that never varies (gates and ceilings excepted) carries zero selection information. *Backward (gap detector):* every architectural difference predicted by the grounding literature must name its detecting test — Kim et al. 2025 (error amplification, saturation) → PL.04–07; CoALA memory effects → writable-memory comparisons; Gallon et al. 2024 (FDIR latency) → MC.01/AU.02. Holes this method has already found and **closed by adoption** (2026-06-10): **M-13 constraint-violation rate** and **M-14 commanding effort** (§5.2).

**ECSS coverage matrix** (every autonomy capability exercised by ≥1 test):

| ECSS autonomy capability (verified §5.7) | Exercised by |
|---|---|
| E1→E2 — ground real-time → on-board scheduler | MC.03, AU.04, PL.01 |
| E2→E3 — pre-planned → adaptive / event-based ops | MC.04, FD.01, FD.02 |
| E3→E4 — adaptive → goal-oriented on-board re-planning | AU.03, AU.05, PL.02, PL.03, DP.02 |
| D1→D2 — essential-data buffering → on-board data independence | MC.02, AU.06, DP.01, DP.03 |
| FDIR — onboard recovery vs ground hand-over (§5.7.5) | AU.01 (gate), AU.02, AU.03 (predictive), FD.02, PL.06 |
| Organisation / scale (multi-agent) | PL.04–PL.07 |
| SAL band spanned | 1 (CG) → 5 (AH·hybrid·agentic), peaking at PL.03 / DP.02 / AU.03 |

**Metric × MOF coverage** (no orphan metric; every exercised MOF measured):

| Metric | MOF measured | Objective axis |
|---|---|---|
| M-01 | all (mission success) | mission value |
| M-02/M-03 | Data Product Management | information timeliness |
| M-04 | Automation (FDIR) | fault resilience |
| M-05 | Monitoring & Control ↔ Manual Operations boundary | autonomy risk / burden (proxy) |
| M-06 | cross-cutting (Planning trade-offs) | efficiency |
| M-07 | Monitoring & Control / Automation | responsiveness |
| M-08 | Analysis | interpretability |
| M-09 | cross-cutting | consistency |
| M-10 | Planning (constellation) | scalability |
| M-11 | Data Product Management | data return |
| M-12 | Data Product Management | information value |
| M-13 | Monitoring & Control (command validity); cross-cutting | safety |
| M-14 | Monitoring & Control ↔ Manual Operations boundary | operations cost |

**Assumptions register (the "not complete" half of the claim, stated):**

1. The CCSDS/ECSS taxonomies capture the operational function/autonomy space (closed-taxonomy assumption; the 4 unexercised MOFs are scoped out, §5.1).
2. The six stressor classes are *declared*, grounded but not derived — a different contingency taxonomy could partition differently (V1 protects coverage *given* the classes, not the classes themselves).
3. Numeric test defaults are engineering values pending sensitivity sweeps (§5.6), not literature constants.
4. One high-fidelity anchor per surrogate region; cross-SSP transfer of scores is an *empirical question* (§4.5 Q1), not an axiom.
5. The anomaly/fault model is a simplification of flight FDIR; AU.02's recovery-task requirement is the floor for it to discriminate substrates at all.
6. M-07 wall-clock is host-relative (relative orderings only).
7. Metric-set minimality is provisional until the V3 pilot correlation check.
8. SAL 0 (fully teleoperated) is not exercised (no such cell exists; CG is SAL 1); *Location* is only partially covered (FD.01).

**Validation loop** (benchmarks are iterated, never derived correct): (1) design-time — V1 table, V3 hierarchy, V2 chains (this document); (2) external anchoring — crosswalk to flight-autonomy heritage: EO-1 Autonomous Sciencecraft (Chien 2005), CASPER continuous planning (Knight et al. 2001), JPL operations-for-autonomous-spacecraft (Castano et al. 2022); a flight-demonstrated capability with no test = gap, a test with no precedent = flagged novel; (3) expert content review by operations engineers (TUM chair / supervisor network); (4) post-pilot — V4 forward check, V3 correlation check, qualitative trace review for differences no metric registered. Changes follow the amendment rule (§5.6).

### 5.6 Analysis protocol (pre-registered)

Fixed *before* the confirmatory sweeps; deviations are exploratory and reported separately.

- **Sample size:** 100 episodes per config (smoke runs never reported); episode length 7 simulated days at 1-minute resolution; constant across a sweep.
- **Variance sources:** launch lottery (RAAN/ArgP/TA per episode) is the Monte-Carlo axis; anomaly injection seeded per episode (`seed + episode_index`).
- **Primary outcome:** mission utility (M-01). Secondary metrics reported descriptively; they do not gate conclusions.
- **Primary test:** paired Wilcoxon signed-rank across launch-lottery seeds against the per-RQ baseline; **effect size** (Cohen's d on paired differences, or paired rank-biserial) reported with every p-value — effect size is the decision-driver, p-values the rejection gate.
- **Multiplicity:** Bonferroni at α = 0.05 per RQ family (chosen over FDR alternatives for transparent defensibility).
- **Descriptive first:** median, IQR, mean ± std per configuration, boxplots by morphological dimension, before any hypothesis test.
- **Minimum effect size:** |d| ≥ 0.5 provisional; revisited after pilot minimum-detectable-effect analysis.
- **Sensitivity sweeps:** all engineering-default thresholds in §5.4 (e.g. \(p_{drop}\), \(CV_{max}\), \(k\)) swept over a declared range; conclusions robust to the sweep or reported as threshold-dependent.
- **Comparison scope:** offline-adaptation contrasts (zero-shot vs prompt-optimised) and the online-learning action (writable memory) are compared *within* a substrate, never across (no unified learning mechanism exists across substrates). The CG planning-delay effect is isolated against the AG counterpart holding representation constant.
- **Trade-off view:** per-architecture scatter plots in objective space, summarised by paired effect sizes; Pareto/hypervolume machinery deferred until pilot data warrant it.
- **Amendment record:** any change to episode count, seeds, test/metric composition, α, or effect thresholds after results are seen is logged in `implementations.md` ("Analysis-plan amendments") with date and rationale.

---

## 6. The Decision Matrix

### Structure

\[
\mathbf{M}[SSP][O\text{-cell}][T] = s \in [0,1]
\]

For a given SSP the slice is a 2D table: rows = applicable O-cells, columns = applicable tests. All cells **TBD** — populated by the multi-fidelity scheme of §4.

### Aggregation

\[
S_{agg}(O, M) = \sum_{k} w_k \cdot \bar{s}_k(O)
\]

\(\bar{s}_k\) = mean score across *applicable* tests in CCSDS MOF group \(k\); \(w_k = 1/K\) by default over the \(K\le6\) MOF groups with ≥1 applicable test in the slice; weights adjustable per SS-A. Tests in structural-zero / control-ceiling / N/A states are excluded from \(\bar{s}_k\) unless the state *is* the result (AU.02's structural zero is the CG/AG verdict and is retained).

### Structural zeroes (paradigm-level failures)

CG and AG score exactly zero on tests requiring onboard per-step response (AU.02; FD.02 for short \(T_{warn}\)). This is a structural consequence of the missing onboard slot, not a performance outcome — these cells stay in the matrix as flagged reference baselines.

---

## 7. Implementation roadmap (design → code)

**Reuse-first policy.** The existing framework already implements the environment physics (Orekit propagation, EventSat 3-pool pipeline, power model), the Pydantic config system, the four-paradigm operations layer, the representation registry, and the PPO training pipeline — and active collaborations build on it (Giulio's RLlib work on the AH onboard slot; the ADCS student models slot into the environment). **Default: adapt this codebase to the spec.** Migration of any component is justified only by a demonstrated case — a concrete comparison showing the adaptation cost exceeds the rebuild cost — presented before any rewrite begins.

**Engine requirements (consolidated from §5.2 *Required signals* + §5.4 *Requires*).** What the test engine must provide, independent of implementation status:

| Subsystem | Requirement | For |
|---|---|---|
| Environment | uplink-command model + silent-drop injection | MC.03 |
| Environment | scheduled / gap-timed anomaly injection | AU.02, PL.02, MC.02 |
| Environment | recovery-task fault model (agent actions determine recovery time) | AU.02 |
| Environment | mitigable resource degradation | AU.03 |
| Environment | ground-outage windows (parametric \(k\,T_{orb}\)) | MC.02, AU.06 |
| Environment | priority-event injection | AU.05 |
| Environment | resource-conflict scenario | PL.03 |
| Environment | GPS denial + onboard propagation | FD.01 |
| Environment | conjunction injection | FD.02 |
| Environment | heterogeneous product values \(V_j\) + per-product pipeline tracking | MC.04, DP.02 |
| Environment | explicit data-loss accounting on buffer overflow | DP.03 |
| Collector | persistent anomaly-state recovery signal (M-04 chain) | AU.02 |
| Collector | per-command execution ledger | MC.03 |
| Collector | per-product age (clock at observation capture) | DP.01, M-02/03 |
| Collector | per-product value delivered/total | DP.02, M-12 |
| Collector | requested-vs-resolved action per step | M-05 |
| Collector | M-09 N/A guard (\(\mu_U\le\varepsilon\)) | M-09 |
| Collector | proposed-action constraint evaluation (violation flag) + repeated-trial harness for \(pass^k\) | M-13 |
| Collector | command / manual-intervention counts per mission-day (shares the MC.03 ledger) | M-14 |
| Analysis | \(U_{max,theoretical}\) from orbital context | PL.01 |
| Analysis | rationale-quality scorer (rubric / LLM-judge) | AN.01 H |
| Analysis | tradespace enumeration script (regenerates §2.1/§3.4 counts from the rule set) | §2, §3 |
| Training | checkpoint saves at fixed fractions of the convergence budget (R2 rungs) + checkpoint library for warm-starts + learning-curve logging | §4.2–§4.3, Q6 |
| Scenario | multi-satellite (Flamingo, N≤12) + ISL model + failure injection + cross-\(N\) harness | PL.04–07 |

**Anchors.** EventSat (SSP-04) first; Vyoma Flamingo (N≤12; activates the MAS organisations) and the large-scale (100+) RQ3 study widen surrogate coverage (§4).

**Ground-planner cores — contract fixed, specifics deferred (guidelines, not implementations).** The AH dual-slot *mechanism* exists (stale ground view, pass-synchronised uplink, planner shared verbatim with AG — the §3.1 isolation property); the non-symbolic planners are currently placeholders delegating to the symbolic greedy planner (`is_placeholder=True`, excluded from headline comparisons), so the ground-substrate axis of the AH pair is not yet a measurable contrast. Decided at the **interface level**:

1. **Uniform output contract.** Every ground planner — symbolic, RL, LLM, agentic — emits the *same artifact*: a schedule \(\{(mode_i,\,t_i)\}\) covering from the current contact until the next one (horizon = inter-pass interval), computed from the stale ground view. Substrates differ **only in how the schedule was produced** — the fair-comparison principle applied to the ground slot: same action space, same horizon, same information for every ground cell, and the §3.1 AH↔AG isolation preserved by construction. One schedule executor, N schedule generators.
2. **Subsymbolic·LLM·agentic ground cell = same contract, no tools.** The agentic loop is pure multi-step reasoning emitting the identical schedule; adding symbolic tools (orbit propagation, pass prediction, buffer model) makes the same loop *hybrid·agentic* per §3.1.
3. **Deliberately deferred to implementation time, per anchor (mission-dependent):** time discretisation and schedule-length caps; validity enforcement; the RL training formulation (credit assignment from episode utility to a whole plan; distinct from the onboard-RL refinement line). The spec constrains the interface, not the solution.

## 8. Where the rest lives (single source per topic)

| Topic | File |
|---|---|
| Component registry, paper basis, design decisions, analysis-plan amendments | `implementations.md` |
| How to add components (step-by-step) | `implementation_guide.md` |
| Scenario specifications | `scenarios.md` |
| Architecture diagram, data flow, directory layout | `architecture.md` |
| EventSat physics; commands; rules | `CLAUDE.md` |
| Research questions (draft) | `papers/expose/` |
| Config schema + documented template | `config_loader.py` · `configs/experiments/template.yaml` |

Metric definitions live in **§5.2 of this document** (`docs/metrics.md` retired 2026-06-10).

## 9. References

| Reference | DOI / URL |
|-----------|-----------|
| CCSDS 520.0-G-3 (2010, Issue 3) — Mission Operations Services Concept; §3.4 Table 3-1 (verified verbatim) | https://ccsds.org/Pubs/520x0g3.pdf |
| ECSS-E-ST-70-11C Rev.1 (2024) — On-board autonomy §5.7 (verified verbatim) | https://ecss.nl/wp-content/uploads/2024/07/ECSS-E-ST-70-11C-Rev.1-DIR1(5July2024).pdf |
| Baker & Phillips (2025) — SAL 0–5 | https://arxiv.org/abs/2503.01049 |
| Amini et al. (2020) — \(\tau_{comm} > \tau_{dyn}\) autonomy requirement | https://doi.org/10.3847/25c2cfeb.a09526a1 |
| Nag et al. (2020) — Operator scaling law; constellation scheduling | https://arxiv.org/abs/2010.09940 |
| Kaul, Yates & Gruteser (2012) — Age of Information | https://ieeexplore.ieee.org/document/6195689 |
| Yates et al. (2021) — AoI survey, IEEE JSAC | https://ieeexplore.ieee.org/document/9380899 |
| Costa, Codreanu & Ephremides (2016) — Peak AoI | https://ieeexplore.ieee.org/document/7307296 |
| Howard (1966) — Information Value Theory (M-12 anchor) | https://doi.org/10.1109/TSSC.1966.300074 |
| Kim et al. (2025) — Agent system scaling S=(A,E,C,Ω) | https://arxiv.org/abs/2503.11935 |
| Casadesus-Vila et al. (2024) — GNN constellation coordination | https://arxiv.org/abs/2403.00692 |
| Gallon et al. (2024) — AI-FDIR <1 s requirement | https://arxiv.org/abs/2410.09126 |
| Forrester, Sóbester & Keane (2008) — Surrogate-based engineering design | https://doi.org/10.1002/9780470770801 |
| Chen et al. (2023) — FrugalGPT (LLM cascade + scoring) | https://arxiv.org/abs/2305.05176 |
| Peherstorfer, Willcox & Gunzburger (2018) — Multifidelity survey, SIAM Review | https://doi.org/10.1137/16M1082469 |
| Gramacy & Lee (2008) — Treed Gaussian processes | https://doi.org/10.1198/016214508000000689 |
| Li et al. (2018) — Hyperband, JMLR 18(185) (truncated training as a fidelity signal; §4.2 R2) | https://jmlr.org/papers/v18/16-558.html |
| Brooks (1991) — Intelligence Without Representation | https://doi.org/10.1016/0004-3702(91)90053-M |
| Kautz (2022) — The Third AI Summer (neurosymbolic taxonomy) | https://doi.org/10.1002/aaai.12036 |
| Garcez & Lamb (2023) — Neurosymbolic AI: The 3rd Wave [Zotero `QMSTQFGI`] | https://doi.org/10.1007/s10462-023-10448-w |
| Sumers et al. (2024) — CoALA, TMLR [Zotero `7X8SRMIG`] | https://arxiv.org/abs/2309.02427 |
| Sellmaier, Uhlig & Schmidhuber (2022) — Spacecraft Operations 2nd ed. | https://link.springer.com/book/10.1007/978-3-030-88593-9 |
| Boden & Larson (2006) — Cost-Effective Space Mission Operations | https://spacetechnologyseries.com/~spacet9/books/Cost-Effective-Space-Mission-Operations.html |
| Palladino et al. (2026) — AI-augmented mission control lifecycle | https://doi.org/10.1109/AERO66936.2026.11519958 |
| Keeney & Raiffa (1976; Cambridge ed. 1993) — Decisions with Multiple Objectives (V3 desiderata) [Zotero `GNCVKP8C`] | ISBN 978-0-521-43883-4 |
| Cronbach & Meehl (1955) — Construct Validity in Psychological Tests, Psychol. Bulletin 52(4):281–302 (V2) | https://doi.org/10.1037/h0040957 |
| Raji et al. (2021) — AI and the Everything in the Whole Wide World Benchmark, NeurIPS D&B (§5.5 framing) | https://arxiv.org/abs/2111.15366 |
| Gray, ed. (1993) — The Benchmark Handbook, 2nd ed., Morgan Kaufmann (relevant · portable · scalable · simple; §5.0) | — (book) |
| Huppler (2009) — The Art of Building a Good Benchmark, TPCTC, Springer LNCS (repeatable · fair · verifiable · economical; §5.0) | — (proceedings) |
| JCGM 200:2012 — International Vocabulary of Metrology (VIM), 3rd ed. (§5.0) | https://www.bipm.org/en/committees/jc/jcgm/publications |
| Chien (2005) — The EO-1 Autonomous Sciencecraft (flight-heritage anchor) [Zotero `WWCTAVDQ`] | — |
| Knight, Rabideau, Chien et al. (2001) — CASPER, IEEE Intelligent Systems (flight-heritage anchor) [Zotero `EMHWUCYG`] | — |
| Castano et al. (2022) — Operations for Autonomous Spacecraft, IEEE Aerospace (flight-heritage anchor) [Zotero `2IJJ7ILS`] | https://doi.org/10.1109/AERO53065.2022.9843352 |
| Yao et al. (2024) — τ-bench (M-13 anchor: domain-rule violation, pass^k) [Zotero `H98GZVYR`] | https://arxiv.org/abs/2406.12045 |
| autops-agentic-framework (branch: migrate/matrix-restructure) | https://github.com/clemenjuan/autops-agentic-framework |
