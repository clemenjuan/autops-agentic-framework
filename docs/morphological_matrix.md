# EventSat Benchmark — the Operations-System (O) Framework

**Scope.** A controlled, reproducible comparison of **cognitive architectures for
autonomous satellite operations**, measured on the **EventSat** mission (an
event-camera CubeSat — physics in `scenarios.md`). The benchmark is
**scenario-specific** (EventSat now; a multi-satellite scenario later) and
**metric-based**: architectures are scored directly on the metrics of §6. There is
no mission tradespace, no multi-fidelity surrogate, and no separate test catalogue.

An operations-system architecture (**O**) is a point defined by **three components**:
**Organisation × Representation × Operational paradigm**. For EventSat the
organisation is fixed at SAS; the comparison varies representation and paradigm.

---

## 1. Organisation

How many reasoning loci and how they relate.

| Token | Organisation |
|-------|--------------|
| `sas` | Single-agent system — **EventSat** |
| `cmas` | Centralised MAS |
| `imas` | Independent MAS |
| `dmas` | Decentralised MAS |
| `hmas` | Hierarchical MAS |

The MAS organisations belong to the multi-satellite scenario and are not exercised
by this benchmark. The first planned multi-satellite sweep is documented in
[`flamingo_mvp.md`](flamingo_mvp.md).

---

## 2. Representation (substrate × action space)

An agent's representation is a combination of a cognitive **substrate** and an
**action space**.

**Substrate:**
- **symbolic** — explicit rules / models
- **subsymbolic · RL** — a learned connectionist policy
- **subsymbolic · LLM** — a language model as the cognitive core
- **hybrid** — a genuine combination of symbolic + subsymbolic, in two modes:
  **RL + symbolic rules**, or **LLM + symbolic rules**

**Action space:**
- **single-shot** — one stimulus → one (constrained) response
- **agentic** — a control loop with internal actions (CoALA — Sumers et al. 2024;
  ReAct — Yao et al. 2023); LLM-bearing only

These yield the **7 representations** of the EventSat benchmark:

| Token | Substrate | Action space |
|-------|-----------|--------------|
| `symb` | symbolic | single-shot |
| `rl` | subsymbolic · RL | single-shot |
| `hrl` | hybrid (RL + symbolic) | single-shot |
| `llm-s` | subsymbolic · LLM | single-shot |
| `llm-a` | subsymbolic · LLM | agentic |
| `hllm-s` | hybrid (LLM + symbolic) | single-shot |
| `hllm-a` | hybrid (LLM + symbolic) | agentic |

(3 non-LLM reactive cells + 4 LLM-based cells. The agentic action space exists only
on the LLM-bearing substrates.)

---

## 3. Operational paradigm

Where and when cognition runs, across two possible slots — an **onboard** core
(closed-loop, per-step) and a **ground planner** (emits the uplinked whole-pass
schedule).

| Token | Paradigm | Slots | Representations (EventSat) |
|-------|----------|-------|----------------------------|
| `conventional` | Conventional | ground (human, one-pass delay) | `symb` only |
| `ag` | Autonomous Ground | ground | all 7 |
| `ao` | Autonomous Onboard | onboard | `symb`, `rl`, `hrl` (no LLM onboard) |
| `ah` | Autonomous Hybrid | onboard **+** ground | onboard {3} × ground {7} |

- **Conventional** runs the *same logic as Autonomous-Ground·symbolic* but with a
  **one-pass delay**: operators must work the fresh telemetry, plan, and prepare the
  schedule before uplinking it at the *next* pass. AG instead plans and uplinks
  within the same pass it receives the downlink.
- **Autonomous Onboard** has no LLM core on EventSat — the platform cannot sustain
  per-step LLM inference — so onboard is restricted to `symb`, `rl`, `hrl`.
- **Autonomous Hybrid** is a **dual-core architecture**: an onboard core *and* a
  ground planner, each with its own representation, which may differ. Onboard draws
  from the 3 onboard-feasible cells, ground from all 7 → **3 × 7 = 21** pairs.

---

## 4. The 32 EventSat experiments (SAS)

| Paradigm | Representations | Count |
|----------|-----------------|:-----:|
| `conventional` | `symb` | **1** |
| `ag` | `symb · rl · hrl · llm-s · llm-a · hllm-s · hllm-a` | **7** |
| `ao` | `symb · rl · hrl` | **3** |
| `ah` | onboard ∈ {`symb · rl · hrl`} × ground ∈ {7 reps} | **21** |
| | | **32** |

---

## 5. Naming convention

```
eventsat_sas_<paradigm>_<representation>
```

For the dual-core `ah`, both cores are named, **onboard first**:

```
eventsat_sas_ah_<onboard>_<ground>
```

Examples:

| Name | Architecture |
|------|--------------|
| `eventsat_sas_conventional_symb` | Conventional, symbolic (one-pass delay) |
| `eventsat_sas_ag_llm-a` | Autonomous Ground, agentic LLM |
| `eventsat_sas_ao_hrl` | Autonomous Onboard, hybrid-RL |
| `eventsat_sas_ah_rl_llm-s` | Autonomous Hybrid — **RL onboard**, **single-shot LLM ground** |
| `eventsat_sas_ah_hrl_hllm-a` | Autonomous Hybrid — hybrid-RL onboard, agentic-hybrid-LLM ground |

The name states the full architecture in one go.

---

## 6. Metrics (M-01 … M-14)

Architectures are compared on these fourteen metrics. Means over the launch-lottery
Monte-Carlo (shared seeds → paired comparisons); robustness is the cross-episode CV.

| ID | Metric | Symbol | Definition | Status |
|----|--------|--------|------------|--------|
| **M-01** | Mission Utility | \(U\) | \(\sum_j w_j\phi_j\) — weighted achievement of EventSat objectives (delivered-data objective by default, optional observation ablation, less anomaly burden), target-normalised to episode length | ✅ measured |
| **M-02** | Mean Age of Information | \(\bar{\Delta}\) | mean data staleness in seconds; downlink resets AoI | ✅ measured |
| **M-03** | Peak Age of Information | \(A_{peak}\) | maximum data staleness in seconds between downlink deliveries | ✅ measured |
| **M-04** | Autonomous Recovery Efficiency | R_FDIR | steps from anomaly onset to (cleared and nominal mode); horizon-censored if unrecovered | ✅ measured |
| **M-05** | Safety-Override Rate | \(OIR\) | \(N_{override}/N_{steps}\) — fraction of steps in protective safe mode (anomaly or critical battery); operator-intervention proxy; disjoint from M-13 | ✅ measured |
| **M-06** | Resource Efficiency | \(\eta_R\) | \(U/E_{consumed}\), utility per Wh consumed | ✅ measured |
| **M-07** | Decision Latency | \(L_{dec}\) | mean wall-clock per decision cycle | ✅ measured |
| **M-08** | Explainability Coverage | \(\xi\) | rationale-bearing decision cycles / decision cycles; presence only | ✅ measured (presence) |
| **M-09** | Robustness | \(CV_U\) | \(\sigma_U/\mu_U\) of utility across episodes (≥30 ep) | ✅ measured |
| **M-10** | Scale Efficiency | \(\eta_{scale}\) | \(U(N)/N\) normalised to \(U(1)\) | 🔭 multi-sat scenario |
| **M-11** | Downlink Efficiency | \(\eta_{dl}\) | delivered / max-achievable through the S-band channel | ✅ measured |
| **M-12** | Value-of-Information | \(\xi_{VoI}\) | raw-equivalent delivered value / raw-equivalent captured value | ✅ measured |
| **M-13** | Constraint-Violation Rate | \(p_{viol}\) | agent requests clamped to charging (preconditions failed); safe-mode steps (anomaly or critical battery) are M-05, not M-13 | ✅ measured |
| **M-14** | Commanding Effort | \(E_{cmd}\) | (N_cmd + w_m N_manual) per mission-day; N_cmd = commanded mode changes, N_manual = anomaly-recovery events (per-event, not per-step dwell) | ✅ measured |

Raw read-outs reported alongside the metrics — **observation hours** and **delivered MB**
per episode — are plotted directly too. Each metric is rendered across the full 32
experiments on the board: a heatmap grid (AH = onboard×ground) plus ranked bars.

---

## 7. Where the rest lives

| Topic | File |
|---|---|
| Component registry · paper basis · design decisions | `implementations.md` |
| How to add a component | `implementation_guide.md` |
| EventSat scenario physics | `scenarios.md` |
| System architecture · data flow · output artifacts | `architecture.md` |
