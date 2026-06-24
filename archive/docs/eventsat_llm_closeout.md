# EventSat LLM Closeout

Date: 2026-06-18

This note freezes the immediate interpretation of the completed 100-episode
EventSat LLM ground runs before the project opens the multi-satellite scenario.
The canonical board has been refreshed at `data/figures/results_board.html`.

## Runs Included

| Run | Episodes | Notes |
|---|---:|---|
| `eventsat_sas_ag_symb` | 100 | symbolic autonomous-ground baseline |
| `eventsat_sas_ag_llm-s` | 100 | single-shot LLM ground scheduler |
| `eventsat_sas_ag_hllm-s` | 100 | hybrid LLM + symbolic shield ground scheduler |

The board currently reports 6 measured EventSat cells out of the 32-cell SAS
matrix. The remaining unmeasured cells remain explicitly marked as not-run or
placeholder.

## Headline Result

The two LLM-bearing autonomous-ground runs are effectively tied on the mission
metrics and sit slightly above the symbolic AG baseline:

| Metric | `ag_symb` | `ag_llm-s` | `ag_hllm-s` |
|---|---:|---:|---:|
| Mission utility | 1.4806 | 1.5269 | 1.5271 |
| Downlinked MB / episode | 25.4516 | 26.2467 | 26.2504 |
| Downlink efficiency | 0.5838 | 0.6020 | 0.6021 |
| Mean AoI, s | 19276.7 | 19465.0 | 19464.9 |
| Constraint-violation rate | 0.0000 | 0.0014 | 0.0011 |
| Explainability coverage | 1.0000 | 1.0000 | 1.0000 |

Interpretation: for the current EventSat AG setup, LLM and hybrid-LLM ground
planning produce a small utility/downlink gain over symbolic AG, but the two LLM
variants do not materially separate on mission performance.

## Latency Caveat

The latency comparison is not apples-to-apples unless cache state is controlled:

| Diagnostic | `ag_llm-s` | `ag_hllm-s` |
|---|---:|---:|
| LLM calls / episode | 20.84 | 20.84 |
| Cache-hit rate | 0.9494 | 0.0035 |
| Mean LLM call latency, s | 2.3476 | 29.8946 |
| Mean decision latency, s | 0.2240 | 4.4692 |

Interpretation: the latency delta is dominated by cache warmness, not necessarily
by architecture. The paper-facing claim should report mission-performance results
separately from live-call cost, and should label cache state clearly.

## Consequence

Do not spend the next increment rerunning more LLM EventSat episodes unless the
question is specifically cache-controlled latency. The scientific next step is to
open the multi-satellite scenario so the organisation axis and M-10 scale
efficiency become measurable. RL/hybrid-RL cells can be integrated when Giulio's
work lands.

