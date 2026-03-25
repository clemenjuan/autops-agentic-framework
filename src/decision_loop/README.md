# Decision Loop Implementations

Each decision loop implementation in this module must **strictly follow**
the corresponding research paper. Do not invent steps or modify the
published algorithm.

## Implementation Checklist

For each new decision loop:

1. **Cite the paper** in the module docstring (authors, year, title, venue).
2. **List the algorithm steps** in comments, referencing specific sections
   or figures in the paper.
3. Subclass `DecisionLoop` from `base.py`.
4. Implement `process()` and `get_metrics()`.
5. Write comprehensive tests in `tests/test_decision_loops.py`.
6. Document any deviations or adaptations from the paper with rationale.

## Decision Loops

| Loop   | Status | Paper / Reference                                              |
|--------|--------|---------------------------------------------------------------|
| SDA    | Done   | Sense-Decide-Act — classical reactive agent pattern           |
| OODA   | Done   | Boyd's OODA cycle; Miller et al. (2021), Hartmann et al. (2024) |
| ReAct  | Done   | Yao et al. (2023) "ReAct", ICLR 2023; Li (2025)              |

All loops produce a `DecisionContext` (see `context.py`) consumed by representations.
See `docs/implementations.md` for full paper basis and design rationale.

**Note:** CoALA (Sumers et al. 2024) is an architecture blueprint, not a decision loop.
It is implemented as a hybrid representation type (`agentic_eventsat`).
See `src/representation/` for details.
