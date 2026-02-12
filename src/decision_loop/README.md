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

## Planned Decision Loops

| Loop   | Paper / Reference                                              |
|--------|---------------------------------------------------------------|
| SDA    | Sense-Decide-Act — classical reactive agent pattern           |
| OODA   | Observe-Orient-Decide-Act — Boyd's decision cycle             |
| CoALA  | Sumers et al. (2023) "Cognitive Architectures for Language Agents", TMLR |
| Others | TBD — researcher will select based on literature review       |
