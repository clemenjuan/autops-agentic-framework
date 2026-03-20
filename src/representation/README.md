# Representation Implementations

Each representation defines *what* knowledge and decision logic is used
inside a decision loop. All representations consume a `DecisionContext`
produced by the decision loop.

## Current Implementations

| Representation | File | Ops Paradigm | Notes |
|---|---|---|---|
| Rule-based EventSat | `rule_based_eventsat.py` | Autonomous Hybrid | OODA-aware rules + ReAct `reason()` |
| Schedule-based EventSat | `schedule_based_eventsat.py` | Autonomous Ground | Greedy cyclic planner, OODA-aware |
| Conventional Schedule EventSat | `conventional_schedule_eventsat.py` | Conventional Ground | Human cognitive constraints (margins, horizon discount, shift handover) |

See `docs/implementations.md` for full paper basis and design rationale.

## Implementation Guidelines

### Symbolic Representations
- Rules, planners, constraint solvers
- Hand-designed by domain experts
- Deterministic given the same inputs
- Must document the rule set or planning formalism used

### Hybrid / Neuro-symbolic Representations
- Combine LLM reasoning with symbolic tools
- May include MARL-based sub-components
- Must clearly separate the neural and symbolic parts
- Document the integration architecture

## Checklist

For each new representation:

1. Subclass `Representation` from `base.py`.
2. Implement `encode_observation()` and `select_action()`.
3. Optionally implement `reason()` for ReAct compatibility.
4. For learned variants, also implement `update()`.
5. Write tests in `tests/test_representations.py`.
6. Document assumptions and design decisions.
