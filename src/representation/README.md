# Representation Implementations

Each representation defines *what* knowledge and decision logic is used
inside a decision loop.

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

### Neural Representations
- Learned policies (RL-trained networks)
- Require training pipeline (see `src/emergence/`)
- Must define observation/action spaces precisely
- Document the network architecture and training procedure

## Checklist

For each new representation:

1. Subclass `Representation` from `base.py`.
2. Implement `encode_observation()` and `select_action()`.
3. For learned variants, also implement `update()`.
4. Write tests in `tests/test_representations.py`.
5. Document assumptions and design decisions.
