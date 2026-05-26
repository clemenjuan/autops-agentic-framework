"""
Representation Module.

Defines how knowledge and decisions are represented within decision loops.
The representation provides the "what" while the decision loop provides the "when/how".

Cognitive paradigms (Brooks 1991, Colelough & Regli 2025):
- Symbolic: Explicit declarative knowledge — rules, planners, constraint solvers.
- Subsymbolic: Implicit learned representations — RL policies, DNNs, base LLMs.
- Hybrid: Integration of symbolic + subsymbolic — LLM + tools/memory, DNN + logic.

Reference-architecture layer: L0 (Foundation Model substrate, where applicable) +
L1 (Reasoning) in the Bhati 2026 mapping. Symbolic variants have no L0 substrate
— see ``docs/implementations.md`` "Layer Mapping (Bhati 2026)" for details.
"""
