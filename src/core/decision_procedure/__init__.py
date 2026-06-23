"""
Decision Loop Module.

Defines the fixed per-step decision driver used by the current benchmarks.

Only SDA is supported. Older OODA/ReAct experiments were retired because the
EventSat O-benchmark now varies representation and operations paradigm while
holding the decision driver fixed. CoALA-style agentic behaviour lives in the
representation layer (`agentic_eventsat`), not as a decision loop.

Reference-architecture layer: L1 (Reasoning / Self-Reflection) in the Bhati 2026
mapping. See ``docs/implementations.md`` "Layer Mapping (Bhati 2026)" for details.
"""
