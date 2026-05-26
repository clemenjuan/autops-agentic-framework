"""
Agent Organization Module.

Defines coordination patterns between agents in a constellation.
Controls how observations are distributed and actions are aggregated.

Full Kim et al. (2025) [FVFQ73RF] taxonomy — "Towards a Science of Scaling
Agent Systems":

Implementations:
- SingleAgentSystem (SAS):   |A|=1, single agent controls entire constellation
- CentralizedMAS:            Orchestrator + local agents, star topology (C)
- DecentralizedMAS:          Peer-to-peer, all-to-all topology (placeholder)
- IndependentMAS:            No inter-agent communication (placeholder)
- HybridMAS:                 Heterogeneous mixed topology (placeholder)

Reference-architecture layer: L4 (Orchestration) in the Bhati 2026 mapping.
See ``docs/implementations.md`` "Layer Mapping (Bhati 2026)" for details.
"""

from src.agent_organization.single_agent_system import SingleAgentSystem
from src.agent_organization.centralized_mas import CentralizedMAS
from src.agent_organization.decentralized_mas import DecentralizedMAS
from src.agent_organization.independent_mas import IndependentMAS
from src.agent_organization.hybrid_mas import HybridMAS

__all__ = [
    "SingleAgentSystem",
    "CentralizedMAS",
    "DecentralizedMAS",
    "IndependentMAS",
    "HybridMAS",
]
