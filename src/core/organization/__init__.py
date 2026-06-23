"""
Agent Organization Module.

Defines coordination patterns between agents in a constellation.
Controls how observations are distributed and actions are aggregated.

Full Kim et al. (2025) [FVFQ73RF] taxonomy — "Towards a Science of Scaling
Agent Systems":

Implementations:
- SingleAgentSystem (SAS):   |A|=1, single agent controls entire constellation
- CentralizedMAS:            Orchestrator + local agents, star topology (C)
- DecentralizedMAS:          Peer-to-peer, all-to-all topology
- IndependentMAS:            No inter-agent communication
- HybridMAS:                 Clustered mixed topology

Reference-architecture layer: L4 (Orchestration) in the Bhati 2026 mapping.
See ``docs/implementations.md`` "Layer Mapping (Bhati 2026)" for details.
"""

from src.core.organization.single_agent_system import SingleAgentSystem
from src.core.organization.centralized_mas import CentralizedMAS
from src.core.organization.decentralized_mas import DecentralizedMAS
from src.core.organization.independent_mas import IndependentMAS
from src.core.organization.hybrid_mas import HybridMAS

__all__ = [
    "SingleAgentSystem",
    "CentralizedMAS",
    "DecentralizedMAS",
    "IndependentMAS",
    "HybridMAS",
]
