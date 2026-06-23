"""
Agent Organization — Abstract Base Class.

Defines abstract coordination patterns between agents in a constellation.
Controls how observations are distributed to individual agents and how
their actions are aggregated before being sent to the environment.

Full Kim et al. (2025) [FVFQ73RF] taxonomy — "Towards a Science of Scaling
Agent Systems":

Implementations:
- ``SingleAgentSystem``:  |A|=1, single agent controls entire constellation.
- ``CentralizedMAS``:     Orchestrator + local satellite agents, star topology.
- ``DecentralizedMAS``:   Peer-to-peer multi-agent with all-to-all topology.
- ``IndependentMAS``:     Multiple agents with no inter-agent communication.
- ``HybridMAS``:          Heterogeneous mixed-topology multi-agent organization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentObservation:
    """Observation tailored for a single agent.

    Attributes:
        agent_id: Unique identifier for this agent.
        local_state: State information visible to this agent.
        messages: Messages received from other agents (if any).
        metadata: Additional context.
    """

    agent_id: str
    local_state: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    """Action produced by a single agent.

    Attributes:
        agent_id: Identifier of the acting agent.
        action: The action payload (type depends on scenario).
        messages: Messages to send to other agents (if any).
        metadata: Additional context / diagnostics.
    """

    agent_id: str
    action: Any = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentOrganization(ABC):
    """Abstract base class for agent coordination patterns.

    The organization layer sits between the environment and the individual
    decision loops. It defines *who sees what* and *how individual actions
    compose* into the environment action dictionary.

    Attributes:
        config: Organization-specific configuration section from YAML.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the organization.

        Args:
            config: Configuration dictionary for this organization type.
        """
        self.config = config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self, constellation_size: int, **kwargs: Any) -> None:
        """Set up the organization for a given constellation size.

        Called once at the beginning of each episode after environment reset.

        Args:
            constellation_size: Number of satellites in the constellation.
            **kwargs: Additional scenario-specific initialization parameters.
        """
        ...

    @abstractmethod
    def distribute_observation(
        self,
        env_observation: Any,
    ) -> Dict[str, AgentObservation]:
        """Map a global environment observation to per-agent observations.

        Args:
            env_observation: The full :class:`EnvironmentObservation` from
                the environment.

        Returns:
            Mapping of agent_id → :class:`AgentObservation`.
        """
        ...

    @abstractmethod
    def collect_actions(
        self,
        agent_actions: Dict[str, AgentAction],
    ) -> Dict[str, Any]:
        """Aggregate individual agent actions into an environment action dict.

        Args:
            agent_actions: Mapping of agent_id → :class:`AgentAction`.

        Returns:
            Action dictionary suitable for ``SatelliteEnvironment.step()``.
        """
        ...

    @abstractmethod
    def get_agents(self) -> List[str]:
        """Return list of all agent identifiers in this organization.

        Returns:
            List of agent_id strings.
        """
        ...

    # ------------------------------------------------------------------
    # Agent ↔ satellite mapping
    # ------------------------------------------------------------------

    def satellite_for_agent(self, agent_id: str) -> str:
        """Return the ``satellite_id`` that ``agent_id`` observes and controls.

        Agents (decision-making entities, e.g. ``central_agent``,
        ``mission_manager``, ``sat_agent_0``) and satellites (physical objects,
        e.g. ``eventsat_0``, ``sat_0``) live in two separate namespaces. The
        RLlib bridge needs this mapping to encode the right satellite into each
        agent's observation, decode its action onto the right satellite, and
        assign it the right per-satellite reward.

        Default: single-satellite organizations map every agent to one
        canonical satellite (overridable via ``agent_organization_config``
        key ``satellite_id``; defaults to ``"eventsat_0"`` to match legacy
        single-satellite behaviour). Multi-satellite organizations override
        this -- see :class:`IndependentMAS`.
        """
        return str(self.config.get("satellite_id", "eventsat_0"))

    # ------------------------------------------------------------------
    # Optional hooks (override as needed)
    # ------------------------------------------------------------------

    def pre_step(self) -> None:
        """Hook called before each environment step. Override if needed."""

    def post_step(self, step_result: Any) -> None:
        """Hook called after each environment step. Override if needed."""

    def get_metrics(self) -> Dict[str, float]:
        """Return organization-level metrics (e.g. communication overhead)."""
        return {}


def validate_agent_satellite_mapping(
    organization: "AgentOrganization",
    environment: Any,
    constellation_size: int,
    scenario: str,
) -> None:
    """Ensure each agent maps onto a satellite that exists in the environment.

    On a multi-satellite scenario, an organization that leaves
    ``satellite_for_agent`` at the single-satellite default silently maps every
    agent to a non-existent satellite (zero observations, ignored actions). This
    turns that silent failure into an explicit error. No-op for N=1.
    """
    if environment is None or constellation_size <= 1:
        return
    mapped = {organization.satellite_for_agent(a) for a in organization.get_agents()}
    env_sats = set(environment.get_observation().constellation_state.satellites)
    unknown = mapped - env_sats
    if unknown:
        raise ValueError(
            f"Organization maps agents to satellites {sorted(unknown)} not "
            f"present in scenario '{scenario}' (has {sorted(env_sats)}). "
            f"Multi-satellite organizations must override satellite_for_agent."
        )
