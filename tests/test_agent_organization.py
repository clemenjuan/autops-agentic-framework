"""
Tests for Agent Organization base class and implementations.
"""

from __future__ import annotations

import pytest

from src.agent_organization.base import AgentAction, AgentObservation, AgentOrganization
from src.agent_organization.centralized import CentralizedOrganization
from src.agent_organization.distributed import DistributedOrganization
from src.agent_organization.hierarchical import HierarchicalOrganization


# ======================================================================
# Data structure tests
# ======================================================================


class TestAgentObservation:
    def test_default(self) -> None:
        obs = AgentObservation(agent_id="a1")
        assert obs.agent_id == "a1"
        assert obs.local_state == {}
        assert obs.messages == []


class TestAgentAction:
    def test_default(self) -> None:
        act = AgentAction(agent_id="a1", action="noop")
        assert act.action == "noop"


# ======================================================================
# ABC contract
# ======================================================================


class TestAgentOrganizationABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            AgentOrganization(config={})  # type: ignore[abstract]


# ======================================================================
# Centralized organization
# ======================================================================


class TestCentralizedOrganization:
    def test_single_agent(self) -> None:
        org = CentralizedOrganization(config={})
        org.initialize(constellation_size=5)
        agents = org.get_agents()
        assert len(agents) == 1
        assert agents[0] == "central_agent"

    def test_distribute_observation(self) -> None:
        org = CentralizedOrganization(config={})
        org.initialize(constellation_size=3)
        obs = org.distribute_observation({"some": "data"})
        assert "central_agent" in obs
        assert obs["central_agent"].agent_id == "central_agent"

    def test_collect_actions(self) -> None:
        org = CentralizedOrganization(config={})
        org.initialize(constellation_size=2)
        actions = {
            "central_agent": AgentAction(
                agent_id="central_agent",
                action={"sat_0": "fire_thruster", "sat_1": "noop"},
            )
        }
        env_actions = org.collect_actions(actions)
        assert env_actions["sat_0"] == "fire_thruster"


# ======================================================================
# Hierarchical organization (placeholder)
# ======================================================================


class TestHierarchicalOrganization:
    def test_agents_include_manager(self) -> None:
        org = HierarchicalOrganization(config={})
        org.initialize(constellation_size=3)
        agents = org.get_agents()
        assert "mission_manager" in agents
        assert len(agents) == 4  # 1 manager + 3 local

    def test_distribute_not_implemented(self) -> None:
        org = HierarchicalOrganization(config={})
        org.initialize(constellation_size=2)
        with pytest.raises(NotImplementedError):
            org.distribute_observation({})


# ======================================================================
# Distributed organization (placeholder)
# ======================================================================


class TestDistributedOrganization:
    def test_agents(self) -> None:
        org = DistributedOrganization(config={})
        org.initialize(constellation_size=4)
        agents = org.get_agents()
        assert len(agents) == 4

    def test_distribute_not_implemented(self) -> None:
        org = DistributedOrganization(config={})
        org.initialize(constellation_size=2)
        with pytest.raises(NotImplementedError):
            org.distribute_observation({})
