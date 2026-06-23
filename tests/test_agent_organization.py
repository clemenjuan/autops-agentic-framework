"""
Tests for Agent Organization base class and implementations.

Kim et al. (2025) [FVFQ73RF] "Towards a Science of Scaling Agent Systems"
taxonomy: SAS, CentralizedMAS, DecentralizedMAS, IndependentMAS, HybridMAS.
"""

from __future__ import annotations

import pytest

from src.agent_organization.base import AgentAction, AgentObservation, AgentOrganization
from src.agent_organization.single_agent_system import SingleAgentSystem
from src.agent_organization.centralized_mas import CentralizedMAS
from src.agent_organization.decentralized_mas import DecentralizedMAS
from src.agent_organization.independent_mas import IndependentMAS
from src.agent_organization.hybrid_mas import HybridMAS
from src.environment.satellite_env import (
    ConstellationState,
    EnvironmentObservation,
    SatelliteState,
)


def _make_obs(satellite_ids: list[str]) -> EnvironmentObservation:
    """Minimal EnvironmentObservation with the given satellites (for org tests)."""
    return EnvironmentObservation(
        constellation_state=ConstellationState(
            timestep=0,
            epoch_seconds=0.0,
            satellites={s: SatelliteState(satellite_id=s) for s in satellite_ids},
        )
    )


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
# SingleAgentSystem (SAS) — Kim et al. 2025 |A|=1
# ======================================================================


class TestSingleAgentSystem:
    def test_single_agent(self) -> None:
        org = SingleAgentSystem(config={})
        org.initialize(constellation_size=5)
        agents = org.get_agents()
        assert len(agents) == 1
        assert agents[0] == "central_agent"

    def test_distribute_observation(self) -> None:
        org = SingleAgentSystem(config={})
        org.initialize(constellation_size=3)
        obs = org.distribute_observation({"some": "data"})
        assert "central_agent" in obs
        assert obs["central_agent"].agent_id == "central_agent"

    def test_collect_actions(self) -> None:
        org = SingleAgentSystem(config={})
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
# CentralizedMAS — Kim et al. 2025 Centralized MAS (star topology)
# ======================================================================


class TestCentralizedMAS:
    def test_agents_include_manager(self) -> None:
        org = CentralizedMAS(config={})
        org.initialize(constellation_size=3)
        agents = org.get_agents()
        assert "mission_manager" in agents
        assert len(agents) == 4  # 1 manager + 3 local

    def test_distribute_observation_no_prior_directive(self) -> None:
        org = CentralizedMAS(config={})
        org.initialize(constellation_size=1)
        obs = _make_obs(["eventsat_0"])
        result = org.distribute_observation(obs)
        assert "mission_manager" in result
        assert "sat_agent_0" in result
        # Centralized MAS: manager AND local both receive the full observation
        # (full observability + hierarchical directive); local has no directive yet.
        assert result["mission_manager"].local_state["full_observation"] is obs
        assert result["mission_manager"].messages == []
        assert result["sat_agent_0"].local_state["full_observation"] is obs
        assert result["sat_agent_0"].messages == []

    def test_distribute_observation_with_prior_directive(self) -> None:
        org = CentralizedMAS(config={})
        org.initialize(constellation_size=1)
        # Simulate a prior collect_actions that stored a directive
        org._last_manager_directive = {"eventsat_0": {"mode": "charging"}}
        result = org.distribute_observation(_make_obs(["eventsat_0"]))
        # Local agent now receives directive as message
        assert len(result["sat_agent_0"].messages) == 1
        assert result["sat_agent_0"].messages[0]["from"] == "mission_manager"
        assert result["sat_agent_0"].messages[0]["directive"] == {"eventsat_0": {"mode": "charging"}}
        # Manager still has no messages
        assert result["mission_manager"].messages == []

    def test_collect_actions_stores_directive_and_uses_local(self) -> None:
        org = CentralizedMAS(config={})
        org.initialize(constellation_size=1)
        actions = {
            "mission_manager": AgentAction(
                agent_id="mission_manager",
                action={"eventsat_0": {"mode": "charging"}},
            ),
            "sat_agent_0": AgentAction(
                agent_id="sat_agent_0",
                action={"eventsat_0": {"mode": "payload_observe"}},
            ),
        }
        env_actions = org.collect_actions(actions)
        # Local agent's action is used as env action
        assert env_actions == {"eventsat_0": {"mode": "payload_observe"}}
        # Manager's action stored as directive for next step
        assert org._last_manager_directive == {"eventsat_0": {"mode": "charging"}}

    def test_collect_actions_fallback_to_manager(self) -> None:
        org = CentralizedMAS(config={})
        org.initialize(constellation_size=1)
        # Only manager action, no local agent action
        actions = {
            "mission_manager": AgentAction(
                agent_id="mission_manager",
                action={"eventsat_0": {"mode": "safe"}},
            ),
        }
        env_actions = org.collect_actions(actions)
        assert env_actions == {"eventsat_0": {"mode": "safe"}}

    def test_initialize_resets_directive(self) -> None:
        org = CentralizedMAS(config={})
        org.initialize(constellation_size=1)
        org._last_manager_directive = {"some": "directive"}
        org.initialize(constellation_size=1)
        assert org._last_manager_directive is None


# ======================================================================
# DecentralizedMAS — placeholder, deferred to constellation scenarios
# ======================================================================


class TestDecentralizedMAS:
    def test_agents(self) -> None:
        org = DecentralizedMAS(config={})
        org.initialize(constellation_size=4)
        agents = org.get_agents()
        assert len(agents) == 4

    def test_distribute_not_implemented(self) -> None:
        org = DecentralizedMAS(config={})
        org.initialize(constellation_size=2)
        with pytest.raises(NotImplementedError):
            org.distribute_observation({})

    def test_collect_not_implemented(self) -> None:
        org = DecentralizedMAS(config={})
        org.initialize(constellation_size=2)
        with pytest.raises(NotImplementedError):
            org.collect_actions({})


# ======================================================================
# IndependentMAS — instantiated for the basemultisat constellation scenario
# ======================================================================


class TestIndependentMAS:
    def test_agents(self) -> None:
        org = IndependentMAS(config={})
        org.initialize(constellation_size=3)
        assert org.get_agents() == ["sat_agent_0", "sat_agent_1", "sat_agent_2"]

    def test_satellite_for_agent_maps_one_to_one(self) -> None:
        org = IndependentMAS(config={})
        org.initialize(constellation_size=3)
        assert org.satellite_for_agent("sat_agent_0") == "sat_0"
        assert org.satellite_for_agent("sat_agent_2") == "sat_2"

    def test_distribute_gives_each_agent_partial_view(self) -> None:
        org = IndependentMAS(config={})
        org.initialize(constellation_size=2)
        per_agent = org.distribute_observation(_make_obs(["sat_0", "sat_1"]))
        assert set(per_agent) == {"sat_agent_0", "sat_agent_1"}
        # C = ∅: each agent perceives ONLY its own satellite (authoritative
        # scoping by the organization) and receives no inter-agent messages.
        for agent_id, obs in per_agent.items():
            sat = org.satellite_for_agent(agent_id)
            view = obs.local_state["full_observation"].constellation_state.satellites
            assert set(view) == {sat}
            assert obs.messages == []
            assert obs.metadata["satellite_id"] == sat

    def test_collect_merges_per_satellite_action_dicts(self) -> None:
        org = IndependentMAS(config={})
        org.initialize(constellation_size=2)
        actions = {
            "sat_agent_0": AgentAction(agent_id="sat_agent_0", action={"sat_0": {"mode": "charging"}}),
            "sat_agent_1": AgentAction(agent_id="sat_agent_1", action={"sat_1": {"mode": "safe"}}),
        }
        env_actions = org.collect_actions(actions)
        assert env_actions == {
            "sat_0": {"mode": "charging"},
            "sat_1": {"mode": "safe"},
        }


# ======================================================================
# HybridMAS — placeholder, deferred to constellation scenarios
# ======================================================================


class TestHybridMAS:
    def test_agents(self) -> None:
        org = HybridMAS(config={})
        org.initialize(constellation_size=5)
        assert len(org.get_agents()) == 5

    def test_distribute_not_implemented(self) -> None:
        org = HybridMAS(config={})
        org.initialize(constellation_size=1)
        with pytest.raises(NotImplementedError):
            org.distribute_observation({})

    def test_collect_not_implemented(self) -> None:
        org = HybridMAS(config={})
        org.initialize(constellation_size=1)
        with pytest.raises(NotImplementedError):
            org.collect_actions({})
