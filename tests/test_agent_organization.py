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
        result = org.distribute_observation({"sensor": 42})
        assert "mission_manager" in result
        assert "sat_agent_0" in result
        # Manager gets full observation, no messages
        assert result["mission_manager"].local_state["full_observation"] == {"sensor": 42}
        assert result["mission_manager"].messages == []
        # Local agent also gets full observation, no directive yet (first step)
        assert result["sat_agent_0"].local_state["full_observation"] == {"sensor": 42}
        assert result["sat_agent_0"].messages == []

    def test_distribute_observation_with_prior_directive(self) -> None:
        org = CentralizedMAS(config={})
        org.initialize(constellation_size=1)
        # Simulate a prior collect_actions that stored a directive
        org._last_manager_directive = {"eventsat_0": {"mode": "charging"}}
        result = org.distribute_observation({"sensor": 99})
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

    def test_distribute_all_to_all_full_observation(self) -> None:
        org = DecentralizedMAS(config={})
        org.initialize(constellation_size=3)
        result = org.distribute_observation({"sensor": 7})
        # Every peer (no manager) receives the full observation.
        assert set(result.keys()) == {"sat_agent_0", "sat_agent_1", "sat_agent_2"}
        for agent_id, obs in result.items():
            assert obs.local_state["full_observation"] == {"sensor": 7}

    def test_collect_reaches_consensus_and_reports_cost(self) -> None:
        org = DecentralizedMAS(config={})
        org.initialize(constellation_size=3)
        # Deterministic peers propose the same global plan -> unanimous consensus.
        plan = {"flamingo_0": {"target_id": "rso_0"}, "flamingo_1": {"target_id": "rso_1"}}
        actions = {
            aid: AgentAction(agent_id=aid, action=dict(plan))
            for aid in org.get_agents()
        }
        env_actions = org.collect_actions(actions)
        assert env_actions == plan
        # All-to-all cost for N=3 is n*(n-1) = 6 messages, one consensus round.
        metrics = org.get_metrics()
        assert metrics["coordination_messages"] == 6.0
        assert metrics["consensus_rounds"] == 1.0

    def test_collect_plurality_breaks_disagreement(self) -> None:
        org = DecentralizedMAS(config={})
        org.initialize(constellation_size=3)
        majority = {"flamingo_0": {"target_id": "rso_0"}}
        minority = {"flamingo_0": {"target_id": "rso_9"}}
        actions = {
            "sat_agent_0": AgentAction(agent_id="sat_agent_0", action=dict(majority)),
            "sat_agent_1": AgentAction(agent_id="sat_agent_1", action=dict(majority)),
            "sat_agent_2": AgentAction(agent_id="sat_agent_2", action=dict(minority)),
        }
        assert org.collect_actions(actions) == majority


# ======================================================================
# IndependentMAS — placeholder, deferred to constellation scenarios
# ======================================================================


class TestIndependentMAS:
    def test_agents(self) -> None:
        org = IndependentMAS(config={})
        org.initialize(constellation_size=3)
        assert len(org.get_agents()) == 3

    def test_distribute_gives_each_agent_only_its_own_satellite(self) -> None:
        from src.environment.satellite_env import (
            ConstellationState,
            EnvironmentObservation,
            SatelliteState,
        )

        env_obs = EnvironmentObservation(
            constellation_state=ConstellationState(
                timestep=0,
                epoch_seconds=0.0,
                satellites={
                    "flamingo_0": SatelliteState(satellite_id="flamingo_0"),
                    "flamingo_1": SatelliteState(satellite_id="flamingo_1"),
                },
            ),
            tasks=[
                {"satellite_id": "flamingo_0", "target_id": "rso_0", "priority": 3.0},
                {"satellite_id": "flamingo_1", "target_id": "rso_1", "priority": 2.0},
            ],
        )

        org = IndependentMAS(config={})
        org.initialize(constellation_size=2)
        result = org.distribute_observation(env_obs)

        # Agent 0 sees only flamingo_0 and only flamingo_0's task (C = ∅).
        local0 = result["sat_agent_0"].local_state["full_observation"]
        assert list(local0.constellation_state.satellites.keys()) == ["flamingo_0"]
        assert [t["target_id"] for t in local0.tasks] == ["rso_0"]
        local1 = result["sat_agent_1"].local_state["full_observation"]
        assert list(local1.constellation_state.satellites.keys()) == ["flamingo_1"]

    def test_collect_merges_without_deconfliction(self) -> None:
        org = IndependentMAS(config={})
        org.initialize(constellation_size=2)
        # Both independent agents pick the same RSO — the merge must keep the
        # collision (no deconfliction), composing one env action per satellite.
        actions = {
            "sat_agent_0": AgentAction(
                agent_id="sat_agent_0", action={"flamingo_0": {"target_id": "rso_0"}}
            ),
            "sat_agent_1": AgentAction(
                agent_id="sat_agent_1", action={"flamingo_1": {"target_id": "rso_0"}}
            ),
        }
        env_actions = org.collect_actions(actions)
        assert env_actions["flamingo_0"] == {"target_id": "rso_0"}
        assert env_actions["flamingo_1"] == {"target_id": "rso_0"}


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
