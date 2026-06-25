"""
Tests for Agent Organization base class and implementations.

Kim et al. (2025) [FVFQ73RF] "Towards a Science of Scaling Agent Systems"
taxonomy: SAS, CentralizedMAS, DecentralizedMAS, IndependentMAS, HybridMAS.
"""

from __future__ import annotations

import pytest

from src.core.organization.base import AgentAction, AgentObservation, AgentOrganization
from src.core.organization.single_agent_system import SingleAgentSystem
from src.core.organization.centralized_mas import CentralizedMAS
from src.core.organization.decentralized_mas import DecentralizedMAS
from src.core.organization.independent_mas import IndependentMAS
from src.core.organization.hybrid_mas import HybridMAS
from src.core.satellite_env import (
    ConstellationState,
    EnvironmentObservation,
    SatelliteState,
)


def _make_obs(
    satellite_ids: list[str],
    tasks: list[dict] | None = None,
) -> EnvironmentObservation:
    """Minimal EnvironmentObservation with the given satellites (for org tests)."""
    return EnvironmentObservation(
        constellation_state=ConstellationState(
            timestep=0,
            epoch_seconds=0.0,
            satellites={s: SatelliteState(satellite_id=s) for s in satellite_ids},
        ),
        tasks=tasks or [],
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
# DecentralizedMAS — SSA organisation implementation
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
        plan = {"sat_0": {"target_id": "rso_0"}, "sat_1": {"target_id": "rso_1"}}
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
        majority = {"sat_0": {"target_id": "rso_0"}}
        minority = {"sat_0": {"target_id": "rso_9"}}
        actions = {
            "sat_agent_0": AgentAction(agent_id="sat_agent_0", action=dict(majority)),
            "sat_agent_1": AgentAction(agent_id="sat_agent_1", action=dict(majority)),
            "sat_agent_2": AgentAction(agent_id="sat_agent_2", action=dict(minority)),
        }
        assert org.collect_actions(actions) == majority


# ======================================================================
# IndependentMAS — local per-satellite views, no deconfliction
# ======================================================================


class TestIndependentMAS:
    def test_agents(self) -> None:
        org = IndependentMAS(config={})
        org.initialize(constellation_size=3)
        assert org.get_agents() == ["sat_agent_0", "sat_agent_1", "sat_agent_2"]

    def test_satellite_for_agent_maps_multieventsat_default(self) -> None:
        org = IndependentMAS(config={})
        org.initialize(constellation_size=3)
        assert org.satellite_for_agent("sat_agent_0") == "sat_0"
        assert org.satellite_for_agent("sat_agent_2") == "sat_2"

    def test_satellite_for_agent_allows_scenario_prefix(self) -> None:
        org = IndependentMAS(config={"satellite_prefix": "demo"})
        org.initialize(constellation_size=2)
        assert org.satellite_for_agent("sat_agent_1") == "demo_1"

    def test_distribute_gives_each_agent_only_its_own_satellite(self) -> None:
        env_obs = _make_obs(
            ["sat_0", "sat_1"],
            tasks=[
                {"satellite_id": "sat_0", "target_id": "rso_0", "priority": 3.0},
                {"satellite_id": "sat_1", "target_id": "rso_1", "priority": 2.0},
            ],
        )

        org = IndependentMAS(config={"satellite_prefix": "demo"})
        org.initialize(constellation_size=2)
        result = org.distribute_observation(env_obs)

        local0 = result["sat_agent_0"].local_state["full_observation"]
        assert list(local0.constellation_state.satellites.keys()) == ["sat_0"]
        assert [t["target_id"] for t in local0.tasks] == ["rso_0"]
        assert result["sat_agent_0"].metadata["satellite_id"] == "sat_0"

        local1 = result["sat_agent_1"].local_state["full_observation"]
        assert list(local1.constellation_state.satellites.keys()) == ["sat_1"]
        assert [t["target_id"] for t in local1.tasks] == ["rso_1"]
        assert result["sat_agent_1"].messages == []

    def test_distribute_falls_back_to_observation_order_for_unknown_prefix(self) -> None:
        org = IndependentMAS(config={})
        org.initialize(constellation_size=2)
        result = org.distribute_observation(_make_obs(["demo_0", "demo_1"]))
        view = result["sat_agent_0"].local_state["full_observation"].constellation_state.satellites
        assert set(view) == {"demo_0"}
        assert result["sat_agent_0"].metadata["satellite_id"] == "demo_0"

    def test_collect_merges_without_deconfliction(self) -> None:
        org = IndependentMAS(config={})
        org.initialize(constellation_size=2)
        actions = {
            "sat_agent_0": AgentAction(
                agent_id="sat_agent_0", action={"sat_0": {"target_id": "rso_0"}}
            ),
            "sat_agent_1": AgentAction(
                agent_id="sat_agent_1", action={"sat_1": {"target_id": "rso_0"}}
            ),
        }
        env_actions = org.collect_actions(actions)
        assert env_actions["sat_0"] == {"target_id": "rso_0"}
        assert env_actions["sat_1"] == {"target_id": "rso_0"}


# ======================================================================
# HybridMAS — clustered: coordinate within, independent across
# ======================================================================


class TestHybridMAS:
    def test_one_cluster_head_per_cluster(self) -> None:
        org = HybridMAS(config={"num_clusters": 2})
        org.initialize(constellation_size=5)
        # 5 satellites into 2 contiguous near-equal clusters -> 2 head agents.
        assert org.get_agents() == ["cluster_agent_0", "cluster_agent_1"]

    def test_explicit_clusters_partition(self) -> None:
        org = HybridMAS(config={"clusters": [[0, 1, 2], [3, 4]]})
        org.initialize(constellation_size=5)
        assert len(org.get_agents()) == 2

    def test_distribute_gives_each_head_only_its_cluster(self) -> None:
        from src.core.satellite_env import (
            ConstellationState,
            EnvironmentObservation,
            SatelliteState,
        )

        env_obs = EnvironmentObservation(
            constellation_state=ConstellationState(
                timestep=0,
                epoch_seconds=0.0,
                satellites={
                    f"sat_{i}": SatelliteState(satellite_id=f"sat_{i}")
                    for i in range(3)
                },
            ),
            tasks=[
                {"satellite_id": f"sat_{i}", "target_id": f"rso_{i}", "priority": 1.0}
                for i in range(3)
            ],
        )
        org = HybridMAS(config={"num_clusters": 2})
        org.initialize(constellation_size=3)
        result = org.distribute_observation(env_obs)
        # Cluster 0 = {sat_0, sat_1}, cluster 1 = {sat_2}.
        head0 = result["cluster_agent_0"].local_state["full_observation"]
        assert set(head0.constellation_state.satellites.keys()) == {
            "sat_0",
            "sat_1",
        }
        head1 = result["cluster_agent_1"].local_state["full_observation"]
        assert set(head1.constellation_state.satellites.keys()) == {"sat_2"}

    def test_collect_merges_clusters_and_reports_localised_cost(self) -> None:
        org = HybridMAS(config={"num_clusters": 2})
        org.initialize(constellation_size=3)
        actions = {
            "cluster_agent_0": AgentAction(
                agent_id="cluster_agent_0",
                action={"sat_0": {"target_id": "rso_0"},
                        "sat_1": {"target_id": "rso_1"}},
            ),
            "cluster_agent_1": AgentAction(
                agent_id="cluster_agent_1",
                action={"sat_2": {"target_id": "rso_0"}},
            ),
        }
        merged = org.collect_actions(actions)
        assert set(merged.keys()) == {"sat_0", "sat_1", "sat_2"}
        # Localised cost: clusters of size 2 and 1 -> 2*1 + 1*0 = 2 messages.
        assert org.get_metrics()["coordination_messages"] == 2.0

    def test_num_clusters_spans_the_spectrum(self) -> None:
        # One cluster -> SAS-like all-to-all cost; singletons -> IMAS-like (zero).
        one = HybridMAS(config={"num_clusters": 1})
        one.initialize(constellation_size=4)
        one.collect_actions({})
        assert one.get_metrics()["coordination_messages"] == 12.0  # 4*3
        singletons = HybridMAS(config={"num_clusters": 4})
        singletons.initialize(constellation_size=4)
        singletons.collect_actions({})
        assert singletons.get_metrics()["coordination_messages"] == 0.0
