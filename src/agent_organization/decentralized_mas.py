"""
Decentralized MAS Organization.

Implements the Decentralized MAS topology from Kim et al. (2025) [FVFQ73RF]
"Towards a Science of Scaling Agent Systems":
  S = (A, E, C, Ω) where
  A = {sat_agent_0, ..., sat_agent_n-1},
  C = {(aᵢ, aⱼ) : ∀i,j, i≠j}  (all-to-all peer exchange),
  Ω = consensus.

Complexity: O(d·n·k) where d = consensus rounds, n = constellation size,
k = max reasoning iterations per agent.

Each satellite has its own peer agent — there is no manager (unlike
CentralizedMAS). Coordination is achieved by all-to-all exchange: every peer
shares what it sees with every other peer, so each agent ends up with the same
global information and, running the shared deterministic protocol, independently
arrives at the same deconflicted assignment. ``collect_actions`` takes the
consensus (plurality) of those proposals.

Consequence for the organisation axis: because the peers share full information,
DecentralizedMAS reaches the *same* deconflicted plan as the coordinated
organisations (SAS / CentralizedMAS) and — unlike IndependentMAS — does **not**
waste capacity on duplicate observations. What it pays is coordination cost: the
all-to-all channel carries ``n·(n-1)`` messages per round (vs the star's O(n)),
surfaced via :meth:`get_metrics`. With a capable global representation the
mission outcome therefore matches SAS/CMAS while the coordination cost is
strictly higher; the place a decentralized org loses *outcome* is when consensus
fails (Kim et al. report 17.2× error amplification), which a single deterministic
round does not trigger here.

First increment uses an all-to-all topology at N≥3. Ring / mesh /
visibility-limited topology ablations come later.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List

from src.agent_organization.base import AgentAction, AgentObservation, AgentOrganization


class DecentralizedMAS(AgentOrganization):
    """Peer-to-peer agents with all-to-all exchange and consensus aggregation.

    No manager: every satellite has an equal peer agent. C = all-to-all,
    Ω = consensus. See Kim et al. (2025) [FVFQ73RF] §4.3.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._agent_ids: List[str] = []
        # Peer proposals from the previous step, replayed as the all-to-all
        # exchange channel (each peer hears every other peer).
        self._last_round_messages: List[Dict[str, Any]] = []
        self._messages_exchanged: int = 0
        self._consensus_rounds: int = 0

    def initialize(self, constellation_size: int, **kwargs: Any) -> None:
        self._agent_ids = [f"sat_agent_{i}" for i in range(constellation_size)]
        self._last_round_messages = []
        self._messages_exchanged = 0
        self._consensus_rounds = 0

    def distribute_observation(
        self,
        env_observation: Any,
    ) -> Dict[str, AgentObservation]:
        """All-to-all exchange: every peer gets the full observation.

        Each agent also receives the other peers' proposals from the previous
        step as ``messages`` (the all-to-all channel C). The symbolic core does
        not read messages, but the full observation already gives every peer the
        global information it needs to reach the shared plan.
        """
        result: Dict[str, AgentObservation] = {}
        for agent_id in self._agent_ids:
            peer_messages = [
                msg for msg in self._last_round_messages if msg["from"] != agent_id
            ]
            result[agent_id] = AgentObservation(
                agent_id=agent_id,
                local_state={"full_observation": env_observation},
                messages=peer_messages,
            )
        return result

    def collect_actions(
        self,
        agent_actions: Dict[str, AgentAction],
    ) -> Dict[str, Any]:
        """Aggregate peer proposals by consensus (plurality, ties by agent order).

        Records the coordination cost of the all-to-all round: ``n·(n-1)``
        messages and one consensus round (deterministic peers converge
        immediately). Returns the agreed environment action dict.
        """
        proposals: List[tuple[str, Dict[str, Any]]] = []
        for agent_id in self._agent_ids:
            action = agent_actions.get(agent_id)
            if action is not None and isinstance(action.action, dict):
                proposals.append((agent_id, action.action))

        n = len(self._agent_ids)
        self._messages_exchanged = n * (n - 1)
        self._consensus_rounds = 1 if proposals else 0

        if not proposals:
            self._last_round_messages = []
            return {}

        keyed = [
            (agent_id, action, json.dumps(action, sort_keys=True, default=str))
            for agent_id, action in proposals
        ]
        counts = Counter(key for _, _, key in keyed)
        # Plurality winner; Counter.most_common preserves insertion order on
        # ties, so the lowest-index agent's proposal wins a tie.
        consensus_key = counts.most_common(1)[0][0]
        consensus = next(action for _, action, key in keyed if key == consensus_key)

        # Stash this round's proposals as next step's all-to-all messages.
        self._last_round_messages = [
            {"from": agent_id, "proposal": action} for agent_id, action, _ in keyed
        ]
        return consensus

    def get_agents(self) -> List[str]:
        return list(self._agent_ids)

    def get_metrics(self) -> Dict[str, float]:
        return {
            "coordination_messages": float(self._messages_exchanged),
            "consensus_rounds": float(self._consensus_rounds),
        }
