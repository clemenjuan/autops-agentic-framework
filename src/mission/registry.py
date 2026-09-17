"""Registry for creating specific missions 

the builders take orbit uniformly even though SlewSequenceMission ignores it,
so that dispatch doesn't have to special-case constructor signatures+
"""
from __future__ import annotations

from typing import Dict, Callable

from src.mission.slew_sequence_mission import SlewSequenceMission
from src.mission.target_track_mission import TargetTrackMission
from src.mission.adcs_mission_base import Mission

from src.environment.orbital.adcs.configs import MissionConfig, OrbitConfig

MISSION_BUILDERS: Dict[str, Callable[[MissionConfig, OrbitConfig], Mission]] ={
    "slew": lambda cfg, orbit: SlewSequenceMission(cfg),
    "target_track": lambda cfg, orbit:TargetTrackMission(cfg,orbit)
}

MISSION_TYPES = tuple(MISSION_BUILDERS)

def build_mission(cfg:MissionConfig, orbit:OrbitConfig) -> Mission:
    try:
        builder = MISSION_BUILDERS[cfg.type]
    except KeyError:
        raise ValueError(
            f"unknown mission type {cfg.type!r}; expected one of {sorted(MISSION_BUILDERS)}") from None
    return builder(cfg, orbit)
