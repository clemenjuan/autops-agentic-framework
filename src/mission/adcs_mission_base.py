"""ADCS mission configuration.

Setup mission dataclasses and layout the abstract mission class 

Here the structure of a generic mission is defined and the specific 
submission can be found in their respective files:

"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


from src.environment.orbital.adcs.control import Setpoint
from src.environment.orbital.adcs.state import SatState
from src.environment.orbital.adcs.configs import MissionConfig
from src.environment.orbital.propagator import EnvironmentData

@dataclass()
class MissionState:
    """Describes state of mission.

    Attributes:
        setpoint: Current target attitude and target angular velocity
        target_idx: Number of targets that have already been tracked 
        hold_timer: Number of seconds spent pointing towards target within tolerance
        phase: Describes the current mission phase 
        extra: Additional parameters needed by specific missions 
    """

    setpoint: Setpoint
    target_idx: int
    hold_timer: float
    phase: str ="nominal"
    extra: Tuple = ()

@dataclass()
class MissionEvents:
    """Describes events like mission done or target tracked.

    Attributes:
        target_cleared
        mission_done 
    """

    target_cleared: bool = False
    mission_done: bool = False



    

class Mission(ABC):
    def __init__(self, cfg:MissionConfig) -> None:
        self.cfg = cfg  

    @abstractmethod
    def reset(self, state: SatState, rng:np.random.Generator) -> MissionState:
        pass

    @abstractmethod
    def update(self, ms: MissionState, state: SatState, env_data: EnvironmentData, pointing_error_deg: float, dt: float) -> Tuple[MissionState, MissionEvents]:
        pass

    @abstractmethod
    def info(self, ms:MissionState) -> Dict[str, Any]:
        return {"mission/target_idx": ms.target_idx, "mission/phase": ms.phase}
