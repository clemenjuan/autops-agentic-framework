"""
Configuration Loader.

Loads, validates, and provides typed access to experiment configurations
stored as YAML files. Every experimental choice is driven by configuration
— no hardcoded decisions.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set

import yaml
from pydantic import BaseModel, Field, field_validator


# ======================================================================
# Pydantic configuration models
# ======================================================================


class MetricsConfig(BaseModel):
    """Metrics collection configuration."""

    enabled: List[str] = Field(
        default=[
            "utility",
            "latency",
            "robustness",
            "resource_efficiency",
            "operator_load",
            "explainability",
            "scale_complexity",
        ]
    )
    collection_frequency: str = Field(default="per_step")
    utility_weights: Dict[str, float] = Field(
        default={"observation": 0.5, "downlink": 0.4, "anomaly_penalty": 0.1}
    )
    utility_targets: Dict[str, float] = Field(
        default={
            "observation_hours": 2.0,
            "downlinked_mb": 240.0,
            "mission_duration_days": 90.0,
        }
    )

    @field_validator("collection_frequency")
    @classmethod
    def _validate_frequency(cls, v: str) -> str:
        allowed = {"per_step", "per_episode"}
        if v not in allowed:
            raise ValueError(f"collection_frequency must be one of {allowed}, got '{v}'")
        return v


class EnvironmentConfig(BaseModel):
    """Environment / scenario configuration."""

    constellation_size: int = Field(default=5, ge=1)
    timestep_seconds: int = Field(default=60, gt=0)
    max_steps: int = Field(default=1440, gt=0)
    scenario: str = Field(default="to_be_defined")
    scenario_config: Dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration.

    One YAML file → one :class:`ExperimentConfig` instance. All
    morphological-matrix dimensions and execution parameters are
    captured here.
    """

    # Identification
    experiment_id: str = Field(default="exp_unnamed")
    description: str = Field(default="")

    # Reproducibility
    seed: int = Field(default=42)

    # Morphological matrix dimensions
    agent_organization: str = Field(default="centralized")
    decision_loop: str = Field(default="sda")
    representation: str = Field(default="symbolic")
    emergence_mode: str = Field(default="hand_designed")
    operations_paradigm: str = Field(default="autonomous_hybrid")

    # Component-specific sub-configs
    agent_organization_config: Dict[str, Any] = Field(default_factory=dict)
    decision_loop_config: Dict[str, Any] = Field(default_factory=dict)
    representation_config: Dict[str, Any] = Field(default_factory=dict)
    emergence_config: Dict[str, Any] = Field(default_factory=dict)
    operations_paradigm_config: Dict[str, Any] = Field(default_factory=dict)

    # Environment
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)

    # Memory
    memory_config: Dict[str, Any] = Field(default_factory=dict)

    # Execution
    num_episodes: int = Field(default=100, gt=0)
    max_steps: int = Field(default=1440, gt=0)

    # Metrics
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    # Output
    output_dir: str = Field(default="data/results")
    save_checkpoints: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    VALID_ORGANIZATIONS: ClassVar[Set[str]] = {"centralized", "hierarchical", "distributed"}
    VALID_REPRESENTATIONS: ClassVar[Set[str]] = {"symbolic", "hybrid", "neural"}
    VALID_EMERGENCE_MODES: ClassVar[Set[str]] = {"hand_designed", "learned"}
    VALID_OPERATIONS_PARADIGMS: ClassVar[Set[str]] = {
        "autonomous_hybrid", "conventional_ground",
    }
    # Decision loops are extensible — no fixed set enforced here.

    @field_validator("agent_organization")
    @classmethod
    def _validate_organization(cls, v: str) -> str:
        if v not in cls.VALID_ORGANIZATIONS:
            raise ValueError(
                f"agent_organization must be one of {cls.VALID_ORGANIZATIONS}, got '{v}'"
            )
        return v

    @field_validator("representation")
    @classmethod
    def _validate_representation(cls, v: str) -> str:
        if v not in cls.VALID_REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {cls.VALID_REPRESENTATIONS}, got '{v}'"
            )
        return v

    @field_validator("emergence_mode")
    @classmethod
    def _validate_emergence_mode(cls, v: str) -> str:
        if v not in cls.VALID_EMERGENCE_MODES:
            raise ValueError(
                f"emergence_mode must be one of {cls.VALID_EMERGENCE_MODES}, got '{v}'"
            )
        return v

    @field_validator("operations_paradigm")
    @classmethod
    def _validate_operations_paradigm(cls, v: str) -> str:
        if v not in cls.VALID_OPERATIONS_PARADIGMS:
            raise ValueError(
                f"operations_paradigm must be one of {cls.VALID_OPERATIONS_PARADIGMS}, got '{v}'"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v_upper


# ======================================================================
# Loader function
# ======================================================================


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A validated :class:`ExperimentConfig` instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        pydantic.ValidationError: If required fields are missing or invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    # Resolve output_dir template variables
    if "output_dir" in raw and "${experiment_id}" in raw["output_dir"]:
        exp_id = raw.get("experiment_id", "exp_unnamed")
        raw["output_dir"] = raw["output_dir"].replace("${experiment_id}", exp_id)

    return ExperimentConfig(**raw)


def apply_overrides(
    config: ExperimentConfig,
    *,
    episodes: int | None = None,
    steps: int | None = None,
    seed: int | None = None,
    output_dir: str | None = None,
) -> ExperimentConfig:
    """Apply CLI overrides to an experiment configuration.

    Args:
        config: The base configuration.
        episodes: Override num_episodes.
        steps: Override max_steps (both top-level and environment).
        seed: Override seed.
        output_dir: Override output_dir.

    Returns:
        A new ExperimentConfig with overrides applied.
    """
    updates: Dict[str, Any] = {}
    if episodes is not None:
        updates["num_episodes"] = episodes
    if steps is not None:
        updates["max_steps"] = steps
        updates["environment"] = config.environment.model_copy(update={"max_steps": steps})
    if seed is not None:
        updates["seed"] = seed
    if output_dir is not None:
        updates["output_dir"] = output_dir

    if not updates:
        return config
    return config.model_copy(update=updates)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Serialise an experiment configuration to YAML.

    Args:
        config: The experiment configuration to save.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump()
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
