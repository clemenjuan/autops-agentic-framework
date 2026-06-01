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
import warnings

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


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

    # Accept both the canonical field name and the legacy alias on input
    # (see validation_alias below); legacy aliases removed after migration.
    model_config = ConfigDict(populate_by_name=True)

    # Identification
    experiment_id: str = Field(default="exp_unnamed")
    description: str = Field(default="")

    # Reproducibility
    seed: int = Field(default=42)

    # Morphological matrix dimensions
    agent_organization: str = Field(default="sas")
    decision_procedure: str = Field(
        default="sda",
        validation_alias=AliasChoices("decision_procedure", "decision_loop"),
    )
    representation: str = Field(default="symbolic")
    behaviour: str = Field(
        default="hand_designed",
        validation_alias=AliasChoices("behaviour", "emergence_mode"),
    )
    operations_paradigm: str = Field(default="autonomous_hybrid")

    # Component-specific sub-configs
    agent_organization_config: Dict[str, Any] = Field(default_factory=dict)
    decision_procedure_config: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("decision_procedure_config", "decision_loop_config"),
    )
    representation_config: Dict[str, Any] = Field(default_factory=dict)
    behaviour_config: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("behaviour_config", "emergence_config"),
    )
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

    # Output. Default includes a template that the model validator
    # substitutes with the actual experiment_id, so direct-constructed
    # configs (tests, ad-hoc scripts) get their own subdir instead of
    # all writing to the same data/results/ root.
    output_dir: str = Field(default="data/results/${experiment_id}")
    save_checkpoints: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    VALID_ORGANIZATIONS: ClassVar[Set[str]] = {
        "sas", "centralized_mas", "decentralized_mas", "independent_mas", "hybrid_mas"
    }
    VALID_REPRESENTATIONS: ClassVar[Set[str]] = {"symbolic", "subsymbolic", "hybrid"}
    # "learned" is the deprecated spelling of "emergent" (accepted during migration).
    VALID_BEHAVIOURS: ClassVar[Set[str]] = {"hand_designed", "emergent", "learned"}
    VALID_ACTION_SPACES: ClassVar[Set[str]] = {"reactive", "agentic"}
    VALID_OPERATIONS_PARADIGMS: ClassVar[Set[str]] = {
        "autonomous_hybrid", "autonomous_ground", "conventional_ground",
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

    @field_validator("behaviour")
    @classmethod
    def _validate_behaviour(cls, v: str) -> str:
        if v not in cls.VALID_BEHAVIOURS:
            raise ValueError(
                f"behaviour must be one of {cls.VALID_BEHAVIOURS}, got '{v}'"
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

    # ------------------------------------------------------------------
    # Learned-emergence mechanism
    # ------------------------------------------------------------------

    VALID_MECHANISMS: ClassVar[Set[str]] = {"ppo", "prompt_optimized", "writable_coala"}

    # Which representations support which mechanisms
    _MECHANISM_REPRESENTATION_RULES: ClassVar[Dict[str, Set[str]]] = {
        "ppo": {"subsymbolic"},
        "prompt_optimized": {"hybrid"},
        "writable_coala": {"hybrid"},
    }
    # writable_coala additionally requires an agentic representation type — either
    # the per-step agentic controller (AH) or its ground-paradigm scheduler stand-in.
    _WRITABLE_COALA_REPR_TYPES: ClassVar[Set[str]] = {
        "agentic_eventsat", "agentic_scheduler_eventsat",
    }

    # Action-space flavor of hybrid representation types (reactive vs agentic).
    _REACTIVE_REPR_TYPES: ClassVar[Set[str]] = {"llm_eventsat", "llm_scheduler_eventsat"}
    _AGENTIC_REPR_TYPES: ClassVar[Set[str]] = {
        "agentic_eventsat", "agentic_scheduler_eventsat",
    }

    # Ground paradigms (AG/CG) execute a `schedule` emitted by the representation
    # between passes. Only these representation types emit one; pairing any other
    # with a ground paradigm yields a degenerate "charge between passes" run.
    _GROUND_PARADIGMS: ClassVar[Set[str]] = {"autonomous_ground", "conventional_ground"}
    _SCHEDULE_PRODUCING_TYPES: ClassVar[Set[str]] = {
        "schedule_based_eventsat",
        "conventional_schedule_eventsat",
        "subsymbolic_scheduler_eventsat",
        "llm_scheduler_eventsat",
        "agentic_scheduler_eventsat",
    }

    # ------------------------------------------------------------------
    # Cross-dimension combination warnings
    # ------------------------------------------------------------------

    # Deterministic representations whose output cannot be improved by
    # iterative loop architectures (OODA orient, ReAct iteration).
    # Future hybrid/subsymbolic representations should NOT be in this set.
    _DETERMINISTIC_REPRESENTATIONS: ClassVar[Set[str]] = {
        "rule_based_eventsat",
        "schedule_based_eventsat",
        "conventional_schedule_eventsat",
    }

    @model_validator(mode="after")
    def _warn_degenerate_combinations(self) -> "ExperimentConfig":
        """Warn about dimension triples that are degenerate given current representations."""
        ops = self.operations_paradigm
        loop = self.decision_procedure
        rep_type = self.representation_config.get("type", "")

        # Action space (hybrid-only flavor: reactive vs agentic). Optional during
        # migration; validated when present and must agree with substrate + type.
        action_space = self.representation_config.get("action_space")
        if action_space is not None:
            if action_space not in self.VALID_ACTION_SPACES:
                raise ValueError(
                    f"representation_config.action_space must be one of "
                    f"{self.VALID_ACTION_SPACES}, got '{action_space}'"
                )
            if action_space == "agentic" and self.representation != "hybrid":
                raise ValueError(
                    f"action_space='agentic' requires representation='hybrid', "
                    f"got '{self.representation}'"
                )
            if rep_type in self._REACTIVE_REPR_TYPES and action_space != "reactive":
                raise ValueError(
                    f"representation_config.type='{rep_type}' is reactive but "
                    f"action_space='{action_space}'"
                )
            if rep_type in self._AGENTIC_REPR_TYPES and action_space != "agentic":
                raise ValueError(
                    f"representation_config.type='{rep_type}' is agentic but "
                    f"action_space='{action_space}'"
                )

        # Ground paradigms execute a `schedule` emitted by the representation
        # between passes. A representation that does not emit one degrades to
        # "charge between every pass" — the representation barely influences the
        # run. Fail loudly so this degenerate cell can't be created silently.
        # Use the *_scheduler_eventsat placeholder types for non-symbolic ground
        # cells (see src/representation/placeholder_schedulers.py).
        if (
            ops in self._GROUND_PARADIGMS
            and rep_type
            and rep_type not in self._SCHEDULE_PRODUCING_TYPES
        ):
            raise ValueError(
                f"operations_paradigm='{ops}' requires a schedule-producing "
                f"representation_config.type (one of {self._SCHEDULE_PRODUCING_TYPES}), "
                f"got '{rep_type}'. Non-schedule representations only act during "
                f"passes and charge between them. For non-symbolic ground cells use "
                f"the placeholder schedulers (e.g. 'llm_scheduler_eventsat')."
            )

        # Deterministic rep + ground paradigm + non-SDA loop:
        # The loop cannot improve a deterministic planner's output, and
        # loop output is discarded between passes (schedule playback).
        # This will NOT apply to future hybrid/subsymbolic representations.
        if (
            ops in ("autonomous_ground", "conventional_ground")
            and loop != "sda"
            and rep_type in self._DETERMINISTIC_REPRESENTATIONS
        ):
            warnings.warn(
                f"Decision loop '{loop}' with deterministic representation "
                f"'{rep_type}' and ground paradigm '{ops}': the loop cannot "
                f"improve a deterministic planner's output, and loop output "
                f"is discarded between passes. Results will match SDA except "
                f"for computational latency. This warning will not apply to "
                f"future hybrid/subsymbolic representations.",
                stacklevel=2,
            )

        # Human-constrained representation with fully autonomous paradigm
        if ops == "autonomous_hybrid" and rep_type == "conventional_schedule_eventsat":
            warnings.warn(
                f"Representation '{rep_type}' models human cognitive "
                f"constraints (conservative margins, shift handover), but "
                f"'{ops}' paradigm is fully autonomous with no human in "
                f"the loop.",
                stacklevel=2,
            )

        # Validate learned-emergence mechanism if specified.
        # "hand_designed" is accepted as an explicit "no learned mechanism"
        # marker (CLAUDE.md lists it as a valid mechanism value); it carries
        # no representation constraints, so treat it like an unset mechanism.
        mechanism = self.behaviour_config.get("mechanism")
        if mechanism == "hand_designed":
            mechanism = None
        if mechanism is not None:
            if mechanism not in self.VALID_MECHANISMS:
                raise ValueError(
                    f"emergence_config.mechanism must be one of "
                    f"{self.VALID_MECHANISMS}, got '{mechanism}'"
                )
            allowed_reps = self._MECHANISM_REPRESENTATION_RULES[mechanism]
            if self.representation not in allowed_reps:
                raise ValueError(
                    f"emergence_config.mechanism='{mechanism}' is only valid with "
                    f"representation in {allowed_reps}, got '{self.representation}'"
                )
            if mechanism == "writable_coala":
                if rep_type not in self._WRITABLE_COALA_REPR_TYPES:
                    raise ValueError(
                        f"behaviour_config.mechanism='writable_coala' requires "
                        f"representation_config.type in {self._WRITABLE_COALA_REPR_TYPES}, "
                        f"got '{rep_type}'"
                    )
                # emergent·memory is gated by the agentic action space (writing is
                # an action). If action_space is declared it must be agentic.
                if action_space is not None and action_space != "agentic":
                    raise ValueError(
                        "behaviour_config.mechanism='writable_coala' requires "
                        f"action_space='agentic', got '{action_space}'"
                    )

        # Warn if emergent hybrid config has no mechanism
        if (
            self.behaviour in ("learned", "emergent")
            and self.representation == "hybrid"
            and mechanism is None
        ):
            warnings.warn(
                f"behaviour='{self.behaviour}' with representation='hybrid' but no "
                f"behaviour_config.mechanism specified. Set mechanism to "
                f"'prompt_optimized' or 'writable_coala'. Defaulting to "
                f"prompt_optimized behaviour at runtime.",
                stacklevel=2,
            )

        # Resolve ${experiment_id} in output_dir for any construction path
        # (YAML load, direct ExperimentConfig(...) in tests, apply_overrides).
        # Keeps each experiment's artifacts in its own subdir instead of
        # all writing to the bare data/results/ root.
        if "${experiment_id}" in self.output_dir:
            self.output_dir = self.output_dir.replace(
                "${experiment_id}", self.experiment_id
            )

        return self

    # ------------------------------------------------------------------
    # Backward-compat read aliases for legacy attribute names.
    # YAML/keyword input aliases are handled by validation_alias above;
    # these cover Python attribute access (config.decision_loop, ...).
    # Removed in Stage 4 of the matrix-restructure migration.
    # ------------------------------------------------------------------

    @property
    def decision_loop(self) -> str:
        return self.decision_procedure

    @property
    def decision_loop_config(self) -> Dict[str, Any]:
        return self.decision_procedure_config

    @property
    def emergence_mode(self) -> str:
        return self.behaviour

    @property
    def emergence_config(self) -> Dict[str, Any]:
        return self.behaviour_config

    @property
    def action_space(self) -> Optional[str]:
        """Resolved action-space flavor: explicit config value, else derived from type."""
        explicit = self.representation_config.get("action_space")
        if explicit is not None:
            return explicit
        rep_type = self.representation_config.get("type", "")
        if rep_type in self._REACTIVE_REPR_TYPES:
            return "reactive"
        if rep_type in self._AGENTIC_REPR_TYPES:
            return "agentic"
        return None


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

    # ${experiment_id} substitution is performed by ExperimentConfig's
    # model validator — no per-loader work needed here.
    return ExperimentConfig(**raw)


def apply_overrides(
    config: ExperimentConfig,
    *,
    episodes: int | None = None,
    steps: int | None = None,
    seed: int | None = None,
    output_dir: str | None = None,
    log_level: str | None = None,
) -> ExperimentConfig:
    """Apply CLI overrides to an experiment configuration.

    Args:
        config: The base configuration.
        episodes: Override num_episodes.
        steps: Override max_steps (both top-level and environment).
        seed: Override seed.
        output_dir: Override output_dir.
        log_level: Override log_level (e.g. "DEBUG" to enable per-step trace).

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
    if log_level is not None:
        updates["log_level"] = log_level

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
