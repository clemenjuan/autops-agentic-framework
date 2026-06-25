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
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


# ======================================================================
# Pydantic configuration models
# ======================================================================

# 7-cell framework vocabulary (morphological_matrix.md §2) -> internal
# (substrate, action_space). A config, or a nested onboard/ground core block,
# may declare `representation: hllm-a`; ``_expand_cell`` rewrites it to the
# substrate + action_space used by downstream resolution and Jetson accounting.
_CELL_TO_INTERNAL: Dict[str, Any] = {
    "symb": ("symbolic", None),
    "rl": ("rl", None),
    "hrl": ("hybrid-rl", None),
    "llm-s": ("llm", "reactive"),
    "llm-a": ("llm", "agentic"),
    "hllm-s": ("hybrid", "reactive"),
    "hllm-a": ("hybrid", "agentic"),
}


def _expand_cell(data: Any) -> Any:
    """Expand a 7-cell ``representation`` token to substrate + action_space."""
    if isinstance(data, dict):
        rep = data.get("representation")
        if rep in _CELL_TO_INTERNAL:
            substrate, action_space = _CELL_TO_INTERNAL[rep]
            data["representation_cell"] = rep
            data["representation"] = substrate
            if action_space is not None:
                rc = dict(data.get("representation_config") or {})
                rc.setdefault("action_space", action_space)
                data["representation_config"] = rc
    return data


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
        # M-01 is calibrated on DELIVERED information: only data downlinked to the
        # ground counts toward mission utility. Raw observation hours are NOT
        # rewarded (observation that never downlinks is worthless and lets a planner
        # inflate utility by hoarding undeliverable data). `observation` is kept at
        # 0.0 as an ablation knob, not removed. See src/eventsat/metrics.py M-01 docstring.
        default={"observation": 0.0, "downlink": 1.0, "anomaly_penalty": 0.1}
    )
    utility_targets: Dict[str, float] = Field(
        default={
            "observation_hours": 2.0,
            # Two hours of 60 s observations compressed: 120 * (9.41 / 5.11) ~= 221 MB.
            "downlinked_mb": 221.0,
            "mission_duration_days": 90.0,
        }
    )
    baseline_utility_n1: float = Field(default=0.0, ge=0.0)

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


class CoreConfig(BaseModel):
    """One reasoning core of a dual-core AH architecture — an onboard core or a
    ground planner — carrying its own representation and config. The
    ``representation`` accepts a 7-cell token or an internal substrate value (same
    vocabulary as ``ExperimentConfig.representation``)."""

    representation: str = Field(default="symbolic")
    representation_cell: Optional[str] = Field(default=None)
    representation_config: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_cell(cls, data: Any) -> Any:
        return _expand_cell(data)


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration.

    One YAML file → one :class:`ExperimentConfig` instance. All
    EventSat benchmark selectors and execution parameters are captured here.
    """

    # Identification
    experiment_id: str = Field(default="exp_unnamed")
    description: str = Field(default="")

    # Reproducibility
    seed: int = Field(default=42)

    # EventSat benchmark selectors
    agent_organization: str = Field(default="sas")
    decision_procedure: str = Field(default="sda")
    representation: str = Field(default="symbolic")
    # 7-cell framework token (symb/rl/hrl/llm-s/llm-a/hllm-s/hllm-a) when the
    # config used the compact cell vocabulary; None for substrate configs. The
    # normalizer expands the cell into the internal substrate + action_space
    # (morphological_matrix.md §2) so all downstream resolution is unchanged.
    representation_cell: Optional[str] = Field(default=None)
    behaviour: str = Field(default="hand_designed")
    operations_paradigm: str = Field(default="autonomous_hybrid")

    @field_validator("decision_procedure")
    @classmethod
    def _validate_decision_procedure(cls, value: str) -> str:
        if value != "sda":
            raise ValueError(
                "decision_procedure must be 'sda'. OODA/ReAct loop implementations "
                "were retired because the current EventSat/SSA benchmarks keep "
                "the decision driver fixed."
            )
        return value

    # Component-specific sub-configs
    agent_organization_config: Dict[str, Any] = Field(default_factory=dict)
    decision_procedure_config: Dict[str, Any] = Field(default_factory=dict)
    representation_config: Dict[str, Any] = Field(default_factory=dict)
    behaviour_config: Dict[str, Any] = Field(default_factory=dict)
    operations_paradigm_config: Dict[str, Any] = Field(default_factory=dict)

    # Dual-core AH: independent onboard + ground cores, each with its own
    # representation + config (morphological_matrix.md §3 — the ah_<onboard>_<ground>
    # pairs). When both are set they drive the two cores; when absent the single
    # `representation` drives both (backward-compatible single-rep AH / AO / AG / CG).
    onboard: Optional[CoreConfig] = Field(default=None)
    ground: Optional[CoreConfig] = Field(default=None)

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
    # Internal substrate vocabulary. Configs may instead declare a 7-cell
    # framework token (symb/rl/hrl/llm-s/llm-a/hllm-s/hllm-a, morphological_matrix.md
    # §2); _normalize_representation_cell expands those into these substrates +
    # action_space before validation. "subsymbolic" remains accepted as the
    # explicit substrate name; "rl" is the compact benchmark token.
    # "hybrid-rl" backs the not-yet-implemented hrl cell (placeholder).
    VALID_REPRESENTATIONS: ClassVar[Set[str]] = {
        "symbolic", "subsymbolic", "rl", "llm", "hybrid", "hybrid-rl",
    }
    VALID_BEHAVIOURS: ClassVar[Set[str]] = {"hand_designed", "emergent"}
    VALID_ACTION_SPACES: ClassVar[Set[str]] = {"reactive", "agentic"}
    VALID_OPERATIONS_PARADIGMS: ClassVar[Set[str]] = {
        "autonomous_onboard", "autonomous_hybrid", "autonomous_ground", "conventional_ground",
    }
    _REMOVED_FIELD_NAMES: ClassVar[Set[str]] = {
        "decision_loop", "decision_loop_config", "emergence_mode", "emergence_config",
    }

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_field_names(cls, data: Any) -> Any:
        """Reject obsolete config field names so stale configs fail loudly."""
        if isinstance(data, dict):
            present = cls._REMOVED_FIELD_NAMES & set(data)
            if present:
                raise ValueError(
                    f"Removed config field(s) {sorted(present)} are no longer supported. "
                    f"Use decision_procedure / decision_procedure_config / behaviour / "
                    f"behaviour_config."
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def _normalize_representation_cell(cls, data: Any) -> Any:
        """Expand the top-level 7-cell representation token (the nested onboard/
        ground cores are expanded by ``CoreConfig``'s own normalizer)."""
        return _expand_cell(data)

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
    # Learned behaviour mechanism
    # ------------------------------------------------------------------

    VALID_MECHANISMS: ClassVar[Set[str]] = {"ppo", "prompt_optimized", "writable_coala"}

    # Which representations support which mechanisms
    _MECHANISM_REPRESENTATION_RULES: ClassVar[Dict[str, Set[str]]] = {
        "ppo": {"subsymbolic", "rl"},
        "prompt_optimized": {"hybrid", "llm"},
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
        "hrl_scheduler_eventsat",
        "llm_single_scheduler_eventsat",
        "llm_agentic_scheduler_eventsat",
    }

    # ------------------------------------------------------------------
    # Cross-dimension combination warnings
    # ------------------------------------------------------------------

    # Deterministic schedule/rule representations used by guardrails.
    _DETERMINISTIC_REPRESENTATIONS: ClassVar[Set[str]] = {
        "rule_based_eventsat",
        "schedule_based_eventsat",
        "conventional_schedule_eventsat",
    }

    @staticmethod
    def _resolve_repr_type(
        representation: str,
        action_space: Optional[str],
        ops: str,
        scenario: str = "eventsat",
    ) -> str:
        """Resolve the concrete representation class from benchmark coordinates.

        The class is determined by (substrate, action_space, operations_paradigm):
        ops picks per-step controller (AH) vs schedule-producer (AG/CG), and for
        hybrids action_space picks reactive vs agentic. An explicit
        ``representation_config.type`` overrides this (see ``resolved_representation_type``).
        """
        _ONBOARD_OPS = ("autonomous_onboard", "autonomous_hybrid")
        if representation == "symbolic":
            if scenario == "ssa" and ops in _ONBOARD_OPS:
                return "rule_based_ssa"
            return {
                "autonomous_onboard": "rule_based_eventsat",
                "autonomous_hybrid": "rule_based_eventsat",
                "autonomous_ground": "schedule_based_eventsat",
                "conventional_ground": "conventional_schedule_eventsat",
            }[ops]
        if representation in ("subsymbolic", "rl"):
            return (
                "subsymbolic_eventsat" if ops in _ONBOARD_OPS
                else "subsymbolic_scheduler_eventsat"
            )
        if representation == "hybrid-rl":
            # hrl cell (hybrid RL+symbolic): not yet implemented → documented
            # placeholder (is_placeholder, morphological_matrix.md §2).
            return (
                "hrl_onboard_eventsat" if ops in _ONBOARD_OPS
                else "hrl_scheduler_eventsat"
            )
        if representation == "llm":
            # Pure-LLM cells (no symbolic layer). Ground schedulers are real;
            # onboard pure-LLM cells remain documented placeholders. The
            # symbolic-guarded LLM lives under the 'hybrid' substrate
            # (hllm-s/hllm-a → llm_eventsat/agentic_eventsat).
            if action_space == "agentic":  # llm-a
                return (
                    "llm_agentic_onboard_eventsat" if ops in _ONBOARD_OPS
                    else "llm_agentic_scheduler_eventsat"
                )
            return (  # llm-s
                "llm_single_onboard_eventsat" if ops in _ONBOARD_OPS
                else "llm_single_scheduler_eventsat"
            )
        if representation == "hybrid":
            if action_space not in ("reactive", "agentic"):
                raise ValueError(
                    "hybrid representation requires representation_config.action_space "
                    "(reactive|agentic) when representation_config.type is not set"
                )
            if action_space == "reactive":
                return "llm_eventsat" if ops == "autonomous_hybrid" else "llm_scheduler_eventsat"
            return "agentic_eventsat" if ops == "autonomous_hybrid" else "agentic_scheduler_eventsat"
        raise ValueError(f"cannot resolve representation type for representation='{representation}'")

    @property
    def resolved_representation_type(self) -> str:
        """Concrete representation class name. Explicit `type` wins; else resolved."""
        explicit = self.representation_config.get("type")
        if explicit:
            return explicit
        return self._resolve_repr_type(
            self.representation,
            self.representation_config.get("action_space"),
            self.operations_paradigm,
            self.environment.scenario,
        )

    def _onboard_core(self) -> tuple[str, Optional[str]]:
        """Effective (substrate, action_space) of the onboard core: the dual-core
        ``onboard`` block when set, else the single top-level representation."""
        if self.onboard is not None:
            return self.onboard.representation, self.onboard.representation_config.get("action_space")
        return self.representation, self.representation_config.get("action_space")

    def _ground_core(self) -> tuple[str, Optional[str]]:
        """Effective (substrate, action_space) of the ground core: the dual-core
        ``ground`` block when set, else the single top-level representation."""
        if self.ground is not None:
            return self.ground.representation, self.ground.representation_config.get("action_space")
        return self.representation, self.representation_config.get("action_space")

    @property
    def resolved_onboard_type(self) -> Optional[str]:
        """Onboard per-step core, for paradigms with an onboard slot (AO, AH).

        Follows the onboard core's substrate (morphological_matrix.md §2 — substrate
        × action *per active core*): symbolic→rule_based, subsymbolic·RL→
        subsymbolic_eventsat, hybrid·reactive (hllm-s)→llm_eventsat, hybrid·agentic
        (hllm-a)→agentic_eventsat. The pure-LLM cells (llm-s/llm-a) and hrl have no
        onboard core yet → placeholders. None for AG/CG.
        """
        if self.operations_paradigm not in ("autonomous_onboard", "autonomous_hybrid"):
            return None
        substrate, action_space = self._onboard_core()
        if substrate == "symbolic":
            return "rule_based_ssa" if self.environment.scenario == "ssa" else "rule_based_eventsat"
        if substrate in ("subsymbolic", "rl"):
            return "subsymbolic_eventsat"
        if substrate == "hybrid-rl":
            return "hrl_onboard_eventsat"  # hrl cell placeholder
        if substrate == "llm":  # pure LLM onboard (no symbolic layer) → placeholders
            if action_space == "agentic":
                return "llm_agentic_onboard_eventsat"  # pure-LLM agentic onboard placeholder
            return "llm_single_onboard_eventsat"  # pure-LLM single-shot onboard placeholder
        # hybrid (LLM + symbolic): reactive (hllm-s) -> llm_eventsat; agentic (hllm-a) -> agentic_eventsat
        if action_space == "agentic":
            return "agentic_eventsat"
        return "llm_eventsat"

    @property
    def resolved_ground_planner_type(self) -> Optional[str]:
        """Ground full-pass planner (schedule producer), for AH/AG/CG. None for AO.

        AH shares AG's *algorithmic* ground planner; CG uses its human-realistic
        planner. Follows the ground core's substrate (the ``ground`` block for
        dual-core AH, else the single representation).
        """
        ops = self.operations_paradigm
        if ops == "autonomous_onboard":
            return None
        ground_ops = "autonomous_ground" if ops == "autonomous_hybrid" else ops
        substrate, action_space = self._ground_core()
        return self._resolve_repr_type(substrate, action_space, ground_ops, self.environment.scenario)

    @property
    def onboard_uses_jetson(self) -> bool:
        """Whether the onboard compute (Jetson) is kept powered for per-step inference.

        True only when an onboard slot is active (AO/AH) **and** the onboard core is
        Jetson-based (subsymbolic RL, or hybrid whose onboard is the RL policy).
        Symbolic onboard rules run on the OBC (3.3 V, sub-watt) → no Jetson overhead.
        Drives `env.onboard_compute_active`. Keys off the *onboard* core's substrate.
        """
        if self.operations_paradigm not in ("autonomous_onboard", "autonomous_hybrid"):
            return False
        substrate, _ = self._onboard_core()
        return substrate in ("subsymbolic", "rl", "hybrid", "hybrid-rl")

    @property
    def onboard_representation_config(self) -> Dict[str, Any]:
        """Representation config for the onboard core (the ``onboard`` block when
        dual-core, else the shared ``representation_config``)."""
        return self.onboard.representation_config if self.onboard is not None else self.representation_config

    @property
    def ground_representation_config(self) -> Dict[str, Any]:
        """Representation config for the ground core (the ``ground`` block when
        dual-core, else the shared ``representation_config``)."""
        return self.ground.representation_config if self.ground is not None else self.representation_config

    @model_validator(mode="after")
    def _validate_dual_core(self) -> "ExperimentConfig":
        """Dual-core onboard/ground blocks: AH-only, both-or-neither, no LLM onboard."""
        if self.onboard is None and self.ground is None:
            return self
        if self.operations_paradigm != "autonomous_hybrid":
            raise ValueError(
                "onboard/ground core blocks are only valid with "
                f"operations_paradigm='autonomous_hybrid', got '{self.operations_paradigm}'."
            )
        if (self.onboard is None) != (self.ground is None):
            raise ValueError(
                "dual-core AH requires BOTH 'onboard' and 'ground' core blocks "
                "(or neither, to drive both cores from the single 'representation')."
            )
        # No per-step LLM onboard (morphological_matrix.md §3): onboard ∈ {symb, rl, hrl}.
        if self.onboard.representation not in ("symbolic", "subsymbolic", "rl", "hybrid-rl"):
            cell = self.onboard.representation_cell or self.onboard.representation
            raise ValueError(
                f"onboard core '{cell}' is not onboard-feasible — AH onboard must be "
                f"symb, rl, or hrl (no per-step LLM onboard); the LLM cells belong in "
                f"the ground slot (morphological_matrix.md §3)."
            )
        return self

    @model_validator(mode="after")
    def _warn_degenerate_combinations(self) -> "ExperimentConfig":
        """Warn about dimension triples that are degenerate given current representations."""
        ops = self.operations_paradigm
        loop = self.decision_procedure
        action_space = self.representation_config.get("action_space")

        # Action space (hybrid-only flavor: reactive vs agentic). Validate value +
        # substrate agreement before resolution.
        if action_space is not None:
            if action_space not in self.VALID_ACTION_SPACES:
                raise ValueError(
                    f"representation_config.action_space must be one of "
                    f"{self.VALID_ACTION_SPACES}, got '{action_space}'"
                )
            if action_space == "agentic" and self.representation not in ("hybrid", "llm"):
                raise ValueError(
                    f"action_space='agentic' requires an LLM-bearing representation "
                    f"('hybrid' or 'llm'), got '{self.representation}'"
                )

        # autonomous_onboard is onboard-only; a hybrid has no onboard core of its
        # own (its LLM is a ground component), so hybrid+ao is degenerate.
        if self.representation == "hybrid" and ops == "autonomous_onboard":
            raise ValueError(
                "representation='hybrid' with operations_paradigm='autonomous_onboard' "
                "is excluded: a hybrid has no standalone onboard core (its LLM is a "
                "ground planner). Use a symbolic or subsymbolic onboard, or paradigm "
                "autonomous_hybrid (onboard + ground)."
            )

        # Resolve the concrete representation class (raises if hybrid lacks action_space).
        rep_type = self.resolved_representation_type

        if (
            self.environment.scenario == "ssa"
            and ops in self._GROUND_PARADIGMS
            and self.agent_organization not in {"sas", "centralized_mas"}
        ):
            raise ValueError(
                "SSA ground paradigms (AG/CG) are only supported with SAS or CMAS "
                "organizations; AO/AH onboard paradigms may use all five organizations."
            )

        # When an explicit type override is given, it must agree with action_space.
        explicit_type = self.representation_config.get("type")
        if explicit_type and action_space is not None:
            if explicit_type in self._REACTIVE_REPR_TYPES and action_space != "reactive":
                raise ValueError(
                    f"representation_config.type='{explicit_type}' is reactive but "
                    f"action_space='{action_space}'"
                )
            if explicit_type in self._AGENTIC_REPR_TYPES and action_space != "agentic":
                raise ValueError(
                    f"representation_config.type='{explicit_type}' is agentic but "
                    f"action_space='{action_space}'"
                )

        # Ground paradigms execute a `schedule` emitted by the representation
        # between passes. A representation that does not emit one degrades to
        # "charge between every pass" — the representation barely influences the
        # run. Fail loudly so this degenerate cell can't be created silently.
        # Use registered *_scheduler_eventsat types for non-symbolic ground cells;
        # only the RL/HRL scheduler entries are placeholder-marked.
        if (
            ops in self._GROUND_PARADIGMS
            and self.environment.scenario in {"eventsat", "ssa"}
            and rep_type
            and rep_type not in self._SCHEDULE_PRODUCING_TYPES
        ):
            raise ValueError(
                f"operations_paradigm='{ops}' requires a schedule-producing "
                f"representation_config.type (one of {self._SCHEDULE_PRODUCING_TYPES}), "
                f"got '{rep_type}'. Non-schedule representations only act during "
                f"passes and charge between them. For non-symbolic ground cells use "
                f"a registered scheduler (e.g. 'llm_scheduler_eventsat')."
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

        # Validate learned-behaviour mechanism if specified.
        # "hand_designed" is accepted as an explicit "no learned mechanism"
        # marker (CLAUDE.md lists it as a valid mechanism value); it carries
        # no representation constraints, so treat it like an unset mechanism.
        mechanism = self.behaviour_config.get("mechanism")
        if mechanism == "hand_designed":
            mechanism = None
        if mechanism is not None:
            if mechanism not in self.VALID_MECHANISMS:
                raise ValueError(
                    f"behaviour_config.mechanism must be one of "
                    f"{self.VALID_MECHANISMS}, got '{mechanism}'"
                )
            allowed_reps = self._MECHANISM_REPRESENTATION_RULES[mechanism]
            # The mechanism trains whichever core matches; for dual-core AH it may be
            # the onboard (ppo on rl) or ground (writable_coala on agentic) core.
            substrates = {self.representation}
            if self.onboard is not None:
                substrates.add(self.onboard.representation)
            if self.ground is not None:
                substrates.add(self.ground.representation)
            if not (substrates & allowed_reps):
                raise ValueError(
                    f"behaviour_config.mechanism='{mechanism}' is only valid with "
                    f"representation in {allowed_reps}, got {sorted(substrates)}"
                )
            if mechanism == "writable_coala":
                # The agentic core that carries writable memory may be the top-level
                # rep (single) or, for dual-core AH, the ground planner.
                core_types = {rep_type}
                if self.onboard is not None or self.ground is not None:
                    core_types |= {self.resolved_onboard_type, self.resolved_ground_planner_type}
                if not (core_types & self._WRITABLE_COALA_REPR_TYPES):
                    raise ValueError(
                        f"behaviour_config.mechanism='writable_coala' requires an agentic "
                        f"core type in {self._WRITABLE_COALA_REPR_TYPES}, "
                        f"got {sorted(t for t in core_types if t)}"
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
            self.behaviour == "emergent"
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
