"""The experiment-config generator emits exactly the 32 EventSat·SAS experiments
(morphological_matrix.md §4) and the on-disk configs stay in sync with it."""
from __future__ import annotations

import importlib.util
import sys
import warnings
from collections import Counter
from pathlib import Path

import pytest

from src.core.config_loader import ExperimentConfig

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "generate_experiment_configs.py"
_spec = importlib.util.spec_from_file_location("generate_experiment_configs", _SCRIPT)
gen = importlib.util.module_from_spec(_spec)
sys.modules["generate_experiment_configs"] = gen
_spec.loader.exec_module(gen)

_SSA_SCRIPT = _ROOT / "scripts" / "generate_ssa_configs.py"
_ssa_spec = importlib.util.spec_from_file_location("generate_ssa_configs", _SSA_SCRIPT)
ssa_gen = importlib.util.module_from_spec(_ssa_spec)
sys.modules["generate_ssa_configs"] = ssa_gen
_ssa_spec.loader.exec_module(ssa_gen)


def _paradigm(eid: str) -> str:
    return eid.split("_")[2]


def test_matrix_counts() -> None:
    m = gen.build_matrix()
    assert len(m) == 32
    assert Counter(_paradigm(e) for e in m) == {"conventional": 1, "ag": 7, "ao": 3, "ah": 21}


def test_every_config_constructs_and_resolves() -> None:
    for eid, cfg in gen.build_matrix().items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ec = ExperimentConfig(**cfg)
        assert ec.experiment_id == eid
        paradigm = _paradigm(eid)
        if paradigm in ("ao", "ah"):
            assert ec.resolved_onboard_type is not None
        if paradigm in ("conventional", "ag", "ah"):
            assert ec.resolved_ground_planner_type is not None


def test_ah_pairs_are_dual_core() -> None:
    ah = {e: c for e, c in gen.build_matrix().items() if _paradigm(e) == "ah"}
    assert len(ah) == 21
    for cfg in ah.values():
        assert "onboard" in cfg and "ground" in cfg
        assert cfg["onboard"]["representation"] in ("symb", "rl", "hrl")  # no LLM onboard


def test_placeholder_cells_present_and_flagged() -> None:
    import src.eventsat.placeholders  # noqa: F401
    import src.eventsat.agentic_scheduler  # noqa: F401  (real llm-a/hllm-a)
    from src.core.behaviour.controller import _REPRESENTATION_REGISTRY
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # hrl ground is still a documented placeholder...
        hrl = ExperimentConfig(**gen.build_matrix()["eventsat_sas_ag_hrl"])
        # ...but the agentic pure-LLM ground cell (llm-a) is now a REAL core.
        llma = ExperimentConfig(**gen.build_matrix()["eventsat_sas_ag_llm-a"])
    assert hrl.resolved_ground_planner_type == "hrl_scheduler_eventsat"
    assert _REPRESENTATION_REGISTRY["hrl_scheduler_eventsat"].is_placeholder is True
    assert llma.resolved_ground_planner_type == "llm_agentic_scheduler_eventsat"
    assert _REPRESENTATION_REGISTRY["llm_agentic_scheduler_eventsat"].is_placeholder is False


def test_disk_configs_match_generator() -> None:
    """Committed configs must equal the generator's output (no drift)."""
    disk = {p.stem for p in (_ROOT / "configs" / "experiments").glob("eventsat_sas_*.yaml")}
    assert disk == set(gen.build_matrix())


def test_ssa_ao_matrix_counts() -> None:
    m = ssa_gen.build_matrix()
    assert len(m) == 20
    assert Counter(e.split("_")[1] for e in m) == {
        "sas": 4,
        "cmas": 4,
        "dmas": 4,
        "imas": 4,
        "hmas": 4,
    }
    assert Counter(e.split("_")[3] for e in m) == {"symb": 10, "rl": 10}
    assert Counter(e.rsplit("_n", 1)[1] for e in m) == {"3": 10, "5": 10}


def test_every_ssa_config_constructs_and_resolves() -> None:
    for eid, cfg in ssa_gen.build_matrix().items():
        ec = ExperimentConfig(**cfg)
        assert ec.experiment_id == eid
        assert ec.environment.scenario == "ssa"
        assert ec.operations_paradigm == "autonomous_onboard"
        assert ec.resolved_ground_planner_type is None
        if eid.split("_")[3] == "symb":
            assert ec.resolved_onboard_type == "rule_based_ssa"
        else:
            assert ec.resolved_onboard_type == "subsymbolic_eventsat"
            assert ec.representation_config["rl_mock"] is True


def test_ssa_disk_configs_match_generator() -> None:
    """Committed SSA configs must equal the SSA generator's output (no drift)."""
    disk = {p.stem for p in (_ROOT / "configs" / "experiments").glob("ssa_*.yaml")}
    assert disk == set(ssa_gen.build_matrix())
