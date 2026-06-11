"""Tests for scripts/enumerate_tradespace.py — the §3.4 counts regenerate from the rules.

These tests pin the published numbers of docs/decision_matrix.md §2.1/§3.4 to the
rule set. If a validity rule changes, the spec numbers and this file must change
together — by design ("counts are regenerated, never asserted").
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "enumerate_tradespace.py"
_spec = importlib.util.spec_from_file_location("enumerate_tradespace", _SCRIPT)
et = importlib.util.module_from_spec(_spec)
sys.modules["enumerate_tradespace"] = et  # dataclass decorator needs the module registered
_spec.loader.exec_module(et)


class TestMAxis:
    def test_unconstrained_space_is_2800(self):
        assert 7 * 4 * 5 * 5 * 4 == 2800

    def test_r_isl_removes_420_profiles(self):
        assert et.count_valid_ssps() == 2380

    def test_isl_on_single_satellite_invalid(self):
        assert not et.ssp_is_valid("A1", "B1", "C1", "D1", "E1")
        assert et.ssp_is_valid("A1", "B1", "C2", "D1", "E1")


class TestComputeGates:
    def test_onboard_cells_per_tier(self):
        # R-COMPUTE1: LLM-bearing onboard at B2+; R-COMPUTE2: agentic onboard at B3+
        assert et.onboard_cells("B1") == 2
        assert et.onboard_cells("B2") == 4
        assert et.onboard_cells("B3") == 6
        assert et.onboard_cells("B4") == 6

    def test_b1_has_no_onboard_llm(self):
        assert et.onboard_llm_cells("B1") == 0


class TestOrgGates:
    def test_cmas_requires_c2(self):
        assert "cmas" not in et.org_counts("B3", "C1", "E0")
        assert "cmas" in et.org_counts("B3", "C2", "E0")

    def test_distributed_require_c3(self):
        orgs_c2 = et.org_counts("B3", "C2", "E3")
        assert not {"dmas", "imas", "hmas"} & set(orgs_c2)
        orgs_c3 = et.org_counts("B3", "C3", "E3")
        assert {"dmas", "imas", "hmas"} <= set(orgs_c3)

    def test_r_org3_dmas_needs_isl(self):
        assert "dmas" not in et.org_counts("B3", "C3", "E0")
        assert "dmas" in et.org_counts("B3", "C3", "E1")
        # without R-ORG3 dmas exists at E0 (comparison rule set)
        assert "dmas" in et.org_counts("B3", "C3", "E0", r_org3=False)

    def test_org_table_at_full_tier(self):
        # §3.4 per-organisation table (B3+), total / llm-bearing / rl-bearing
        counts = et.org_counts("B3", "C3", "E1")
        assert (counts["sas"].total, counts["sas"].llm, counts["sas"].rl) == (54, 44, 14)
        assert (counts["cmas"].total, counts["cmas"].llm, counts["cmas"].rl) == (36, 32, 11)
        assert (counts["dmas"].total, counts["dmas"].llm, counts["dmas"].rl) == (54, 44, 14)


class TestSpecNumbers:
    def test_adopted_totals(self):
        oc = et.total_mxo(r_org3=True)
        assert (oc.total, oc.llm, oc.rl) == (364980, 286020, 108780)

    def test_comparison_totals_without_r_org3(self):
        oc = et.total_mxo(r_org3=False)
        assert (oc.total, oc.llm, oc.rl) == (383250, 300090, 114030)

    def test_partition_identity(self):
        # llm and rl overlap on mixed AH pairs: total = llm + rl - both + symbolic-only
        oc = et.total_mxo(r_org3=True)
        assert oc.total - oc.llm - oc.rl + oc.both == et.SPEC["symbolic_only"] == 30240

    @pytest.mark.parametrize("name,expected", sorted(et.SPEC["ssp"].items()))
    def test_reference_ssp_counts(self, name, expected):
        a, b, c, d, e = et.REFERENCE_SSPS[name]
        oc = et.o_cells(b, c, e)
        assert (oc.total, oc.llm, oc.rl) == expected

    def test_check_runs_clean(self):
        et.check()
