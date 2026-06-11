"""Tradespace enumeration — regenerates the M×O counts of docs/decision_matrix.md.

Encodes the validity rules of decision_matrix.md §2.1 (M axis) and §3.2 (O axis,
O×M gates) and derives every count in §3.4 from them. Counts are *regenerated,
never asserted*: if a rule changes, this script (and its test) is the single
place the numbers come from.

Rules encoded (hard gates only; soft flags steer sampling, never validity):
  R-ISL       SS-E > E0  requires  SS-C >= C2
  R-ORG1      cmas exists only as AH, at SS-C >= C2 (identified with SAS·AH at C1)
  R-ORG2      dmas / imas / hmas require SS-C >= C3
  R-ORG3      dmas additionally requires SS-E >= E1
  R-COMPUTE1  onboard LLM core (subsymbolic·LLM / hybrid) requires SS-B >= B2
  R-COMPUTE2  onboard agentic loop requires SS-B >= B3

Per-core cells (substrate × action): symbolic·re, RL·re, LLM·re, LLM·ag,
hybrid·re, hybrid·ag — 6 cells; LLM-bearing = all except {symbolic·re, RL·re}
(hybrid counted LLM-bearing conservatively, §3.2).

Usage:
    uv run python scripts/enumerate_tradespace.py          # report
    uv run python scripts/enumerate_tradespace.py --check  # assert spec numbers
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

# ---------------------------------------------------------------- M axis
SS_A = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
SS_B = ["B1", "B2", "B3", "B4"]
SS_C = ["C1", "C2", "C3", "C4", "C5"]
SS_D = ["D1", "D2", "D3", "D4", "D5"]
SS_E = ["E0", "E1", "E2", "E3"]

C_NUM = {c: i + 1 for i, c in enumerate(SS_C)}   # ordinal rank, 1-based
B_NUM = {b: i + 1 for i, b in enumerate(SS_B)}
E_NUM = {e: i for i, e in enumerate(SS_E)}        # E0 -> 0


def ssp_is_valid(a: str, b: str, c: str, d: str, e: str) -> bool:
    """R-ISL is the only hard M-level gate (§2.1)."""
    if E_NUM[e] > 0 and C_NUM[c] < 2:
        return False
    return True


def count_valid_ssps() -> int:
    return sum(
        ssp_is_valid(a, b, c, d, e)
        for a in SS_A for b in SS_B for c in SS_C for d in SS_D for e in SS_E
    )


# ---------------------------------------------------------------- O axis
# Per-core cells under the SS-B compute gates (R-COMPUTE1/2), onboard slot only.
def onboard_cells(b: str) -> int:
    """symbolic·re + RL·re always; +LLM·re, hybrid·re at B2+; +LLM·ag, hybrid·ag at B3+."""
    n = 2
    if B_NUM[b] >= 2:
        n += 2
    if B_NUM[b] >= 3:
        n += 2
    return n


def onboard_llm_cells(b: str) -> int:
    return onboard_cells(b) - 2  # minus symbolic·re, RL·re


GROUND_CELLS = 6          # ground core is never SS-B-constrained
GROUND_LLM = 4


@dataclass(frozen=True)
class OrgCount:
    total: int
    llm: int          # contains an LLM-bearing core in >=1 active slot
    rl: int           # contains a subsymbolic-RL core in >=1 active slot
    both: int         # mixed AH pairs needing BOTH ladders (llm and rl overlap)


def org_counts(b: str, c: str, e: str, r_org3: bool = True) -> dict[str, OrgCount]:
    """Valid O configurations per organisation for one (SS-B, SS-C, SS-E)."""
    ob = onboard_cells(b)
    ob_llm = onboard_llm_cells(b)

    # AH pair block: ob onboard cells x 6 ground cells
    ah_total = ob * GROUND_CELLS
    ah_llm = ah_total - 2 * 2                      # minus pairs with no LLM in either slot
    ah_rl = ah_total - (ob - 1) * (GROUND_CELLS - 1)  # minus pairs with no RL in either slot
    ah_both = 1 * GROUND_LLM + ob_llm * 1          # RL onboard x LLM ground + LLM onboard x RL ground

    # single-core paradigms: CG, AG (ground core) + AO (onboard core); AH = pair
    sas_total = GROUND_CELLS + GROUND_CELLS + ob + ah_total
    sas_llm = GROUND_LLM + GROUND_LLM + ob_llm + ah_llm
    sas_rl = 1 + 1 + 1 + ah_rl                     # one RL·re cell per single-core paradigm

    sas_oc = OrgCount(sas_total, sas_llm, sas_rl, ah_both)
    out = {"sas": sas_oc}

    # R-ORG1: cmas = AH only, C >= C2 (identified with SAS·AH at C1 -> counted once)
    if C_NUM[c] >= 2:
        out["cmas"] = OrgCount(ah_total, ah_llm, ah_rl, ah_both)
    # R-ORG2 (+ R-ORG3 for dmas)
    if C_NUM[c] >= 3:
        out["imas"] = sas_oc
        out["hmas"] = sas_oc                        # soft flag at E0, still valid
        if not r_org3 or E_NUM[e] >= 1:
            out["dmas"] = sas_oc
    return out


def o_cells(b: str, c: str, e: str, r_org3: bool = True) -> OrgCount:
    counts = org_counts(b, c, e, r_org3)
    return OrgCount(
        sum(v.total for v in counts.values()),
        sum(v.llm for v in counts.values()),
        sum(v.rl for v in counts.values()),
        sum(v.both for v in counts.values()),
    )


def total_mxo(r_org3: bool = True) -> OrgCount:
    total = llm = rl = both = 0
    for a in SS_A:
        for b in SS_B:
            for c in SS_C:
                for d in SS_D:
                    for e in SS_E:
                        if not ssp_is_valid(a, b, c, d, e):
                            continue
                        oc = o_cells(b, c, e, r_org3)
                        total += oc.total
                        llm += oc.llm
                        rl += oc.rl
                        both += oc.both
    return OrgCount(total, llm, rl, both)


# ---------------------------------------------------------------- reference SSPs (§2.2)
REFERENCE_SSPS = {
    "SSP-01 GEO telecomms":      ("A2", "B4", "C1", "D2", "E0"),
    "SSP-03 Mega-constellation": ("A2", "B2", "C5", "D1", "E3"),
    "SSP-04 EventSat":           ("A1", "B1", "C1", "D1", "E0"),
    "SSP-05 Agile 1-sat":        ("A1", "B2", "C1", "D1", "E0"),
    "SSP-06 Agile med. const.":  ("A1", "B2", "C3", "D1", "E0"),
    "SSP-09 SSA small const.":   ("A5", "B2", "C2", "D1", "E2"),
    "SSP-10 SSA large const.":   ("A5", "B2", "C3", "D1", "E3"),
    "SSP-11 Formation flying":   ("A4", "B3", "C2", "D1", "E1"),
    "SSP-12 Mars orbiter":       ("A6", "B3", "C1", "D4", "E0"),
    "SSP-15 TechDemo":           ("A7", "B1", "C2", "D1", "E0"),
}

# Published numbers (decision_matrix.md §2.1 / §3.4) — what --check asserts.
SPEC = {
    "valid_ssps": 2380,
    # (total, llm-bearing, rl-bearing) — llm/rl overlap on mixed AH pairs
    "total": (364980, 286020, 108780),  # adopted rule set (with R-ORG3)
    "total_no_r_org3": (383250, 300090, 114030),
    "symbolic_only": 30240,             # adopted rule set: total - llm - rl + both
    "ssp": {
        "SSP-04 EventSat": (26, 16, 10),
        "SSP-15 TechDemo": (38, 24, 17),
        "SSP-05 Agile 1-sat": (40, 30, 12),
        "SSP-01 GEO telecomms": (54, 44, 14),
        "SSP-12 Mars orbiter": (54, 44, 14),
        "SSP-09 SSA small const.": (64, 50, 21),
        "SSP-11 Formation flying": (90, 76, 25),
        "SSP-06 Agile med. const.": (144, 110, 45),
        "SSP-03 Mega-constellation": (184, 140, 57),
        "SSP-10 SSA large const.": (184, 140, 57),
    },
}


def report() -> None:
    print(f"valid SSPs (R-ISL): {count_valid_ssps()}  (of {7*4*5*5*4} unconstrained)")
    adopted = total_mxo(r_org3=True)
    legacy = total_mxo(r_org3=False)
    sym_only = adopted.total - adopted.llm - adopted.rl + adopted.both
    print(f"M×O cells, adopted rule set : {adopted.total:>7,}  ({adopted.llm:,} LLM-bearing,"
          f" {adopted.rl:,} RL-bearing, {sym_only:,} symbolic-only;"
          f" {adopted.both:,} mixed AH pairs need both ladders)")
    print(f"M×O cells, without R-ORG3   : {legacy.total:>7,}  ({legacy.llm:,} LLM-bearing,"
          f" {legacy.rl:,} RL-bearing)")
    print("\nPer reference SSP (O-cells / LLM-bearing / RL-bearing):")
    for name, (a, b, c, d, e) in REFERENCE_SSPS.items():
        oc = o_cells(b, c, e)
        print(f"  {name:<28} {a}/{b}/{c}/{d}/{e}   {oc.total:>3} / {oc.llm} / {oc.rl}")
    print("\nPer organisation at B3+/C3+/E1+ (unique O configs, total/llm/rl):")
    for org, oc in org_counts("B3", "C3", "E1").items():
        print(f"  {org:<5} {oc.total:>3} / {oc.llm} / {oc.rl}")


def check() -> None:
    assert count_valid_ssps() == SPEC["valid_ssps"]
    adopted = total_mxo(r_org3=True)
    assert (adopted.total, adopted.llm, adopted.rl) == SPEC["total"], (adopted, SPEC["total"])
    assert adopted.total - adopted.llm - adopted.rl + adopted.both == SPEC["symbolic_only"]
    legacy = total_mxo(r_org3=False)
    assert (legacy.total, legacy.llm, legacy.rl) == SPEC["total_no_r_org3"]
    for name, expected in SPEC["ssp"].items():
        a, b, c, d, e = REFERENCE_SSPS[name]
        oc = o_cells(b, c, e)
        assert (oc.total, oc.llm, oc.rl) == expected, (name, oc, expected)
    print("OK — all spec numbers regenerate from the rule set")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        report()
