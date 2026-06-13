"""Generate SSP-variant scenario realizations and experiment configs — standard nomenclature.

Naming standard (until the token-rename migration unifies everything):
  scenario realization : data/ssp/<sspcode>.yaml          e.g. a1b2c1d1e0.yaml
  experiment variant   : data/experiments/<sspcode>_<org>_<proc>_<repr>_<beh>_<ops>[_lf<size>].yaml
The anchor (a1b1c1d1e0) keeps its canonical configs under configs/experiments/.

SS-B realizations scale the platform power class (assumption, stated: solar peak and
battery capacity scale together; B2 ≈ ×5, B3 ≈ ×20 of the EventSat baseline) — to be
replaced by per-tier engineering baselines per the SSP realization sheets (§2.2 to-do).
"""
import copy, yaml
from pathlib import Path

BASE_SC = yaml.safe_load(open("configs/scenarios/eventsat.yaml"))
TIERS = {"a1b2c1d1e0": 5.0, "a1b3c1d1e0": 20.0}
# cell token -> (base config, schema overrides). New-nomenclature tokens
# (decision_matrix §3.1): symb · srl · sllm_re · sllm_ag · hyb_re · hyb_ag.
EXPS = {
    "symb_hd_ao":    ("configs/experiments/eventsat_sas_symbolic_ao.yaml", {}),
    "symb_hd_ah":    ("configs/experiments/eventsat_sas_symbolic_ah.yaml", {}),
    "sllm_re_hd_ah": ("configs/experiments/eventsat_sas_llm_ah.yaml",
                      {"representation": "llm"}),
}

for ssp, k in TIERS.items():
    sc = copy.deepcopy(BASE_SC)
    sc["power"]["solar_panels"]["generation_peak_w"] = 24.0 * k
    sc["power"]["battery"]["capacity_wh"] = 84.0 * k
    yaml.safe_dump(sc, open(f"data/ssp/{ssp}.yaml", "w"))
    for cell, (path, overrides) in EXPS.items():
        ex = copy.deepcopy(yaml.safe_load(open(path)))
        ex.update(overrides)
        eid = f"{ssp}_sas_sda_{cell}"
        ex["experiment_id"] = eid
        ex["description"] = f"SSP {ssp.upper()}: {cell} (B-tier power x{k} realization)"
        ex["environment"]["scenario_config"] = {"scenario_file": f"data/ssp/{ssp}.yaml"}
        ex["num_episodes"] = 30
        ex["output_dir"] = f"data/results/{eid}"
        yaml.safe_dump(ex, open(f"data/experiments/{eid}.yaml", "w"))

# LF rung variant at the anchor (4B model), paired seeds with the HF config
lf = copy.deepcopy(yaml.safe_load(open(EXPS["sllm_re_hd_ah"][0])))
lf["representation"] = "llm"
lf["experiment_id"] = "a1b1c1d1e0_sas_sda_sllm_re_hd_ah_lf4b"
lf["description"] = "LF rung L1 (qwen3.5:4b) at the anchor, paired seeds with HF"
lf["representation_config"]["llm_model"] = "qwen3.5:4b"
lf["num_episodes"] = 3
lf["max_steps"] = 1440
lf["log_level"] = "DEBUG"
lf["output_dir"] = "data/results/a1b1c1d1e0_sas_sda_sllm_re_hd_ah_lf4b"
yaml.safe_dump(lf, open("data/experiments/a1b1c1d1e0_sas_sda_sllm_re_hd_ah_lf4b.yaml", "w"))
print("generated:", sorted(p.name for p in Path("data/ssp").glob("*.yaml")),
      "+", len(list(Path("data/experiments").glob("*.yaml"))), "experiment configs")
