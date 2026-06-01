"""
Generate Experiment Configurations.

Generates YAML configuration files for all (or selected) combinations
of morphological matrix dimensions from a template.

Usage::

    python scripts/generate_experiment_configs.py \\
        --template configs/experiments/template.yaml \\
        --output-dir configs/experiments/generated/
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import yaml


# Default morphological matrix dimensions
ORGANIZATIONS = ["centralized", "hierarchical", "distributed"]
DECISION_LOOPS = ["sda", "ooda", "react"]
REPRESENTATIONS = ["symbolic", "subsymbolic", "hybrid"]
EMERGENCE_MODES = ["hand_designed", "learned"]


def generate_configs(
    template_path: str,
    output_dir: str,
    organizations: list[str] | None = None,
    decision_loops: list[str] | None = None,
    representations: list[str] | None = None,
    emergence_modes: list[str] | None = None,
    seed: int = 42,
) -> list[Path]:
    """Generate experiment config files for all dimension combinations.

    Args:
        template_path: Path to the template YAML file.
        output_dir: Directory to write generated configs.
        organizations: List of organization types (default: all).
        decision_loops: List of decision loop types (default: all).
        representations: List of representation types (default: all).
        emergence_modes: List of emergence modes (default: all).
        seed: Base random seed.

    Returns:
        List of paths to generated config files.
    """
    template_path = Path(template_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(template_path, "r", encoding="utf-8") as f:
        template = yaml.safe_load(f)

    orgs = organizations or ORGANIZATIONS
    loops = decision_loops or DECISION_LOOPS
    reprs = representations or REPRESENTATIONS
    modes = emergence_modes or EMERGENCE_MODES

    generated: list[Path] = []
    exp_num = 1

    for org, loop, rep, mode in itertools.product(orgs, loops, reprs, modes):
        config = dict(template)
        config["experiment_id"] = f"exp_{exp_num:03d}_{org}_{loop}_{rep}_{mode}"
        config["description"] = (
            f"Organization={org}, Loop={loop}, Representation={rep}, Emergence={mode}"
        )
        config["seed"] = seed
        config["agent_organization"] = org
        config["decision_procedure"] = loop
        config["representation"] = rep
        config["behaviour"] = mode
        config["output_dir"] = f"data/results/{config['experiment_id']}"

        filename = f"{config['experiment_id']}.yaml"
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        generated.append(filepath)
        exp_num += 1

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate experiment configurations from a template."
    )
    parser.add_argument(
        "--template",
        default="configs/experiments/template.yaml",
        help="Path to the template YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        default="configs/experiments/generated/",
        help="Directory to write generated configs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    args = parser.parse_args()

    generated = generate_configs(
        template_path=args.template,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(f"Generated {len(generated)} experiment configurations in {args.output_dir}")
    for path in generated:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
