#!/usr/bin/env python3
"""Run both manuscript-method stages without copying or exporting source data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/reproducibility.json"),
    )
    parser.add_argument(
        "--methods-config",
        type=Path,
        default=Path("config/manuscript_methods.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--allow-missing-optional-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data = args.data.resolve()
    config = (
        (root / args.config).resolve() if not args.config.is_absolute() else args.config
    )
    methods_config = (
        (root / args.methods_config).resolve()
        if not args.methods_config.is_absolute()
        else args.methods_config
    )
    output = args.output_dir.resolve()

    methods_command = [
        sys.executable,
        str(root / "scripts" / "run_manuscript_methods.py"),
        "--config",
        str(config),
        "--methods-config",
        str(methods_config),
        "--data",
        str(data),
        "--output-dir",
        str(output / "manuscript_methods"),
    ]
    if args.allow_missing_optional_models:
        methods_command.append("--allow-missing-optional-models")
    subprocess.run(methods_command, cwd=root, check=True)

    decision_command = [
        sys.executable,
        str(root / "scripts" / "run_pipeline.py"),
        "--config",
        str(config),
        "--data",
        str(data),
        "--output-dir",
        str(output / "decision_pipeline"),
    ]
    subprocess.run(decision_command, cwd=root, check=True)


if __name__ == "__main__":
    main()
