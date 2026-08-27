#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def variants(mode: str) -> List[tuple[str, Dict[str, object]]]:
    if mode == "components":
        return [
            ("no_sim_no_align", {
                "trigger.trajectory_mode": "last",
                "trigger.lambda_align": 0.0,
                "refinement.enabled": False,
            }),
            ("sim_only", {
                "trigger.trajectory_mode": "all",
                "trigger.lambda_align": 0.0,
            }),
            ("align_only", {
                "trigger.trajectory_mode": "last",
                "trigger.lambda_align": 1.0,
                "refinement.enabled": False,
            }),
            ("full", {
                "trigger.trajectory_mode": "all",
                "trigger.lambda_align": 1.0,
            }),
        ]
    if mode == "trajectory":
        return [(f"proxy_tasks_{m}", {"proxy.num_tasks": m}) for m in [1, 3, 5, 10]]
    if mode == "lambda":
        return [(f"lambda_{value}", {"trigger.lambda_align": value}) for value in [0, 0.1, 1.0, 5.0, 10.0]]
    if mode == "memory":
        return [(f"memory_{value}", {"victim.memory_size": value}) for value in [500, 1000, 2000, 5000]]
    if mode == "proxy_size":
        return [(f"proxy_size_{value}", {"proxy.samples_per_class": value}) for value in [1000, 2000, 5000, 10000]]
    raise ValueError(mode)


def yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch PBTO ablation sweeps.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode", required=True,
        choices=["components", "trajectory", "lambda", "memory", "proxy_size"]
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", action="append", default=[], help="Additional KEY=VALUE override")
    args = parser.parse_args()

    script = Path(__file__).with_name("run_pbto.py")
    output_root = Path(args.output_root)
    for name, override_map in variants(args.mode):
        command = [sys.executable, str(script), "--config", args.config]
        command += ["--set", f"experiment.output_dir={output_root / name}"]
        for key, value in override_map.items():
            command += ["--set", f"{key}={yaml_scalar(value)}"]
        for item in args.extra:
            command += ["--set", item]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
