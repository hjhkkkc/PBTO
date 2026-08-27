#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch independent PBTO seeds.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", action="append", default=[])
    args = parser.parse_args()
    runner = Path(__file__).with_name("run_pbto.py")
    for seed in args.seeds:
        output = Path(args.output_root) / f"seed_{seed}"
        command = [
            sys.executable, str(runner), "--config", args.config,
            "--set", f"experiment.seed={seed}",
            "--set", f"experiment.output_dir={output}",
        ]
        for item in args.extra:
            command += ["--set", item]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
