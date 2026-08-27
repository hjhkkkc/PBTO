#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate task-wise PBTO metrics over seeds.")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    grouped = defaultdict(lambda: defaultdict(list))
    for run_dir in args.run_dirs:
        path = Path(run_dir) / "victim_taskwise_metrics.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                task = int(row["task"])
                for metric in ["benign_accuracy", "asr", "target_clean_accuracy"]:
                    grouped[task][metric].append(float(row[metric]))

    rows = []
    for task in sorted(grouped):
        row = {"task": task, "num_runs": len(grouped[task]["asr"])}
        for metric, values in grouped[task].items():
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} task summaries to {output}")


if __name__ == "__main__":
    main()
