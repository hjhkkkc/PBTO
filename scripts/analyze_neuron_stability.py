#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pbto_repro.data import build_dataset_bundle, make_task_views
from pbto_repro.neuron_stability import analyze_neuron_stability
from pbto_repro.utils import configure_torch_threads, load_yaml, make_loader, resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10])
    parser.add_argument("--fisher-batches", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir / "resolved_config.yaml")
    seed = int(cfg.get("experiment", {}).get("seed", 0))
    seed_everything(seed)
    device = resolve_device(str(cfg.get("experiment", {}).get("device", "auto")))
    configure_torch_threads(int(cfg.get("experiment", {}).get("cpu_threads", 4)))
    data_cfg = cfg["dataset"]
    bundle = build_dataset_bundle(
        data_cfg["name"], data_cfg.get("root", "./data"),
        download=bool(data_cfg.get("download", True)), image_size=data_cfg.get("image_size")
    )
    order_file = run_dir / "class_orders.json"
    if order_file.exists():
        order = json.loads(order_file.read_text(encoding="utf-8"))["victim_class_order_ids"]
    else:
        order = json.loads((run_dir / "class_order.json").read_text(encoding="utf-8"))["ids"]
    task1 = make_task_views(bundle, order, int(data_cfg["num_tasks"]), train=False, train_augment=False)[0]
    loader = make_loader(task1, 128, False, int(cfg.get("evaluation", {}).get("num_workers", 4)), seed)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (
        run_dir / "trajectory" if (run_dir / "trajectory").exists() else run_dir / "victim_trajectory"
    )
    checkpoints = sorted(checkpoint_dir.glob("task_*.pt"))
    rows = analyze_neuron_stability(
        checkpoints, loader, device, fractions=args.fractions, fisher_batches=args.fisher_batches
    )
    output = run_dir / "neuron_stability.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
