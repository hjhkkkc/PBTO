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
from pbto_repro.subspace import analyze_subspace_trajectory
from pbto_repro.utils import configure_torch_threads, ensure_dir, load_yaml, make_loader, resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CKA, Grassmann distance and variance retention.")
    parser.add_argument("--run-dir", required=True, help="Directory produced by run_clean_cil.py or run_pbto.py")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--layers", nargs="+", default=["layer1", "layer2", "layer3", "layer4"])
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--explained-variance", type=float, default=0.90)
    parser.add_argument("--max-samples", type=int, default=1000)
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
    tasks = make_task_views(bundle, order, int(data_cfg["num_tasks"]), train=False, train_augment=False)
    loader = make_loader(
        tasks[0], batch_size=256, shuffle=False,
        num_workers=int(cfg.get("evaluation", {}).get("num_workers", 4)), seed=seed
    )

    if args.checkpoint_dir is not None:
        checkpoint_dir = Path(args.checkpoint_dir)
    elif (run_dir / "trajectory").exists():
        checkpoint_dir = run_dir / "trajectory"
    else:
        checkpoint_dir = run_dir / "victim_trajectory"
    checkpoints = sorted(checkpoint_dir.glob("task_*.pt"))
    rows = analyze_subspace_trajectory(
        checkpoint_paths=checkpoints,
        loader=loader,
        layers=args.layers,
        device=device,
        rank=args.rank,
        explained_variance=args.explained_variance,
        max_samples=args.max_samples,
    )
    output = Path(args.output) if args.output else run_dir / "subspace_metrics.csv"
    ensure_dir(output.parent)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
