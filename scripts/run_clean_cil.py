#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pbto_repro.data import build_class_order, build_dataset_bundle, make_class_views, make_task_views
from pbto_repro.memory import ExemplarMemory
from pbto_repro.metrics import evaluate_benign_accuracy
from pbto_repro.models import model_from_checkpoint
from pbto_repro.trajectory import train_icarl_trajectory
from pbto_repro.utils import (
    append_csv,
    configure_torch_threads,
    ensure_dir,
    load_yaml,
    make_loader,
    merge_dicts,
    parse_overrides,
    resolve_device,
    save_json,
    save_yaml,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = merge_dicts(load_yaml(args.config), parse_overrides(args.overrides))
    exp = cfg.get("experiment", {})
    seed = int(exp.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(str(exp.get("device", "auto")))
    configure_torch_threads(int(exp.get("cpu_threads", 4)))
    output = ensure_dir(exp.get("output_dir", "outputs/clean_cil"))
    save_yaml(cfg, output / "resolved_config.yaml")

    data_cfg = cfg["dataset"]
    bundle = build_dataset_bundle(
        data_cfg["name"], data_cfg.get("root", "./data"),
        download=bool(data_cfg.get("download", True)), image_size=data_cfg.get("image_size")
    )
    explicit = data_cfg.get("class_order")
    order = build_class_order(len(bundle.class_names), seed, explicit)
    num_tasks = int(data_cfg["num_tasks"])
    classes_per_task = len(order) // num_tasks
    train_tasks = make_task_views(bundle, order, num_tasks, train=True, train_augment=True)
    test_tasks = make_task_views(bundle, order, num_tasks, train=False, train_augment=False)
    class_views = make_class_views(bundle, order, train=True, train_augment=False)
    save_json({"ids": order, "names": [bundle.class_names[i] for i in order]}, output / "class_order.json")

    victim_cfg = cfg["victim"]
    checkpoints, _ = train_icarl_trajectory(
        task_datasets=train_tasks,
        class_views=class_views,
        dataset_name=bundle.name,
        image_size=bundle.image_size,
        classes_per_task=classes_per_task,
        memory_size=int(victim_cfg.get("memory_size", 2000)),
        train_config=victim_cfg.get("train", {}),
        device=device,
        seed=seed,
        output_dir=output / "trajectory",
        model_name=victim_cfg.get("model", "resnet18"),
        poison_trigger=None,
    )

    eval_cfg = cfg.get("evaluation", {})
    inference = str(eval_cfg.get("inference", "nme"))
    for task_id, checkpoint_path in enumerate(checkpoints):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = model_from_checkpoint(payload, device)
        combined = ConcatDataset(test_tasks[: task_id + 1])
        loader = make_loader(
            combined,
            batch_size=int(eval_cfg.get("batch_size", 256)),
            shuffle=False,
            num_workers=int(eval_cfg.get("num_workers", 4)),
            seed=seed + task_id,
        )
        class_means = None
        if inference == "nme":
            memory = ExemplarMemory.from_state_dict(payload["memory_state"])
            class_means = memory.compute_class_means(
                model, device,
                batch_size=int(eval_cfg.get("batch_size", 256)),
                num_workers=int(eval_cfg.get("num_workers", 4)),
                seed=seed + 1000 + task_id,
            )
        ba = evaluate_benign_accuracy(
            model, loader, device, int(payload["seen_classes"]), inference, class_means
        )
        append_csv({"task": task_id + 1, "benign_accuracy": ba}, output / "clean_metrics.csv")
        print(f"Task {task_id + 1}: BA={100 * ba:.2f}%")


if __name__ == "__main__":
    main()
