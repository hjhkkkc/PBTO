#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import ConcatDataset

# Allow running directly from a source checkout without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pbto_repro.data import (  # noqa: E402
    ExcludeLabelDataset,
    build_class_order,
    build_dataset_bundle,
    limit_per_class,
    make_class_views,
    make_task_views,
)
from pbto_repro.memory import ExemplarMemory  # noqa: E402
from pbto_repro.metrics import (  # noqa: E402
    evaluate_asr,
    evaluate_benign_accuracy,
    evaluate_target_clean_accuracy,
)
from pbto_repro.models import model_from_checkpoint  # noqa: E402
from pbto_repro.pbto import run_iterative_refinement  # noqa: E402
from pbto_repro.trajectory import train_icarl_trajectory  # noqa: E402
from pbto_repro.utils import (  # noqa: E402
    append_csv,
    configure_torch_threads,
    describe_environment,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PBTO reproduction on controlled benchmark datasets.")
    parser.add_argument("--config", required=True, help="YAML experiment configuration.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a setting, e.g. --set refinement.max_rounds=1",
    )
    return parser.parse_args()


def _resolve_class_id(value: Any, class_names: Sequence[str]) -> int:
    if isinstance(value, str) and not value.isdigit():
        if value not in class_names:
            raise ValueError(f"Unknown class name {value!r}. Available examples: {class_names[:10]}")
        return class_names.index(value)
    return int(value)


def victim_order(cfg: Mapping[str, Any], class_names: Sequence[str], seed: int) -> List[int]:
    explicit = cfg.get("class_order")
    explicit_ids = None
    if explicit is not None:
        explicit_ids = [_resolve_class_id(item, class_names) for item in explicit]
    order = build_class_order(len(class_names), seed=seed, explicit=explicit_ids)
    target = cfg.get("target_original_class")
    if target is not None:
        target_id = _resolve_class_id(target, class_names)
        target_position = order.index(target_id)
        order[0], order[target_position] = order[target_position], order[0]
    return order


def proxy_order(cfg: Mapping[str, Any], class_names: Sequence[str], seed: int) -> List[int]:
    total = int(cfg["num_tasks"]) * int(cfg["classes_per_task"])
    explicit = cfg.get("class_order")
    if explicit is not None:
        order = [_resolve_class_id(item, class_names) for item in explicit]
        if len(order) != total or len(set(order)) != total:
            raise ValueError(f"proxy.class_order must contain {total} unique classes.")
    else:
        generator = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(len(class_names), generator=generator)[:total].tolist()
    target = cfg.get("target_class")
    if target is not None:
        target_id = _resolve_class_id(target, class_names)
        if target_id not in order:
            order[-1] = target_id
        position = order.index(target_id)
        order[0], order[position] = order[position], order[0]
    return order


def evaluate_victim_checkpoints(
    checkpoint_paths: Sequence[Path],
    test_tasks,
    trigger: torch.Tensor,
    target_label: int,
    device: torch.device,
    cfg: Mapping[str, Any],
    output_dir: Path,
    seed: int,
) -> None:
    eval_cfg = cfg.get("evaluation", {})
    inference = str(eval_cfg.get("inference", "nme"))
    batch_size = int(eval_cfg.get("batch_size", 256))
    num_workers = int(eval_cfg.get("num_workers", 4))
    metrics_path = output_dir / "victim_taskwise_metrics.csv"

    for task_id, checkpoint_path in enumerate(checkpoint_paths):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = model_from_checkpoint(payload, device)
        seen_classes = int(payload["seen_classes"])
        combined = ConcatDataset(list(test_tasks[: task_id + 1]))
        loader = make_loader(
            combined,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=seed + task_id,
        )
        class_means = None
        if inference == "nme":
            memory = ExemplarMemory.from_state_dict(payload["memory_state"])
            class_means = memory.compute_class_means(
                model=model,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                seed=seed + 20_000 + task_id,
            )
        ba = evaluate_benign_accuracy(
            model,
            loader,
            device=device,
            seen_classes=seen_classes,
            inference=inference,
            class_means=class_means,
        )
        asr = evaluate_asr(
            model,
            loader,
            trigger=trigger,
            target_label=target_label,
            device=device,
            seen_classes=seen_classes,
            inference=inference,
            class_means=class_means,
            exclude_true_target=True,
        )
        target_clean = evaluate_target_clean_accuracy(
            model,
            loader,
            target_label=target_label,
            device=device,
            seen_classes=seen_classes,
            inference=inference,
            class_means=class_means,
        )
        append_csv(
            {
                "task": task_id + 1,
                "seen_classes": seen_classes,
                "benign_accuracy": ba,
                "asr": asr,
                "target_clean_accuracy": target_clean,
                "inference": inference,
                "checkpoint": str(checkpoint_path),
            },
            metrics_path,
        )
        print(
            f"[victim task {task_id + 1}] BA={100 * ba:.2f}% "
            f"ASR={100 * asr:.2f}% target-clean={100 * target_clean:.2f}%"
        )


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.config)
    cfg = merge_dicts(base_cfg, parse_overrides(args.overrides))
    experiment_cfg = cfg.get("experiment", {})
    seed = int(experiment_cfg.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(str(experiment_cfg.get("device", "auto")))
    configure_torch_threads(int(experiment_cfg.get("cpu_threads", 4)))
    output_dir = ensure_dir(experiment_cfg.get("output_dir", "outputs/pbto"))
    save_yaml(cfg, output_dir / "resolved_config.yaml")
    save_json(describe_environment(), output_dir / "environment.json")

    dataset_cfg = cfg["dataset"]
    victim_bundle = build_dataset_bundle(
        name=str(dataset_cfg["name"]),
        root=str(dataset_cfg.get("root", "./data")),
        download=bool(dataset_cfg.get("download", True)),
        image_size=dataset_cfg.get("image_size"),
    )
    order = victim_order(dataset_cfg, victim_bundle.class_names, seed)
    num_victim_tasks = int(dataset_cfg["num_tasks"])
    classes_per_victim_task = len(order) // num_victim_tasks
    victim_train_tasks = make_task_views(
        victim_bundle, order, num_victim_tasks, train=True, train_augment=True
    )
    victim_train_tasks_eval = make_task_views(
        victim_bundle, order, num_victim_tasks, train=True, train_augment=False
    )
    victim_test_tasks = make_task_views(
        victim_bundle, order, num_victim_tasks, train=False, train_augment=False
    )
    victim_class_views = make_class_views(victim_bundle, order, train=True, train_augment=False)
    target_label = 0

    proxy_cfg = cfg["proxy"]
    proxy_bundle = build_dataset_bundle(
        name=str(proxy_cfg.get("source", "imagefolder")),
        root=str(proxy_cfg["root"]),
        download=bool(proxy_cfg.get("download", False)),
        image_size=int(proxy_cfg.get("image_size", victim_bundle.image_size)),
    )
    p_order = proxy_order(proxy_cfg, proxy_bundle.class_names, seed + 1_000)
    num_proxy_tasks = int(proxy_cfg["num_tasks"])
    classes_per_proxy_task = int(proxy_cfg["classes_per_task"])
    proxy_train_tasks = make_task_views(
        proxy_bundle, p_order, num_proxy_tasks, train=True, train_augment=True
    )
    proxy_class_views = make_class_views(proxy_bundle, p_order, train=True, train_augment=False)
    max_per_class = proxy_cfg.get("samples_per_class")
    if max_per_class is not None:
        proxy_train_tasks = [
            limit_per_class(task, int(max_per_class), seed + 2_000 + i)
            for i, task in enumerate(proxy_train_tasks)
        ]
        proxy_class_views = {
            class_id: limit_per_class(view, int(max_per_class), seed + 3_000 + class_id)
            for class_id, view in proxy_class_views.items()
            if class_id < num_proxy_tasks * classes_per_proxy_task
        }

    save_json(
        {
            "victim_class_order_ids": order,
            "victim_class_order_names": [victim_bundle.class_names[i] for i in order],
            "proxy_class_order_ids": p_order,
            "proxy_class_order_names": [proxy_bundle.class_names[i] for i in p_order],
            "victim_target_original_id": order[0],
            "victim_target_name": victim_bundle.class_names[order[0]],
            "proxy_target_original_id": p_order[0],
            "proxy_target_name": proxy_bundle.class_names[p_order[0]],
        },
        output_dir / "class_orders.json",
    )

    source_dataset = ExcludeLabelDataset(victim_train_tasks_eval[0], excluded_label=target_label)
    reference_dataset = proxy_class_views[target_label]
    trigger_batch_size = int(cfg.get("trigger", {}).get("batch_size", 64))
    workers = int(cfg.get("trigger", {}).get("num_workers", 4))
    source_loader = make_loader(
        source_dataset,
        batch_size=trigger_batch_size,
        shuffle=True,
        num_workers=workers,
        seed=seed + 4_000,
        drop_last=False,
    )
    reference_loader = make_loader(
        reference_dataset,
        batch_size=trigger_batch_size,
        shuffle=True,
        num_workers=workers,
        seed=seed + 5_000,
        drop_last=False,
    )

    proxy_train_cfg = proxy_cfg.get("train", cfg.get("victim", {}).get("train", {}))
    clean_proxy_dir = output_dir / "proxy_clean_trajectory"
    clean_checkpoints, _ = train_icarl_trajectory(
        task_datasets=proxy_train_tasks,
        class_views=proxy_class_views,
        dataset_name=victim_bundle.name,
        image_size=victim_bundle.image_size,
        classes_per_task=classes_per_proxy_task,
        memory_size=int(proxy_cfg.get("memory_size", 2_000)),
        train_config=proxy_train_cfg,
        device=device,
        seed=seed + 6_000,
        output_dir=clean_proxy_dir,
        model_name=str(proxy_cfg.get("model", "resnet18")),
        poison_trigger=None,
        poison_task=int(cfg.get("attack", {}).get("poison_task", 0)),
        poison_rate=float(cfg.get("attack", {}).get("poison_rate", 0.05)),
        target_label=target_label,
        metadata={"role": "clean_proxy"},
    )

    def build_poisoned_proxy_trajectory(
        current_trigger: Optional[torch.Tensor], round_dir: Path, round_id: int
    ) -> Sequence[Path]:
        if current_trigger is None:
            raise ValueError("Refinement trajectory requires a trigger.")
        checkpoints, _ = train_icarl_trajectory(
            task_datasets=proxy_train_tasks,
            class_views=proxy_class_views,
            dataset_name=victim_bundle.name,
            image_size=victim_bundle.image_size,
            classes_per_task=classes_per_proxy_task,
            memory_size=int(proxy_cfg.get("memory_size", 2_000)),
            train_config=proxy_train_cfg,
            device=device,
            seed=seed + 6_000,  # fixed initialization across rounds
            output_dir=round_dir,
            model_name=str(proxy_cfg.get("model", "resnet18")),
            poison_trigger=current_trigger,
            poison_task=int(cfg.get("attack", {}).get("poison_task", 0)),
            poison_rate=float(cfg.get("attack", {}).get("poison_rate", 0.05)),
            target_label=target_label,
            metadata={"role": "poisoned_proxy", "refinement_round": round_id},
        )
        return checkpoints

    final_trigger = run_iterative_refinement(
        clean_checkpoint_paths=clean_checkpoints,
        trajectory_builder=build_poisoned_proxy_trajectory,
        source_loader=source_loader,
        reference_loader=reference_loader,
        target_label=target_label,
        image_size=victim_bundle.image_size,
        device=device,
        trigger_config=cfg.get("trigger", {}),
        refinement_config=cfg.get("refinement", {}),
        output_dir=output_dir / "pbto_trigger",
        seed=seed + 7_000,
    )

    victim_cfg = cfg["victim"]
    victim_checkpoints, _ = train_icarl_trajectory(
        task_datasets=victim_train_tasks,
        class_views=victim_class_views,
        dataset_name=victim_bundle.name,
        image_size=victim_bundle.image_size,
        classes_per_task=classes_per_victim_task,
        memory_size=int(victim_cfg.get("memory_size", 2_000)),
        train_config=victim_cfg.get("train", {}),
        device=device,
        seed=seed,
        output_dir=output_dir / "victim_trajectory",
        model_name=str(victim_cfg.get("model", "resnet18")),
        poison_trigger=final_trigger,
        poison_task=int(cfg.get("attack", {}).get("poison_task", 0)),
        poison_rate=float(cfg.get("attack", {}).get("poison_rate", 0.05)),
        target_label=target_label,
        metadata={"role": "victim"},
    )
    evaluate_victim_checkpoints(
        checkpoint_paths=victim_checkpoints,
        test_tasks=victim_test_tasks,
        trigger=final_trigger,
        target_label=target_label,
        device=device,
        cfg=cfg,
        output_dir=output_dir,
        seed=seed,
    )


if __name__ == "__main__":
    main()
