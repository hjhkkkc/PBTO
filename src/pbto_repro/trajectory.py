from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data import PoisonedDataset, SingleLabelDataset
from .memory import ExemplarMemory
from .models import ExpandableResNet18, build_model
from .trainer import ICaRLTrainer
from .utils import ensure_dir, save_json


def train_icarl_trajectory(
    task_datasets: Sequence[Dataset[Tuple[Tensor, int]]],
    class_views: Mapping[int, Dataset[Tuple[Tensor, int]]],
    dataset_name: str,
    image_size: int,
    classes_per_task: int,
    memory_size: int,
    train_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    output_dir: str | Path,
    model_name: str = "resnet18",
    poison_trigger: Optional[Tensor] = None,
    poison_task: int = 0,
    poison_rate: float = 0.05,
    target_label: int = 0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Path], ICaRLTrainer]:
    """Train a class-incremental trajectory and save every task checkpoint."""

    if not task_datasets:
        raise ValueError("task_datasets must not be empty")
    output_dir = ensure_dir(output_dir)
    model = build_model(
        name=model_name,
        num_classes=classes_per_task,
        dataset=dataset_name,
        small_input=image_size <= 64,
    )
    memory = ExemplarMemory(budget=int(memory_size), image_size=int(image_size))
    trainer = ICaRLTrainer(
        model=model,
        memory=memory,
        device=device,
        train_config=train_config,
        seed=seed,
        seen_classes=0,
    )

    checkpoints: List[Path] = []
    for task_id, clean_dataset in enumerate(task_datasets):
        train_dataset = clean_dataset
        poisoned = False
        poison_count = 0
        if poison_trigger is not None and task_id == int(poison_task):
            train_dataset = PoisonedDataset(
                dataset=clean_dataset,
                trigger=poison_trigger,
                target_label=target_label,
                poison_rate=poison_rate,
                seed=seed + 31_337,
                exclude_target_samples=True,
            )
            poisoned = True
            poison_count = len(train_dataset.poison_indices)  # type: ignore[attr-defined]
        new_classes = list(
            range(task_id * classes_per_task, (task_id + 1) * classes_per_task)
        )

        # For the poisoned task, exemplar herding must see the same replacement-
        # poisoned and relabeled samples that were used for task training. This
        # lets target-class memory contain selected trigger-stamped samples and
        # removes those samples from their original source-class candidate sets.
        memory_class_views: Mapping[int, Dataset[Tuple[Tensor, int]]] = class_views
        if poisoned:
            memory_class_views = dict(class_views)
            for class_id in new_classes:
                memory_class_views[class_id] = SingleLabelDataset(
                    train_dataset,
                    label=class_id,
                    use_raw=True,
                )

        checkpoint = trainer.fit_task(
            task_dataset=train_dataset,
            new_class_ids=new_classes,
            class_views=memory_class_views,
            task_id=task_id,
            checkpoint_dir=output_dir,
            checkpoint_metadata={
                **dict(metadata or {}),
                "poisoned": poisoned,
                "poison_count": poison_count,
                "poison_rate": float(poison_rate) if poisoned else 0.0,
                "target_label": int(target_label),
            },
        )
        checkpoints.append(checkpoint)

    save_json(
        {
            "checkpoints": [str(path) for path in checkpoints],
            "num_tasks": len(task_datasets),
            "classes_per_task": int(classes_per_task),
            "memory_size": int(memory_size),
            "poison_task": int(poison_task),
            "poison_rate": float(poison_rate),
            "target_label": int(target_label),
            "poisoned": poison_trigger is not None,
        },
        output_dir / "trajectory.json",
    )
    return checkpoints, trainer
