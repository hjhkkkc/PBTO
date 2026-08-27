from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import ConcatDataset, Dataset
from tqdm.auto import tqdm

from .memory import ExemplarMemory
from .models import ExpandableResNet18, freeze_model
from .utils import append_csv, cpu_state_dict, ensure_dir, make_loader


class ICaRLTrainer:
    """Compact iCaRL implementation for class-incremental experiments.

    Training follows the original one-vs-rest BCE formulation with distillation
    targets supplied by the previous model. Exemplar means can be used for NME
    inference through :mod:`pbto_repro.metrics`.
    """

    def __init__(
        self,
        model: ExpandableResNet18,
        memory: ExemplarMemory,
        device: torch.device,
        train_config: Mapping[str, Any],
        seed: int,
        seen_classes: int = 0,
    ) -> None:
        self.model = model.to(device)
        self.memory = memory
        self.device = device
        self.cfg = dict(train_config)
        self.seed = int(seed)
        self.seen_classes = int(seen_classes)

    @property
    def batch_size(self) -> int:
        return int(self.cfg.get("batch_size", 128))

    @property
    def num_workers(self) -> int:
        return int(self.cfg.get("num_workers", 4))

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, epochs: int):
        scheduler_name = str(self.cfg.get("scheduler", "multistep")).lower()
        if scheduler_name == "cosine":
            return CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        if scheduler_name == "none":
            return None
        milestones = [int(x) for x in self.cfg.get("milestones", [49, 63])]
        gamma = float(self.cfg.get("gamma", 0.1))
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    def fit_task(
        self,
        task_dataset: Dataset[Tuple[Tensor, int]],
        new_class_ids: Sequence[int],
        class_views: Mapping[int, Dataset[Tuple[Tensor, int]]],
        task_id: int,
        checkpoint_dir: str | Path,
        checkpoint_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        new_class_ids = [int(x) for x in new_class_ids]
        expected = list(range(self.seen_classes, self.seen_classes + len(new_class_ids)))
        if new_class_ids != expected:
            raise ValueError(
                f"New class ids must be the next contiguous range {expected}; got {new_class_ids}."
            )

        old_model: Optional[ExpandableResNet18] = None
        old_classes = self.seen_classes
        if old_classes > 0:
            old_model = copy.deepcopy(self.model).to(self.device)
            freeze_model(old_model)

        new_total = old_classes + len(new_class_ids)
        self.model.expand_classifier(new_total)
        self.model.to(self.device)

        datasets: List[Dataset] = [task_dataset]
        if len(self.memory) > 0:
            datasets.append(self.memory.as_dataset(train_augment=True))
        train_dataset: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        loader = make_loader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            seed=self.seed + task_id,
            drop_last=False,
        )

        epochs = int(self.cfg.get("epochs", 70))
        optimizer = SGD(
            self.model.parameters(),
            lr=float(self.cfg.get("lr", 0.01)),
            momentum=float(self.cfg.get("momentum", 0.9)),
            weight_decay=float(self.cfg.get("weight_decay", 5e-4)),
            nesterov=bool(self.cfg.get("nesterov", True)),
        )
        scheduler = self._build_scheduler(optimizer, epochs)
        criterion = nn.BCEWithLogitsLoss()

        checkpoint_dir = ensure_dir(checkpoint_dir)
        log_path = checkpoint_dir / "train_log.csv"
        start = time.perf_counter()
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            progress = tqdm(loader, desc=f"task {task_id + 1} epoch {epoch + 1}/{epochs}", leave=False)
            for images, labels in progress:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                targets = torch.zeros_like(logits)
                targets.scatter_(1, labels.view(-1, 1), 1.0)
                if old_model is not None:
                    with torch.no_grad():
                        old_probabilities = torch.sigmoid(old_model(images))
                    targets[:, :old_classes] = old_probabilities
                loss = criterion(logits, targets)
                loss.backward()
                max_grad_norm = self.cfg.get("max_grad_norm")
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(max_grad_norm))
                optimizer.step()
                running_loss += float(loss.item()) * labels.shape[0]
                predictions = logits[:, :new_total].argmax(dim=1)
                correct += int((predictions == labels).sum().item())
                total += int(labels.numel())
                progress.set_postfix(loss=f"{loss.item():.4f}")
            if scheduler is not None:
                scheduler.step()
            append_csv(
                {
                    "task": task_id + 1,
                    "epoch": epoch + 1,
                    "loss": running_loss / max(total, 1),
                    "train_accuracy": correct / max(total, 1),
                    "lr": optimizer.param_groups[0]["lr"],
                },
                log_path,
            )

        self.seen_classes = new_total
        self._update_memory(new_class_ids, class_views, task_id)

        metadata = dict(checkpoint_metadata or {})
        metadata.update(
            {
                "task_id": int(task_id),
                "seen_classes": int(self.seen_classes),
                "old_classes": int(old_classes),
                "new_class_ids": new_class_ids,
                "training_seconds": time.perf_counter() - start,
            }
        )
        checkpoint_path = checkpoint_dir / f"task_{task_id + 1:02d}.pt"
        torch.save(
            {
                "model_spec": self.model.export_spec(),
                "model_state": cpu_state_dict(self.model),
                "memory_state": self.memory.state_dict(),
                "seen_classes": int(self.seen_classes),
                "train_config": self.cfg,
                "seed": int(self.seed),
                "metadata": metadata,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def _update_memory(
        self,
        new_class_ids: Sequence[int],
        class_views: Mapping[int, Dataset[Tuple[Tensor, int]]],
        task_id: int,
    ) -> None:
        quota = self.memory.per_class_quota(self.seen_classes)
        self.memory.reduce(quota)
        if quota <= 0:
            return
        for class_id in new_class_ids:
            if class_id not in class_views:
                raise KeyError(f"Missing clean class view for class {class_id}.")
            images = self.memory.construct_class_exemplars(
                model=self.model,
                dataset=class_views[class_id],
                class_id=class_id,
                quota=quota,
                batch_size=int(self.cfg.get("herding_batch_size", self.batch_size)),
                num_workers=self.num_workers,
                device=self.device,
                seed=self.seed + 10_000 + task_id * 100 + class_id,
                max_candidates=self.cfg.get("herding_max_candidates"),
            )
            self.memory.add_class(class_id, images)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model: ExpandableResNet18,
        device: torch.device,
        train_config: Optional[Mapping[str, Any]] = None,
    ) -> "ICaRLTrainer":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        memory = ExemplarMemory.from_state_dict(checkpoint["memory_state"])
        return cls(
            model=model,
            memory=memory,
            device=device,
            train_config=train_config or checkpoint.get("train_config", {}),
            seed=int(checkpoint.get("seed", 0)),
            seen_classes=int(checkpoint.get("seen_classes", model.num_classes)),
        )
