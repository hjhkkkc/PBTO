from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


def predict_logits(model: nn.Module, images: Tensor, seen_classes: int) -> Tensor:
    logits = model(images)
    if seen_classes <= 0 or seen_classes > logits.shape[1]:
        raise ValueError("seen_classes is inconsistent with classifier output size.")
    return logits[:, :seen_classes].argmax(dim=1)


def predict_nme(
    model: nn.Module,
    images: Tensor,
    class_means: Mapping[int, Tensor],
    seen_classes: int,
) -> Tensor:
    features = model.extract_features(images, normalize=True)  # type: ignore[attr-defined]
    available = [class_id for class_id in range(seen_classes) if class_id in class_means]
    if len(available) != seen_classes:
        missing = sorted(set(range(seen_classes)).difference(available))
        raise ValueError(f"Missing NME means for classes: {missing}")
    means = torch.stack([class_means[class_id].to(images.device) for class_id in available], dim=0)
    distances = torch.cdist(features, means, p=2)
    nearest = distances.argmin(dim=1)
    return torch.tensor(available, device=images.device)[nearest]


@torch.no_grad()
def evaluate_benign_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    seen_classes: int,
    inference: str = "logits",
    class_means: Optional[Mapping[int, Tensor]] = None,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if inference == "nme":
            if class_means is None:
                raise ValueError("NME evaluation requires class_means.")
            predictions = predict_nme(model, images, class_means, seen_classes)
        else:
            predictions = predict_logits(model, images, seen_classes)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
    return float(correct / total) if total else float("nan")


@torch.no_grad()
def evaluate_asr(
    model: nn.Module,
    loader: DataLoader,
    trigger: Tensor,
    target_label: int,
    device: torch.device,
    seen_classes: int,
    inference: str = "logits",
    class_means: Optional[Mapping[int, Tensor]] = None,
    exclude_true_target: bool = True,
) -> float:
    """Attack success rate on triggered samples.

    By default, examples whose true label already equals the target are excluded,
    preventing target-class prevalence from inflating ASR.
    """

    model.eval()
    success = 0
    total = 0
    trigger = trigger.to(device)
    for images, labels in loader:
        if exclude_true_target:
            mask = labels != int(target_label)
            if not bool(mask.any()):
                continue
            images = images[mask]
            labels = labels[mask]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        current_trigger = trigger
        if current_trigger.shape[-2:] != images.shape[-2:]:
            current_trigger = torch.nn.functional.interpolate(
                current_trigger,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        poisoned = (images + current_trigger).clamp(0.0, 1.0)
        if inference == "nme":
            if class_means is None:
                raise ValueError("NME evaluation requires class_means.")
            predictions = predict_nme(model, poisoned, class_means, seen_classes)
        else:
            predictions = predict_logits(model, poisoned, seen_classes)
        success += int((predictions == int(target_label)).sum().item())
        total += int(labels.numel())
    return float(success / total) if total else float("nan")


@torch.no_grad()
def evaluate_target_clean_accuracy(
    model: nn.Module,
    loader: DataLoader,
    target_label: int,
    device: torch.device,
    seen_classes: int,
    inference: str = "logits",
    class_means: Optional[Mapping[int, Tensor]] = None,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        mask = labels == int(target_label)
        if not bool(mask.any()):
            continue
        images = images[mask].to(device, non_blocking=True)
        labels = labels[mask].to(device, non_blocking=True)
        if inference == "nme":
            if class_means is None:
                raise ValueError("NME evaluation requires class_means.")
            predictions = predict_nme(model, images, class_means, seen_classes)
        else:
            predictions = predict_logits(model, images, seen_classes)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
    return float(correct / total) if total else float("nan")
