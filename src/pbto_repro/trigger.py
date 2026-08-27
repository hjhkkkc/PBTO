from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .models import ExpandableResNet18, freeze_model, model_from_checkpoint
from .utils import append_csv, infinite


@dataclass
class TriggerOptimizationResult:
    trigger: Tensor
    history: List[Dict[str, float]]
    final_trajectory_asr: float


class UniversalAdditiveTrigger(nn.Module):
    def __init__(
        self,
        image_size: int,
        epsilon: float,
        init: str = "uniform",
        initial_delta: Optional[Tensor] = None,
        seed: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        shape = (1, 3, int(image_size), int(image_size))
        if initial_delta is not None:
            delta = initial_delta.detach().float().clone()
            if tuple(delta.shape) != shape:
                delta = F.interpolate(delta, size=(image_size, image_size), mode="bilinear", align_corners=False)
        elif init == "zero":
            delta = torch.zeros(shape)
        elif init == "uniform":
            generator = torch.Generator().manual_seed(int(seed))
            delta = torch.empty(shape).uniform_(-self.epsilon, self.epsilon, generator=generator)
        else:
            raise ValueError(f"Unknown trigger initialization: {init}")
        delta = delta.clamp(-self.epsilon, self.epsilon)
        if device is not None:
            delta = delta.to(device)
        self.delta = nn.Parameter(delta)

    def forward(self, images: Tensor) -> Tensor:
        delta = self.delta
        if delta.shape[-2:] != images.shape[-2:]:
            delta = F.interpolate(delta, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return (images + delta).clamp(0.0, 1.0)

    @torch.no_grad()
    def project_(self) -> None:
        self.delta.clamp_(-self.epsilon, self.epsilon)


def gram_matrix(features: Tensor, normalize: bool = True) -> Tensor:
    """Channel-correlation Gram matrix for BCHW or BC features."""
    if features.ndim == 2:
        features = features[:, :, None, None]
    if features.ndim != 4:
        raise ValueError(f"Expected BCHW or BC features, got {tuple(features.shape)}")
    batch, channels, height, width = features.shape
    flattened = features.reshape(batch, channels, height * width)
    gram = torch.bmm(flattened, flattened.transpose(1, 2))
    if normalize:
        gram = gram / float(channels * height * width)
    return gram


def load_trajectory_models(
    checkpoint_paths: Sequence[str | Path],
    device: torch.device,
) -> List[ExpandableResNet18]:
    models: List[ExpandableResNet18] = []
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = model_from_checkpoint(checkpoint, device)
        freeze_model(model)
        models.append(model)
    if not models:
        raise ValueError("At least one trajectory checkpoint is required.")
    return models


@torch.no_grad()
def compute_reference_grams(
    models: Sequence[ExpandableResNet18],
    reference_loader: DataLoader,
    anchor_layers: Sequence[str],
    device: torch.device,
    max_samples: Optional[int] = None,
    mode: str = "mean_gram",
    normalize_gram: bool = True,
) -> List[Dict[str, Tensor]]:
    """Precompute target-class Gram statistics for each trajectory snapshot."""
    if mode not in {"mean_gram", "fixed_sample"}:
        raise ValueError("reference mode must be mean_gram or fixed_sample")
    results: List[Dict[str, Tensor]] = []
    for model in models:
        sums: Dict[str, Tensor] = {}
        count = 0
        for images, _ in reference_loader:
            images = images.to(device, non_blocking=True)
            if max_samples is not None:
                remaining = int(max_samples) - count
                if remaining <= 0:
                    break
                images = images[:remaining]
            _, activations = model.forward_with_features(images, layers=anchor_layers)
            for layer, feature in activations.items():
                grams = gram_matrix(feature, normalize=normalize_gram)
                if mode == "fixed_sample":
                    sums[layer] = grams[:1].detach()
                else:
                    current = grams.sum(dim=0, keepdim=True)
                    sums[layer] = sums.get(layer, torch.zeros_like(current)) + current
            count += int(images.shape[0])
            if mode == "fixed_sample" or (max_samples is not None and count >= int(max_samples)):
                break
        if count == 0:
            raise ValueError("Reference loader is empty.")
        if mode == "mean_gram":
            sums = {layer: value / float(count) for layer, value in sums.items()}
        results.append(sums)
    return results


def _select_model_indices(
    num_models: int,
    checkpoints_per_step: int | str,
    step: int,
    seed: int,
) -> List[int]:
    if checkpoints_per_step == "all":
        return list(range(num_models))
    count = min(int(checkpoints_per_step), num_models)
    generator = torch.Generator().manual_seed(int(seed) + int(step))
    return torch.randperm(num_models, generator=generator)[:count].tolist()


def optimize_universal_trigger(
    models: Sequence[ExpandableResNet18],
    source_loader: DataLoader,
    reference_loader: DataLoader,
    target_label: int,
    image_size: int,
    device: torch.device,
    config: Mapping[str, Any],
    output_log: Optional[str | Path] = None,
    initial_delta: Optional[Tensor] = None,
    seed: int = 0,
) -> TriggerOptimizationResult:
    """Optimize Eq. (4)+(5) with projected sign-gradient descent.

    ``models`` represents the trajectory Ω. The classifier loss is averaged over
    snapshots and the anchoring loss aligns Gram matrices at configured layers.
    """

    if not models:
        raise ValueError("models must not be empty")
    for model in models:
        freeze_model(model)
        if target_label >= model.num_classes:
            raise ValueError(
                f"Target label {target_label} is absent from a trajectory model with "
                f"{model.num_classes} classes. Put the target proxy class in proxy task 1."
            )

    epsilon = float(config.get("epsilon", 8.0 / 255.0))
    steps = int(config.get("steps", 200))
    step_size = float(config.get("step_size", 1.0 / 255.0))
    lambda_align = float(config.get("lambda_align", 1.0))
    anchor_layers = [str(x) for x in config.get("anchor_layers", ["layer3"])]
    normalize_gram = bool(config.get("normalize_gram", True))
    gram_reduction = str(config.get("gram_reduction", "mean"))
    reference_mode = str(config.get("reference_mode", "mean_gram"))
    reference_samples = config.get("reference_samples", 128)
    checkpoints_per_step = config.get("checkpoints_per_step", "all")

    trigger = UniversalAdditiveTrigger(
        image_size=image_size,
        epsilon=epsilon,
        init=str(config.get("init", "uniform")),
        initial_delta=initial_delta,
        seed=seed,
        device=device,
    )
    reference_grams = compute_reference_grams(
        models=models,
        reference_loader=reference_loader,
        anchor_layers=anchor_layers,
        device=device,
        max_samples=None if reference_samples is None else int(reference_samples),
        mode=reference_mode,
        normalize_gram=normalize_gram,
    )

    source_iterator = infinite(source_loader)
    history: List[Dict[str, float]] = []
    progress = tqdm(range(steps), desc="optimizing PBTO trigger")
    for step in progress:
        images, _ = next(source_iterator)
        images = images.to(device, non_blocking=True)
        trigger.zero_grad(set_to_none=True)
        poisoned = trigger(images)
        selected_indices = _select_model_indices(
            len(models), checkpoints_per_step=checkpoints_per_step, step=step, seed=seed
        )
        classifier_loss = torch.zeros((), device=device)
        alignment_loss = torch.zeros((), device=device)
        targets = torch.full((images.shape[0],), int(target_label), device=device, dtype=torch.long)

        for model_index in selected_indices:
            model = models[model_index]
            logits, activations = model.forward_with_features(poisoned, layers=anchor_layers)
            classifier_loss = classifier_loss + F.cross_entropy(logits, targets)
            if lambda_align > 0.0:
                for layer in anchor_layers:
                    source_gram = gram_matrix(activations[layer], normalize=normalize_gram)
                    target_gram = reference_grams[model_index][layer].expand_as(source_gram)
                    squared = (source_gram - target_gram).pow(2)
                    if gram_reduction == "sum":
                        alignment_loss = alignment_loss + squared.flatten(1).sum(dim=1).mean()
                    elif gram_reduction == "mean":
                        alignment_loss = alignment_loss + squared.mean()
                    else:
                        raise ValueError("gram_reduction must be mean or sum")

        classifier_loss = classifier_loss / float(len(selected_indices))
        if lambda_align > 0.0:
            alignment_loss = alignment_loss / float(len(selected_indices) * len(anchor_layers))
        total_loss = classifier_loss + lambda_align * alignment_loss
        total_loss.backward()

        if trigger.delta.grad is None:
            raise RuntimeError("Trigger gradient is missing.")
        with torch.no_grad():
            # PGD for minimization: move against the sign of the gradient.
            trigger.delta.add_(-step_size * trigger.delta.grad.sign())
            trigger.project_()

        row = {
            "step": float(step + 1),
            "loss": float(total_loss.detach().item()),
            "classifier_loss": float(classifier_loss.detach().item()),
            "alignment_loss": float(alignment_loss.detach().item()),
            "linf": float(trigger.delta.detach().abs().max().item()),
        }
        history.append(row)
        if output_log is not None:
            append_csv(row, output_log)
        progress.set_postfix(
            loss=f"{row['loss']:.4f}", ce=f"{row['classifier_loss']:.4f}", gram=f"{row['alignment_loss']:.4f}"
        )

    asr = trajectory_asr(
        models=models,
        loader=source_loader,
        trigger=trigger.delta.detach(),
        target_label=target_label,
        device=device,
    )
    return TriggerOptimizationResult(
        trigger=trigger.delta.detach().cpu(),
        history=history,
        final_trajectory_asr=asr,
    )


@torch.no_grad()
def trajectory_asr(
    models: Sequence[ExpandableResNet18],
    loader: DataLoader,
    trigger: Tensor,
    target_label: int,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    if not models:
        return float("nan")
    success = 0
    total = 0
    trigger = trigger.to(device)
    for batch_index, (images, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        current_trigger = trigger
        if current_trigger.shape[-2:] != images.shape[-2:]:
            current_trigger = F.interpolate(
                current_trigger,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        poisoned = (images + current_trigger).clamp(0.0, 1.0)
        for model in models:
            predictions = model(poisoned).argmax(dim=1)
            success += int((predictions == int(target_label)).sum().item())
            total += int(predictions.numel())
    return float(success / total) if total else float("nan")


def save_trigger(trigger: Tensor, path: str | Path, metadata: Optional[Mapping[str, Any]] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"delta": trigger.detach().cpu(), "metadata": dict(metadata or {})}, path)


def load_trigger(path: str | Path) -> Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Tensor):
        return payload.float()
    if isinstance(payload, Mapping) and "delta" in payload:
        return torch.as_tensor(payload["delta"]).float()
    raise ValueError(f"Invalid trigger file: {path}")
