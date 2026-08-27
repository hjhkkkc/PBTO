from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .models import ExpandableResNet18, model_from_checkpoint


ChannelKey = Tuple[str, int]


def _conv_modules(model: nn.Module) -> Dict[str, nn.Conv2d]:
    return {name: module for name, module in model.named_modules() if isinstance(module, nn.Conv2d)}


def _flatten_channel_scores(scores: Mapping[str, Tensor]) -> Tuple[List[ChannelKey], Tensor]:
    keys: List[ChannelKey] = []
    values: List[Tensor] = []
    for name in sorted(scores):
        score = scores[name].detach().cpu().flatten()
        keys.extend((name, index) for index in range(score.numel()))
        values.append(score)
    return keys, torch.cat(values) if values else torch.empty(0)


def compute_channel_fisher(
    model: ExpandableResNet18,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Diagonal Fisher proxy aggregated per convolution output channel."""
    model.train(False)
    convs = _conv_modules(model)
    fisher = {name: torch.zeros(module.out_channels, device=device) for name, module in convs.items()}
    sample_count = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        log_prob = torch.log_softmax(model(images), dim=1)
        loss = -log_prob.gather(1, labels[:, None]).mean()
        loss.backward()
        for name, module in convs.items():
            if module.weight.grad is None:
                continue
            channel_score = module.weight.grad.detach().pow(2).flatten(1).mean(dim=1)
            fisher[name] += channel_score * images.shape[0]
        sample_count += int(images.shape[0])
    if sample_count == 0:
        raise ValueError("Fisher loader is empty.")
    return {name: value.detach().cpu() / float(sample_count) for name, value in fisher.items()}


def channel_parameter_drift(
    reference: ExpandableResNet18,
    current: ExpandableResNet18,
    relative: bool = True,
) -> Dict[str, Tensor]:
    ref_convs = _conv_modules(reference)
    cur_convs = _conv_modules(current)
    if ref_convs.keys() != cur_convs.keys():
        raise ValueError("Models do not have matching convolutional modules.")
    drift: Dict[str, Tensor] = {}
    for name in ref_convs:
        ref = ref_convs[name].weight.detach().cpu().flatten(1)
        cur = cur_convs[name].weight.detach().cpu().flatten(1)
        numerator = torch.linalg.vector_norm(cur - ref, dim=1)
        if relative:
            numerator = numerator / torch.linalg.vector_norm(ref, dim=1).clamp_min(1e-12)
        drift[name] = numerator
    return drift


def overlap_ratio(
    fisher_scores: Mapping[str, Tensor],
    drift_scores: Mapping[str, Tensor],
    fraction: float,
) -> float:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    fisher_keys, fisher_values = _flatten_channel_scores(fisher_scores)
    drift_keys, drift_values = _flatten_channel_scores(drift_scores)
    if fisher_keys != drift_keys:
        raise ValueError("Fisher and drift channel layouts differ.")
    count = max(1, int(round(fraction * len(fisher_keys))))
    critical_indices = torch.topk(fisher_values, k=count, largest=True).indices.tolist()
    stable_indices = torch.topk(drift_values, k=count, largest=False).indices.tolist()
    critical = set(critical_indices)
    stable = set(stable_indices)
    return len(critical.intersection(stable)) / float(count)


def analyze_neuron_stability(
    checkpoint_paths: Sequence[str | Path],
    task1_loader: DataLoader,
    device: torch.device,
    fractions: Sequence[float] = (0.01, 0.05, 0.10),
    fisher_batches: Optional[int] = None,
) -> List[Dict[str, float | int]]:
    if len(checkpoint_paths) < 2:
        raise ValueError("At least Task-1 and one later checkpoint are required.")
    first_payload = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    reference = model_from_checkpoint(first_payload, device)
    fisher = compute_channel_fisher(reference, task1_loader, device, max_batches=fisher_batches)
    rows: List[Dict[str, float | int]] = []
    for task_index, checkpoint_path in enumerate(checkpoint_paths[1:], start=2):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        current = model_from_checkpoint(payload, device)
        drift = channel_parameter_drift(reference, current, relative=True)
        for fraction in fractions:
            rows.append(
                {
                    "task": task_index,
                    "fraction": float(fraction),
                    "overlap_ratio": overlap_ratio(fisher, drift, fraction),
                }
            )
    return rows
