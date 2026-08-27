from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .models import model_from_checkpoint


def center_features(features: Tensor) -> Tensor:
    if features.ndim != 2:
        raise ValueError("Features must be a two-dimensional N x D matrix.")
    return features - features.mean(dim=0, keepdim=True)


def linear_cka(x: Tensor, y: Tensor, eps: float = 1e-12) -> float:
    """Linear centered kernel alignment without materializing N x N kernels."""
    x = center_features(x.double())
    y = center_features(y.double())
    cross = torch.linalg.norm(x.T @ y, ord="fro").pow(2)
    x_norm = torch.linalg.norm(x.T @ x, ord="fro")
    y_norm = torch.linalg.norm(y.T @ y, ord="fro")
    return float((cross / (x_norm * y_norm + eps)).item())


def principal_basis(
    features: Tensor,
    rank: Optional[int] = None,
    explained_variance: float = 0.90,
) -> Tuple[Tensor, int]:
    """Return a feature-space principal basis V_r for an N x D matrix."""
    centered = center_features(features.double())
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    max_rank = int(vh.shape[0])
    if rank is None:
        energy = singular_values.pow(2)
        cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp_min(1e-12)
        rank = int(torch.searchsorted(cumulative, torch.tensor(explained_variance, dtype=cumulative.dtype)).item()) + 1
    rank = max(1, min(int(rank), max_rank))
    return vh[:rank].T.contiguous(), rank


def grassmann_distance(
    basis_a: Tensor,
    basis_b: Tensor,
    normalize: bool = True,
) -> float:
    """Geodesic Grassmann distance based on principal angles.

    When ``normalize`` is true, the root-mean-square angle is divided by π/2,
    producing a convenient [0, 1] scale. The paper does not specify its exact
    normalization, so the report records this definition explicitly.
    """
    rank = min(basis_a.shape[1], basis_b.shape[1])
    a = torch.linalg.qr(basis_a[:, :rank].double(), mode="reduced").Q
    b = torch.linalg.qr(basis_b[:, :rank].double(), mode="reduced").Q
    singular_values = torch.linalg.svdvals(a.T @ b).clamp(-1.0, 1.0)
    angles = torch.arccos(singular_values)
    distance = torch.sqrt(torch.mean(angles.pow(2)))
    if normalize:
        distance = distance / (torch.pi / 2.0)
    return float(distance.item())


def variance_retention(features: Tensor, reference_basis: Tensor) -> float:
    centered = center_features(features.double())
    basis = reference_basis.double()
    projected = centered @ basis @ basis.T
    return float((projected.pow(2).sum() / centered.pow(2).sum().clamp_min(1e-12)).item())


@torch.no_grad()
def extract_feature_matrices(
    checkpoint_path: str | Path,
    loader: DataLoader,
    layers: Sequence[str],
    device: torch.device,
    max_samples: Optional[int] = 1000,
) -> Dict[str, Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint, device=device)
    model.eval()
    collected: Dict[str, List[Tensor]] = {layer: [] for layer in layers}
    count = 0
    for images, _ in loader:
        if max_samples is not None:
            remaining = int(max_samples) - count
            if remaining <= 0:
                break
            images = images[:remaining]
        images = images.to(device, non_blocking=True)
        _, activations = model.forward_with_features(images, layers=layers)
        for layer, activation in activations.items():
            if activation.ndim == 4:
                activation = activation.mean(dim=(2, 3))
            collected[layer].append(activation.detach().cpu())
        count += int(images.shape[0])
    if count == 0:
        raise ValueError("Feature loader is empty.")
    return {layer: torch.cat(chunks, dim=0) for layer, chunks in collected.items()}


def analyze_subspace_trajectory(
    checkpoint_paths: Sequence[str | Path],
    loader: DataLoader,
    layers: Sequence[str],
    device: torch.device,
    rank: Optional[int] = None,
    explained_variance: float = 0.90,
    max_samples: Optional[int] = 1000,
) -> List[Dict[str, float | int | str]]:
    if not checkpoint_paths:
        raise ValueError("No checkpoints were provided.")
    all_features = [
        extract_feature_matrices(path, loader, layers, device, max_samples=max_samples)
        for path in checkpoint_paths
    ]
    reference = all_features[0]
    reference_bases: Dict[str, Tensor] = {}
    ranks: Dict[str, int] = {}
    for layer in layers:
        basis, selected_rank = principal_basis(reference[layer], rank, explained_variance)
        reference_bases[layer] = basis
        ranks[layer] = selected_rank

    rows: List[Dict[str, float | int | str]] = []
    for task_index, current in enumerate(all_features, start=1):
        for layer in layers:
            current_basis, _ = principal_basis(current[layer], ranks[layer], explained_variance)
            rows.append(
                {
                    "task": task_index,
                    "layer": layer,
                    "rank": ranks[layer],
                    "cka": linear_cka(reference[layer], current[layer]),
                    "grassmann": grassmann_distance(reference_bases[layer], current_basis),
                    "variance_retention": variance_retention(current[layer], reference_bases[layer]),
                }
            )
    return rows
