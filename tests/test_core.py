from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pbto_repro.data import PoisonedDataset
from pbto_repro.models import ExpandableResNet18
from pbto_repro.subspace import grassmann_distance, linear_cka, principal_basis, variance_retention
from pbto_repro.trigger import UniversalAdditiveTrigger, gram_matrix


def test_classifier_expansion_preserves_old_weights() -> None:
    model = ExpandableResNet18(num_classes=2, dataset="cifar10", small_input=True)
    old_weight = model.classifier.weight.detach().clone()
    old_bias = model.classifier.bias.detach().clone()
    model.expand_classifier(5)
    assert model.num_classes == 5
    assert torch.allclose(model.classifier.weight[:2], old_weight)
    assert torch.allclose(model.classifier.bias[:2], old_bias)


def test_gram_matrix_shape_and_symmetry() -> None:
    features = torch.randn(4, 8, 5, 5)
    gram = gram_matrix(features)
    assert gram.shape == (4, 8, 8)
    assert torch.allclose(gram, gram.transpose(1, 2), atol=1e-6)


def test_trigger_projection() -> None:
    trigger = UniversalAdditiveTrigger(32, epsilon=8 / 255, init="uniform", seed=1)
    with torch.no_grad():
        trigger.delta.fill_(1.0)
    trigger.project_()
    assert float(trigger.delta.detach().abs().max()) <= 8 / 255 + 1e-7


def test_poisoned_dataset_relabels_and_clips() -> None:
    images = torch.zeros(10, 3, 8, 8)
    labels = torch.arange(10) % 2
    dataset = TensorDataset(images, labels)
    trigger = torch.ones(1, 3, 8, 8)
    poisoned = PoisonedDataset(
        dataset, trigger=trigger, target_label=1, poison_rate=0.5, seed=0,
        exclude_target_samples=True
    )
    assert len(poisoned.poison_indices) == 5
    for index in poisoned.poison_indices:
        image, label = poisoned[index]
        assert label == 1
        assert float(image.max()) <= 1.0


def test_subspace_metrics_identical_features() -> None:
    x = torch.randn(64, 16)
    basis, _ = principal_basis(x, rank=8)
    assert linear_cka(x, x) > 0.999
    assert grassmann_distance(basis, basis) < 1e-6
    assert 0.0 <= variance_retention(x, basis) <= 1.0 + 1e-8
