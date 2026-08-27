from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pbto_repro.models import ExpandableResNet18


def test_forward_feature_layers() -> None:
    model = ExpandableResNet18(num_classes=10, dataset="cifar10", small_input=True)
    images = torch.rand(2, 3, 32, 32)
    logits, features = model.forward_with_features(images, ["layer1", "layer2", "layer3", "layer4"])
    assert logits.shape == (2, 10)
    assert list(features) == ["layer1", "layer2", "layer3", "layer4"]
    assert features["layer1"].shape[-1] == 32
    assert features["layer4"].shape[-1] == 4
