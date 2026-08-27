from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torchvision.models import resnet18


_DATASET_STATS: Mapping[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "tinyimagenet": ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    "imagefolder": ((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
}


class NormalizeLayer(nn.Module):
    """Normalize raw [0, 1] image tensors inside the network.

    Keeping normalization inside the model makes it unambiguous that a trigger is
    added in pixel space before normalization.
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


@dataclass(frozen=True)
class ModelSpec:
    name: str
    dataset: str
    num_classes: int
    small_input: bool = True


class ExpandableResNet18(nn.Module):
    """ResNet-18 with a dynamically expandable classifier and feature hooks.

    The model accepts raw image tensors in [0, 1]. For CIFAR-sized inputs the
    7x7/stride-2 stem is replaced by a 3x3/stride-1 stem and max-pooling is
    removed, which is standard in CIFAR ResNet reproductions.
    """

    valid_feature_layers = ("stem", "layer1", "layer2", "layer3", "layer4", "penultimate")

    def __init__(
        self,
        num_classes: int,
        dataset: str = "cifar100",
        small_input: bool = True,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        dataset_key = dataset.lower().replace("-", "")
        if mean is None or std is None:
            stats_key = dataset_key if dataset_key in _DATASET_STATS else "imagefolder"
            default_mean, default_std = _DATASET_STATS[stats_key]
            mean = default_mean if mean is None else mean
            std = default_std if std is None else std

        base = resnet18(weights=None)
        if small_input:
            base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity()

        self.normalize = NormalizeLayer(mean, std)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.feature_dim = int(base.fc.in_features)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.dataset = dataset
        self.small_input = small_input

    @property
    def num_classes(self) -> int:
        return int(self.classifier.out_features)

    def expand_classifier(self, new_num_classes: int) -> None:
        """Expand the classifier while preserving existing class weights."""
        if new_num_classes < self.num_classes:
            raise ValueError(
                f"Cannot shrink classifier from {self.num_classes} to {new_num_classes}."
            )
        if new_num_classes == self.num_classes:
            return
        old = self.classifier
        new = nn.Linear(old.in_features, new_num_classes, device=old.weight.device, dtype=old.weight.dtype)
        nn.init.kaiming_normal_(new.weight, nonlinearity="linear")
        nn.init.zeros_(new.bias)
        with torch.no_grad():
            new.weight[: old.out_features].copy_(old.weight)
            new.bias[: old.out_features].copy_(old.bias)
        self.classifier = new

    def forward_activations(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.normalize(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        stem = x
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        pooled = torch.flatten(self.avgpool(x4), 1)
        return {
            "stem": stem,
            "layer1": x1,
            "layer2": x2,
            "layer3": x3,
            "layer4": x4,
            "penultimate": pooled,
        }

    def extract_features(self, x: Tensor, normalize: bool = False) -> Tensor:
        features = self.forward_activations(x)["penultimate"]
        if normalize:
            features = torch.nn.functional.normalize(features, dim=1)
        return features

    def forward_with_features(
        self, x: Tensor, layers: Sequence[str] = ("layer3",)
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        invalid = set(layers).difference(self.valid_feature_layers)
        if invalid:
            raise ValueError(f"Unknown feature layer(s): {sorted(invalid)}")
        activations = self.forward_activations(x)
        logits = self.classifier(activations["penultimate"])
        return logits, {name: activations[name] for name in layers}

    def forward(self, x: Tensor) -> Tensor:
        features = self.extract_features(x)
        return self.classifier(features)

    def export_spec(self) -> Dict[str, object]:
        return {
            "name": "resnet18",
            "dataset": self.dataset,
            "num_classes": self.num_classes,
            "small_input": self.small_input,
        }


def build_model(
    name: str,
    num_classes: int,
    dataset: str,
    small_input: Optional[bool] = None,
) -> ExpandableResNet18:
    normalized = name.lower().replace("_", "").replace("-", "")
    if normalized != "resnet18":
        raise ValueError(
            f"This reference implementation currently supports ResNet-18 only; got {name!r}. "
            "Cross-architecture experiments should use the adapter interface described in README.md."
        )
    if small_input is None:
        small_input = dataset.lower() in {"cifar10", "cifar100", "tinyimagenet", "tiny-imagenet"}
    return ExpandableResNet18(
        num_classes=num_classes,
        dataset=dataset,
        small_input=small_input,
    )


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def model_from_checkpoint(checkpoint: Mapping[str, object], device: torch.device) -> ExpandableResNet18:
    spec = checkpoint.get("model_spec")
    if not isinstance(spec, Mapping):
        raise KeyError("Checkpoint does not contain a model_spec mapping.")
    model = build_model(
        name=str(spec.get("name", "resnet18")),
        num_classes=int(spec["num_classes"]),
        dataset=str(spec.get("dataset", "cifar100")),
        small_input=bool(spec.get("small_input", True)),
    )
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise KeyError("Checkpoint does not contain model_state.")
    model.load_state_dict(state)  # type: ignore[arg-type]
    return model.to(device)
