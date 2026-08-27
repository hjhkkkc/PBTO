from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import functional as TF


@dataclass
class DatasetBundle:
    train: Dataset
    test: Dataset
    class_names: List[str]
    name: str
    image_size: int


def _dataset_targets(dataset: Dataset) -> List[int]:
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if isinstance(targets, Tensor):
            return [int(x) for x in targets.tolist()]
        return [int(x) for x in targets]
    if hasattr(dataset, "samples"):
        return [int(label) for _, label in getattr(dataset, "samples")]
    raise TypeError(f"Cannot retrieve targets from dataset type {type(dataset)!r}")


def build_dataset_bundle(
    name: str,
    root: str | Path,
    download: bool = True,
    image_size: Optional[int] = None,
) -> DatasetBundle:
    """Build an untransformed train/test dataset pair.

    Tiny-ImageNet is expected in an ImageFolder-compatible layout:
    ``root/train/<class>/*`` and ``root/val/<class>/*``.
    """

    normalized = name.lower().replace("-", "")
    root = Path(root)
    if normalized == "cifar10":
        train = datasets.CIFAR10(root=str(root), train=True, download=download, transform=None)
        test = datasets.CIFAR10(root=str(root), train=False, download=download, transform=None)
        return DatasetBundle(train, test, list(train.classes), "cifar10", image_size or 32)
    if normalized == "cifar100":
        train = datasets.CIFAR100(root=str(root), train=True, download=download, transform=None)
        test = datasets.CIFAR100(root=str(root), train=False, download=download, transform=None)
        return DatasetBundle(train, test, list(train.classes), "cifar100", image_size or 32)
    if normalized in {"tinyimagenet", "imagefolder"}:
        train_root = root / "train"
        test_root = root / "val"
        if normalized == "imagefolder" and not train_root.exists():
            # For a proxy ImageFolder, a single root is allowed and reused as
            # both train and test. The experiment script will create its own
            # deterministic split.
            train_root = root
            test_root = root
        if not train_root.exists() or not test_root.exists():
            raise FileNotFoundError(
                f"Expected ImageFolder directories {train_root} and {test_root}."
            )
        train = datasets.ImageFolder(str(train_root), transform=None)
        test = datasets.ImageFolder(str(test_root), transform=None)
        if train.class_to_idx != test.class_to_idx:
            raise ValueError("Train and test ImageFolder class mappings differ.")
        inferred_size = image_size or (64 if normalized == "tinyimagenet" else 224)
        return DatasetBundle(train, test, list(train.classes), normalized, inferred_size)
    raise ValueError(f"Unsupported dataset: {name}")


def to_float_tensor(image: Image.Image | Tensor | np.ndarray, image_size: Optional[int] = None) -> Tensor:
    if isinstance(image, Tensor):
        tensor = image.detach().clone()
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div(255.0)
        else:
            tensor = tensor.float()
    else:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        tensor = TF.pil_to_tensor(image).float().div(255.0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW image, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if image_size is not None and tensor.shape[-2:] != (image_size, image_size):
        tensor = TF.resize(tensor, [image_size, image_size], antialias=True)
    return tensor.clamp(0.0, 1.0)


class TensorTrainAugment:
    def __init__(self, image_size: int, padding: int = 4, horizontal_flip: bool = True) -> None:
        self.image_size = int(image_size)
        self.padding = int(padding)
        self.horizontal_flip = bool(horizontal_flip)

    def __call__(self, image: Tensor) -> Tensor:
        if self.padding > 0:
            image = TF.pad(image, [self.padding] * 4, padding_mode="reflect")
        top = random.randint(0, image.shape[-2] - self.image_size)
        left = random.randint(0, image.shape[-1] - self.image_size)
        image = TF.crop(image, top, left, self.image_size, self.image_size)
        if self.horizontal_flip and random.random() < 0.5:
            image = TF.hflip(image)
        return image


class IncrementalView(Dataset[Tuple[Tensor, int]]):
    """A class-filtered view whose labels follow an incremental class order."""

    def __init__(
        self,
        base: Dataset,
        original_classes: Sequence[int],
        class_to_incremental: Mapping[int, int],
        image_size: int,
        train_augment: bool = False,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.base = base
        self.original_classes = [int(x) for x in original_classes]
        self.class_to_incremental = {int(k): int(v) for k, v in class_to_incremental.items()}
        self.image_size = int(image_size)
        self.augment = TensorTrainAugment(image_size) if train_augment else None
        targets = _dataset_targets(base)
        allowed = set(self.original_classes)
        if indices is None:
            self.indices = [idx for idx, target in enumerate(targets) if target in allowed]
        else:
            self.indices = [int(idx) for idx in indices if targets[int(idx)] in allowed]
        self._targets = targets

    def __len__(self) -> int:
        return len(self.indices)

    def raw_item(self, local_index: int) -> Tuple[Tensor, int]:
        base_index = self.indices[local_index]
        image, original_label = self.base[base_index]
        image_tensor = to_float_tensor(image, self.image_size)
        mapped = self.class_to_incremental[int(original_label)]
        return image_tensor, mapped

    def __getitem__(self, local_index: int) -> Tuple[Tensor, int]:
        image, label = self.raw_item(local_index)
        if self.augment is not None:
            image = self.augment(image)
        return image, label

    @property
    def labels(self) -> List[int]:
        return [self.class_to_incremental[self._targets[idx]] for idx in self.indices]


class RemappedImageFolderView(IncrementalView):
    """Alias kept for readability in proxy-dataset code."""


class ExcludeLabelDataset(Dataset[Tuple[Tensor, int]]):
    def __init__(self, dataset: Dataset[Tuple[Tensor, int]], excluded_label: int) -> None:
        self.dataset = dataset
        self.excluded_label = int(excluded_label)
        labels: List[int]
        if hasattr(dataset, "labels"):
            labels = [int(x) for x in getattr(dataset, "labels")]
        else:
            labels = [int(dataset[i][1]) for i in range(len(dataset))]
        self.indices = [i for i, label in enumerate(labels) if label != self.excluded_label]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        return self.dataset[self.indices[index]]


class SingleLabelDataset(Dataset[Tuple[Tensor, int]]):
    def __init__(
        self,
        dataset: Dataset[Tuple[Tensor, int]],
        label: int,
        use_raw: bool = False,
    ) -> None:
        self.dataset = dataset
        self.label = int(label)
        self.use_raw = bool(use_raw)
        if hasattr(dataset, "labels"):
            labels = [int(x) for x in getattr(dataset, "labels")]
        else:
            labels = [int(dataset[i][1]) for i in range(len(dataset))]
        self.indices = [i for i, item_label in enumerate(labels) if item_label == self.label]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        dataset_index = self.indices[index]
        if self.use_raw and hasattr(self.dataset, "raw_item"):
            image, label = getattr(self.dataset, "raw_item")(dataset_index)
            return image, int(label)
        image, label = self.dataset[dataset_index]
        return image, int(label)


class PoisonedDataset(Dataset[Tuple[Tensor, int]]):
    """Replace a fixed subset of samples with trigger-stamped, relabeled samples."""

    def __init__(
        self,
        dataset: Dataset[Tuple[Tensor, int]],
        trigger: Tensor,
        target_label: int,
        poison_rate: float,
        seed: int,
        exclude_target_samples: bool = True,
        poison_indices: Optional[Sequence[int]] = None,
    ) -> None:
        if not 0.0 <= poison_rate <= 1.0:
            raise ValueError("poison_rate must be in [0, 1]")
        self.dataset = dataset
        # Trigger optimization stores a universal perturbation as NCHW with a
        # singleton batch dimension: [1, 3, H, W].  Dataset items, however, are
        # individual CHW images: [3, H, W].  Remove only that singleton batch
        # dimension here; otherwise poisoned samples become [1, 3, H, W] while
        # clean samples remain [3, H, W], and DataLoader cannot stack them.
        normalized_trigger = trigger.detach().cpu().float()
        if normalized_trigger.ndim == 4:
            if normalized_trigger.shape[0] != 1:
                raise ValueError(
                    "PoisonedDataset expects a CHW trigger or a singleton-batch "
                    f"NCHW trigger, got {tuple(normalized_trigger.shape)}"
                )
            normalized_trigger = normalized_trigger.squeeze(0)
        if normalized_trigger.ndim != 3:
            raise ValueError(
                "PoisonedDataset expects a CHW trigger or a singleton-batch "
                f"NCHW trigger, got {tuple(normalized_trigger.shape)}"
            )
        self.trigger = normalized_trigger
        self.target_label = int(target_label)
        base_labels = (
            [int(x) for x in getattr(dataset, "labels")]
            if hasattr(dataset, "labels")
            else [int(dataset[i][1]) for i in range(len(dataset))]
        )
        eligible = [
            i
            for i, label in enumerate(base_labels)
            if not exclude_target_samples or label != self.target_label
        ]
        if poison_indices is None:
            generator = torch.Generator().manual_seed(int(seed))
            count = int(round(poison_rate * len(dataset)))
            count = min(count, len(eligible))
            if count > 0:
                order = torch.randperm(len(eligible), generator=generator)[:count].tolist()
                poison_indices = [eligible[i] for i in order]
            else:
                poison_indices = []
        self.poison_indices = frozenset(int(i) for i in poison_indices)
        self._labels = [
            self.target_label if index in self.poison_indices else int(label)
            for index, label in enumerate(base_labels)
        ]

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def labels(self) -> List[int]:
        """Labels after replacement poisoning and target relabeling."""

        return list(self._labels)

    def _apply_poison(self, image: Tensor, label: int, index: int) -> Tuple[Tensor, int]:
        if index in self.poison_indices:
            trigger = self.trigger
            if trigger.shape[-2:] != image.shape[-2:]:
                trigger = TF.resize(trigger, list(image.shape[-2:]), antialias=True)
            image = (image + trigger).clamp(0.0, 1.0)
            label = self.target_label
        return image, int(label)

    def raw_item(self, index: int) -> Tuple[Tensor, int]:
        """Return an unaugmented item while preserving poisoning semantics."""

        if hasattr(self.dataset, "raw_item"):
            image, label = getattr(self.dataset, "raw_item")(index)
        else:
            image, label = self.dataset[index]
        return self._apply_poison(image, int(label), index)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        image, label = self.dataset[index]
        return self._apply_poison(image, int(label), index)


class ExemplarTensorDataset(Dataset[Tuple[Tensor, int]]):
    def __init__(
        self,
        samples: Sequence[Tuple[Tensor, int]],
        image_size: int,
        train_augment: bool,
    ) -> None:
        self.samples = [(image.detach().cpu(), int(label)) for image, label in samples]
        self.image_size = int(image_size)
        self.augment = TensorTrainAugment(image_size) if train_augment else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        image, label = self.samples[index]
        image = to_float_tensor(image, self.image_size)
        if self.augment is not None:
            image = self.augment(image)
        return image, label


def build_class_order(num_classes: int, seed: int, explicit: Optional[Sequence[int]] = None) -> List[int]:
    if explicit is not None:
        order = [int(x) for x in explicit]
        if sorted(order) != list(range(num_classes)):
            raise ValueError("Explicit class order must be a permutation of all class ids.")
        return order
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(num_classes, generator=generator).tolist()


def make_task_views(
    bundle: DatasetBundle,
    class_order: Sequence[int],
    num_tasks: int,
    train: bool,
    train_augment: bool,
) -> List[IncrementalView]:
    if len(class_order) % num_tasks != 0:
        raise ValueError("Number of classes must be divisible by num_tasks.")
    per_task = len(class_order) // num_tasks
    mapping = {int(original): incremental for incremental, original in enumerate(class_order)}
    base = bundle.train if train else bundle.test
    tasks: List[IncrementalView] = []
    for task_id in range(num_tasks):
        originals = class_order[task_id * per_task : (task_id + 1) * per_task]
        tasks.append(
            IncrementalView(
                base=base,
                original_classes=originals,
                class_to_incremental=mapping,
                image_size=bundle.image_size,
                train_augment=train_augment,
            )
        )
    return tasks


def make_class_views(
    bundle: DatasetBundle,
    class_order: Sequence[int],
    train: bool,
    train_augment: bool = False,
) -> Dict[int, IncrementalView]:
    mapping = {int(original): incremental for incremental, original in enumerate(class_order)}
    base = bundle.train if train else bundle.test
    return {
        incremental: IncrementalView(
            base=base,
            original_classes=[original],
            class_to_incremental=mapping,
            image_size=bundle.image_size,
            train_augment=train_augment,
        )
        for incremental, original in enumerate(class_order)
    }


def limit_per_class(
    view: IncrementalView,
    max_per_class: int,
    seed: int,
) -> IncrementalView:
    """Return a deterministic subsample with at most max_per_class examples per class."""
    if max_per_class <= 0:
        raise ValueError("max_per_class must be positive")
    groups: Dict[int, List[int]] = {}
    for local_idx, label in enumerate(view.labels):
        groups.setdefault(int(label), []).append(local_idx)
    generator = torch.Generator().manual_seed(int(seed))
    selected_base_indices: List[int] = []
    for label in sorted(groups):
        locals_for_label = groups[label]
        permutation = torch.randperm(len(locals_for_label), generator=generator).tolist()
        for local_idx in [locals_for_label[i] for i in permutation[:max_per_class]]:
            selected_base_indices.append(view.indices[local_idx])
    return IncrementalView(
        base=view.base,
        original_classes=view.original_classes,
        class_to_incremental=view.class_to_incremental,
        image_size=view.image_size,
        train_augment=view.augment is not None,
        indices=selected_base_indices,
    )


def concatenate_labels(datasets_: Sequence[Dataset[Tuple[Tensor, int]]]) -> List[int]:
    labels: List[int] = []
    for dataset in datasets_:
        if hasattr(dataset, "labels"):
            labels.extend(int(x) for x in getattr(dataset, "labels"))
        else:
            labels.extend(int(dataset[i][1]) for i in range(len(dataset)))
    return labels
