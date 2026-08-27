from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .data import ExemplarTensorDataset, to_float_tensor
from .utils import make_loader


@dataclass
class ExemplarMemory:
    budget: int
    image_size: int
    exemplars: Dict[int, List[Tensor]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.budget < 0:
            raise ValueError("Memory budget must be non-negative.")

    def __len__(self) -> int:
        return sum(len(items) for items in self.exemplars.values())

    @property
    def classes(self) -> List[int]:
        return sorted(self.exemplars)

    def per_class_quota(self, num_seen_classes: int) -> int:
        if num_seen_classes <= 0 or self.budget == 0:
            return 0
        return self.budget // int(num_seen_classes)

    def reduce(self, quota: int) -> None:
        quota = max(0, int(quota))
        for class_id in list(self.exemplars):
            self.exemplars[class_id] = self.exemplars[class_id][:quota]
            if not self.exemplars[class_id]:
                del self.exemplars[class_id]

    def add_class(self, class_id: int, images: Sequence[Tensor]) -> None:
        self.exemplars[int(class_id)] = [self._compress(image) for image in images]

    @staticmethod
    def _compress(image: Tensor) -> Tensor:
        image = to_float_tensor(image)
        return (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()

    def as_dataset(self, train_augment: bool) -> ExemplarTensorDataset:
        samples: List[Tuple[Tensor, int]] = []
        for class_id in sorted(self.exemplars):
            samples.extend((image, class_id) for image in self.exemplars[class_id])
        return ExemplarTensorDataset(samples, self.image_size, train_augment=train_augment)

    def state_dict(self) -> Dict[str, object]:
        return {
            "budget": int(self.budget),
            "image_size": int(self.image_size),
            "exemplars": {
                int(class_id): [image.cpu() for image in images]
                for class_id, images in self.exemplars.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "ExemplarMemory":
        memory = cls(budget=int(state["budget"]), image_size=int(state["image_size"]))
        raw = state.get("exemplars", {})
        if not isinstance(raw, Mapping):
            raise TypeError("Invalid exemplar memory state.")
        memory.exemplars = {
            int(class_id): [torch.as_tensor(image).to(torch.uint8).cpu() for image in images]
            for class_id, images in raw.items()
        }
        return memory

    @torch.no_grad()
    def construct_class_exemplars(
        self,
        model: nn.Module,
        dataset: Dataset[Tuple[Tensor, int]],
        class_id: int,
        quota: int,
        batch_size: int,
        num_workers: int,
        device: torch.device,
        seed: int,
        max_candidates: Optional[int] = None,
    ) -> List[Tensor]:
        """Select exemplars using iCaRL's feature-mean herding rule."""

        quota = min(int(quota), len(dataset))
        if quota <= 0:
            return []
        loader = make_loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=seed,
        )
        all_images: List[Tensor] = []
        all_features: List[Tensor] = []
        model.eval()
        for images, labels in tqdm(loader, desc=f"herding class {class_id}", leave=False):
            if not torch.all(labels == int(class_id)):
                raise ValueError("Class exemplar dataset contains unexpected labels.")
            images_device = images.to(device, non_blocking=True)
            features = model.extract_features(images_device, normalize=True)  # type: ignore[attr-defined]
            all_images.append(images.cpu())
            all_features.append(features.cpu())
        images_tensor = torch.cat(all_images, dim=0)
        features_tensor = torch.cat(all_features, dim=0)

        if max_candidates is not None and features_tensor.shape[0] > max_candidates:
            generator = torch.Generator().manual_seed(int(seed))
            indices = torch.randperm(features_tensor.shape[0], generator=generator)[: int(max_candidates)]
            images_tensor = images_tensor[indices]
            features_tensor = features_tensor[indices]
            quota = min(quota, int(max_candidates))

        class_mean = torch.nn.functional.normalize(features_tensor.mean(dim=0, keepdim=True), dim=1)[0]
        selected: List[int] = []
        selected_mask = torch.zeros(features_tensor.shape[0], dtype=torch.bool)
        running_sum = torch.zeros_like(class_mean)
        for k in range(quota):
            candidate_means = (features_tensor + running_sum.unsqueeze(0)) / float(k + 1)
            distances = torch.sum((candidate_means - class_mean.unsqueeze(0)) ** 2, dim=1)
            distances[selected_mask] = float("inf")
            chosen = int(torch.argmin(distances).item())
            selected.append(chosen)
            selected_mask[chosen] = True
            running_sum += features_tensor[chosen]
        return [images_tensor[index] for index in selected]

    @torch.no_grad()
    def compute_class_means(
        self,
        model: nn.Module,
        device: torch.device,
        batch_size: int,
        num_workers: int,
        seed: int,
    ) -> Dict[int, Tensor]:
        model.eval()
        means: Dict[int, Tensor] = {}
        for class_id in self.classes:
            samples = [(image, class_id) for image in self.exemplars[class_id]]
            dataset = ExemplarTensorDataset(samples, self.image_size, train_augment=False)
            loader = make_loader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                seed=seed,
            )
            features: List[Tensor] = []
            for images, _ in loader:
                features.append(
                    model.extract_features(images.to(device, non_blocking=True), normalize=True).cpu()  # type: ignore[attr-defined]
                )
            if features:
                mean = torch.cat(features, dim=0).mean(dim=0)
                means[class_id] = torch.nn.functional.normalize(mean, dim=0)
        return means
