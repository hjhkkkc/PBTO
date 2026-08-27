from __future__ import annotations

import csv
import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence, TypeVar

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

T = TypeVar("T")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def configure_torch_threads(num_threads: int) -> None:
    """Limit CPU thread oversubscription for small vision workloads."""
    num_threads = max(1, int(num_threads))
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(max(1, min(num_threads, 4)))
    except RuntimeError:
        # Inter-op threads can only be set before parallel work starts.
        pass


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def save_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    import yaml

    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False, allow_unicode=True)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def append_csv(row: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))


def infinite(loader: Iterable[T]) -> Iterator[T]:
    while True:
        yield from loader



def vision_collate(batch):
    """Collate image/label pairs even when labels mix Python ints and 0-D tensors."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    images, labels = zip(*batch)
    image_batch = torch.stack([torch.as_tensor(image) for image in images], dim=0)
    label_batch = torch.tensor([int(torch.as_tensor(label).item()) for label in labels], dtype=torch.long)
    return image_batch, label_batch


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    drop_last: bool = False,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
        generator=generator,
        persistent_workers=bool(num_workers > 0),
        collate_fn=vision_collate,
    )


def cpu_state_dict(module: torch.nn.Module) -> Dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def nested_get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def nested_set(mapping: MutableMapping[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: MutableMapping[str, Any] = mapping
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def parse_overrides(items: Sequence[str]) -> Dict[str, Any]:
    import yaml

    result: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        nested_set(result, key.strip(), yaml.safe_load(raw))
    return result


def describe_environment() -> Dict[str, Any]:
    return {
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
