from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .trigger import (
    TriggerOptimizationResult,
    load_trajectory_models,
    optimize_universal_trigger,
    save_trigger,
)
from .utils import append_csv, ensure_dir, save_json


TrajectoryBuilder = Callable[[Optional[Tensor], Path, int], Sequence[Path]]


def _select_trajectory_models(models, mode: str):
    mode = mode.lower()
    if mode == "all":
        return models
    if mode == "first":
        return models[:1]
    if mode == "last":
        return models[-1:]
    raise ValueError("trigger.trajectory_mode must be all, first, or last")


def run_iterative_refinement(
    clean_checkpoint_paths: Sequence[Path],
    trajectory_builder: TrajectoryBuilder,
    source_loader: DataLoader,
    reference_loader: DataLoader,
    target_label: int,
    image_size: int,
    device: torch.device,
    trigger_config: Mapping[str, Any],
    refinement_config: Mapping[str, Any],
    output_dir: str | Path,
    seed: int,
) -> Tensor:
    """Run PBTO initialization and the alternating refinement in Eq. (6)-(9)."""

    output_dir = ensure_dir(output_dir)
    round_log = output_dir / "refinement.csv"

    clean_models = load_trajectory_models(clean_checkpoint_paths, device=device)
    clean_models = _select_trajectory_models(clean_models, str(trigger_config.get("trajectory_mode", "all")))
    initial = optimize_universal_trigger(
        models=clean_models,
        source_loader=source_loader,
        reference_loader=reference_loader,
        target_label=target_label,
        image_size=image_size,
        device=device,
        config=trigger_config,
        output_log=output_dir / "trigger_round_000.csv",
        initial_delta=None,
        seed=seed,
    )
    del clean_models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    trigger = initial.trigger
    save_trigger(
        trigger,
        output_dir / "trigger_round_000.pt",
        metadata={"round": 0, "trajectory_asr": initial.final_trajectory_asr},
    )
    append_csv(
        {
            "round": 0,
            "trajectory_asr": initial.final_trajectory_asr,
            "improvement": float("nan"),
        },
        round_log,
    )

    if not bool(refinement_config.get("enabled", True)):
        save_trigger(trigger, output_dir / "trigger_final.pt", metadata={"round": 0})
        return trigger

    max_rounds = int(refinement_config.get("max_rounds", 60))
    asr_threshold = float(refinement_config.get("asr_threshold", 0.95))
    tolerance = float(refinement_config.get("tolerance", 0.001))
    min_rounds = int(refinement_config.get("min_rounds", 1))
    keep_poisoned_trajectories = bool(refinement_config.get("keep_poisoned_trajectories", False))
    previous_asr = initial.final_trajectory_asr
    final_round = 0

    for round_id in range(1, max_rounds + 1):
        round_dir = ensure_dir(output_dir / f"poisoned_trajectory_round_{round_id:03d}")
        checkpoint_paths = list(trajectory_builder(trigger, round_dir, round_id))
        models = load_trajectory_models(checkpoint_paths, device=device)
        models = _select_trajectory_models(models, str(trigger_config.get("trajectory_mode", "all")))
        result = optimize_universal_trigger(
            models=models,
            source_loader=source_loader,
            reference_loader=reference_loader,
            target_label=target_label,
            image_size=image_size,
            device=device,
            config=trigger_config,
            output_log=output_dir / f"trigger_round_{round_id:03d}.csv",
            initial_delta=trigger,
            seed=seed + round_id,
        )
        trigger = result.trigger
        improvement = result.final_trajectory_asr - previous_asr
        append_csv(
            {
                "round": round_id,
                "trajectory_asr": result.final_trajectory_asr,
                "improvement": improvement,
            },
            round_log,
        )
        save_trigger(
            trigger,
            output_dir / f"trigger_round_{round_id:03d}.pt",
            metadata={
                "round": round_id,
                "trajectory_asr": result.final_trajectory_asr,
                "improvement": improvement,
            },
        )
        final_round = round_id
        should_stop = (
            round_id >= min_rounds
            and (
                result.final_trajectory_asr >= asr_threshold
                or abs(improvement) < tolerance
            )
        )
        previous_asr = result.final_trajectory_asr
        del models
        if not keep_poisoned_trajectories:
            shutil.rmtree(round_dir, ignore_errors=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if should_stop:
            break

    save_trigger(
        trigger,
        output_dir / "trigger_final.pt",
        metadata={"round": final_round, "trajectory_asr": previous_asr},
    )
    save_json(
        {
            "final_round": final_round,
            "final_trajectory_asr": previous_asr,
            "asr_threshold": asr_threshold,
            "tolerance": tolerance,
            "max_rounds": max_rounds,
        },
        output_dir / "refinement_summary.json",
    )
    return trigger
