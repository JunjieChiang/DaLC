"""Difficulty-aware Label Completion (DaLC).

Public API:
    DifficultyAwareCompletionGNN  -- the GNN encoder/decoder (Section 3.3).
    train                          -- training loop for the joint objective.
    compute_difficulty_info        -- NCS-based difficulty estimation (Section 3.1).
    DifficultyInfo, RoundOutput    -- result containers used by the pipeline.
    bucket_metrics_rows            -- per-subset accuracy / F1 / AUC rows.
    set_seed, write_csv            -- helpers.
"""
from __future__ import annotations

from .core import (
    DifficultyInfo,
    RoundOutput,
    bucket_metrics_rows,
    compute_difficulty_info,
    set_seed,
    write_csv,
)
from .model import DifficultyAwareCompletionGNN, train

__all__ = [
    "DifficultyAwareCompletionGNN",
    "DifficultyInfo",
    "RoundOutput",
    "bucket_metrics_rows",
    "compute_difficulty_info",
    "set_seed",
    "train",
    "write_csv",
]
