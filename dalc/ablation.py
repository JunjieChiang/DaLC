"""
DaLC ablation variants.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np

from dalc._data_utils import ObjectRecord, majority_one_hot, normalize_distribution


VARIANTS = (
    "full",
    "no_ncs",
    "random_buckets",
    "no_cond_worker_clean",
    "global_worker_reliability",
    "no_edge_omega_cond",
    # Loss-component ablations (paper Table 5, rows w/o L_mv / L_obs / L_tgt).
    # Each zeroes the corresponding loss weight at parse_args; model architecture
    # and feature modes match "full".
    "no_loss_mv",
    "no_loss_obs",
    "no_loss_tgt",
)


def override_records_for_variant(
    records: Dict[int, ObjectRecord],
    variant: str,
    seed: int = 42,
) -> Dict[int, ObjectRecord]:
    if variant == "no_ncs":
        # Collapse the partition: everything looks "easy" so per-subset machinery is bypassed.
        return {obj: replace(rec, bucket="easy") for obj, rec in records.items()}
    if variant == "random_buckets":
        import random
        rng = random.Random(seed)
        buckets = [rec.bucket for rec in records.values()]
        rng.shuffle(buckets)
        return {obj: replace(rec, bucket=b) for (obj, rec), b in zip(records.items(), buckets)}
    return records


def flatten_conditioned_profiles(
    conditioned: Dict[int, Dict[str, float]],
    base: Dict[int, Dict[str, float]],
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for worker, prof in conditioned.items():
        omega = float(base.get(worker, {"omega": 0.5}).get("omega", 0.5))
        row = dict(prof)
        row["omega_global"] = omega
        row["omega_easy"] = omega
        row["omega_ambiguous"] = omega
        row["omega_hard"] = omega
        out[worker] = row
    return out


def base_reliability_dict(base_profiles: Dict[int, Dict[str, float]]) -> Dict[int, float]:
    return {w: float(prof.get("omega", 0.5)) for w, prof in base_profiles.items()}


def reliability_weighted_mv(
    records: Dict[int, ObjectRecord],
    votes: Dict[int, List[Tuple[int, int]]],
    object_ids: List[int],
    worker_reliability: Dict[int, float],
    num_classes: int,
    bucket_filter: str | None = "ambiguous",
) -> np.ndarray:
    targets = np.zeros((len(object_ids), num_classes), dtype=np.float32)
    for idx, obj in enumerate(object_ids):
        rec = records[obj]
        if bucket_filter is not None and rec.bucket != bucket_filter:
            targets[idx] = majority_one_hot(rec.majority, num_classes)
            continue
        weights = np.zeros(num_classes, dtype=np.float32)
        for worker, label in votes[obj]:
            weights[label] += float(worker_reliability.get(worker, 0.5))
        targets[idx] = normalize_distribution(weights, fallback_label=rec.majority)
    return targets


def disables_cond_worker(variant: str) -> bool:
    """Drop ω^(d) from worker.x AND from vote-edge features."""
    return variant in {"no_ncs", "no_cond_worker_clean", "global_worker_reliability"}


def disables_hard_amb_transfer(variant: str) -> bool:
    """Replace hard-conditioned worker scores and the ambiguous-backoff target with base ω."""
    return variant in {"no_ncs", "no_cond_worker_clean", "global_worker_reliability"}


def drops_edge_cond_omega(variant: str) -> bool:
    """Drop ω^(d) from vote-edge features only (worker.x may still carry it)."""
    return variant == "no_edge_omega_cond" or disables_cond_worker(variant)


def uses_global_ds_reliability(variant: str) -> bool:
    """Replace easy-only base ω (paper Eq.~10) with DS-style global ω across all training objects."""
    return variant == "global_worker_reliability"


def compute_global_ds_reliability(
    records: Dict[int, ObjectRecord],
    votes: Dict[int, List[Tuple[int, int]]],
    object_ids: List[int],
    train_mask: np.ndarray,
    num_classes: int,
) -> Dict[int, float]:
    """DS-style worker reliability initialized from agreement with MV across ALL training objects."""
    correct = defaultdict(float)
    total = defaultdict(float)
    for idx, obj in enumerate(object_ids):
        if not train_mask[idx]:
            continue
        mv_label = records[obj].majority
        for worker, label in votes[obj]:
            correct[worker] += float(label == mv_label)
            total[worker] += 1.0
    out: Dict[int, float] = {}
    workers = set(total) | {worker for obj in object_ids for worker, _ in votes[obj]}
    for worker in workers:
        n = float(total.get(worker, 0.0))
        out[worker] = (float(correct.get(worker, 0.0)) + 1.0) / (n + float(num_classes))
    return out
