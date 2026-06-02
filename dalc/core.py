"""DaLC core utilities.

Maps to the paper as follows:
    * :func:`compute_difficulty_info`               -- NCS difficulty estimation (Sec. 3.1).
        ``bucket`` (code) == ``subset(i)`` (paper, Eq.~8).
    * :func:`build_worker_similarity_matrix`        -- worker similarity ``a_{rr'}`` (Sec. 3.3, Eq.~18).
    * :func:`build_completion_initializer`          -- missing-label initialization ``z_{ir}^{(0)}``
        (Sec. 3.3, Eq.~19/20/21). The internal name ``q_init`` corresponds to ``z`` in the paper.
    * :func:`aggregate_missing_completion_prior`    -- object-level completion prior ``p_i^{comp}``.
    * :func:`build_object_object_edges` /
      :func:`build_vote_edge_features`              -- heterogeneous graph edges fed to the GNN.

Per-subset metrics are written with column ``subset \\in {easy, ambiguous, hard}``.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dalc._data_utils import ObjectRecord, anchor_strength, majority_one_hot, normalize_distribution

# Difficulty-aware completion-init and pseudo-target blend coefficients (paper
# Sec. 3.3). These were originally per-bucket (easy / ambiguous / hard) triples,
# but an A/B comparison (5 datasets x 3 seeds) showed that collapsing them to a
# single shared value changes overall accuracy by <0.1pp on average -- well
# inside seed noise. The paper only reports the self-neighbor lambda triple as
# bucket-conditional, so the implementation matches that now.
_MV_WEIGHT = 0.50           # q_init: how much of the prior (MV / backoff) to mix with local-similarity dist
_CONSENSUS_FLOOR = 0.10     # consensus_trust lower bound

# Hard-bucket-specific mechanism constants (no easy/ambiguous counterpart):
_HARD_LOCAL_WEIGHT = 0.30
_MAJORITY_FLOOR_WEIGHT = 0.10
# Margin-based consensus-trust schedule used by `consensus_completion_trust`.
_CONSENSUS_MARGIN_LOW = 0.40
_CONSENSUS_MARGIN_HIGH = 0.80


@dataclass
class DifficultyInfo:
    bucket: str
    ncs: float
    teacher_distribution: np.ndarray
    teacher_label: int
    easy_neighbor_count: int
    teacher_margin: float
    teacher_entropy: float
    teacher_confident: bool


@dataclass
class RoundOutput:
    completion_vote_dist: np.ndarray
    completion_mv_pred: np.ndarray
    missing_prob: np.ndarray
    missing_object_idx: np.ndarray
    data: object


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def build_train_val_masks(
    n_objects: int,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_objects, dtype=np.int64)
    rng.shuffle(indices)

    val_size = max(1, int(round(n_objects * val_frac)))
    val_size = min(val_size, max(n_objects - 1, 1))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    if len(train_idx) == 0:
        train_idx = val_idx[:1]
        val_idx = val_idx[1:]

    train_mask = np.zeros(n_objects, dtype=bool)
    val_mask = np.zeros(n_objects, dtype=bool)
    test_mask = np.zeros(n_objects, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    return train_mask, val_mask, test_mask


def safe_auc(y_true: np.ndarray, y_score: np.ndarray, num_classes: int) -> float:
    try:
        if num_classes == 2:
            return float(roc_auc_score(y_true, y_score[:, 1]))
        return float(roc_auc_score(y_true, y_score, multi_class="ovr", labels=list(range(num_classes))))
    except Exception:
        return float("nan")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, num_classes: int) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"accuracy": float("nan"), "f1": float("nan"), "auc": float("nan")}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "auc": safe_auc(y_true, y_score, num_classes),
    }


def bucket_metrics_rows(
    method: str,
    object_ids: List[int],
    records: Dict[int, ObjectRecord],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    num_classes: int,
    eval_mask: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for subset in ("all", "easy", "ambiguous", "hard"):
        idxs = [
            i
            for i, obj in enumerate(object_ids)
            if eval_mask[i] and (subset == "all" or records[obj].bucket == subset)
        ]
        metrics = evaluate_predictions(y_true[idxs], y_pred[idxs], y_score[idxs], num_classes)
        rows.append(
            {
                "method": method,
                "subset": subset,
                "n_objects": len(idxs),
                "accuracy": f"{metrics['accuracy']:.10f}" if not math.isnan(metrics["accuracy"]) else "",
                "f1": f"{metrics['f1']:.10f}" if not math.isnan(metrics["f1"]) else "",
                "auc": f"{metrics['auc']:.10f}" if not math.isnan(metrics["auc"]) else "",
            }
        )
    return rows


def distribution_margin(dist: np.ndarray) -> float:
    vals = np.sort(np.asarray(dist, dtype=np.float32))[::-1]
    top1 = float(vals[0]) if vals.size else 0.0
    top2 = float(vals[1]) if vals.size > 1 else 0.0
    return top1 - top2


def normalized_entropy(dist: np.ndarray) -> float:
    probs = np.asarray(dist, dtype=np.float32)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 1.0
    ent = float(-np.sum(probs * np.log(probs)))
    max_ent = float(np.log(len(dist))) if len(dist) > 1 else 1.0
    return ent / max(max_ent, 1e-12)


def consensus_margin_score(rec: ObjectRecord) -> float:
    return float(rec.gap_norm)


def _blend_prior_with_local(
    consensus_trust: float,
    mv_weight: float,
    majority_dist: np.ndarray,
    local_dist: np.ndarray,
    fallback_label: int,
) -> np.ndarray:
    """MV-weighted blend of majority vote and local similarity distribution.

    Used for easy and ambiguous q_init: higher mv_weight → trusts MV more.
    """
    effective_mv = 1.0 - consensus_trust * (1.0 - mv_weight)
    blended = effective_mv * majority_dist + (1.0 - effective_mv) * local_dist
    return normalize_distribution(blended, fallback_label=fallback_label)


def consensus_completion_trust(rec: ObjectRecord) -> float:
    margin = consensus_margin_score(rec)
    if margin <= _CONSENSUS_MARGIN_LOW:
        raw = 1.0
    elif margin >= _CONSENSUS_MARGIN_HIGH:
        raw = 0.0
    else:
        raw = 1.0 - (margin - _CONSENSUS_MARGIN_LOW) / max(_CONSENSUS_MARGIN_HIGH - _CONSENSUS_MARGIN_LOW, 1e-8)
    return float(_CONSENSUS_FLOOR + (1.0 - _CONSENSUS_FLOOR) * raw)


def build_worker_similarity_matrix(
    records: Dict[int, ObjectRecord],
    votes: Dict[int, List[Tuple[int, int]]],
    object_ids: List[int],
    cluster_of: Dict[int, int],
    num_classes: int,
) -> Tuple[List[int], Dict[int, int], np.ndarray]:
    workers = sorted({worker for obj in object_ids for worker, _ in votes[obj]})
    worker_to_idx = {worker: idx for idx, worker in enumerate(workers)}
    n_workers = len(workers)
    n_clusters = max(cluster_of.values()) + 1 if cluster_of else 1
    bucket_names = ("easy", "ambiguous", "hard")

    label_hist = np.zeros((n_workers, num_classes), dtype=np.float32)
    bucket_hist = np.zeros((n_workers, len(bucket_names), num_classes), dtype=np.float32)
    bucket_support = np.zeros((n_workers, len(bucket_names)), dtype=np.float32)
    bucket_agree = np.zeros((n_workers, len(bucket_names)), dtype=np.float32)
    cluster_hist = np.zeros((n_workers, n_clusters), dtype=np.float32)
    worker_labels: List[Dict[int, int]] = [dict() for _ in workers]

    bucket_to_idx = {name: idx for idx, name in enumerate(bucket_names)}
    for obj in object_ids:
        rec = records[obj]
        cidx = cluster_of[obj]
        bidx = bucket_to_idx[rec.bucket]
        for worker, label in votes[obj]:
            widx = worker_to_idx[worker]
            label_hist[widx, label] += 1.0
            bucket_hist[widx, bidx, label] += 1.0
            bucket_support[widx, bidx] += 1.0
            bucket_agree[widx, bidx] += float(label == rec.majority)
            cluster_hist[widx, cidx] += 1.0
            worker_labels[widx][obj] = label

    features: List[np.ndarray] = []
    for widx in range(n_workers):
        total = max(float(label_hist[widx].sum()), 1.0)
        global_probs = label_hist[widx] / total
        bucket_probs = []
        bucket_agree_rate = []
        bucket_ratio = []
        for bidx in range(len(bucket_names)):
            supp = max(float(bucket_support[widx, bidx]), 1.0)
            bucket_probs.append(bucket_hist[widx, bidx] / supp)
            bucket_agree_rate.append(np.asarray([bucket_agree[widx, bidx] / supp], dtype=np.float32))
            bucket_ratio.append(np.asarray([bucket_support[widx, bidx] / total], dtype=np.float32))
        cluster_ratio = cluster_hist[widx] / max(float(cluster_hist[widx].sum()), 1.0)
        features.append(
            np.concatenate(
                [global_probs] + bucket_probs + bucket_agree_rate + bucket_ratio + [cluster_ratio],
                axis=0,
            ).astype(np.float32)
        )
    feature_matrix = np.vstack(features) if features else np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    cosine = (feature_matrix / norms) @ (feature_matrix / norms).T if n_workers else np.zeros((0, 0), dtype=np.float32)

    similarity = np.zeros((n_workers, n_workers), dtype=np.float32)
    for i in range(n_workers):
        for j in range(i + 1, n_workers):
            shared = sorted(set(worker_labels[i]) & set(worker_labels[j]))
            if shared:
                agreement = float(np.mean([worker_labels[i][obj] == worker_labels[j][obj] for obj in shared]))
                overlap_weight = len(shared) / (len(shared) + 5.0)
                sim = (1.0 - overlap_weight) * float(cosine[i, j]) + overlap_weight * agreement
            else:
                sim = float(cosine[i, j])
            similarity[i, j] = sim
            similarity[j, i] = sim
    np.fill_diagonal(similarity, 0.0)
    np.clip(similarity, 0.0, 1.0, out=similarity)
    return workers, worker_to_idx, similarity


def compute_difficulty_info(
    records: Dict[int, ObjectRecord],
    neighbors: Dict[int, List[Tuple[float, int]]],
    object_ids: List[int],
    num_classes: int,
    min_easy_neighbors: int = 3,
    min_margin: float = 0.20,
    max_entropy: float = 0.75,
) -> List[DifficultyInfo]:
    rows: List[Dict[str, object]] = []
    for obj in object_ids:
        rec = records[obj]
        easy_neighbors = [(sim, src) for sim, src in neighbors[obj] if records[src].bucket == "easy"]
        if easy_neighbors:
            dist = np.zeros(num_classes, dtype=np.float32)
            for sim, src in easy_neighbors:
                weight = float(max(sim, 0.0))
                dist[records[src].majority] += weight
            teacher_dist = normalize_distribution(dist, fallback_label=rec.majority)
        else:
            teacher_dist = majority_one_hot(rec.majority, num_classes)
        easy_neighbor_count = len(easy_neighbors)
        teacher_margin = distribution_margin(teacher_dist)
        teacher_entropy = normalized_entropy(teacher_dist)
        rows.append(
            {
                "bucket": rec.bucket,
                "ncs": rec.ncs,
                "teacher_distribution": teacher_dist,
                "teacher_label": int(np.argmax(teacher_dist)),
                "easy_neighbor_count": easy_neighbor_count,
                "teacher_margin": teacher_margin,
                "teacher_entropy": teacher_entropy,
            }
        )

    count_thresh = int(min_easy_neighbors)
    margin_thresh = float(min_margin)
    entropy_thresh = float(max_entropy)
    hard_rows = [row for row in rows if row["bucket"] == "hard"]

    out: List[DifficultyInfo] = []
    for row in rows:
        teacher_confident = (
            row["bucket"] == "hard"
            and row["easy_neighbor_count"] >= count_thresh
            and row["teacher_margin"] >= margin_thresh
            and row["teacher_entropy"] <= entropy_thresh
        )
        out.append(
            DifficultyInfo(
                bucket=str(row["bucket"]),
                ncs=float(row["ncs"]),
                teacher_distribution=np.asarray(row["teacher_distribution"], dtype=np.float32),
                teacher_label=int(row["teacher_label"]),
                easy_neighbor_count=int(row["easy_neighbor_count"]),
                teacher_margin=float(row["teacher_margin"]),
                teacher_entropy=float(row["teacher_entropy"]),
                teacher_confident=teacher_confident,
            )
        )
    if hard_rows:
        final_coverage = float(np.mean([row.teacher_confident for row in out if row.bucket == "hard"]))
        print(
            "hard_teacher_gate"
            f" count_thresh={count_thresh}"
            f" margin_thresh={margin_thresh:.4f}"
            f" entropy_thresh={entropy_thresh:.4f}"
            f" hard_coverage={final_coverage:.4f}",
            flush=True,
        )
    return out


def build_completion_initializer(
    records: Dict[int, ObjectRecord],
    votes: Dict[int, List[Tuple[int, int]]],
    object_ids: List[int],
    workers: List[int],
    worker_to_idx: Dict[int, int],
    worker_similarity: np.ndarray,
    difficulty: List[DifficultyInfo],
    object_prior: np.ndarray,
    num_classes: int,
    ambiguous_worker_backoff_target: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    q_init = np.zeros((len(object_ids), len(workers), num_classes), dtype=np.float32)
    missing_mask = np.ones((len(object_ids), len(workers)), dtype=bool)
    for idx, obj in enumerate(object_ids):
        rec = records[obj]
        observed_items = sorted(votes[obj])
        if not observed_items:
            continue
        observed_workers = np.asarray([worker_to_idx[worker] for worker, _ in observed_items], dtype=np.int64)
        observed_labels = np.asarray([label for _, label in observed_items], dtype=np.int64)
        majority_dist = object_prior[idx]
        teacher_dist = difficulty[idx].teacher_distribution
        consensus_trust = consensus_completion_trust(rec)
        for worker, label in observed_items:
            widx = worker_to_idx[worker]
            q_init[idx, widx, label] = 1.0
            missing_mask[idx, widx] = False
        missing_workers = np.where(missing_mask[idx])[0]
        for widx in missing_workers:
            weights = worker_similarity[widx, observed_workers]
            local_dist = np.bincount(observed_labels, weights=weights, minlength=num_classes).astype(np.float32)
            local_dist = normalize_distribution(local_dist, fallback_label=rec.majority)
            if rec.bucket == "easy":
                init = _blend_prior_with_local(consensus_trust, _MV_WEIGHT, majority_dist, local_dist, rec.majority)
            elif rec.bucket == "hard":
                if difficulty[idx].teacher_confident:
                    teacher_weight = (1.0 - _HARD_LOCAL_WEIGHT) * consensus_trust
                    non_teacher_weight = 1.0 - teacher_weight
                    init = normalize_distribution(
                        non_teacher_weight * majority_dist
                        + teacher_weight * teacher_dist
                        + _MAJORITY_FLOOR_WEIGHT * (1.0 - consensus_trust) * majority_dist,
                        fallback_label=difficulty[idx].teacher_label,
                    )
                else:
                    init = normalize_distribution(
                        consensus_trust * local_dist + (1.0 - consensus_trust) * majority_dist,
                        fallback_label=rec.majority,
                    )
            else:
                if ambiguous_worker_backoff_target is not None:
                    init = _blend_prior_with_local(
                        consensus_trust, _MV_WEIGHT, ambiguous_worker_backoff_target[idx], local_dist, rec.majority,
                    )
                else:
                    init = _blend_prior_with_local(consensus_trust, _MV_WEIGHT, majority_dist, local_dist, rec.majority)
            q_init[idx, widx] = init
    return q_init, missing_mask


def aggregate_missing_completion_prior(
    q_init: np.ndarray,
    missing_mask: np.ndarray,
    object_ids: List[int],
    records: Dict[int, ObjectRecord],
    num_classes: int,
) -> np.ndarray:
    priors = np.zeros((len(object_ids), num_classes), dtype=np.float32)
    for idx, obj in enumerate(object_ids):
        missing_workers = np.where(missing_mask[idx])[0]
        if missing_workers.size == 0:
            priors[idx] = majority_one_hot(records[obj].majority, num_classes)
        else:
            priors[idx] = normalize_distribution(
                np.mean(q_init[idx, missing_workers], axis=0),
                fallback_label=records[obj].majority,
            )
    return priors


def build_vote_edge_features(
    workers: List[int],
    worker_to_idx: Dict[int, int],
    conditioned_profiles: Dict[int, Dict[str, float]],
    records: Dict[int, ObjectRecord],
    votes: Dict[int, List[Tuple[int, int]]],
    object_ids: List[int],
    num_classes: int,
    vote_edge_feature_mode: str = "drop_omega_global",
) -> Tuple[List[List[int]], List[List[int]], List[List[float]], List[List[float]]]:
    """Build worker->object and object->worker edge features for the heterogeneous graph.

    Supported ``vote_edge_feature_mode`` values:
        ``drop_omega_global``     : [label_onehot, NCS, anchor_strength, omega_subset]
                                    -- the default for the full DaLC model.
        ``drop_per_bucket_omega`` : [label_onehot, NCS, anchor_strength, omega_global]
                                    -- used by ablations that drop omega^(d) from edges
                                    (no_edge_omega_cond / no_cond_worker_clean / etc.).
        ``label_only``            : [label_onehot]
                                    -- used by the no_ncs ablation.
    """
    worker_obj_edge_index: List[List[int]] = [[], []]
    obj_worker_edge_index: List[List[int]] = [[], []]
    worker_obj_edge_attr: List[List[float]] = []
    obj_worker_edge_attr: List[List[float]] = []

    default_omega = 0.5
    for obj_idx, obj in enumerate(object_ids):
        rec = records[obj]
        target_anchor = anchor_strength(rec.bucket)
        for worker, label in votes[obj]:
            widx = worker_to_idx[worker]
            prof = conditioned_profiles.get(
                worker,
                {
                    "omega_global": default_omega,
                    "omega_easy": default_omega,
                    "omega_ambiguous": default_omega,
                    "omega_hard": default_omega,
                    "bias": 0.0,
                },
            )
            current_bucket_score = float(prof.get(f"omega_{rec.bucket}", prof["omega_global"]))
            label_one_hot = np.zeros(num_classes, dtype=np.float32)
            label_one_hot[label] = 1.0
            feat = label_one_hot.tolist()
            if vote_edge_feature_mode == "drop_omega_global":
                feat.extend([rec.ncs, target_anchor, current_bucket_score])
            elif vote_edge_feature_mode == "drop_per_bucket_omega":
                feat.extend([rec.ncs, target_anchor, prof["omega_global"]])
            elif vote_edge_feature_mode == "label_only":
                pass
            else:
                raise ValueError(f"Unknown vote_edge_feature_mode: {vote_edge_feature_mode!r}")
            worker_obj_edge_index[0].append(widx)
            worker_obj_edge_index[1].append(obj_idx)
            worker_obj_edge_attr.append(feat)
            obj_worker_edge_index[0].append(obj_idx)
            obj_worker_edge_index[1].append(widx)
            obj_worker_edge_attr.append(feat)
    return worker_obj_edge_index, obj_worker_edge_index, worker_obj_edge_attr, obj_worker_edge_attr


def build_object_object_edges(
    records: Dict[int, ObjectRecord],
    neighbors: Dict[int, List[Tuple[float, int]]],
    object_ids: List[int],
    difficulty: List[DifficultyInfo],
    object_edge_feature_mode: str = "full",
) -> Tuple[List[List[int]], List[List[float]]]:
    """Build object-object similarity edges for the heterogeneous graph.

    Supported ``object_edge_feature_mode`` values:
        ``full``            : [sim, src_NCS, dst_NCS, src_anchor, dst_anchor,
                               src_neighbor_evidence_confident, dst_neighbor_evidence_confident]
                              -- the default for the full DaLC model.
        ``similarity_only`` : [sim]
                              -- used by the no_ncs ablation.
    """
    object_to_idx = {obj: idx for idx, obj in enumerate(object_ids)}
    edge_index: List[List[int]] = [[], []]
    edge_attr: List[List[float]] = []
    difficulty_by_obj = {obj: difficulty[idx] for idx, obj in enumerate(object_ids)}
    for dst_obj in object_ids:
        dst = object_to_idx[dst_obj]
        dst_rec = records[dst_obj]
        dst_diff = difficulty_by_obj[dst_obj]
        for sim, src_obj in neighbors[dst_obj]:
            src = object_to_idx[src_obj]
            src_rec = records[src_obj]
            src_diff = difficulty_by_obj[src_obj]
            edge_index[0].append(src)
            edge_index[1].append(dst)
            feat = [float(sim)]
            if object_edge_feature_mode == "full":
                feat.extend(
                    [
                        src_rec.ncs,
                        dst_rec.ncs,
                        anchor_strength(src_rec.bucket),
                        anchor_strength(dst_rec.bucket),
                        float(src_diff.teacher_confident),
                        float(dst_diff.teacher_confident),
                    ]
                )
            elif object_edge_feature_mode == "similarity_only":
                pass
            else:
                raise ValueError(f"Unknown object_edge_feature_mode: {object_edge_feature_mode!r}")
            edge_attr.append(feat)
    return edge_index, edge_attr


def build_completed_label_vote_distribution(
    records: Dict[int, ObjectRecord],
    votes: Dict[int, List[Tuple[int, int]]],
    object_ids: List[int],
    num_classes: int,
    missing_object_idx: np.ndarray,
    missing_prob: np.ndarray,
) -> np.ndarray:
    dist = np.zeros((len(object_ids), num_classes), dtype=np.float32)
    for idx, obj in enumerate(object_ids):
        for _, label in votes[obj]:
            dist[idx, label] += 1.0
    for edge_idx, obj_idx in enumerate(missing_object_idx):
        hard_label = int(np.argmax(missing_prob[edge_idx]))
        dist[int(obj_idx), hard_label] += 1.0
    out = np.zeros_like(dist)
    for idx, obj in enumerate(object_ids):
        out[idx] = normalize_distribution(dist[idx], fallback_label=records[obj].majority)
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
