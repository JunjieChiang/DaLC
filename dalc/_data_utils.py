"""
Shared data-loading and feature-construction utilities used by DaLC.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, pairwise_distances, roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv

from ._ncs_split import assign_ncs_buckets
from ._io_paths import result_path_for_dataset

MISSING_TRUTH = -1


@dataclass
class ObjectRecord:
    obj: int
    truth: int
    vote_hist: Tuple[int, ...]
    total_votes: int
    majority: int
    majority_share: float
    runnerup_share: float
    gap_norm: float
    entropy: float
    ncs: float
    bucket: str

    @property
    def mv_correct(self) -> int:
        return int(self.majority == self.truth)

    @property
    def has_truth(self) -> bool:
        return self.truth != MISSING_TRUTH


@dataclass
class FeatureSpace:
    vectors: Dict[int, np.ndarray]
    is_nominal: np.ndarray
    numeric_ranges: np.ndarray
    nominal_cardinalities: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


VARIANT_SEED_OFFSETS = {
    "x_base": 11,
    "x_cred": 23,
    "x_conditioned": 37,
}
FIXED_SEED = 42


def anchor_strength(bucket: str) -> float:
    if bucket == "easy":
        return 1.0
    if bucket == "ambiguous":
        return 0.5
    return 0.1


def ncs_bucket(score: float, easy_threshold: float, ambiguous_threshold: float) -> str:
    if score >= easy_threshold:
        return "easy"
    if score >= ambiguous_threshold:
        return "ambiguous"
    return "hard"


def self_vote_lambda(bucket: str, args: argparse.Namespace) -> float:
    if bucket == "easy":
        return args.lambda_easy
    if bucket == "ambiguous":
        return args.lambda_ambiguous
    return args.lambda_hard


def output_paths(dataset_dir: Path, args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    suffix = f"_{args.output_prefix}" if args.output_prefix else ""
    return (
        result_path_for_dataset(dataset_dir, f"pyg_hetero_multiclass{suffix}_metrics.csv"),
        result_path_for_dataset(dataset_dir, f"pyg_hetero_multiclass{suffix}_predictions.csv"),
        result_path_for_dataset(dataset_dir, f"pyg_hetero_multiclass{suffix}_truth_vote_summary.csv"),
    )


def _looks_like_header(row: Sequence[str]) -> bool:
    try:
        int(row[0])
        return False
    except Exception:
        return True


def _read_csv_rows(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append([cell.strip() for cell in row])
    return rows


def load_truth(path: Path) -> Dict[int, int]:
    rows = _read_csv_rows(path)
    if not rows:
        return {}
    truth: Dict[int, int] = {}
    if _looks_like_header(rows[0]):
        for row in rows[1:]:
            truth[int(row[0])] = int(row[1])
    else:
        for row in rows:
            truth[int(row[0])] = int(row[1])
    return truth


def load_votes(path: Path) -> Dict[int, List[Tuple[int, int]]]:
    rows = _read_csv_rows(path)
    if not rows:
        return {}
    votes: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    start = 1 if _looks_like_header(rows[0]) else 0
    for row in rows[start:]:
        obj = int(row[0])
        worker = int(row[1])
        label = int(row[2])
        votes[obj].append((worker, label))
    return dict(votes)


def parse_arff(path: Path) -> Tuple[np.ndarray, List[str], List[str], List[int] | None]:
    attr_names: List[str] = []
    attr_specs: List[str] = []
    rows: List[List[float]] = []
    in_data = False
    in_id_map = False
    id_map: List[int] = []
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            lower = line.lower()
            if lower.startswith("@id-map"):
                in_id_map = True
                in_data = False
                continue
            if lower.startswith("@attribute"):
                rest = line[len("@attribute"):].lstrip()
                if rest.startswith("'") or rest.startswith('"'):
                    q = rest[0]
                    close = rest.index(q, 1)
                    attr_names.append(rest[1:close])
                    attr_specs.append(rest[close + 1:].strip())
                else:
                    parts = rest.split(None, 1)
                    attr_names.append(parts[0] if parts else "")
                    attr_specs.append(parts[1].strip() if len(parts) >= 2 else "")
                continue
            if lower.startswith("@data"):
                in_data = True
                in_id_map = False
                continue
            if in_id_map:
                id_map.append(int(line))
                continue
            if not in_data:
                continue
            if line.startswith("@"):
                continue
            cells = [cell.strip() for cell in line.split(",")]
            if attr_names and len(cells) != len(attr_names):
                continue
            values: List[float] = []
            for col_idx, cell in enumerate(cells):
                if cell == "?":
                    values.append(0.0)
                    continue
                spec = attr_specs[col_idx].strip() if col_idx < len(attr_specs) else ""
                if spec.startswith("{") and spec.endswith("}"):
                    raw_choices = spec[1:-1].split(",")
                    choices = [choice.strip().strip("'").strip('"') for choice in raw_choices]
                    normalized = cell.strip().strip("'").strip('"')
                    numeric_like = True
                    for choice in choices:
                        try:
                            float(choice)
                        except ValueError:
                            numeric_like = False
                            break
                    if numeric_like:
                        values.append(float(normalized))
                    else:
                        try:
                            # Reserve 0.0 for missing values and map unordered nominal values to 1..K.
                            values.append(float(choices.index(normalized) + 1))
                        except ValueError as exc:
                            raise ValueError(
                                f"unknown nominal value {cell!r} for attribute {attr_names[col_idx]!r} in {path}"
                            ) from exc
                else:
                    values.append(float(cell))
            rows.append(values)
    if not rows:
        raise ValueError(f"no ARFF rows found in {path}")
    return np.asarray(rows, dtype=np.float32), attr_names, attr_specs, (id_map if id_map else None)


def find_arff_path(dataset_dir: Path) -> Path:
    paths = sorted(dataset_dir.glob("*.arff"))
    if not paths:
        raise FileNotFoundError(f"no .arff file found in {dataset_dir}")
    return paths[0]


def load_feature_space(dataset_dir: Path, object_ids: List[int]) -> FeatureSpace:
    try:
        arff_path = find_arff_path(dataset_dir)
    except FileNotFoundError:
        raise
    matrix, _, attr_specs, id_map = parse_arff(arff_path)
    if matrix.shape[1] < 2:
        raise ValueError(f"ARFF file must contain at least one feature column and one class column: {arff_path}")
    feature_matrix = matrix[:, :-1]
    feature_specs = attr_specs[:-1]
    row_of_object = {obj: idx for idx, obj in enumerate(id_map)} if id_map is not None else None
    vectors: Dict[int, np.ndarray] = {}
    for obj in object_ids:
        if row_of_object is not None:
            if obj not in row_of_object:
                raise KeyError(f"object id {obj} is missing from ARFF @ID-MAP in {arff_path}")
            row_idx = row_of_object[obj]
        else:
            if obj < 0 or obj >= feature_matrix.shape[0]:
                raise IndexError(f"object id {obj} is outside ARFF row range 0..{feature_matrix.shape[0]-1}")
            row_idx = obj
        vectors[obj] = np.asarray(feature_matrix[row_idx], dtype=np.float32)
    is_nominal_flags: List[bool] = []
    for spec in feature_specs:
        spec = spec.strip()
        nominal = spec.startswith("{") and spec.endswith("}")
        if nominal:
            raw_choices = spec[1:-1].split(",")
            choices = [choice.strip().strip("'").strip('"') for choice in raw_choices if choice.strip()]
            numeric_like = bool(choices)
            for choice in choices:
                try:
                    float(choice)
                except ValueError:
                    numeric_like = False
                    break
            nominal = not numeric_like
        is_nominal_flags.append(nominal)
    is_nominal = np.asarray(is_nominal_flags, dtype=bool)
    numeric_ranges = np.ones(len(feature_specs), dtype=np.float32)
    nominal_cardinalities = np.zeros(len(feature_specs), dtype=np.int64)
    if feature_matrix.size:
        numeric_idx = np.where(~is_nominal)[0]
        if numeric_idx.size:
            numeric_block = feature_matrix[:, numeric_idx]
            mins = np.nanmin(numeric_block, axis=0)
            maxs = np.nanmax(numeric_block, axis=0)
            ranges = np.asarray(maxs - mins, dtype=np.float32)
            ranges[ranges < 1e-6] = 1.0
            numeric_ranges[numeric_idx] = ranges
    for idx, spec in enumerate(feature_specs):
        spec = spec.strip()
        if spec.startswith("{") and spec.endswith("}") and is_nominal[idx]:
            raw_choices = spec[1:-1].split(",")
            nominal_cardinalities[idx] = len([choice for choice in raw_choices if choice.strip()])
    return FeatureSpace(
        vectors=vectors,
        is_nominal=is_nominal,
        numeric_ranges=numeric_ranges,
        nominal_cardinalities=nominal_cardinalities,
    )


def infer_num_classes(truth: Dict[int, int], votes: Dict[int, List[Tuple[int, int]]]) -> int:
    labels = set(truth.values())
    for rows in votes.values():
        for _, label in rows:
            labels.add(label)
    return max(labels) + 1


def resolve_object_ids(
    truth: Dict[int, int],
    votes: Dict[int, List[Tuple[int, int]]],
    source: str,
) -> List[int]:
    if source == "intersection":
        return sorted(set(truth) & set(votes))
    if source == "truth":
        return sorted(truth)
    if source == "answer":
        return sorted(votes)
    if source == "union":
        return sorted(set(truth) | set(votes))
    raise ValueError(f"unsupported object-id-source: {source}")


def build_basic_vote_stats(
    truth: Dict[int, int],
    votes: Dict[int, List[Tuple[int, int]]],
    num_classes: int,
    object_ids: Sequence[int] | None = None,
) -> Dict[int, Dict[str, object]]:
    info: Dict[int, Dict[str, object]] = {}
    target_objects = list(object_ids) if object_ids is not None else sorted(truth)
    for obj in target_objects:
        y = truth.get(obj, MISSING_TRUTH)
        vr = votes.get(obj, [])
        hist = np.zeros(num_classes, dtype=np.int64)
        for _, label in vr:
            hist[label] += 1
        total = int(hist.sum())
        majority = int(np.argmax(hist))
        sorted_counts = np.sort(hist)[::-1]
        top = int(sorted_counts[0]) if len(sorted_counts) else 0
        runner = int(sorted_counts[1]) if len(sorted_counts) > 1 else 0
        majority_share = float(top / total) if total else 0.0
        runnerup_share = float(runner / total) if total else 0.0
        gap_norm = majority_share - runnerup_share
        entropy = 0.0
        if total:
            for count in hist:
                if count == 0:
                    continue
                p = count / total
                entropy -= p * math.log(p, 2)
            if num_classes > 1:
                entropy /= math.log(num_classes, 2)
        info[obj] = {
            "truth": int(y),
            "vote_hist": tuple(int(v) for v in hist.tolist()),
            "total_votes": total,
            "majority": majority,
            "majority_share": majority_share,
            "runnerup_share": runnerup_share,
            "gap_norm": gap_norm,
            "entropy": entropy,
        }
    return info


def build_global_neighbor_map(
    object_ids: List[int],
    feature_space: FeatureSpace,
    top_k: int,
    metric: str,
) -> Dict[int, List[Tuple[float, int]]]:
    x = np.vstack([feature_space.vectors[obj] for obj in object_ids]).astype(np.float32, copy=False)
    neighbors: Dict[int, List[Tuple[float, int]]] = {}
    if metric == "cosine":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        x_norm = x / norms
        sims = x_norm @ x_norm.T
        for i, obj in enumerate(object_ids):
            order = np.argsort(-sims[i])
            pairs: List[Tuple[float, int]] = []
            for j in order:
                if i == j:
                    continue
                pairs.append((float(sims[i, j]), object_ids[j]))
                if len(pairs) >= top_k:
                    break
            neighbors[obj] = pairs
        return neighbors
    if metric == "euclidean":
        sq_norm = np.sum(x * x, axis=1, keepdims=True)
        dist_sq = np.maximum(sq_norm + sq_norm.T - 2.0 * (x @ x.T), 0.0)
        dists = np.sqrt(dist_sq).astype(np.float32)
        for i, obj in enumerate(object_ids):
            order = np.argsort(dists[i])
            pairs: List[Tuple[float, int]] = []
            for j in order:
                if i == j:
                    continue
                affinity = 1.0 / (1.0 + float(dists[i, j]))
                pairs.append((affinity, object_ids[j]))
                if len(pairs) >= top_k:
                    break
            neighbors[obj] = pairs
        return neighbors
    if metric == "heom":
        num_features = x.shape[1]
        if num_features == 0:
            raise ValueError("HEOM requires at least one feature column")
        distance_sq = np.zeros((x.shape[0], x.shape[0]), dtype=np.float32)
        numeric_idx = np.where(~feature_space.is_nominal)[0]
        if numeric_idx.size:
            x_num = x[:, numeric_idx] / feature_space.numeric_ranges[numeric_idx]
            distance_sq += pairwise_distances(x_num, metric="sqeuclidean").astype(np.float32)
        nominal_idx = np.where(feature_space.is_nominal)[0]
        if nominal_idx.size:
            x_nom = x[:, nominal_idx]
            nominal_mismatch = pairwise_distances(x_nom, metric="hamming").astype(np.float32) * nominal_idx.size
            distance_sq += nominal_mismatch
        dists = np.sqrt(distance_sq / float(num_features)).astype(np.float32)
        for i, obj in enumerate(object_ids):
            order = np.argsort(dists[i])
            pairs: List[Tuple[float, int]] = []
            for j in order:
                if i == j:
                    continue
                affinity = 1.0 / (1.0 + float(dists[i, j]))
                pairs.append((affinity, object_ids[j]))
                if len(pairs) >= top_k:
                    break
            neighbors[obj] = pairs
        return neighbors
    raise ValueError(f"unsupported neighbor metric: {metric}")


def build_cluster_neighbor_map(
    object_ids: List[int],
    vectors: Dict[int, np.ndarray],
    cluster_of: Dict[int, int],
    top_k: int,
) -> Dict[int, List[Tuple[float, int]]]:
    by_cluster: Dict[int, List[int]] = defaultdict(list)
    for obj in object_ids:
        by_cluster[cluster_of[obj]].append(obj)

    neighbors: Dict[int, List[Tuple[float, int]]] = {}
    for objs in by_cluster.values():
        cluster_vecs = np.vstack([vectors[obj] for obj in objs])
        sims = cluster_vecs @ cluster_vecs.T
        for i, obj in enumerate(objs):
            order = np.argsort(-sims[i])
            pairs: List[Tuple[float, int]] = []
            for j in order:
                if i == j:
                    continue
                pairs.append((float(sims[i, j]), objs[j]))
                if len(pairs) >= top_k:
                    break
            neighbors[obj] = pairs
    return neighbors


def assign_semantic_clusters(
    object_ids: List[int],
    vectors: Dict[int, np.ndarray],
    seed: int,
    n_clusters: int = 8,
) -> Dict[int, int]:
    actual_clusters = min(max(1, n_clusters), len(object_ids))
    x = np.vstack([vectors[obj] for obj in object_ids])
    km = KMeans(n_clusters=actual_clusters, n_init=20, random_state=seed)
    labels = km.fit_predict(x)
    return {obj: int(label) for obj, label in zip(object_ids, labels)}


def load_records(
    truth: Dict[int, int],
    votes: Dict[int, List[Tuple[int, int]]],
    num_classes: int,
    neighbors: Dict[int, List[Tuple[float, int]]],
    easy_threshold: float,
    ambiguous_threshold: float,
    object_ids: Sequence[int] | None = None,
    bucket_strategy: str = "fixed",
) -> Dict[int, ObjectRecord]:
    basic = build_basic_vote_stats(truth, votes, num_classes, object_ids=object_ids)
    ncs_by_obj: Dict[int, float] = {}
    ordered_objects = sorted(basic)
    for obj in ordered_objects:
        row = basic[obj]
        current_majority = int(row["majority"])
        first_hop = neighbors[obj]
        if first_hop:
            match_count = sum(
                int(basic[src_obj]["majority"] == current_majority)
                for _sim, src_obj in first_hop
            )
            ncs = match_count / len(first_hop)
        else:
            ncs = 0.0
        ncs_by_obj[obj] = float(ncs)

    bucket_list = assign_ncs_buckets(
        [ncs_by_obj[obj] for obj in ordered_objects],
        easy_threshold=easy_threshold,
        ambiguous_threshold=ambiguous_threshold,
        strategy=bucket_strategy,
    )
    bucket_by_obj = {obj: bucket for obj, bucket in zip(ordered_objects, bucket_list)}

    records: Dict[int, ObjectRecord] = {}
    for obj, row in basic.items():
        current_majority = int(row["majority"])
        ncs = ncs_by_obj[obj]
        bucket = bucket_by_obj[obj]
        records[obj] = ObjectRecord(
            obj=obj,
            truth=int(row["truth"]),
            vote_hist=tuple(int(v) for v in row["vote_hist"]),
            total_votes=int(row["total_votes"]),
            majority=current_majority,
            majority_share=float(row["majority_share"]),
            runnerup_share=float(row["runnerup_share"]),
            gap_norm=float(row["gap_norm"]),
            entropy=float(row["entropy"]),
            ncs=float(ncs),
            bucket=bucket,
        )
    return records


def weighted_label_distribution(
    weighted_neighbors: List[Tuple[float, int]],
    records: Dict[int, ObjectRecord],
    num_classes: int,
) -> np.ndarray:
    dist = np.zeros(num_classes, dtype=np.float32)
    for weight, src_obj in weighted_neighbors:
        if weight <= 0.0:
            continue
        dist[records[src_obj].majority] += float(weight)
    total = float(dist.sum())
    if total > 0.0:
        dist /= total
    return dist


def normalize_distribution(dist: np.ndarray, fallback_label: int | None = None) -> np.ndarray:
    out = np.asarray(dist, dtype=np.float32).copy()
    total = float(out.sum())
    if total > 0.0:
        out /= total
        return out
    if fallback_label is None:
        return out
    out[:] = 0.0
    out[int(fallback_label)] = 1.0
    return out


def majority_one_hot(label: int, num_classes: int) -> np.ndarray:
    dist = np.zeros(num_classes, dtype=np.float32)
    dist[label] = 1.0
    return dist


def dist_to_label(dist: np.ndarray, fallback: int) -> int:
    if float(dist.sum()) <= 0.0:
        return int(fallback)
    return int(np.argmax(dist))


def profile_workers_easy(
    records: Dict[int, ObjectRecord],
    votes: Dict[int, List[Tuple[int, int]]],
    train_mask: np.ndarray,
    object_ids: List[int],
    num_classes: int,
) -> Dict[int, Dict[str, float]]:
    stats: Dict[int, Dict[str, object]] = defaultdict(
        lambda: {
            "correct": 0.0,
            "total": 0.0,
            "label_hist": np.zeros(num_classes, dtype=np.float32),
        }
    )
    for idx, obj in enumerate(object_ids):
        if not train_mask[idx]:
            continue
        rec = records[obj]
        if rec.bucket != "easy":
            continue
        teacher = rec.majority
        for worker, label in votes[obj]:
            s = stats[worker]
            s["total"] = float(s["total"]) + 1.0
            s["correct"] = float(s["correct"]) + float(label == teacher)
            label_hist = np.asarray(s["label_hist"], dtype=np.float32)
            label_hist[label] += 1.0
            s["label_hist"] = label_hist

    profiles: Dict[int, Dict[str, float]] = {}
    for worker, s in stats.items():
        total = float(s["total"])
        label_hist = np.asarray(s["label_hist"], dtype=np.float32)
        omega = (float(s["correct"]) + 1.0) / (total + float(num_classes)) if total else (1.0 / num_classes)
        probs = label_hist / total if total else np.full(num_classes, 1.0 / num_classes, dtype=np.float32)
        vote_entropy = 0.0
        for p in probs:
            if p > 0:
                vote_entropy -= float(p) * math.log(float(p), 2)
        if num_classes > 1:
            vote_entropy /= math.log(num_classes, 2)
        dominant_rate = float(np.max(probs)) if total else (1.0 / num_classes)
        bias = dominant_rate - (1.0 / num_classes)
        profiles[worker] = {
            "omega": omega,
            "bias": bias,
            "dominant_rate": dominant_rate,
            "vote_entropy": vote_entropy,
            "total": total,
        }
    return profiles


def profile_workers_conditioned(
    records: Dict[int, ObjectRecord],
    votes: Dict[int, List[Tuple[int, int]]],
    train_mask: np.ndarray,
    object_ids: List[int],
    pseudo_target: np.ndarray,
    base_profiles: Dict[int, Dict[str, float]],
) -> Dict[int, Dict[str, float]]:
    shrink_strength = 5.0
    bucket_stats: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"correct": 0.0, "total": 0.0})
    )
    bucket_names = ("easy", "ambiguous", "hard")

    for idx, obj in enumerate(object_ids):
        if not train_mask[idx]:
            continue
        teacher = np.asarray(pseudo_target[idx], dtype=np.float32)
        bucket = records[obj].bucket
        for worker, label in votes[obj]:
            s = bucket_stats[worker][bucket]
            s["total"] += 1.0
            s["correct"] += float(teacher[label]) if 0 <= label < len(teacher) else 0.0

    conditioned: Dict[int, Dict[str, float]] = {}
    all_workers = sorted(set(base_profiles) | set(bucket_stats))
    for worker in all_workers:
        base = base_profiles.get(
            worker,
            {
                "omega": 0.5,
                "bias": 0.0,
                "dominant_rate": 0.5,
                "vote_entropy": 1.0,
                "total": 0.0,
            },
        )
        row = dict(base)
        row["omega_global"] = float(base["omega"])
        for bucket in bucket_names:
            local = bucket_stats.get(worker, {}).get(bucket, {"correct": 0.0, "total": 0.0})
            total = float(local["total"])
            score = (float(local["correct"]) + float(base["omega"]) * shrink_strength) / (total + shrink_strength)
            row[f"omega_{bucket}"] = score
        conditioned[worker] = row
    return conditioned


def build_object_features(
    records: Dict[int, ObjectRecord],
    neighbors: Dict[int, List[Tuple[float, int]]],
    object_ids: List[int],
    train_mask: np.ndarray,
    vectors: Dict[int, np.ndarray] | None = None,
    append_object_vectors: bool = False,
) -> np.ndarray:
    feats: List[List[float]] = []
    for obj in object_ids:
        rec = records[obj]
        first_hop = neighbors[obj]
        k = max(len(first_hop), 1)
        easy_ratio = sum(1 for _, src_obj in first_hop if records[src_obj].bucket == "easy") / k
        amb_ratio = sum(1 for _, src_obj in first_hop if records[src_obj].bucket == "ambiguous") / k
        hard_ratio = sum(1 for _, src_obj in first_hop if records[src_obj].bucket == "hard") / k
        feats.append(
            [
                float(rec.total_votes),
                rec.majority_share,
                rec.runnerup_share,
                rec.gap_norm,
                rec.entropy,
                rec.ncs,
                float(rec.bucket == "easy"),
                float(rec.bucket == "ambiguous"),
                float(rec.bucket == "hard"),
                easy_ratio,
                amb_ratio,
                hard_ratio,
                anchor_strength(rec.bucket),
            ]
        )
    if append_object_vectors:
        if vectors is None:
            raise ValueError("vectors are required when append_object_vectors=True")
        raw_x = np.vstack([vectors[obj] for obj in object_ids]).astype(np.float32)
        x = np.concatenate([np.asarray(feats, dtype=np.float32), raw_x], axis=1)
    else:
        x = np.asarray(feats, dtype=np.float32)
    train_x = x[train_mask]
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (x - mean) / std


def build_worker_features(
    workers: List[int],
    profiles: Dict[int, Dict[str, float]],
    train_worker_ids: Sequence[int],
    variant: str,
) -> np.ndarray:
    omegas = [profiles[w].get("omega_global", profiles[w].get("omega", 0.5)) for w in train_worker_ids if w in profiles]
    dom_rates = [profiles[w]["dominant_rate"] for w in train_worker_ids if w in profiles]
    entropies = [profiles[w]["vote_entropy"] for w in train_worker_ids if w in profiles]
    default_omega = float(sum(omegas) / len(omegas)) if omegas else 0.5
    default_dom_rate = float(sum(dom_rates) / len(dom_rates)) if dom_rates else 0.5
    default_entropy = float(sum(entropies) / len(entropies)) if entropies else 1.0

    feats: List[List[float]] = []
    worker_pos = {worker: idx for idx, worker in enumerate(workers)}
    for worker in workers:
        prof = profiles.get(
            worker,
            {
                "omega": default_omega,
                "omega_global": default_omega,
                "omega_easy": default_omega,
                "omega_ambiguous": default_omega,
                "omega_hard": default_omega,
                "bias": 0.0,
                "dominant_rate": default_dom_rate,
                "vote_entropy": default_entropy,
                "total": 0.0,
            },
        )
        # Supported ``variant`` values (set automatically from ``--ablation-variant``):
        #   ``global``            -- [total, dominant_rate, vote_entropy, omega_global]
        #                            for no_cond_worker_clean / global_worker_reliability.
        #   ``omega_only``        -- [omega_easy, omega_ambiguous, omega_hard]
        #                            for the full DaLC model.
        #   ``omega_global_only`` -- [omega_global]
        #                            for the no_ncs ablation.
        if variant == "global":
            base = [
                prof["total"],
                prof["dominant_rate"],
                prof["vote_entropy"],
                prof.get("omega_global", prof.get("omega", default_omega)),
            ]
        elif variant == "omega_only":
            base = [
                prof.get("omega_easy", prof.get("omega", default_omega)),
                prof.get("omega_ambiguous", prof.get("omega", default_omega)),
                prof.get("omega_hard", prof.get("omega", default_omega)),
            ]
        elif variant == "omega_global_only":
            base = [prof.get("omega_global", prof.get("omega", default_omega))]
        else:
            raise ValueError(f"Unknown worker-feature variant: {variant!r}")
        feats.append(base)

    x = np.asarray(feats, dtype=np.float32)
    train_idxs = [worker_pos[w] for w in train_worker_ids if w in worker_pos]
    train_x = x[train_idxs] if train_idxs else x
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (x - mean) / std


def pick_label_from_weighted_neighbors(
    weighted_neighbors: List[Tuple[float, int]],
    records: Dict[int, ObjectRecord],
    num_classes: int,
    fallback: int,
) -> int:
    dist = weighted_label_distribution(weighted_neighbors, records, num_classes)
    return dist_to_label(dist, fallback=fallback)


def compute_easy_neighbor_pseudo_labels(
    records: Dict[int, ObjectRecord],
    neighbors: Dict[int, List[Tuple[float, int]]],
    object_ids: List[int],
    num_classes: int,
) -> Tuple[np.ndarray, List[str]]:
    pseudo_labels: List[np.ndarray] = []
    strategies: List[str] = []
    for obj in object_ids:
        rec = records[obj]
        if rec.bucket != "easy":
            pseudo_labels.append(majority_one_hot(rec.majority, num_classes))
            strategies.append("not_easy")
            continue
        easy_neighbors = [
            (max(sim, 0.0) * anchor_strength(records[src_obj].bucket), src_obj)
            for sim, src_obj in neighbors[obj]
            if records[src_obj].bucket == "easy"
        ]
        if easy_neighbors:
            pseudo_labels.append(
                normalize_distribution(
                    weighted_label_distribution(easy_neighbors, records, num_classes),
                    fallback_label=rec.majority,
                )
            )
            strategies.append("easy_first_hop_teacher")
        else:
            pseudo_labels.append(majority_one_hot(rec.majority, num_classes))
            strategies.append("self_fallback")
    return np.vstack(pseudo_labels).astype(np.float32), strategies


def compute_ambiguous_neighbor_pseudo_labels(
    records: Dict[int, ObjectRecord],
    neighbors: Dict[int, List[Tuple[float, int]]],
    object_ids: List[int],
    num_classes: int,
) -> Tuple[np.ndarray, List[str]]:
    pseudo_labels: List[np.ndarray] = []
    strategies: List[str] = []
    for obj in object_ids:
        rec = records[obj]
        if rec.bucket != "ambiguous":
            pseudo_labels.append(majority_one_hot(rec.majority, num_classes))
            strategies.append("not_ambiguous")
            continue
        first_hop = neighbors[obj]
        easy_first = [
            (max(sim, 0.0) * anchor_strength(records[src_obj].bucket), src_obj)
            for sim, src_obj in first_hop
            if records[src_obj].bucket == "easy"
        ]
        if len(easy_first) > len(first_hop) / 2:
            pseudo_labels.append(
                normalize_distribution(
                    weighted_label_distribution(easy_first, records, num_classes),
                    fallback_label=rec.majority,
                )
            )
            strategies.append("first_hop_easy_majority")
        else:
            second_hop_easy: List[Tuple[float, int]] = []
            for sim1, src_obj in first_hop:
                if records[src_obj].bucket == "easy":
                    continue
                for sim2, hop2_obj in neighbors[src_obj]:
                    if hop2_obj == obj or records[hop2_obj].bucket != "easy":
                        continue
                    chain = max(sim1, 0.0) * max(sim2, 0.0) * anchor_strength(records[hop2_obj].bucket)
                    second_hop_easy.append((chain, hop2_obj))
            if second_hop_easy:
                pseudo_labels.append(
                    normalize_distribution(
                        weighted_label_distribution(second_hop_easy, records, num_classes),
                        fallback_label=rec.majority,
                    )
                )
                strategies.append("second_hop_easy_bootstrap")
            else:
                pseudo_labels.append(majority_one_hot(rec.majority, num_classes))
                strategies.append("self_fallback")
    return np.vstack(pseudo_labels).astype(np.float32), strategies


def compute_hard_bootstrap_pseudo_target(
    obj: int,
    records: Dict[int, ObjectRecord],
    bootstrap_neighbors: Dict[int, List[Tuple[float, int]]],
    num_classes: int,
) -> np.ndarray:
    rec = records[obj]
    first_hop = bootstrap_neighbors[obj]
    easy_first = [
        (max(sim, 0.0) * anchor_strength(records[src_obj].bucket), src_obj)
        for sim, src_obj in first_hop
        if records[src_obj].bucket == "easy"
    ]
    if len(easy_first) > len(first_hop) / 2:
        return normalize_distribution(
            weighted_label_distribution(easy_first, records, num_classes),
            fallback_label=rec.majority,
        )
    return majority_one_hot(rec.majority, num_classes)


def init_worker_reliability_stats() -> Dict[int, Dict[str, float]]:
    return defaultdict(lambda: {"correct": 0.0, "total": 0.0})


def build_worker_backoff_stats(
    records: Dict[int, ObjectRecord],
    votes: Dict[int, List[Tuple[int, int]]],
    object_ids: List[int],
    train_mask: np.ndarray,
    cluster_of: Dict[int, int],
    hard_pseudo_target: np.ndarray,
) -> Dict[str, Dict]:
    easy_global = init_worker_reliability_stats()
    hard_global = init_worker_reliability_stats()
    easy_cluster = defaultdict(init_worker_reliability_stats)
    hard_cluster = defaultdict(init_worker_reliability_stats)

    for idx, obj in enumerate(object_ids):
        if not train_mask[idx]:
            continue
        rec = records[obj]
        cluster = cluster_of[obj]
        if rec.bucket == "easy":
            teacher = rec.majority
            for worker, label in votes[obj]:
                easy_global[worker]["correct"] += float(label == teacher)
                easy_global[worker]["total"] += 1.0
                easy_cluster[cluster][worker]["correct"] += float(label == teacher)
                easy_cluster[cluster][worker]["total"] += 1.0
        elif rec.bucket == "hard":
            teacher = np.asarray(hard_pseudo_target[idx], dtype=np.float32)
            for worker, label in votes[obj]:
                hard_global[worker]["correct"] += float(teacher[label])
                hard_global[worker]["total"] += 1.0
                hard_cluster[cluster][worker]["correct"] += float(teacher[label])
                hard_cluster[cluster][worker]["total"] += 1.0

    return {
        "easy_global": easy_global,
        "hard_global": hard_global,
        "easy_cluster": easy_cluster,
        "hard_cluster": hard_cluster,
    }


def worker_backoff_score(
    worker: int,
    cluster: int,
    stats: Dict[str, Dict],
    smoothing_strength: float,
    min_support: float,
) -> float:
    easy_global_row = stats["easy_global"].get(worker, {"correct": 0.0, "total": 0.0})
    hard_global_row = stats["hard_global"].get(worker, {"correct": 0.0, "total": 0.0})
    easy_global_score = (easy_global_row["correct"] + 0.5 * smoothing_strength) / (easy_global_row["total"] + smoothing_strength)
    hard_global_score = (hard_global_row["correct"] + 0.5 * smoothing_strength) / (hard_global_row["total"] + smoothing_strength)

    easy_cluster_row = stats["easy_cluster"][cluster].get(worker, {"correct": 0.0, "total": 0.0})
    hard_cluster_row = stats["hard_cluster"][cluster].get(worker, {"correct": 0.0, "total": 0.0})
    easy_cluster_score = (easy_cluster_row["correct"] + easy_global_score * smoothing_strength) / (
        easy_cluster_row["total"] + smoothing_strength
    )
    hard_cluster_score = (hard_cluster_row["correct"] + hard_global_score * smoothing_strength) / (
        hard_cluster_row["total"] + smoothing_strength
    )
    if hard_cluster_row["total"] >= min_support:
        return float(hard_cluster_score)
    return float(easy_cluster_score)


def compute_hard_worker_reliability_scores(
    records: Dict[int, ObjectRecord],
    votes: Dict[int, List[Tuple[int, int]]],
    object_ids: List[int],
    train_mask: np.ndarray,
    hard_pseudo_target: np.ndarray,
    shrink_strength: float = 10.0,
) -> Dict[int, float]:
    agree = Counter()
    total = Counter()
    workers = sorted({worker for obj in object_ids for worker, _ in votes[obj]})
    for idx, obj in enumerate(object_ids):
        if not train_mask[idx] or records[obj].bucket != "hard":
            continue
        teacher = np.asarray(hard_pseudo_target[idx], dtype=np.float32)
        for worker, label in votes[obj]:
            total[worker] += 1
            agree[worker] += float(teacher[label])

    total_votes = sum(total.values())
    prior = (sum(agree.values()) / total_votes) if total_votes else 0.5
    scores: Dict[int, float] = {}
    for worker in workers:
        n = total[worker]
        if n > 0:
            scores[worker] = (agree[worker] + prior * shrink_strength) / (n + shrink_strength)
        else:
            scores[worker] = prior
    return scores


def compute_worker_reliability_targets(
    records: Dict[int, ObjectRecord],
    votes: Dict[int, List[Tuple[int, int]]],
    object_ids: List[int],
    worker_reliability_scores: Dict[int, float],
    num_classes: int,
) -> np.ndarray:
    targets = np.zeros((len(object_ids), num_classes), dtype=np.float32)
    for idx, obj in enumerate(object_ids):
        rec = records[obj]
        label_weights = np.zeros(num_classes, dtype=np.float32)
        for worker, label in votes[obj]:
            score = float(worker_reliability_scores.get(worker, 0.5))
            label_weights[label] += score
        targets[idx] = normalize_distribution(label_weights, fallback_label=rec.majority)
    return targets


def compute_ambiguous_worker_backoff_pseudo_labels(
    records: Dict[int, ObjectRecord],
    votes: Dict[int, List[Tuple[int, int]]],
    object_ids: List[int],
    train_mask: np.ndarray,
    cluster_of: Dict[int, int],
    hard_pseudo_target: np.ndarray,
    num_classes: int,
    smoothing_strength: float = 5.0,
    min_support: float = 1.0,
    stats: Dict[str, Dict] | None = None,
) -> np.ndarray:
    if stats is None:
        stats = build_worker_backoff_stats(
            records=records,
            votes=votes,
            object_ids=object_ids,
            train_mask=train_mask,
            cluster_of=cluster_of,
            hard_pseudo_target=hard_pseudo_target,
        )
    targets = np.zeros((len(object_ids), num_classes), dtype=np.float32)
    for idx, obj in enumerate(object_ids):
        rec = records[obj]
        if rec.bucket != "ambiguous":
            targets[idx] = majority_one_hot(rec.majority, num_classes)
            continue
        cluster = cluster_of[obj]
        label_weights = np.zeros(num_classes, dtype=np.float32)
        for worker, label in votes[obj]:
            score = worker_backoff_score(
                worker=worker,
                cluster=cluster,
                stats=stats,
                smoothing_strength=smoothing_strength,
                min_support=min_support,
            )
            label_weights[label] += float(score)
        targets[idx] = normalize_distribution(label_weights, fallback_label=rec.majority)
    return targets


def compute_neighbor_targets(
    records: Dict[int, ObjectRecord],
    neighbors: Dict[int, List[Tuple[float, int]]],
    object_ids: List[int],
    args: argparse.Namespace,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    neighbor_target_list: List[np.ndarray] = []
    self_lambda_list: List[float] = []
    strategy_list: List[str] = []
    for obj in object_ids:
        rec = records[obj]
        lam = self_vote_lambda(rec.bucket, args)
        first_hop = neighbors[obj]
        easy_count = sum(1 for _, src_obj in first_hop if records[src_obj].bucket == "easy")
        easy_first = [
            (max(sim, 0.0) * anchor_strength(records[src_obj].bucket), src_obj)
            for sim, src_obj in first_hop
            if records[src_obj].bucket == "easy"
        ]
        if rec.bucket in {"easy", "ambiguous"}:
            all_weighted = [
                (max(sim, 0.0) * anchor_strength(records[src_obj].bucket), src_obj)
                for sim, src_obj in first_hop
            ]
            neighbor_dist = weighted_label_distribution(all_weighted, records, num_classes)
            strategy = "all_neighbors"
        elif easy_count > len(first_hop) / 2:
            neighbor_dist = weighted_label_distribution(easy_first, records, num_classes)
            strategy = "first_hop_easy_majority"
        else:
            neighbor_dist = majority_one_hot(rec.majority, num_classes)
            lam = 1.0
            strategy = "self_fallback"

        neighbor_target_list.append(normalize_distribution(neighbor_dist, fallback_label=rec.majority))
        self_lambda_list.append(lam)
        strategy_list.append(strategy)
    return (
        np.vstack(neighbor_target_list).astype(np.float32),
        np.asarray(self_lambda_list, dtype=np.float32),
        strategy_list,
    )