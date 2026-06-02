""""
Difficulty-aware label completion pipeline.

Usage::

    python -m dalc.pipeline --dataset-dir labelme --device cpu

The script:
    1. Loads ``answer.csv``/``truth.csv`` and object features (``*.arff``).
    2. Computes the Neighbor Consistency Score (NCS) and assigns each object to the
       easy / ambiguous / hard *subset* (paper Eq.~8, code field ``ObjectRecord.bucket``).
    3. Builds the heterogeneous graph (object-object similarity edges +
       worker-object vote edges).
    4. Initializes missing labels with the difficulty-aware prior
       ``z_{ir}^{(0)}`` (code variable ``q_init``).
    5. Trains the GNN with the completion objective (observed + missing-prior
       losses) plus an auxiliary majority-vote head.
    6. Writes per-subset metrics to ``results/<dataset>/*_metrics.csv`` and the
       completed worker-object matrix to ``--completed-answer-path`` (optional).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

from dalc.ablation import *
from dalc.core import *
from dalc._data_utils import *
from dalc.model import train
from dalc._io_paths import DATA_ROOT as PROJECT_DATA_ROOT, result_path_for_dataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Difficulty-aware label completion for truth inference.")
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory path or dataset name under data/.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--neighbor-metric", choices=["euclidean", "cosine", "heom"], default="cosine")
    parser.add_argument("--easy-threshold", type=float, default=0.8)
    parser.add_argument("--ambiguous-threshold", type=float, default=0.6)
    parser.add_argument("--bucket-strategy", choices=["fixed", "quantile", "beta_mix", "beta_mix_guarded"], default="fixed", help=("How to split objects into easy/ambiguous/hard buckets from NCS. "))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-easy", type=float, default=0.9)
    parser.add_argument("--lambda-ambiguous", type=float, default=0.6)
    parser.add_argument("--lambda-hard", type=float, default=0.2)
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--completion-observed-weight", type=float, default=1.0)
    parser.add_argument("--completion-prior-weight", type=float, default=0.6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-prefix", default="dalc", help=argparse.SUPPRESS)
    parser.add_argument("--completed-answer-path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ablation-variant", choices=list(VARIANTS), default="full")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.seed is None:
        args.seed = FIXED_SEED
    if args.bucket_strategy == "fixed" and args.easy_threshold <= args.ambiguous_threshold:
        raise ValueError("--easy-threshold must be greater than --ambiguous-threshold (only required for --bucket-strategy=fixed)")
    # Auto-derive edge / worker feature modes from the ablation variant.
    if args.ablation_variant == "no_ncs":
        # The model has zero NCS knowledge: drop NCS scalar from object and vote edges,
        # and replace ω^(d) with ω_global on worker.x.
        args.object_edge_feature_mode = "similarity_only"
        args.vote_edge_feature_mode = "label_only"
        args.worker_feature_mode = "omega_global_only"
    elif disables_cond_worker(args.ablation_variant):
        # Section 3.2 ablation: drop ω^(d) from worker.x AND from vote edges.
        args.object_edge_feature_mode = "full"
        args.vote_edge_feature_mode = "drop_per_bucket_omega"
        args.worker_feature_mode = "global"
    elif args.ablation_variant == "no_edge_omega_cond":
        # Table 8 row: keep ω^(d) in worker.x, drop it from vote edges only.
        args.object_edge_feature_mode = "full"
        args.vote_edge_feature_mode = "drop_per_bucket_omega"
        args.worker_feature_mode = "omega_only"
    else:
        # Full DaLC default (also covers no_loss_* variants -- they keep the full
        # architecture and only zero one loss term below).
        args.object_edge_feature_mode = "full"
        args.vote_edge_feature_mode = "drop_omega_global"
        args.worker_feature_mode = "omega_only"
    # Loss-component ablations (paper Table 5).
    if args.ablation_variant == "no_loss_mv":
        args.aux_weight = 0.0
    elif args.ablation_variant == "no_loss_obs":
        args.completion_observed_weight = 0.0
    elif args.ablation_variant == "no_loss_tgt":
        args.completion_prior_weight = 0.0
    return args


def resolve_dataset_dir(dataset_dir: str) -> Path:
    dataset_path = Path(dataset_dir).expanduser()
    if dataset_path.exists():
        return dataset_path.resolve()

    search_roots = (
        PROJECT_DATA_ROOT,
        PROJECT_DATA_ROOT / "simulation",
        PROJECT_DATA_ROOT / "simulation_generated",
    )
    for root in search_roots:
        candidate = root / dataset_path
        if candidate.exists():
            return candidate.resolve()

    searched = ", ".join(str(root) for root in search_roots)
    raise FileNotFoundError(
        f"Dataset path '{dataset_dir}' does not exist and was not found under any supported root: {searched}"
    )

def output_paths(dataset_dir: Path, args: argparse.Namespace) -> Tuple[Path, Path]:
    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    return (
        result_path_for_dataset(dataset_dir, f"{prefix}metrics.csv"),
        result_path_for_dataset(dataset_dir, f"{prefix}predictions.csv"),
    )

def build_completion_data(
    records: Dict[int, ObjectRecord],
    votes: Dict[int, List[Tuple[int, int]]],
    neighbors: Dict[int, List[Tuple[float, int]]],
    hard_bootstrap_neighbors: Dict[int, List[Tuple[float, int]]],
    cluster_of: Dict[int, int],
    object_ids: List[int],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    args: argparse.Namespace,
    num_classes: int,
    object_prior: np.ndarray,
    feature_space: FeatureSpace | None = None,
    vectors: Dict[int, np.ndarray] | None = None,
) -> HeteroData:
    if vectors is None:
        vectors = feature_space.vectors if feature_space is not None else {}
    workers, worker_to_idx, worker_similarity = build_worker_similarity_matrix(
        records,
        votes,
        object_ids,
        cluster_of,
        num_classes,
    )
    difficulty = compute_difficulty_info(records, hard_bootstrap_neighbors, object_ids, num_classes)
    neighbor_target, self_lambda, neighbor_strategy = compute_neighbor_targets(records, neighbors, object_ids, args, num_classes)
    y_majority = np.asarray([records[obj].majority for obj in object_ids], dtype=np.int64)
    hard_pseudo_target = np.asarray(
        [compute_hard_bootstrap_pseudo_target(obj, records, hard_bootstrap_neighbors, num_classes) for obj in object_ids],
        dtype=np.float32,
    )
    worker_backoff_stats = build_worker_backoff_stats(
        records=records,
        votes=votes,
        object_ids=object_ids,
        train_mask=train_mask,
        cluster_of=cluster_of,
        hard_pseudo_target=hard_pseudo_target,
    )
    base_profiles = profile_workers_easy(records, votes, train_mask, object_ids, num_classes)
    if uses_global_ds_reliability(args.ablation_variant):
        global_ds = compute_global_ds_reliability(records, votes, object_ids, train_mask, num_classes)
        base_profiles = {
            worker: {**prof, "omega": float(global_ds.get(worker, prof.get("omega", 0.5)))}
            for worker, prof in base_profiles.items()
        }
        for worker, score in global_ds.items():
            if worker not in base_profiles:
                base_profiles[worker] = {
                    "omega": float(score),
                    "bias": 0.0,
                    "dominant_rate": 0.5,
                    "vote_entropy": 1.0,
                    "total": 0.0,
                }
    if disables_hard_amb_transfer(args.ablation_variant):
        hard_worker_scores = base_reliability_dict(base_profiles)
    else:
        hard_worker_scores = compute_hard_worker_reliability_scores(
            records=records,
            votes=votes,
            object_ids=object_ids,
            train_mask=train_mask,
            hard_pseudo_target=hard_pseudo_target,
        )
    if disables_hard_amb_transfer(args.ablation_variant):
        ambiguous_worker_backoff_target = reliability_weighted_mv(
            records=records,
            votes=votes,
            object_ids=object_ids,
            worker_reliability=hard_worker_scores,
            num_classes=num_classes,
            bucket_filter="ambiguous",
        )
    else:
        ambiguous_worker_backoff_target = compute_ambiguous_worker_backoff_pseudo_labels(
            records=records,
            votes=votes,
            object_ids=object_ids,
            train_mask=train_mask,
            cluster_of=cluster_of,
            hard_pseudo_target=hard_pseudo_target,
            num_classes=num_classes,
            stats=worker_backoff_stats,
        )
    q_init, missing_mask = build_completion_initializer(
        records,
        votes,
        object_ids,
        workers,
        worker_to_idx,
        worker_similarity,
        difficulty,
        object_prior,
        num_classes,
        ambiguous_worker_backoff_target=ambiguous_worker_backoff_target,
    )
    completion_prior = aggregate_missing_completion_prior(q_init, missing_mask, object_ids, records, num_classes)
    worker_reliability_target = compute_worker_reliability_targets(
        records=records,
        votes=votes,
        object_ids=object_ids,
        worker_reliability_scores=hard_worker_scores,
        num_classes=num_classes,
    )
    # Worker conditioning target: self_lambda-weighted blend of MV and neighbor signal
    # (paper's self-neighbor fusion). Replaces the older `combine_pseudo_targets`
    # output, which additionally mixed in bucket-specific completion-prior etc. --
    # an A/B comparison showed the elaborate version did not move the needle.
    conditioning_target = (
        self_lambda[:, None] * object_prior + (1.0 - self_lambda[:, None]) * neighbor_target
    ).astype(np.float32)

    conditioned_profiles = profile_workers_conditioned(
        records=records,
        votes=votes,
        train_mask=train_mask,
        object_ids=object_ids,
        pseudo_target=conditioning_target,
        base_profiles=base_profiles,
    )
    if disables_cond_worker(args.ablation_variant):
        conditioned_profiles = flatten_conditioned_profiles(conditioned_profiles, base_profiles)
    use_mixed_encoder = False
    if feature_space is not None:
        nominal_idx = np.where(feature_space.is_nominal)[0]
        use_mixed_encoder = nominal_idx.size > 0
    object_x = np.zeros((len(object_ids), 0), dtype=np.float32)
    if feature_space is not None and not use_mixed_encoder:
        raw_x = np.vstack([feature_space.vectors[obj] for obj in object_ids]).astype(np.float32)
        train_raw = raw_x[train_mask]
        mean = train_raw.mean(axis=0, keepdims=True)
        std = train_raw.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        object_x = (raw_x - mean) / std
    if feature_space is not None and use_mixed_encoder:
        raw_x = np.vstack([feature_space.vectors[obj] for obj in object_ids]).astype(np.float32)
        numeric_idx = np.where(~feature_space.is_nominal)[0]
        if numeric_idx.size:
            object_numeric_x = raw_x[:, numeric_idx].astype(np.float32)
            train_numeric = object_numeric_x[train_mask]
            mean = train_numeric.mean(axis=0, keepdims=True)
            std = train_numeric.std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            object_numeric_x = (object_numeric_x - mean) / std
        else:
            object_numeric_x = np.zeros((len(object_ids), 0), dtype=np.float32)
        if nominal_idx.size:
            object_nominal_x = raw_x[:, nominal_idx].astype(np.int64)
            nominal_cardinalities = feature_space.nominal_cardinalities[nominal_idx].astype(np.int64)
        else:
            object_nominal_x = np.zeros((len(object_ids), 0), dtype=np.int64)
            nominal_cardinalities = np.zeros(0, dtype=np.int64)
    else:
        object_numeric_x = np.zeros((len(object_ids), 0), dtype=np.float32)
        object_nominal_x = np.zeros((len(object_ids), 0), dtype=np.int64)
        nominal_cardinalities = np.zeros(0, dtype=np.int64)
    train_workers = sorted({worker for idx, obj in enumerate(object_ids) if train_mask[idx] for worker, _ in votes[obj]})
    worker_x = build_worker_features(workers, conditioned_profiles, train_workers, variant=args.worker_feature_mode)
    obj_obj_edge_index, obj_obj_edge_attr = build_object_object_edges(
        records,
        neighbors,
        object_ids,
        difficulty,
        object_edge_feature_mode=args.object_edge_feature_mode,
    )
    worker_obj_edge_index, obj_worker_edge_index, worker_obj_edge_attr, obj_worker_edge_attr = build_vote_edge_features(
        workers,
        worker_to_idx,
        conditioned_profiles,
        records,
        votes,
        object_ids,
        num_classes,
        vote_edge_feature_mode=args.vote_edge_feature_mode,
    )

    observed_worker_idx: List[int] = []
    observed_object_idx: List[int] = []
    observed_label: List[int] = []
    missing_worker_idx: List[int] = []
    missing_object_idx: List[int] = []
    missing_q_init: List[np.ndarray] = []
    for obj_idx, obj in enumerate(object_ids):
        observed_workers = {worker_to_idx[worker] for worker, _ in votes[obj]}
        for worker, label in votes[obj]:
            observed_worker_idx.append(worker_to_idx[worker])
            observed_object_idx.append(obj_idx)
            observed_label.append(label)
        for widx in range(len(workers)):
            if widx in observed_workers:
                continue
            missing_worker_idx.append(widx)
            missing_object_idx.append(obj_idx)
            missing_q_init.append(q_init[obj_idx, widx])

    data = HeteroData()
    data["object"].x = torch.tensor(object_x, dtype=torch.float32)
    data["object"].x_numeric = torch.tensor(object_numeric_x, dtype=torch.float32)
    data["object"].x_nominal = torch.tensor(object_nominal_x, dtype=torch.long)
    data["object"].y_majority = torch.tensor(y_majority, dtype=torch.long)
    data["object"].completion_prior = torch.tensor(completion_prior, dtype=torch.float32)
    data["object"].worker_reliability_target = torch.tensor(worker_reliability_target.argmax(axis=1), dtype=torch.long)
    data["object"].self_lambda = torch.tensor(self_lambda, dtype=torch.float32)
    data["object"].bucket_id = torch.tensor(
        [0 if records[obj].bucket == "easy" else 1 if records[obj].bucket == "ambiguous" else 2 for obj in object_ids],
        dtype=torch.long,
    )
    data["object"].train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data["object"].val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data["object"].test_mask = torch.tensor(test_mask, dtype=torch.bool)

    data["worker"].x = torch.tensor(worker_x, dtype=torch.float32)

    data["object", "similar", "object"].edge_index = torch.tensor(obj_obj_edge_index, dtype=torch.long)
    data["object", "similar", "object"].edge_attr = torch.tensor(obj_obj_edge_attr, dtype=torch.float32)
    data["worker", "votes", "object"].edge_index = torch.tensor(worker_obj_edge_index, dtype=torch.long)
    data["worker", "votes", "object"].edge_attr = torch.tensor(worker_obj_edge_attr, dtype=torch.float32)
    data["object", "rev_votes", "worker"].edge_index = torch.tensor(obj_worker_edge_index, dtype=torch.long)
    data["object", "rev_votes", "worker"].edge_attr = torch.tensor(obj_worker_edge_attr, dtype=torch.float32)

    data.observed_worker_idx = torch.tensor(observed_worker_idx, dtype=torch.long)
    data.observed_object_idx = torch.tensor(observed_object_idx, dtype=torch.long)
    data.observed_label = torch.tensor(observed_label, dtype=torch.long)
    data.missing_worker_idx = torch.tensor(missing_worker_idx, dtype=torch.long)
    data.missing_object_idx = torch.tensor(missing_object_idx, dtype=torch.long)
    # reshape(-1, num_classes) keeps a 2D shape when missing_q_init is empty (fully-dense matrix),
    # otherwise np.asarray([]) yields shape (0,) and breaks downstream broadcasting in soft_cross_entropy.
    data.missing_q_init = torch.tensor(
        np.asarray(missing_q_init, dtype=np.float32).reshape(-1, num_classes),
        dtype=torch.float32,
    )
    data.object_id = torch.tensor(object_ids, dtype=torch.long)
    data.worker_id = torch.tensor(workers, dtype=torch.long)
    data.num_classes = num_classes
    data.object_nominal_cardinalities = torch.tensor(nominal_cardinalities, dtype=torch.long)
    data.object_edge_feature_mode = args.object_edge_feature_mode
    data.vote_edge_feature_mode = args.vote_edge_feature_mode
    data.neighbor_strategy = neighbor_strategy
    data.pseudo_target_mode = ["" for _ in object_ids]
    data.difficulty_teacher_confident = torch.tensor([int(row.teacher_confident) for row in difficulty], dtype=torch.long)
    data.difficulty_teacher_label = torch.tensor([row.teacher_label for row in difficulty], dtype=torch.long)
    data.difficulty_teacher_margin = torch.tensor([row.teacher_margin for row in difficulty], dtype=torch.float32)
    data.difficulty_easy_neighbor_count = torch.tensor([row.easy_neighbor_count for row in difficulty], dtype=torch.long)
    return data


def run_round(
    records: Dict[int, ObjectRecord],
    votes: Dict[int, List[Tuple[int, int]]],
    neighbors: Dict[int, List[Tuple[float, int]]],
    hard_bootstrap_neighbors: Dict[int, List[Tuple[float, int]]],
    semantic_cluster_of: Dict[int, int],
    object_ids: List[int],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    args: argparse.Namespace,
    num_classes: int,
    device: torch.device,
    object_prior: np.ndarray,
    feature_space: FeatureSpace | None = None,
    vectors: Dict[int, np.ndarray] | None = None,
) -> RoundOutput:
    if vectors is None:
        vectors = feature_space.vectors if feature_space is not None else {}
    data = build_completion_data(
        records=records,
        votes=votes,
        neighbors=neighbors,
        hard_bootstrap_neighbors=hard_bootstrap_neighbors,
        cluster_of=semantic_cluster_of,
        object_ids=object_ids,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        args=args,
        num_classes=num_classes,
        feature_space=feature_space,
        vectors=vectors,
        object_prior=object_prior,
    )
    _truth_prob, missing_prob, missing_object_idx = train(
        data=data,
        args=args,
        device=device,
        seed=args.seed,
    )
    completion_vote_dist = build_completed_label_vote_distribution(
        records=records,
        votes=votes,
        object_ids=object_ids,
        num_classes=num_classes,
        missing_object_idx=missing_object_idx,
        missing_prob=missing_prob,
    )
    completion_mv_pred = completion_vote_dist.argmax(axis=1).astype(np.int64)
    return RoundOutput(
        completion_vote_dist=completion_vote_dist,
        completion_mv_pred=completion_mv_pred,
        missing_prob=missing_prob,
        missing_object_idx=missing_object_idx,
        data=data,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    device = torch.device(args.device)
    out_metrics, out_pred = output_paths(dataset_dir, args)

    truth_all = load_truth(dataset_dir / "truth.csv")
    votes_all = load_votes(dataset_dir / "answer.csv")
    object_ids = resolve_object_ids(truth_all, votes_all, "intersection")
    truth = {obj: truth_all[obj] for obj in object_ids if obj in truth_all}
    votes = {obj: votes_all.get(obj, []) for obj in object_ids}
    num_classes = infer_num_classes(truth, votes)
    feature_space: FeatureSpace = load_feature_space(dataset_dir, object_ids)
    vectors = feature_space.vectors
    neighbors = build_global_neighbor_map(object_ids, feature_space, top_k=args.top_k, metric=args.neighbor_metric)
    records = load_records(
        truth,
        votes,
        num_classes,
        neighbors,
        easy_threshold=args.easy_threshold,
        ambiguous_threshold=args.ambiguous_threshold,
        object_ids=object_ids,
        bucket_strategy=args.bucket_strategy,
    )
    object_ids = sorted(records)
    records_eval = records
    if args.ablation_variant in {"no_ncs", "random_buckets"}:
        records = override_records_for_variant(records, args.ablation_variant, seed=args.seed)

    import os
    if os.environ.get("DALC_NO_CLUSTER", "0") == "1":
        semantic_cluster_of = {obj: 0 for obj in object_ids}
    else:
        semantic_cluster_of = assign_semantic_clusters(object_ids, vectors, args.seed)
    hard_bootstrap_neighbors = neighbors

    train_mask, val_mask, test_mask = build_train_val_masks(
        len(object_ids),
        val_frac=0.15,
        seed=args.seed,
    )
    y_true = np.asarray([records[obj].truth for obj in object_ids], dtype=np.int64)
    has_truth_mask = y_true != MISSING_TRUTH
    mv_pred = np.asarray([records[obj].majority for obj in object_ids], dtype=np.int64)
    mv_score = np.vstack([majority_one_hot(records[obj].majority, num_classes) for obj in object_ids]).astype(np.float32)
    round_output = run_round(
        records=records,
        votes=votes,
        neighbors=neighbors,
        hard_bootstrap_neighbors=hard_bootstrap_neighbors,
        semantic_cluster_of=semantic_cluster_of,
        object_ids=object_ids,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        args=args,
        num_classes=num_classes,
        feature_space=feature_space,
        vectors=vectors,
        device=device,
        object_prior=mv_score.copy(),
    )

    eval_mask = has_truth_mask.copy()
    metric_rows: List[Dict[str, object]] = []
    metric_rows.extend(bucket_metrics_rows("mv", object_ids, records_eval, y_true, mv_pred, mv_score, num_classes, eval_mask=eval_mask))
    metric_rows.extend(
        bucket_metrics_rows(
            "completion_mv",
            object_ids,
            records_eval,
            y_true,
            round_output.completion_mv_pred,
            round_output.completion_vote_dist,
            num_classes,
            eval_mask=eval_mask,
        )
    )

    pred_rows: List[Dict[str, object]] = []
    split_role = np.full(len(object_ids), "train", dtype=object)
    split_role[val_mask] = "val"
    final_data = round_output.data
    teacher_confident = final_data.difficulty_teacher_confident.numpy()
    teacher_label = final_data.difficulty_teacher_label.numpy()
    teacher_margin = final_data.difficulty_teacher_margin.numpy()
    teacher_easy_neighbors = final_data.difficulty_easy_neighbor_count.numpy()
    for idx, obj in enumerate(object_ids):
        pred_rows.append(
            {
                "object": obj,
                "truth": records_eval[obj].truth if has_truth_mask[idx] else "",
                "has_truth": int(has_truth_mask[idx]),
                "split": str(split_role[idx]),
                "bucket": records_eval[obj].bucket,
                "ncs": f"{records_eval[obj].ncs:.10f}",
                "majority": records_eval[obj].majority,
                "teacher_label": int(teacher_label[idx]),
                "teacher_confident": int(teacher_confident[idx]),
                "teacher_margin": float(teacher_margin[idx]),
                "teacher_easy_neighbor_count": int(teacher_easy_neighbors[idx]),
                "mv_pred": int(mv_pred[idx]),
                "completion_mv_pred": int(round_output.completion_mv_pred[idx]),
                "completion_mv_conf": float(np.max(round_output.completion_vote_dist[idx])),
            }
        )

    write_csv(out_metrics, metric_rows)
    write_csv(out_pred, pred_rows)
    print(f"wrote metrics to {out_metrics}", flush=True)
    print(f"wrote predictions to {out_pred}", flush=True)
    for row in metric_rows:
        if row["subset"] == "all":
            acc = float(row["accuracy"]) if row["accuracy"] else float("nan")
            f1 = float(row["f1"]) if row["f1"] else float("nan")
            auc = float(row["auc"]) if row["auc"] else float("nan")
            print(
                f"{str(row['method']):>16s} "
                f"acc={acc:.4f} "
                f"f1={f1:.4f} "
                f"auc={auc:.4f}",
                flush=True,
            )
    if args.completed_answer_path is not None:
        completed_rows: List[Tuple[int, int, int]] = []
        with (dataset_dir / "answer.csv").open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.reader(handle):
                if len(row) < 3:
                    continue
                try:
                    completed_rows.append((int(row[0]), int(row[1]), int(row[2])))
                except ValueError:
                    continue
        data = round_output.data
        worker_ids = data.worker_id.cpu().numpy().tolist()
        missing_worker_idx = data.missing_worker_idx.cpu().numpy()
        for row_idx, object_pos in enumerate(round_output.missing_object_idx):
            object_id = int(object_ids[int(object_pos)])
            worker_id = int(worker_ids[int(missing_worker_idx[row_idx])])
            label = int(np.argmax(round_output.missing_prob[row_idx]))
            completed_rows.append((object_id, worker_id, label))
        args.completed_answer_path.parent.mkdir(parents=True, exist_ok=True)
        with args.completed_answer_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["object", "worker", "label"])
            for object_id, worker_id, label in sorted(completed_rows):
                writer.writerow([object_id, worker_id, label])
        print(f"wrote completed answers to {args.completed_answer_path}", flush=True)


if __name__ == "__main__":
    main()
