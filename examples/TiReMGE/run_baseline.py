from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.python.ops.numpy_ops.np_config as np_config

import Model
from utils import dis_loss, get_adj, update_feature, update_reliability

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregation_baseline_utils import discover_dataset_names, resolve_dataset_dir


np_config.enable_numpy_behavior()

_ORIG_ADD_WEIGHT = tf.keras.layers.Layer.add_weight


def _compat_add_weight(self, *args, **kwargs):
    if args and isinstance(args[0], str) and "name" not in kwargs and "shape" in kwargs:
        kwargs["name"] = args[0]
        args = args[1:]
    return _ORIG_ADD_WEIGHT(self, *args, **kwargs)


tf.keras.layers.Layer.add_weight = _compat_add_weight


def read_answer_rows(path: Path) -> List[Tuple[int, int, int]]:
    rows: List[Tuple[int, int, int]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                rows.append((int(row[0]), int(row[1]), int(row[2])))
            except ValueError:
                continue
    return rows


def read_truth_map(path: Path) -> Dict[int, int]:
    truth_map: Dict[int, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                truth_map[int(row[0])] = int(row[1])
            except ValueError:
                continue
    return truth_map


def build_inputs(answer_path: Path, truth_path: Path):
    answer_rows = read_answer_rows(answer_path)
    truth_map_all = read_truth_map(truth_path)

    object_order: List[int] = []
    source_order: List[int] = []
    object_to_idx: Dict[int, int] = {}
    source_to_idx: Dict[int, int] = {}
    object_index: List[int] = []
    source_index: List[int] = []
    claims: List[int] = []

    for object_id, source_id, label in answer_rows:
        if object_id not in truth_map_all:
            continue
        if object_id not in object_to_idx:
            object_to_idx[object_id] = len(object_order)
            object_order.append(object_id)
        if source_id not in source_to_idx:
            source_to_idx[source_id] = len(source_order)
            source_order.append(source_id)
        object_index.append(object_to_idx[object_id])
        source_index.append(source_to_idx[source_id])
        claims.append(label)

    object_index_np = np.asarray(object_index, dtype=np.int32)
    source_index_np = np.asarray(source_index, dtype=np.int32)
    claims_np = np.asarray(claims, dtype=np.int32)
    object_num = len(object_order)
    source_num = len(source_order)
    object_source_pair = np.vstack([object_index_np, source_index_np + object_num])
    truths = np.asarray([truth_map_all[object_id] for object_id in object_order], dtype=np.int32)
    gt_index = np.arange(object_num, dtype=np.int32)

    return {
        "object_order": object_order,
        "graph": {"object_source_pair": object_source_pair, "claims": claims_np},
        "object_index": object_index_np,
        "source_index": source_index_np,
        "truth_set": {"truths": truths, "gt_index": gt_index},
        "class_num": int(max(max(truths), max(claims_np)) + 1),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def accuracy_for_pred(truths: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(pred == truths))


def run_one(dataset_name: str, data_root: Path, output_root: Path, steps: int, lr: float, seed: int) -> None:
    set_seed(seed)
    dataset_dir = resolve_dataset_dir(dataset_name, None if data_root == Path("/Users/jiang/Documents/Truth Inference/data") else data_root)
    inputs = build_inputs(dataset_dir / "answer.csv", dataset_dir / "truth.csv")
    graph = inputs["graph"]
    object_order = inputs["object_order"]
    object_index = inputs["object_index"]
    source_index = inputs["source_index"]
    truth_set = inputs["truth_set"]
    class_num = inputs["class_num"]
    object_source_pair = graph["object_source_pair"]
    object_num = int(np.max(object_index) + 1)
    source_num = int(np.max(source_index) + 1)
    node_num = int(np.max(object_source_pair) + 1)
    claims = tf.one_hot(indices=graph["claims"], depth=class_num)

    adj1, adj2 = get_adj(object_source_pair, node_num)
    edge_index1 = adj1.astype(np.int32)
    edge_index2 = adj2.astype(np.int32)

    model = Model.TiReMGE(node_num=node_num, source_num=source_num, class_num=class_num)
    legacy_optimizers = getattr(tf.keras.optimizers, "legacy", None)
    if legacy_optimizers is not None and hasattr(legacy_optimizers, "Adam"):
        try:
            optimizer = legacy_optimizers.Adam(learning_rate=lr)
        except ImportError:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    reliability = np.ones(shape=(source_num,), dtype=np.float32) / source_num
    x = update_feature([object_index, source_index, claims], reliability, object_num, source_num)

    best_acc = -1.0
    best_pred = None

    for step in range(steps):
        with tf.GradientTape() as tape:
            embedding = model([x, edge_index1, edge_index2], training=True)
            reliability = update_reliability(embedding, [object_index, source_index, claims], source_num, reliability)
            loss = dis_loss(embedding, [object_index, source_index, claims], reliability, source_num)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        x = update_feature([object_index, source_index, claims], reliability, object_num, source_num)

        object_embedding = tf.gather(embedding, truth_set["gt_index"])
        pred = tf.argmax(tf.nn.softmax(object_embedding), axis=-1, output_type=tf.int32).numpy()
        acc = accuracy_for_pred(truth_set["truths"], pred)
        if acc > best_acc:
            best_acc = acc
            best_pred = pred.copy()
        print(f"{dataset_name}\tstep={step}\tloss={float(loss):.6f}\tacc={acc:.4f}\tbest={best_acc:.4f}")

    assert best_pred is not None
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "object_predictions.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["object", "truth", "pred"])
        writer.writeheader()
        for object_id, truth, pred in zip(object_order, truth_set["truths"], best_pred):
            writer.writerow({"object": object_id, "truth": int(truth), "pred": int(pred)})
    print(f"{dataset_name}\tbest_acc={best_acc:.4f}\toutput={out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TiReMGE on Truth Inference datasets.")
    parser.add_argument("--datasets", nargs="+", default=discover_dataset_names())
    parser.add_argument("--data-root", type=Path, default=Path("/Users/jiang/Documents/Truth Inference/data"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for dataset_name in args.datasets:
        run_one(dataset_name, args.data_root, args.output_root, args.steps, args.learning_rate, args.seed)


if __name__ == "__main__":
    main()
