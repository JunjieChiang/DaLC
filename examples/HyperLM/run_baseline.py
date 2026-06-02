from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Dict

import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.aggregation_baseline_utils import compact_ids, discover_dataset_names, load_dataset, write_object_predictions


def resolve_torch_device(device_name: str) -> str:
    requested = device_name.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested device 'cuda' is unavailable on this machine")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise ValueError("requested device 'mps' is unavailable on this machine")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported torch device: {device_name}")
    return requested


def load_hyperlm_class():
    orig_torch_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_torch_load(*args, **kwargs)

    torch.load = patched_load
    try:
        module = importlib.import_module("hyperlm")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "hyperlm is not installed in the current Python environment. "
            "Install it with `pip install hyperlm` in the same environment you use to run this script."
        ) from exc
    return module.HyperLabelModel


def build_label_matrix(dataset_name: str, data_root: Path) -> tuple[np.ndarray, Dict[int, int], Dict[int, int], object]:
    dataset = load_dataset(dataset_name, data_root)
    object_to_idx = {object_id: idx for idx, object_id in enumerate(dataset.object_ids)}
    worker_to_idx = {worker_id: idx for idx, worker_id in enumerate(dataset.worker_ids)}
    label_forward, label_backward = compact_ids(dataset.label_values)
    X = np.full((len(dataset.object_ids), len(dataset.worker_ids)), -1, dtype=np.int64)
    for object_id, worker_id, label in dataset.answer_rows:
        X[object_to_idx[object_id], worker_to_idx[worker_id]] = label_forward[label]
    return X, label_backward, object_to_idx, dataset


def run_one(dataset_name: str, data_root: Path, output_root: Path, device_name: str) -> None:
    X, label_backward, object_to_idx, dataset = build_label_matrix(dataset_name, data_root)
    HyperLabelModel = load_hyperlm_class()
    device = resolve_torch_device(device_name)
    model = HyperLabelModel(device=device)
    try:
        pred_compact = np.asarray(model.infer(X), dtype=np.int64)
    except RuntimeError as exc:
        if device != "cpu" and "out of memory" in str(exc).lower():
            print(f"[HyperLM] {dataset_name}: {device} OOM, retrying on cpu", flush=True)
            device = "cpu"
            model = HyperLabelModel(device=device)
            pred_compact = np.asarray(model.infer(X), dtype=np.int64)
        else:
            raise
    pred_map = {
        object_id: int(label_backward[int(pred_compact[object_to_idx[object_id]])])
        for object_id in dataset.object_ids
    }
    output_dir = output_root / dataset_name
    output_path = output_dir / "object_predictions.csv"
    write_object_predictions(output_path, dataset, pred_map)
    matched = len(dataset.object_ids)
    acc = sum(int(pred_map[obj] == dataset.truth_map[obj]) for obj in dataset.object_ids) / matched if matched else float("nan")
    print(f"{dataset_name}\tdevice={device}\tacc={acc:.4f}\toutput={output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HyperLM on Truth Inference datasets.")
    parser.add_argument("--datasets", nargs="+", default=discover_dataset_names())
    parser.add_argument("--data-root", type=Path, default=Path("/Users/jiang/Documents/Truth Inference/data"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    for dataset_name in args.datasets:
        run_one(dataset_name, args.data_root, args.output_root, args.device)


if __name__ == "__main__":
    main()
