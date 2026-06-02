from __future__ import annotations

import argparse
import importlib
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.aggregation_baseline_utils import compact_ids, discover_dataset_names, load_dataset, write_object_predictions


DEFAULT_CROWDFM_ROOT = Path(__file__).resolve().parent / "source"


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_crowdfm_classes(crowdfm_root: Path):
    src_root = crowdfm_root / "src"
    if not src_root.exists():
        raise FileNotFoundError(
            f"CrowdFM source not found at {crowdfm_root}. "
            "Clone https://github.com/liiuhaao/CrowdFM there or pass --crowdfm-root."
        )
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    try:
        crowd_data_module = importlib.import_module("cfm.data.crowd_data")
        cfm_module = importlib.import_module("cfm.model.CFM")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CrowdFM dependencies are unavailable. Install the CrowdFM dependencies "
            "in this environment, especially torch-geometric."
        ) from exc
    return crowd_data_module.CrowdData, cfm_module.CFM


def build_crowd_data(dataset, crowd_data_cls, dim: int):
    object_to_idx = {object_id: idx for idx, object_id in enumerate(dataset.object_ids)}
    worker_to_idx = {worker_id: idx for idx, worker_id in enumerate(dataset.worker_ids)}
    label_forward, label_backward = compact_ids(dataset.label_values)

    worker_ids = []
    option_ids = []
    task_ids = []
    for object_id, worker_id, label in dataset.answer_rows:
        worker_ids.append(worker_to_idx[worker_id])
        option_ids.append(label_forward[label])
        task_ids.append(object_to_idx[object_id])

    task_y = [
        label_forward[dataset.truth_map[object_id]]
        for object_id in dataset.object_ids
    ]

    data = crowd_data_cls(dim=dim)
    data.triple = torch.tensor([worker_ids, option_ids, task_ids], dtype=torch.long)
    data.task_y = torch.tensor(task_y, dtype=torch.long)
    data.num_worker = len(worker_to_idx)
    data.num_task = len(object_to_idx)
    data.num_option = len(label_backward)
    data.setup()
    return data, label_backward


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: str) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"CrowdFM checkpoint not found at {checkpoint_path}. "
            "Pass --checkpoint to the downloaded checkpoint.pt."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)


@torch.no_grad()
def logits_for_seed(
    seed: int,
    dataset,
    crowd_data_cls,
    cfm_cls,
    checkpoint_path: Path,
    device: str,
    dim: int,
    layer: int,
    head: int,
    dropout: float,
) -> Tuple[np.ndarray, Dict[int, int]]:
    set_seed(seed)
    data, label_backward = build_crowd_data(dataset, crowd_data_cls, dim=dim)
    data = data.to(device)
    model = cfm_cls(dim=dim, device=device, layer=layer, head=head, dropout=dropout).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    out = model(data)
    return out["hat_task_option"].detach().cpu().numpy(), label_backward


def parse_seeds(seeds: Iterable[int]) -> Tuple[int, ...]:
    parsed = tuple(int(seed) for seed in seeds)
    if not parsed:
        raise ValueError("at least one seed is required")
    return parsed


def run_one(
    dataset_name: str,
    data_root: Path,
    output_root: Path,
    crowd_data_cls,
    cfm_cls,
    checkpoint_path: Path,
    device_name: str,
    seeds: Tuple[int, ...],
    dim: int,
    layer: int,
    head: int,
    dropout: float,
) -> None:
    dataset = load_dataset(dataset_name, data_root)
    device = resolve_torch_device(device_name)

    logits_sum = None
    label_backward = None
    for seed in seeds:
        seed_logits, seed_label_backward = logits_for_seed(
            seed=seed,
            dataset=dataset,
            crowd_data_cls=crowd_data_cls,
            cfm_cls=cfm_cls,
            checkpoint_path=checkpoint_path,
            device=device,
            dim=dim,
            layer=layer,
            head=head,
            dropout=dropout,
        )
        logits_sum = seed_logits if logits_sum is None else logits_sum + seed_logits
        label_backward = seed_label_backward

    assert logits_sum is not None
    assert label_backward is not None
    pred_options = np.argmax(logits_sum / len(seeds), axis=1)
    pred_map = {
        object_id: int(label_backward[int(pred_options[idx])])
        for idx, object_id in enumerate(dataset.object_ids)
    }

    output_path = output_root / dataset_name / "object_predictions.csv"
    write_object_predictions(output_path, dataset, pred_map)
    matched = len(dataset.object_ids)
    acc = sum(int(pred_map[obj] == dataset.truth_map[obj]) for obj in dataset.object_ids) / matched if matched else float("nan")
    print(
        f"{dataset_name}\tdevice={device}\tseeds={','.join(str(seed) for seed in seeds)}"
        f"\tacc={acc:.4f}\toutput={output_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CrowdFM on Truth Inference datasets.")
    parser.add_argument("--datasets", nargs="+", default=discover_dataset_names())
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--crowdfm-root", type=Path, default=DEFAULT_CROWDFM_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    crowd_data_cls, cfm_cls = load_crowdfm_classes(args.crowdfm_root.resolve())
    checkpoint_path = args.checkpoint.resolve() if args.checkpoint is not None else args.crowdfm_root.resolve() / "checkpoint.pt"
    seeds = parse_seeds(args.seeds)

    for dataset_name in args.datasets:
        run_one(
            dataset_name=dataset_name,
            data_root=args.data_root,
            output_root=args.output_root,
            crowd_data_cls=crowd_data_cls,
            cfm_cls=cfm_cls,
            checkpoint_path=checkpoint_path,
            device_name=args.device,
            seeds=seeds,
            dim=args.dim,
            layer=args.layer,
            head=args.head,
            dropout=args.dropout,
        )


if __name__ == "__main__":
    main()
