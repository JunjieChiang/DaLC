"""DaLC + 7 aggregators on 5 real datasets.

For each dataset:
  1. Runs ``dalc.pipeline`` to produce a densified worker-object matrix.
  2. Runs each downstream aggregator (MV, DS, GTIC, IWBVT, TiReMGE, HyperLM,
     CrowdFM) on the DaLC-completed answer.csv.
  3. Reports per-aggregator overall + easy/ambiguous/hard accuracy, using
     DaLC's bucket assignments from its predictions CSV.

Usage::

    python experiments/main_experiment.py
    python experiments/main_experiment.py --datasets income
    python experiments/main_experiment.py --aggregators MV DS
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dalc._io_paths import RESULTS_ROOT, result_path_for_dataset  # noqa: E402
from dalc.pipeline import resolve_dataset_dir  # noqa: E402

DEFAULT_DATASETS: Tuple[str, ...] = ("labelme", "ruters", "leaves", "music_genre", "income")
DEFAULT_AGGREGATORS: Tuple[str, ...] = ("MV", "DS", "GTIC", "IWBVT", "TiReMGE", "HyperLM", "CrowdFM")

SUMMARY_PATH = RESULTS_ROOT / "_main_summary" / "main_experiment_dalc.csv"
STAGING_ROOT = _REPO_ROOT / "outputs" / "main_experiment" / "completed_answers" / "DaLC"
AGG_OUTPUT_ROOT = _REPO_ROOT / "outputs" / "main_experiment" / "aggregation_predictions" / "DaLC"


def output_prefix_for(seed: int) -> str:
    return f"main_dalc_seed{seed}"


def dalc_metrics_path(dataset_name: str, seed: int) -> Path:
    filename = f"{output_prefix_for(seed)}_metrics.csv"
    return result_path_for_dataset(resolve_dataset_dir(dataset_name), filename, create_parent=False)


def dalc_predictions_path(dataset_name: str, seed: int) -> Path:
    filename = f"{output_prefix_for(seed)}_predictions.csv"
    return result_path_for_dataset(resolve_dataset_dir(dataset_name), filename, create_parent=False)


def run_dalc(python: str, dataset: str, seed: int, device: str, staging_answer_path: Path) -> int:
    cmd = [
        python, "-m", "dalc.pipeline",
        "--dataset-dir", dataset,
        "--device", device,
        "--seed", str(seed),
        "--output-prefix", output_prefix_for(seed),
        "--completed-answer-path", str(staging_answer_path),
    ]
    print(">>>", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def stage_dataset(dataset: str, staging_dir: Path) -> None:
    """Copy truth.csv from the original dataset to the staging dir; answer.csv is
    expected to have been written by DaLC's --completed-answer-path."""
    src = resolve_dataset_dir(dataset)
    staging_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "truth.csv", staging_dir / "truth.csv")


def build_aggregator_cmd(
    aggregator: str,
    dataset: str,
    python: str,
    staging_root: Path,
    feature_root: Path,
    output_root: Path,
) -> Tuple[List[str], Dict[str, str] | None]:
    out = output_root / aggregator
    env: Dict[str, str] | None = None
    if aggregator in {"MV", "DS", "GTIC"}:
        cmd = [
            python, str(_REPO_ROOT / "examples" / "run_java_aggregation.py"),
            "--method", aggregator,
            "--datasets", dataset,
            "--data-root", str(staging_root),
            "--feature-root", str(feature_root),
            "--output-root", str(out),
        ]
    elif aggregator == "IWBVT":
        cmd = [
            python, str(_REPO_ROOT / "examples" / "IWBVT" / "run_java_baseline.py"),
            "--datasets", dataset,
            "--data-root", str(staging_root),
            "--feature-root", str(feature_root),
            "--output-root", str(out),
        ]
    elif aggregator == "TiReMGE":
        cmd = [
            python, str(_REPO_ROOT / "examples" / "TiReMGE" / "run_baseline.py"),
            "--datasets", dataset,
            "--data-root", str(staging_root),
            "--output-root", str(out),
            "--steps", "150",
            "--learning-rate", "0.01",
            "--seed", "42",
        ]
        mpl_dir = out / ".matplotlib"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, MPLCONFIGDIR=str(mpl_dir))
    elif aggregator == "HyperLM":
        cmd = [
            python, str(_REPO_ROOT / "examples" / "HyperLM" / "run_baseline.py"),
            "--datasets", dataset,
            "--data-root", str(staging_root),
            "--output-root", str(out),
            "--device", "cpu",
        ]
    elif aggregator == "CrowdFM":
        cmd = [
            python, str(_REPO_ROOT / "examples" / "CrowdFM" / "run_baseline.py"),
            "--datasets", dataset,
            "--data-root", str(staging_root),
            "--output-root", str(out),
            "--device", "cpu",
            "--seeds", "42",
        ]
    else:
        raise ValueError(f"unsupported aggregator: {aggregator}")
    return cmd, env


def run_aggregator(aggregator: str, dataset: str, cmd: List[str], env: Dict[str, str] | None) -> int:
    print(f"[{aggregator}] {dataset} >>>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env).returncode


def read_predictions(path: Path) -> Dict[int, int]:
    out: Dict[int, int] = {}
    with path.open(newline="") as h:
        for row in csv.DictReader(h):
            if row.get("pred") in {"", None}:
                continue
            out[int(row["object"])] = int(row["pred"])
    return out


def read_bucket_map(predictions_csv: Path) -> Dict[int, str]:
    """Read object→bucket mapping from DaLC's predictions CSV."""
    out: Dict[int, str] = {}
    with predictions_csv.open(newline="") as h:
        for row in csv.DictReader(h):
            out[int(row["object"])] = row["bucket"]
    return out


def read_truth(dataset: str) -> Dict[int, int]:
    src = resolve_dataset_dir(dataset)
    out: Dict[int, int] = {}
    with (src / "truth.csv").open(newline="", encoding="utf-8-sig") as h:
        for i, row in enumerate(csv.reader(h)):
            if len(row) < 2:
                continue
            if i == 0 and not row[0].lstrip("-").isdigit():
                continue
            try:
                out[int(row[0])] = int(row[1])
            except ValueError:
                continue
    return out


def bucket_accuracy(pred: Dict[int, int], truth: Dict[int, int], buckets: Dict[int, str], bucket: str | None) -> Tuple[float, int]:
    matched = [o for o in truth if o in pred and (bucket is None or buckets.get(o) == bucket)]
    if not matched:
        return float("nan"), 0
    acc = sum(pred[o] == truth[o] for o in matched) / len(matched)
    return acc, len(matched)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--aggregators", nargs="+", default=list(DEFAULT_AGGREGATORS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--skip-dalc", action="store_true", help="Skip DaLC run; reuse existing completed answer.csv.")
    p.add_argument("--aggregate-only", action="store_true", help="Skip both DaLC and aggregator runs; just aggregate.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, object]] = []
    failures: List[str] = []

    for ds in args.datasets:
        staging_dir = STAGING_ROOT / ds
        staging_answer = staging_dir / "answer.csv"
        if not args.aggregate_only and not args.skip_dalc:
            stage_dataset(ds, staging_dir)
            if run_dalc(args.python, ds, args.seed, args.device, staging_answer) != 0:
                failures.append(f"DaLC/{ds}")
                continue
        if not staging_answer.exists():
            print(f"!!! missing completed answer.csv for {ds}: {staging_answer}", flush=True)
            failures.append(f"DaLC/{ds}/missing")
            continue

        for agg in args.aggregators:
            if not args.aggregate_only:
                cmd, env = build_aggregator_cmd(
                    aggregator=agg,
                    dataset=ds,
                    python=args.python,
                    staging_root=STAGING_ROOT,
                    feature_root=resolve_dataset_dir(ds).parent,
                    output_root=AGG_OUTPUT_ROOT,
                )
                if run_aggregator(agg, ds, cmd, env) != 0:
                    failures.append(f"{agg}/{ds}")
                    continue

            pred_path = AGG_OUTPUT_ROOT / agg / ds / "object_predictions.csv"
            if not pred_path.exists():
                print(f"!!! missing prediction CSV for {agg}/{ds}: {pred_path}", flush=True)
                failures.append(f"{agg}/{ds}/missing")
                continue

            pred = read_predictions(pred_path)
            truth = read_truth(ds)
            buckets = read_bucket_map(dalc_predictions_path(ds, args.seed))

            acc_all, n_all = bucket_accuracy(pred, truth, buckets, None)
            acc_easy, n_easy = bucket_accuracy(pred, truth, buckets, "easy")
            acc_amb, n_amb = bucket_accuracy(pred, truth, buckets, "ambiguous")
            acc_hard, n_hard = bucket_accuracy(pred, truth, buckets, "hard")

            rows.append({
                "aggregator": agg,
                "dataset": ds,
                "easy": f"{acc_easy * 100:.2f}",
                "ambiguous": f"{acc_amb * 100:.2f}",
                "hard": f"{acc_hard * 100:.2f}",
                "all": f"{acc_all * 100:.2f}",
                "n_easy": n_easy,
                "n_ambiguous": n_amb,
                "n_hard": n_hard,
                "n_all": n_all,
            })
            print(
                f"[result] {agg:8s} {ds:12s}  easy={acc_easy*100:6.2f}  "
                f"amb={acc_amb*100:6.2f}  hard={acc_hard*100:6.2f}  all={acc_all*100:6.2f}",
                flush=True,
            )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="") as h:
        writer = csv.DictWriter(
            h,
            fieldnames=["aggregator", "dataset", "easy", "ambiguous", "hard", "all",
                        "n_easy", "n_ambiguous", "n_hard", "n_all"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {SUMMARY_PATH}", flush=True)

    # Pretty-print: aggregator × dataset table of overall accuracy.
    print(f"\n=== Overall accuracy (%) — DaLC + aggregator ===\n", flush=True)
    hdr = f"{'aggregator':10s} | " + " | ".join(f"{d:>12s}" for d in args.datasets) + f" | {'avg':>6s}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for agg in args.aggregators:
        cells = []
        vals: List[float] = []
        for ds in args.datasets:
            r = next((r for r in rows if r["aggregator"] == agg and r["dataset"] == ds), None)
            if r is None:
                cells.append("")
                continue
            cells.append(str(r["all"]))
            try:
                vals.append(float(r["all"]))
            except (TypeError, ValueError):
                pass
        avg = f"{sum(vals)/len(vals):.2f}" if vals else ""
        print(f"{agg:10s} | " + " | ".join(f"{c:>12s}" for c in cells) + f" | {avg:>6s}", flush=True)

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", flush=True)


if __name__ == "__main__":
    main()
