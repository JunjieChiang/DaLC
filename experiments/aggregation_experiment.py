"""
Run label aggregation baselines directly on the original datasets.

Example:
    python aggregation_experiment.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
EXPERIMENT_ROOT = PROJECT_ROOT / "outputs" / "aggregation_experiment"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "_aggregation_summary" / "aggregation_results.csv"
DEFAULT_OUTPUT_TEX = PROJECT_ROOT / "results" / "_aggregation_summary" / "aggregation_results.tex"
DEFAULT_DATASETS = ["labelme", "ruters", "leaves", "music_genre", "income"]
DEFAULT_AGGREGATION_MODELS = ["MV", "DS", "GTIC", "IWBVT", "TiReMGE", "HyperLM", "CrowdFM"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run aggregation-only experiments and export a LaTeX accuracy table."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--aggregation-methods", nargs="+", default=DEFAULT_AGGREGATION_MODELS)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_OUTPUT_TEX)
    parser.add_argument("--python", default=sys.executable, help="Python executable used for subprocess baselines.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ds-iterations", type=int, default=20)
    parser.add_argument("--tiremge-steps", type=int, default=150)
    parser.add_argument("--tiremge-learning-rate", type=float, default=1e-2)
    parser.add_argument("--hyperlm-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--crowdfm-root", type=Path, default=PROJECT_ROOT / "examples" / "CrowdFM" / "source")
    parser.add_argument("--crowdfm-checkpoint", type=Path, default=None)
    parser.add_argument("--crowdfm-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--crowdfm-seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--java-force-recompile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining method/dataset pairs if one aggregation command fails.",
    )
    return parser.parse_args()


def read_truth_map(path: Path) -> Dict[int, int]:
    truth_map: Dict[int, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            try:
                truth_map[int(row[0])] = int(row[1])
            except ValueError:
                continue
    return truth_map


def read_answer_rows(path: Path) -> List[tuple[int, int, int]]:
    rows: List[tuple[int, int, int]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            try:
                rows.append((int(row[0]), int(row[1]), int(row[2])))
            except ValueError:
                continue
    return rows


def has_arff_like_file(dataset_dir: Path) -> bool:
    return any(path.suffix.lower() in {".arff", ".arffx"} for path in dataset_dir.rglob("*") if path.is_file())


def resolve_dataset_dir(dataset_name: str) -> Path:
    for root in (DATA_ROOT, DATA_ROOT / "simulation", DATA_ROOT / "simulation_generated"):
        dataset_dir = root / dataset_name
        if dataset_dir.exists():
            return dataset_dir.resolve()
    raise FileNotFoundError(f"Dataset '{dataset_name}' not found under data/, data/simulation/, or data/simulation_generated/")


def majority_vote_predictions(answer_rows: List[tuple[int, int, int]]) -> Dict[int, int]:
    counts: Dict[int, Dict[int, int]] = {}
    for object_id, _worker_id, label in answer_rows:
        label_counts = counts.setdefault(object_id, {})
        label_counts[label] = label_counts.get(label, 0) + 1

    pred_map: Dict[int, int] = {}
    for object_id, label_counts in counts.items():
        max_count = max(label_counts.values())
        pred_map[object_id] = min(label for label, count in label_counts.items() if count == max_count)
    return pred_map


def dawid_skene_predictions(
    answer_rows: List[tuple[int, int, int]],
    truth_map: Dict[int, int],
    max_iters: int = 100,
    tol: float = 1e-6,
) -> Dict[int, int]:
    object_ids = sorted(truth_map)
    worker_ids = sorted({worker for _obj, worker, _label in answer_rows})
    label_values = sorted({label for _obj, _worker, label in answer_rows} | set(truth_map.values()))

    object_to_idx = {obj: idx for idx, obj in enumerate(object_ids)}
    worker_to_idx = {worker: idx for idx, worker in enumerate(worker_ids)}
    label_to_idx = {label: idx for idx, label in enumerate(label_values)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    num_items = len(object_ids)
    num_workers = len(worker_ids)
    num_classes = len(label_values)

    obs_by_item: List[List[tuple[int, int]]] = [[] for _ in range(num_items)]
    vote_counts = [[0.0] * num_classes for _ in range(num_items)]
    for object_id, worker_id, label in answer_rows:
        if object_id not in object_to_idx:
            continue
        item_idx = object_to_idx[object_id]
        worker_idx = worker_to_idx[worker_id]
        label_idx = label_to_idx[label]
        obs_by_item[item_idx].append((worker_idx, label_idx))
        vote_counts[item_idx][label_idx] += 1.0

    posterior = []
    for counts in vote_counts:
        total = sum(counts) + num_classes
        posterior.append([(count + 1.0) / total for count in counts])

    worker_cm = [
        [[1.0 / num_classes for _ in range(num_classes)] for _ in range(num_classes)]
        for _ in range(num_workers)
    ]
    prev_ll = None

    for _ in range(max_iters):
        class_prior = [0.0] * num_classes
        for post in posterior:
            for class_idx, value in enumerate(post):
                class_prior[class_idx] += value
        class_prior = [value / num_items for value in class_prior]

        numer = [
            [[1.0 for _ in range(num_classes)] for _ in range(num_classes)]
            for _ in range(num_workers)
        ]
        denom = [[float(num_classes) for _ in range(num_classes)] for _ in range(num_workers)]
        for item_idx, obs in enumerate(obs_by_item):
            post = posterior[item_idx]
            for worker_idx, observed_label in obs:
                for true_label in range(num_classes):
                    numer[worker_idx][true_label][observed_label] += post[true_label]
                    denom[worker_idx][true_label] += post[true_label]
        for worker_idx in range(num_workers):
            for true_label in range(num_classes):
                denom_value = denom[worker_idx][true_label]
                worker_cm[worker_idx][true_label] = [
                    value / denom_value for value in numer[worker_idx][true_label]
                ]

        ll = 0.0
        new_posterior = []
        for obs in obs_by_item:
            logp = [math.log(max(prior, 1e-12)) for prior in class_prior]
            for worker_idx, observed_label in obs:
                for true_label in range(num_classes):
                    logp[true_label] += math.log(max(worker_cm[worker_idx][true_label][observed_label], 1e-12))
            max_logp = max(logp)
            probs = [math.exp(value - max_logp) for value in logp]
            prob_sum = sum(probs)
            new_posterior.append([value / prob_sum for value in probs])
            ll += max_logp + math.log(prob_sum)

        posterior = new_posterior
        if prev_ll is not None and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    pred_map: Dict[int, int] = {}
    for object_id, post in zip(object_ids, posterior):
        pred_map[object_id] = idx_to_label[max(range(num_classes), key=lambda idx: post[idx])]
    return pred_map


def read_prediction_csv(path: Path) -> Dict[int, int]:
    pred_map: Dict[int, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            pred_token = row.get("pred", "")
            if pred_token in {"", None}:
                continue
            pred_map[int(row["object"])] = int(pred_token)
    return pred_map


def write_prediction_csv(path: Path, truth_map: Dict[int, int], pred_map: Dict[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["object", "truth", "pred"])
        writer.writeheader()
        for object_id in sorted(truth_map):
            writer.writerow(
                {
                    "object": int(object_id),
                    "truth": int(truth_map[object_id]),
                    "pred": int(pred_map[object_id]) if object_id in pred_map else "",
                }
            )


def count_accuracy(pred_map: Dict[int, int], truth_map: Dict[int, int]) -> float:
    matched = [object_id for object_id in sorted(truth_map) if object_id in pred_map]
    if not matched:
        return float("nan")
    return float(sum(int(pred_map[obj] == truth_map[obj]) for obj in matched) / len(matched))


def format_accuracy(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.10f}"


def format_percent(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "--"
    return f"{value * 100:.1f}"


def format_latex_percent(value: float | None) -> str:
    formatted = format_percent(value)
    if formatted == "--":
        return formatted
    return formatted + r"\%"


def latex_escape(value: str) -> str:
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_")


def write_results_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(path: Path, datasets: List[str], methods: List[str], acc: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    column_spec = "l" + "c" * len(datasets)
    header = "Aggregation Model & " + " & ".join(latex_escape(dataset) for dataset in datasets) + r" \\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Accuracy of direct label aggregation on the default datasets.}",
        r"\label{tab:aggregation-only}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for method in methods:
        values = [format_latex_percent(acc.get(method, {}).get(dataset)) for dataset in datasets]
        lines.append(latex_escape(method) + " & " + " & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def print_latex_table(datasets: List[str], methods: List[str], acc: Dict[str, Dict[str, float]]) -> None:
    column_spec = "l" + "c" * len(datasets)
    print(r"\begin{tabular}{" + column_spec + "}")
    print(r"\toprule")
    print("Aggregation Model & " + " & ".join(latex_escape(dataset) for dataset in datasets) + r" \\")
    print(r"\midrule")
    for method in methods:
        values = [format_latex_percent(acc.get(method, {}).get(dataset)) for dataset in datasets]
        print(latex_escape(method) + " & " + " & ".join(values) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def run_one(method: str, dataset_name: str, output_root: Path, args: argparse.Namespace) -> Dict[int, int]:
    script_map = {
        "MV": PROJECT_ROOT / "examples" / "run_java_aggregation.py",
        "DS": PROJECT_ROOT / "examples" / "run_java_aggregation.py",
        "GTIC": PROJECT_ROOT / "examples" / "run_java_aggregation.py",
        "IWBVT": PROJECT_ROOT / "examples" / "IWBVT" / "run_java_baseline.py",
        "TiReMGE": PROJECT_ROOT / "examples" / "TiReMGE" / "run_baseline.py",
        "HyperLM": PROJECT_ROOT / "examples" / "HyperLM" / "run_baseline.py",
        "CrowdFM": PROJECT_ROOT / "examples" / "CrowdFM" / "run_baseline.py",
    }
    if method not in script_map:
        raise ValueError(f"unsupported aggregation method: {method}")

    method_output_root = output_root / method
    output_path = method_output_root / dataset_name / "object_predictions.csv"
    if args.skip_existing and output_path.exists():
        print(f"[aggregation:skip] {method} on {dataset_name}", flush=True)
        return read_prediction_csv(output_path)

    dataset_dir = resolve_dataset_dir(dataset_name)
    if method in {"MV", "DS"} and not has_arff_like_file(dataset_dir):
        print(f"[aggregation:python] {method} on {dataset_name}", flush=True)
        truth_map = read_truth_map(dataset_dir / "truth.csv")
        answer_rows = read_answer_rows(dataset_dir / "answer.csv")
        pred_map = (
            majority_vote_predictions(answer_rows)
            if method == "MV"
            else dawid_skene_predictions(answer_rows, truth_map, max_iters=args.ds_iterations)
        )
        write_prediction_csv(output_path, truth_map, pred_map)
        return pred_map
    if method in {"GTIC", "IWBVT"} and not has_arff_like_file(dataset_dir):
        raise FileNotFoundError(f"{method} requires an ARFF/ARFFX feature file under {dataset_dir}")

    print(f"[aggregation] {method} on {dataset_name}", flush=True)
    cmd = [args.python, str(script_map[method]), "--datasets", dataset_name, "--output-root", str(method_output_root)]
    env = None
    if method in {"MV", "DS", "GTIC"}:
        cmd.extend(
            [
                "--method",
                method,
                "--data-root",
                    str(dataset_dir.parent),
                "--feature-root",
                    str(dataset_dir.parent),
                "--ds-iterations",
                str(args.ds_iterations),
            ]
        )
        if args.java_force_recompile:
            cmd.append("--force-recompile")
    elif method == "IWBVT":
        cmd.extend(["--data-root", str(dataset_dir.parent), "--feature-root", str(dataset_dir.parent)])
        if args.java_force_recompile:
            cmd.append("--force-recompile")
    elif method == "TiReMGE":
        cmd.extend(
            [
                "--data-root",
                str(dataset_dir.parent),
                "--steps",
                str(args.tiremge_steps),
                "--learning-rate",
                str(args.tiremge_learning_rate),
                "--seed",
                str(args.seed),
            ]
        )
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(args.experiment_root / ".matplotlib")
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    elif method == "HyperLM":
        cmd.extend(["--data-root", str(dataset_dir.parent), "--device", args.hyperlm_device])
    elif method == "CrowdFM":
        cmd.extend(
            [
                "--data-root",
                str(dataset_dir.parent),
                "--crowdfm-root",
                str(args.crowdfm_root),
                "--device",
                args.crowdfm_device,
                "--seeds",
                *[str(seed) for seed in args.crowdfm_seeds],
            ]
        )
        if args.crowdfm_checkpoint is not None:
            cmd.extend(["--checkpoint", str(args.crowdfm_checkpoint)])

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)
    return read_prediction_csv(method_output_root / dataset_name / "object_predictions.csv")


def main() -> None:
    args = parse_args()
    args.experiment_root = args.experiment_root.resolve()
    args.output_csv = args.output_csv.resolve()
    args.output_tex = args.output_tex.resolve()

    random.seed(args.seed)

    prediction_root = args.experiment_root / "aggregation_predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    acc_by_method: Dict[str, Dict[str, float]] = {method: {} for method in args.aggregation_methods}

    for method in args.aggregation_methods:
        for dataset_name in args.datasets:
            dataset_dir = resolve_dataset_dir(dataset_name)
            truth_map = read_truth_map(dataset_dir / "truth.csv")
            prediction_path = prediction_root / method / dataset_name / "object_predictions.csv"
            try:
                pred_map = run_one(method, dataset_name, prediction_root, args)
                acc = count_accuracy(pred_map, truth_map)
                write_prediction_csv(prediction_path, truth_map, pred_map)
                error = ""
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                pred_map = {}
                acc = float("nan")
                error = f"{type(exc).__name__}: {exc}"
                print(f"[error] {method} on {dataset_name}: {error}", flush=True)

            acc_by_method[method][dataset_name] = acc
            rows.append(
                {
                    "aggregation_method": method,
                    "dataset": dataset_name,
                    "acc": format_accuracy(acc),
                    "acc_percent": format_percent(acc),
                    "n_objects": len(truth_map),
                    "prediction_path": str(prediction_path.resolve()),
                    "error": error,
                }
            )
            write_results_csv(args.output_csv, rows)
            write_latex_table(args.output_tex, args.datasets, args.aggregation_methods, acc_by_method)
            print(f"[result] aggregation={method} dataset={dataset_name} acc={format_percent(acc)}", flush=True)

    print_latex_table(args.datasets, args.aggregation_methods, acc_by_method)
    print(f"[saved] csv={args.output_csv}", flush=True)
    print(f"[saved] latex={args.output_tex}", flush=True)


if __name__ == "__main__":
    main()
