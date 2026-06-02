"""DaLC on 34 simulated CEKA datasets (paper Table 4, DaLC column).

For each simulated dataset, runs `dalc.pipeline` with default config and reads
back the ``completion_mv`` accuracy. Reported number is MV aggregation on top
of DaLC's completed worker-object matrix --- equivalent to "DaLC + MV" in
paper Table 4.

Usage::

    python experiments/simulation_experiment.py
    python experiments/simulation_experiment.py --datasets iris segment
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dalc._io_paths import RESULTS_ROOT, result_path_for_dataset  # noqa: E402
from dalc.pipeline import resolve_dataset_dir  # noqa: E402

DEFAULT_DATASETS: Tuple[str, ...] = (
    "anneal", "audiology", "autos", "balance-scale", "biodeg",
    "breast-cancer", "breast-w", "car", "credit-a", "credit-g",
    "diabetes", "heart-c", "heart-h", "heart-statlog", "hepatitis",
    "horse-colic", "hypothyroid", "ionosphere", "iris", "kr-vs-kp",
    "labor", "letter", "lymph", "mushroom", "segment",
    "sick", "sonar", "spambase", "tic-tac-toe", "vehicle",
    "vote", "vowel", "waveform", "zoo",
)
SUMMARY_PATH = RESULTS_ROOT / "_simulation_summary" / "simulation_experiment_dalc.csv"


def output_prefix_for(seed: int) -> str:
    return f"sim_dalc_seed{seed}"


def metrics_path(dataset_name: str, seed: int) -> Path:
    filename = f"{output_prefix_for(seed)}_metrics.csv"
    try:
        resolved_dir = resolve_dataset_dir(dataset_name)
        return result_path_for_dataset(resolved_dir, filename, create_parent=False)
    except FileNotFoundError:
        return result_path_for_dataset(Path(dataset_name), filename, create_parent=False)


def run_one(python: str, dataset: str, seed: int, device: str) -> int:
    cmd = [
        python, "-m", "dalc.pipeline",
        "--dataset-dir", dataset,
        "--device", device,
        "--seed", str(seed),
        "--output-prefix", output_prefix_for(seed),
    ]
    print(">>>", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def read_completion_mv(metrics_csv: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with metrics_csv.open(newline="") as h:
        for row in csv.DictReader(h):
            if row.get("method") != "completion_mv":
                continue
            sub = row.get("subset", "")
            if not sub or not row.get("accuracy"):
                continue
            out[sub] = float(row["accuracy"])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--aggregate-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    failures: List[str] = []
    if not args.aggregate_only:
        for ds in args.datasets:
            if run_one(args.python, ds, args.seed, args.device) != 0:
                failures.append(ds)
                print(f"!!! failed {ds}", flush=True)

    rows: List[Dict[str, object]] = []
    for ds in args.datasets:
        path = metrics_path(ds, args.seed)
        if not path.exists():
            print(f"missing metrics for {ds} -> {path}", flush=True)
            continue
        m = read_completion_mv(path)
        rows.append({
            "dataset": ds,
            "easy": f"{m.get('easy', 0) * 100:.2f}" if m.get("easy") else "",
            "ambiguous": f"{m.get('ambiguous', 0) * 100:.2f}" if m.get("ambiguous") else "",
            "hard": f"{m.get('hard', 0) * 100:.2f}" if m.get("hard") else "",
            "all": f"{m.get('all', 0) * 100:.2f}" if m.get("all") else "",
        })

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["dataset", "easy", "ambiguous", "hard", "all"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {SUMMARY_PATH}", flush=True)
    print(f"\n{'dataset':16s} | {'Easy':>6s} {'Amb.':>6s} {'Hard':>6s} {'Acc.':>6s}", flush=True)
    print("-" * 54, flush=True)
    for r in rows:
        print(f"{str(r['dataset']):16s} | {str(r['easy']):>6s} {str(r['ambiguous']):>6s} {str(r['hard']):>6s} {str(r['all']):>6s}", flush=True)
    if failures:
        print(f"\n{len(failures)} dataset(s) failed: {failures}", flush=True)


if __name__ == "__main__":
    main()
