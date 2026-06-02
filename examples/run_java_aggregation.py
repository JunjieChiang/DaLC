from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.java_aggregation_utils import VALID_METHODS, run_java_aggregation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CEKA Java aggregation baselines and export object predictions.")
    parser.add_argument("--method", choices=sorted(VALID_METHODS), required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ds-iterations", type=int, default=20)
    parser.add_argument("--force-recompile", action="store_true")
    args = parser.parse_args()

    for dataset_name in args.datasets:
        output_csv = run_java_aggregation(
            method=args.method,
            completed_dataset_dir=(args.data_root / dataset_name).resolve(),
            feature_dataset_dir=(args.feature_root / dataset_name).resolve(),
            output_dir=(args.output_root / dataset_name).resolve(),
            ds_iterations=args.ds_iterations,
            force_recompile=args.force_recompile,
        )
        print(f"{dataset_name}\toutput={output_csv}", flush=True)


if __name__ == "__main__":
    main()
