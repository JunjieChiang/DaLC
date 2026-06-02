from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.java_baseline_utils import (
    compilation_up_to_date,
    materialize_ceka_dataset_from_completed,
    require_java_binary,
    touch_compilation_stamp,
)


CEKA_ROOT = PROJECT_ROOT / "examples" / "Ceka-v1.0.1"
CEKA_CLASSES = CEKA_ROOT / "build" / "classes"
CEKA_LIB = CEKA_ROOT / "lib"
IWBVT_ROOT = PROJECT_ROOT / "examples" / "IWBVT"
IWBVT_BUILD = IWBVT_ROOT / "build" / "java_baseline"
IWBVT_STAMP = IWBVT_BUILD / "classes" / ".compile.stamp"


def compile_runner(javac_bin: str, force_recompile: bool = False) -> Path:
    class_dir = IWBVT_BUILD / "classes"
    sources = [
        IWBVT_ROOT / "LinearRegression.java",
        IWBVT_ROOT / "IWBVT.java",
        IWBVT_ROOT / "RunIWBVTPrediction.java",
    ]
    compile_inputs = [*sources, CEKA_CLASSES]
    if not force_recompile and compilation_up_to_date(IWBVT_STAMP, compile_inputs):
        return class_dir
    class_dir.mkdir(parents=True, exist_ok=True)
    classpath = ":".join([str(CEKA_CLASSES), str(CEKA_LIB / "*")])
    subprocess.run(
        [
            javac_bin,
            "-encoding",
            "UTF-8",
            "-cp",
            classpath,
            "-d",
            str(class_dir),
            *[str(source) for source in sources],
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    touch_compilation_stamp(IWBVT_STAMP)
    return class_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the original Java IWBVT baseline and export object predictions.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force-recompile", action="store_true")
    args = parser.parse_args()

    java_bin = require_java_binary("java")
    javac_bin = require_java_binary("javac")
    class_dir = compile_runner(javac_bin, force_recompile=args.force_recompile)
    classpath = ":".join([str(class_dir), str(CEKA_CLASSES), str(CEKA_LIB / "*")])

    for dataset_name in args.datasets:
        completed_dir = (args.data_root / dataset_name).resolve()
        feature_dir = (args.feature_root / dataset_name).resolve()
        run_dir = (args.output_root / dataset_name).resolve()
        ceka_dir = run_dir / "ceka_dataset"
        materialize_ceka_dataset_from_completed(completed_dir, feature_dir, ceka_dir)
        output_csv = run_dir / "object_predictions.csv"
        log_path = run_dir / "iwbvt_java.log"
        run_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                java_bin,
                "-Xmx4g",
                "-cp",
                classpath,
                "ceka.IWBVT.RunIWBVTPrediction",
                str(ceka_dir),
                str(output_csv),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        log_path.write_text(result.stdout + ("\n" + result.stderr if result.stderr else ""), encoding="utf-8")
        print(f"{dataset_name}\toutput={output_csv}", flush=True)


if __name__ == "__main__":
    main()
