from __future__ import annotations

import subprocess
from pathlib import Path

from examples.java_baseline_utils import (
    compilation_up_to_date,
    materialize_ceka_dataset_from_completed,
    require_java_binary,
    touch_compilation_stamp,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CEKA_ROOT = PROJECT_ROOT / "examples" / "Ceka-v1.0.1"
CEKA_CLASSES = CEKA_ROOT / "build" / "classes"
CEKA_LIB = CEKA_ROOT / "lib"
RUNNER_SOURCE = CEKA_ROOT / "src" / "ceka" / "integration" / "RunAggregationPrediction.java"
RUNNER_BUILD = CEKA_ROOT / "build" / "java_aggregation"
RUNNER_STAMP = RUNNER_BUILD / "classes" / ".compile.stamp"
VALID_METHODS = {"MV", "DS", "GTIC"}


def compile_runner(javac_bin: str, force_recompile: bool = False) -> Path:
    class_dir = RUNNER_BUILD / "classes"
    compile_inputs = [RUNNER_SOURCE, CEKA_CLASSES]
    if not force_recompile and compilation_up_to_date(RUNNER_STAMP, compile_inputs):
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
            str(RUNNER_SOURCE),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    touch_compilation_stamp(RUNNER_STAMP)
    return class_dir

def run_java_aggregation(
    method: str,
    completed_dataset_dir: Path,
    feature_dataset_dir: Path,
    output_dir: Path,
    ds_iterations: int = 20,
    force_recompile: bool = False,
) -> Path:
    if method not in VALID_METHODS:
        raise ValueError(f"Unsupported Java aggregation method: {method}")

    java_bin = require_java_binary("java")
    javac_bin = require_java_binary("javac")
    class_dir = compile_runner(javac_bin, force_recompile=force_recompile)
    classpath = ":".join([str(class_dir), str(CEKA_CLASSES), str(CEKA_LIB / "*")])

    run_dir = output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ceka_dir = run_dir / "ceka_dataset"
    materialize_ceka_dataset_from_completed(completed_dataset_dir.resolve(), feature_dataset_dir.resolve(), ceka_dir)

    output_csv = run_dir / "object_predictions.csv"
    work_dir = run_dir / "java_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{method.lower()}_java.log"
    java_cmd = [java_bin]
    if method == "GTIC":
        # GTIC's comparator trips TimSort on newer JDKs; legacy mergesort matches the older runtime behavior.
        java_cmd.append("-Djava.util.Arrays.useLegacyMergeSort=true")
    java_cmd.extend(
        [
            "-cp",
            classpath,
            "ceka.integration.RunAggregationPrediction",
            method,
            str(ceka_dir),
            str(output_csv),
            str(work_dir),
            str(ds_iterations),
        ]
    )
    try:
        result = subprocess.run(
            java_cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        log_text = result.stdout + ("\n" + result.stderr if result.stderr else "")
    except subprocess.CalledProcessError as exc:
        log_text = (exc.stdout or "") + ("\n" + exc.stderr if exc.stderr else "")
        log_path.write_text(log_text, encoding="utf-8")
        raise
    log_path.write_text(log_text, encoding="utf-8")
    return output_csv
