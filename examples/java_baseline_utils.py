from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def require_java_binary(name: str) -> str:
    candidates = []
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidates.append(Path(java_home) / "bin" / name)
    candidates.append(Path("/opt/homebrew/opt/openjdk/bin") / name)
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate and candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise FileNotFoundError(f"Could not find executable for {name}. Set JAVA_HOME or add it to PATH.")


def find_latest_matlab_root() -> Path:
    env_root = os.environ.get("MATLAB_ROOT")
    if env_root:
        path = Path(env_root).expanduser().resolve()
        if path.exists():
            return path
    candidates = sorted(Path("/Applications").glob("MATLAB_R*.app"))
    if not candidates:
        raise FileNotFoundError("Could not find a MATLAB installation under /Applications. Set MATLAB_ROOT.")
    return candidates[-1].resolve()


def find_existing_path(description: str, candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    formatted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {description}. Checked: {formatted}")


def compilation_up_to_date(stamp_path: Path, inputs: list[Path]) -> bool:
    if not stamp_path.exists():
        return False
    resolved_inputs = [path.resolve() for path in inputs if path.exists()]
    if not resolved_inputs:
        return False
    stamp_mtime = stamp_path.stat().st_mtime
    return all(path.stat().st_mtime <= stamp_mtime for path in resolved_inputs)


def touch_compilation_stamp(stamp_path: Path) -> None:
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.touch()


def find_arff_like_file(dataset_dir: Path) -> Path:
    candidates = sorted(dataset_dir.glob("*.arffx")) + sorted(dataset_dir.glob("*.arff")) + sorted(dataset_dir.glob("*.ARFF")) + sorted(dataset_dir.glob("*.ARFFX"))
    if not candidates:
        raise FileNotFoundError(f"Could not find an ARFF/ARFFX file under {dataset_dir}")
    return candidates[0].resolve()


def find_dataset_file(dataset_dir: Path, suffixes: list[str]) -> Path:
    lowered = [(suffix, suffix.lower()) for suffix in suffixes]
    candidates = sorted(path for path in dataset_dir.iterdir() if path.is_file())
    for _raw_suffix, suffix in lowered:
        for candidate in candidates:
            if candidate.name.lower().endswith(suffix):
                return candidate.resolve()
    formatted = ", ".join(suffixes)
    raise FileNotFoundError(f"Could not find a dataset file ending with one of [{formatted}] under {dataset_dir}")


def _split_ceka_line(raw: str) -> list[str]:
    line = raw.strip()
    if not line:
        return []
    parts = [token for token in line.replace(",", " ").split() if token]
    return parts


def materialize_completed_csv_from_ceka_raw(dataset_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_src = find_dataset_file(dataset_dir, [".gold.txt"])
    truth_csv = output_dir / "truth.csv"
    with gold_src.open("r", encoding="utf-8", errors="ignore") as src, truth_csv.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        writer = csv.writer(dst)
        writer.writerow(["object", "truth"])
        for raw in src:
            parts = _split_ceka_line(raw)
            if len(parts) < 2:
                continue
            try:
                writer.writerow([int(float(parts[0])), int(float(parts[1]))])
            except ValueError:
                continue

    response_src = find_dataset_file(dataset_dir, [".response.txt"])
    answer_csv = output_dir / "answer.csv"
    with response_src.open("r", encoding="utf-8", errors="ignore") as src, answer_csv.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        writer = csv.writer(dst)
        writer.writerow(["object", "worker", "label"])
        for raw in src:
            parts = _split_ceka_line(raw)
            if len(parts) < 3:
                continue
            try:
                worker_id = int(float(parts[0]))
                object_id = int(float(parts[1]))
                label = int(float(parts[2]))
            except ValueError:
                continue
            writer.writerow([object_id, worker_id, label])

    return truth_csv, answer_csv


def materialize_java_raw_dataset(dataset_dir: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    arff_src = find_arff_like_file(dataset_dir)
    response_src = find_dataset_file(dataset_dir, [".response.txt"])
    gold_src = find_dataset_file(dataset_dir, [".gold.txt"])

    has_id_map = False
    with arff_src.open("r", encoding="utf-8", errors="ignore") as src:
        for raw in src:
            if raw.strip().lower().startswith("@id-map"):
                has_id_map = True
                break

    arff_name = arff_src.stem + ".arffx" if has_id_map else arff_src.name
    arff_dst = output_dir / arff_name
    shutil.copyfile(arff_src, arff_dst)
    response_dst = output_dir / response_src.name
    gold_dst = output_dir / gold_src.name
    shutil.copyfile(response_src, response_dst)
    shutil.copyfile(gold_src, gold_dst)
    return arff_dst, gold_dst, response_dst


def materialize_ceka_dataset_from_completed(
    completed_dataset_dir: Path,
    feature_dataset_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    arff_src = find_arff_like_file(feature_dataset_dir)
    has_id_map = False
    with arff_src.open("r", encoding="utf-8", errors="ignore") as src:
        for raw in src:
            if raw.strip().lower().startswith("@id-map"):
                has_id_map = True
                break

    arff_name = arff_src.stem + ".arffx" if has_id_map else arff_src.name
    arff_dst = output_dir / arff_name
    shutil.copyfile(arff_src, arff_dst)

    gold_dst = output_dir / "completed.gold.txt"
    with (completed_dataset_dir / "truth.csv").open(newline="", encoding="utf-8-sig") as src, gold_dst.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.reader(src)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                object_id = int(row[0])
                truth = int(row[1])
            except ValueError:
                continue
            dst.write(f"{object_id}\t{truth}\n")

    response_dst = output_dir / "completed.response.txt"
    with (completed_dataset_dir / "answer.csv").open(newline="", encoding="utf-8-sig") as src, response_dst.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.reader(src)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                object_id = int(row[0])
                worker_id = int(row[1])
                label = int(row[2])
            except ValueError:
                continue
            dst.write(f"{worker_id}\t{object_id}\t{label}\n")

    return arff_dst, gold_dst, response_dst
