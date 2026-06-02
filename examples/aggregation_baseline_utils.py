from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
SIMULATION_DATA_ROOT = DATA_ROOT / "simulation"
SIMULATION_GENERATED_DATA_ROOT = DATA_ROOT / "simulation_generated"


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    answer_path: Path
    truth_path: Path
    object_ids: List[int]
    worker_ids: List[int]
    label_values: List[int]
    answer_rows: List[Tuple[int, int, int]]
    truth_map: Dict[int, int]

    @property
    def num_classes(self) -> int:
        return len(self.label_values)


def candidate_data_roots() -> List[Path]:
    roots = [DATA_ROOT]
    if SIMULATION_DATA_ROOT.exists():
        roots.append(SIMULATION_DATA_ROOT)
    if SIMULATION_GENERATED_DATA_ROOT.exists():
        roots.append(SIMULATION_GENERATED_DATA_ROOT)
    return roots


def discover_dataset_names() -> List[str]:
    names = set()
    for root in candidate_data_roots():
        if not root.exists():
            continue
        for dataset_dir in root.iterdir():
            if not dataset_dir.is_dir():
                continue
            if (dataset_dir / "answer.csv").exists() and (dataset_dir / "truth.csv").exists():
                names.add(dataset_dir.name)
    return sorted(names)


def resolve_dataset_dir(name: str, data_root: Path | None = None) -> Path:
    if data_root is not None:
        dataset_dir = data_root / name
        if dataset_dir.exists():
            return dataset_dir
        raise FileNotFoundError(f"Dataset '{name}' not found under {data_root}")

    for root in candidate_data_roots():
        dataset_dir = root / name
        if dataset_dir.exists():
            return dataset_dir
    searched = ", ".join(str(root) for root in candidate_data_roots())
    raise FileNotFoundError(f"Dataset '{name}' not found under any supported data root: {searched}")


def _numeric_row(row: Sequence[str], min_len: int) -> bool:
    if len(row) < min_len:
        return False
    try:
        for token in row[:min_len]:
            int(token)
    except ValueError:
        return False
    return True


def read_answer_rows(path: Path) -> List[Tuple[int, int, int]]:
    rows: List[Tuple[int, int, int]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not _numeric_row(row, 3):
                continue
            rows.append((int(row[0]), int(row[1]), int(row[2])))
    return rows


def read_truth_map(path: Path) -> Dict[int, int]:
    truth_map: Dict[int, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not _numeric_row(row, 2):
                continue
            truth_map[int(row[0])] = int(row[1])
    return truth_map


def load_dataset(name: str, data_root: Path = DATA_ROOT) -> DatasetBundle:
    dataset_dir = resolve_dataset_dir(name, None if data_root == DATA_ROOT else data_root)
    answer_path = dataset_dir / "answer.csv"
    truth_path = dataset_dir / "truth.csv"
    answer_rows_all = read_answer_rows(answer_path)
    truth_map_all = read_truth_map(truth_path)

    answer_objects = {obj for obj, _worker, _label in answer_rows_all}
    object_ids = sorted(obj for obj in truth_map_all if obj in answer_objects)
    truth_map = {obj: truth_map_all[obj] for obj in object_ids}
    valid_objects = set(object_ids)
    answer_rows = [row for row in answer_rows_all if row[0] in valid_objects]
    worker_ids = sorted({worker for _obj, worker, _label in answer_rows})
    label_values = sorted({label for _obj, _worker, label in answer_rows} | set(truth_map.values()))

    return DatasetBundle(
        name=name,
        answer_path=answer_path,
        truth_path=truth_path,
        object_ids=object_ids,
        worker_ids=worker_ids,
        label_values=label_values,
        answer_rows=answer_rows,
        truth_map=truth_map,
    )


def accuracy_for_predictions(dataset: DatasetBundle, pred_map: Dict[int, int]) -> float:
    matched = [obj for obj in dataset.object_ids if obj in pred_map]
    if not matched:
        return 0.0
    correct = sum(int(pred_map[obj] == dataset.truth_map[obj]) for obj in matched)
    return correct / len(matched)


def write_object_predictions(
    output_path: Path,
    dataset: DatasetBundle,
    pred_map: Dict[int, int],
    fieldnames: Sequence[str] | None = None,
    extra_rows: Dict[int, Dict[str, object]] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = ("object", "truth", "pred")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for object_id in dataset.object_ids:
            row = {
                "object": object_id,
                "truth": dataset.truth_map[object_id],
                "pred": pred_map.get(object_id),
            }
            if extra_rows is not None and object_id in extra_rows:
                row.update(extra_rows[object_id])
            writer.writerow(row)


def compact_ids(values: Iterable[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    originals = sorted(set(values))
    forward = {value: idx for idx, value in enumerate(originals)}
    backward = {idx: value for value, idx in forward.items()}
    return forward, backward
