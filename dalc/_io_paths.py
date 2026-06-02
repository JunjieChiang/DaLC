from __future__ import annotations

from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _relative_data_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(DATA_ROOT.resolve())
    except ValueError:
        return Path(resolved.name)


def result_dir_for_dataset(dataset_dir: str | Path, create: bool = True) -> Path:
    dataset_path = _as_path(dataset_dir)
    result_dir = RESULTS_ROOT / _relative_data_path(dataset_path)
    if create:
        result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def result_path_for_dataset(dataset_dir: str | Path, filename: str, create_parent: bool = True) -> Path:
    result_dir = result_dir_for_dataset(dataset_dir, create=create_parent)
    return result_dir / filename


def top_level_result_path(filename: str, create_parent: bool = True) -> Path:
    path = RESULTS_ROOT / filename
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def preferred_result_path(dataset_dir: str | Path, filename: str) -> Path:
    result_path = result_path_for_dataset(dataset_dir, filename, create_parent=False)
    legacy_path = _as_path(dataset_dir) / filename
    if result_path.exists():
        return result_path
    if legacy_path.exists():
        return legacy_path
    return result_path


def preferred_result_paths(dataset_dir: str | Path, filenames: Iterable[str]) -> tuple[Path, ...]:
    return tuple(preferred_result_path(dataset_dir, filename) for filename in filenames)
