from __future__ import annotations

from pathlib import Path

RESULTS_ROOT_NAME = "results"
IMAGES_ZIP_NAME = "images.zip"
RUN_ZIP_NAME = "run.zip"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def normalize_result_dir(path: str) -> str:
    raw = (path or "").strip() or f"./{RESULTS_ROOT_NAME}/results"
    candidate = Path(raw)
    if candidate.is_absolute():
        return raw
    rel = candidate.as_posix().removeprefix("./")
    if rel == RESULTS_ROOT_NAME or rel.startswith(f"{RESULTS_ROOT_NAME}/"):
        return raw if raw.startswith(".") else f"./{rel}"
    return f"./{RESULTS_ROOT_NAME}/{rel}"


def results_root(cwd: Path | None = None) -> Path:
    base = cwd or Path.cwd()
    return (base / RESULTS_ROOT_NAME).resolve()


def list_result_run_dirs(results_root_path: Path | None = None) -> list[Path]:
    root = results_root_path or results_root()
    if not root.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("results_")
    )
