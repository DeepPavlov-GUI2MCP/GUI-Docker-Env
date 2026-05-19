from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

SYNTHETIC_SPLITS = frozenset({"golden", "train", "test"})


def make_rollout_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_synthetic_task_layout(cfg_path: str) -> Optional[Dict[str, str]]:
    path = Path(cfg_path).resolve()
    parts = path.parts
    for index, part in enumerate(parts):
        if part != "tasks" or index == 0:
            continue
        split = parts[index - 1]
        if split not in SYNTHETIC_SPLITS:
            continue
        dataset_root = Path(*parts[: index - 1])
        return {
            "dataset_root": str(dataset_root),
            "dataset_name": dataset_root.name,
            "split": split,
            "task_id": path.stem,
            "task_config_path": str(path),
        }
    return None


def resolve_rollout_save_dir(
    *,
    result_dir: str,
    domain: str,
    task_id: str,
    task_config_path: str,
    rollout_id: Optional[str],
) -> str:
    layout = parse_synthetic_task_layout(task_config_path)
    if layout:
        rollout_name = rollout_id or make_rollout_id()
        return str(
            Path(layout["dataset_root"])
            / layout["split"]
            / "rollouts"
            / layout["task_id"]
            / rollout_name
        )
    return str(Path(result_dir) / "coact" / f"{domain}/{task_id}")
