#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path


DEFAULT_ROOT = Path("/home/pitchblack/dart-gui/GUI-Docker-Env")
RUNNING_STALE_SECONDS = 120.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
    )
    parser.add_argument(
        "--results-glob",
        type=str,
    )
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Eval Progress Monitor",
    )
    parser.add_argument(
        "--all-matches",
        action="store_true",
        help="Monitor every directory matched by --results-glob instead of only the latest one",
    )
    args = parser.parse_args()
    if args.results_dir is None and not args.results_glob:
        parser.error("one of --results-dir or --results-glob is required")
    return args


def load_tasks(eval_json: Path) -> list[tuple[str, str]]:
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("eval json must be a domain-to-task-id mapping")
    tasks: list[tuple[str, str]] = []
    for domain in sorted(payload):
        task_ids = payload[domain]
        if not isinstance(task_ids, list):
            raise ValueError("each domain entry must be a list of task ids")
        for task_id in task_ids:
            if isinstance(task_id, str):
                tasks.append((domain, task_id))
    return tasks


def task_path_exists(result_dir: Path, domain: str, task_id: str) -> bool:
    return any(path.is_dir() for path in result_dir.glob(f"**/{domain}/{task_id}"))


def result_file_paths(result_dir: Path, domain: str, task_id: str) -> list[Path]:
    return sorted(path for path in result_dir.glob(f"**/{domain}/{task_id}/result.txt") if path.is_file())


def task_root_paths(result_dir: Path, domain: str, task_id: str) -> list[Path]:
    return sorted(path for path in result_dir.glob(f"**/{domain}/{task_id}") if path.is_dir())


def done_result_value(result_file: Path) -> str:
    raw = result_file.read_text(encoding="utf-8", errors="replace").strip()
    if raw == "1.0":
        return "1"
    if raw == "0.0":
        return "0"
    try:
        value = float(raw)
    except ValueError:
        return raw or "-"
    return "1" if value == 1.0 else "0" if value == 0.0 else raw


def resolve_result_dirs(root: Path, results_dir: Path | None, results_glob: str | None, all_matches: bool) -> list[Path]:
    if results_dir is not None:
        candidate = results_dir if results_dir.is_absolute() else root / results_dir
        candidate = candidate.resolve()
        return [candidate] if candidate.is_dir() else []
    assert results_glob is not None
    matches = [path.resolve() for path in root.glob(results_glob) if path.is_dir()]
    if not matches:
        return []
    matches.sort(key=lambda path: (path.stat().st_mtime, path.name))
    return matches if all_matches else [matches[-1]]


def state_for(result_dirs: list[Path], domain: str, task_id: str) -> tuple[str, str]:
    found_incomplete = False
    latest_incomplete_mtime = None
    for result_dir in result_dirs:
        if not task_path_exists(result_dir, domain, task_id):
            continue
        result_files = result_file_paths(result_dir, domain, task_id)
        if result_files:
            return "DONE", done_result_value(result_files[-1])
        found_incomplete = True
        for task_root in task_root_paths(result_dir, domain, task_id):
            try:
                mtime = task_root.stat().st_mtime
            except FileNotFoundError:
                continue
            if latest_incomplete_mtime is None or mtime > latest_incomplete_mtime:
                latest_incomplete_mtime = mtime
    if found_incomplete:
        now = time.time()
        if latest_incomplete_mtime is not None and now - latest_incomplete_mtime <= RUNNING_STALE_SECONDS:
            return "RUNN", "-"
        return "ERR ", "-"
    return "TODO", "-"


def collect_rows(tasks: list[tuple[str, str]], result_dirs: list[Path]) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    rows = []
    done = running = todo = err = 0
    for domain, task_id in tasks:
        state, result = state_for(result_dirs, domain, task_id)
        if state == "DONE":
            done += 1
        elif state == "RUNN":
            running += 1
        elif state == "ERR ":
            err += 1
        else:
            todo += 1
        rows.append((task_id, state, result))
    counts = {"DONE": done, "RUNN": running, "ERR ": err, "TODO": todo}
    return rows, counts


def render(
    tasks: list[tuple[str, str]],
    result_dirs: list[Path],
    title: str,
    refresh_seconds: float,
) -> str:
    rows, counts = collect_rows(tasks, result_dirs)
    done = counts["DONE"]
    running = counts["RUNN"]
    err = counts["ERR "]
    todo = counts["TODO"]
    id_width = max(len("TASK ID"), max(len(task_id) for task_id, _state, _result in rows))
    state_width = len("STATE")
    result_width = len("RESULT")
    lines = [
        "\x1bc",
        f"{title}  refresh={refresh_seconds:g}s",
        f"watching={', '.join(path.name for path in result_dirs) if result_dirs else '-'}",
        f"done={done}  running={running}  err={err}  todo={todo}",
        "",
        f"{'TASK ID':<{id_width}}  {'STATE':<{state_width}}  {'RESULT':<{result_width}}",
        f"{'-' * id_width}  {'-' * state_width}  {'-' * result_width}",
    ]
    for task_id, state, result in rows:
        lines.append(f"{task_id:<{id_width}}  {state:<{state_width}}  {result:<{result_width}}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    eval_json = args.eval_json.resolve()
    tasks = load_tasks(eval_json)
    while True:
        result_dirs = resolve_result_dirs(root, args.results_dir, args.results_glob, args.all_matches)
        rendered = render(
            tasks=tasks,
            result_dirs=result_dirs,
            title=args.title,
            refresh_seconds=args.refresh_seconds,
        )
        print(rendered, end="", flush=True)
        time.sleep(args.refresh_seconds)


if __name__ == "__main__":
    main()
