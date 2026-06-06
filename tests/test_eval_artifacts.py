from __future__ import annotations

from pathlib import Path

from lib_eval_artifacts import (
    example_for_reset_with_deferred_preflight,
    record_eval_step,
    save_step_a11y_dump,
    save_step_screenshot,
    split_a11y_preflight_config,
    step_a11y_filename,
    step_screenshot_filename,
)


class _FakeController:
    def __init__(self, tree: str) -> None:
        self.tree = tree

    def get_accessibility_tree(self) -> str:
        return self.tree


class _FakeEnv:
    def __init__(self, tree: str) -> None:
        self.controller = _FakeController(tree)


def test_save_step_a11y_dump_uses_flat_result_dir(tmp_path: Path) -> None:
    result_dir = str(tmp_path)
    env = _FakeEnv("<fallback/>")

    step_a11y = save_step_a11y_dump(result_dir, 1, "20260607@120000", {"accessibility_tree": "<root/>"}, env)
    assert step_a11y == step_a11y_filename(1, "20260607@120000")
    assert (tmp_path / step_a11y_filename(1, "20260607@120000")).read_text(encoding="utf-8") == "<root/>"

    step_a11y = save_step_a11y_dump(result_dir, 2, "20260607@120001", {"accessibility_tree": None}, env)
    assert (tmp_path / step_a11y_filename(2, "20260607@120001")).read_text(encoding="utf-8") == "<fallback/>"


def test_save_step_screenshot(tmp_path: Path) -> None:
    result_dir = str(tmp_path)
    filename = save_step_screenshot(result_dir, 2, "20260607@120001", {"screenshot": b"img"})
    assert filename == step_screenshot_filename(2, "20260607@120001")
    assert (tmp_path / filename).read_bytes() == b"img"


def test_split_a11y_preflight_config() -> None:
    config = [
        {"type": "open", "parameters": {"path": "/tmp/a.docx"}},
        {"type": "a11y_preflight", "parameters": {"steps": [{"op": "sleep", "seconds": 0.1}]}},
    ]
    base_config, preflight = split_a11y_preflight_config(config)
    assert len(base_config) == 1
    assert base_config[0]["type"] == "open"
    assert preflight == {"steps": [{"op": "sleep", "seconds": 0.1}]}


def test_example_for_reset_with_deferred_preflight() -> None:
    example = {
        "id": "task-1",
        "config": [
            {"type": "open", "parameters": {"path": "/tmp/a.docx"}},
            {"type": "a11y_preflight", "parameters": {"steps": [{"op": "click", "selector": {"role": "menu"}}]}},
        ],
    }
    reset_example, preflight = example_for_reset_with_deferred_preflight(example)
    assert len(reset_example["config"]) == 1
    assert preflight is not None
    assert preflight["steps"][0]["op"] == "click"


def test_record_eval_step_writes_traj(tmp_path: Path) -> None:
    result_dir = str(tmp_path)
    record_eval_step(
        result_dir,
        1,
        "20260607@120000",
        {"screenshot": b"img", "accessibility_tree": "<root/>"},
    )
    lines = (tmp_path / "traj.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = __import__("json").loads(lines[0])
    assert payload["step_num"] == 1
    assert payload["screenshot_file"] == step_screenshot_filename(1, "20260607@120000")
    assert payload["a11y_file"] == step_a11y_filename(1, "20260607@120000")
