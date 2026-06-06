from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from desktop_env.controllers.a11y_preflight import (
    A11yPreflightExecutor,
    A11yTreeIndex,
    bbox_center,
    pyautogui_click_command,
    pyautogui_move_command,
)


SAMPLE_TREE = """
<desktop-frame name="desktop"
  xmlns:st="https://accessibility.ubuntu.example.org/ns/state"
  xmlns:cp="https://accessibility.ubuntu.example.org/ns/component"
  st:visible="true">
  <frame name="Doc - LibreOffice Writer" cp:screencoord="(100, 80)" cp:size="(800, 600)" st:visible="true" st:showing="true" st:enabled="true">
    <menu name="View" cp:screencoord="(137, 50)" cp:size="(48, 25)" st:visible="true" st:showing="true" st:enabled="true"/>
    <push-button name="Bold" cp:screencoord="(200, 120)" cp:size="(30, 30)" st:visible="true" st:showing="true" st:enabled="true"/>
  </frame>
</desktop-frame>
""".strip()


@pytest.fixture(autouse=True)
def _inject_namespaces() -> None:
    # lxml needs namespaces declared on root for prefixed attributes in tests
    pass


def test_bbox_center() -> None:
    assert bbox_center([100, 80, 40, 20]) == (120, 90)


def test_match_selector_by_role_and_name() -> None:
    index = A11yTreeIndex(SAMPLE_TREE)
    matches = index.match_selector({"role": "push-button", "name": "Bold"})
    assert len(matches) == 1
    assert matches[0].center == (215, 135)


def test_match_selector_name_contains() -> None:
    index = A11yTreeIndex(SAMPLE_TREE)
    matches = index.match_selector({
        "role": "frame",
        "name_contains": "LibreOffice Writer",
        "require_bbox": False,
    })
    assert len(matches) == 1


def test_pyautogui_click_command() -> None:
    command = pyautogui_click_command(10, 20)
    assert "pyautogui.click(10, 20)" in command


def test_pyautogui_move_command() -> None:
    command = pyautogui_move_command(10, 20)
    assert "pyautogui.moveTo(10, 20)" in command


def test_executor_runs_click_and_assert_steps() -> None:
    executor = A11yPreflightExecutor("http://127.0.0.1:5000")
    steps = [
        {"op": "click", "selector": {"role": "menu", "name": "View"}},
        {"op": "assert", "selector": {"role": "push-button", "name": "Bold"}},
    ]

    with patch.object(executor, "fetch_accessibility_tree", return_value=SAMPLE_TREE):
        with patch.object(executor, "execute_python") as execute_python:
            executor.run(steps=steps, timeout_seconds=5)

    execute_python.assert_called_once()
    assert "pyautogui.click(161, 62)" in execute_python.call_args[0][0]


def test_executor_runs_move_bbox_step() -> None:
    executor = A11yPreflightExecutor("http://127.0.0.1:5000")

    with patch.object(executor, "fetch_accessibility_tree", return_value=SAMPLE_TREE):
        with patch.object(executor, "execute_python") as execute_python:
            executor.run(steps=[{"op": "move_bbox", "bbox": [10, 20, 30, 40]}], timeout_seconds=5)

    execute_python.assert_called_once()
    assert "pyautogui.moveTo(25, 40)" in execute_python.call_args[0][0]


def test_executor_invokes_on_step_complete() -> None:
    executor = A11yPreflightExecutor("http://127.0.0.1:5000")
    steps = [
        {"op": "click", "selector": {"role": "menu", "name": "View"}},
        {"op": "sleep", "seconds": 0.01},
    ]
    completed: list[tuple[int, dict]] = []

    with patch.object(executor, "fetch_accessibility_tree", return_value=SAMPLE_TREE):
        with patch.object(executor, "execute_python"):
            executor.run(
                steps=steps,
                timeout_seconds=5,
                on_step_complete=lambda index, step: completed.append((index, step)),
            )

    assert len(completed) == 2
    assert completed[0][1]["op"] == "click"
    assert completed[1][1]["op"] == "sleep"


def test_executor_wait_for_retries_until_match() -> None:
    executor = A11yPreflightExecutor("http://127.0.0.1:5000", poll_interval=0.01)
    steps = [{"op": "wait_for", "selector": {"role": "push-button", "name": "Bold"}}]
    responses = [
        """
        <desktop-frame name="desktop"
          xmlns:st="https://accessibility.ubuntu.example.org/ns/state"
          xmlns:cp="https://accessibility.ubuntu.example.org/ns/component">
          <frame name="Doc - LibreOffice Writer" cp:screencoord="(100, 80)" cp:size="(800, 600)" st:visible="true" st:showing="true"/>
        </desktop-frame>
        """.strip(),
        SAMPLE_TREE,
    ]

    with patch.object(executor, "fetch_accessibility_tree", side_effect=responses):
        executor.run(steps=steps, timeout_seconds=1)
