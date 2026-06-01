"""Unit tests for native Holo action parsing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

GUI_DOCKER_ENV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUI_DOCKER_ENV))

from mm_agents.holo_agent import holo_to_pyautogui, parse_holo_response

RH, RW = 540, 960
SH, SW = 1080, 1920

SMOKE_TRAJ = (
    GUI_DOCKER_ENV
    / "results_holo_smoke_writer_first_hist2_20260601_003640"
    / "pyautogui/screenshot/Hcompany/Holo3-35B-A3B/libreoffice_writer"
    / "0810415c-bde4-4443-9047-d5f70165a697/traj.jsonl"
)


@pytest.mark.parametrize(
    "response,expected_kind",
    [
        (
            "Thought: select all\nAction: hotkey(key='ctrl a')",
            "hotkey",
        ),
        (
            "Thought: click font\nAction: click(start_box='\t289, 132)')",
            "click",
        ),
        (
            "Action: drag(start_box='335, 362', end_box='401, 487')",
            "drag",
        ),
        (
            "Okay...\n</think>\n\nThought: done\nAction: hotkey(key='ctrl a')",
            "hotkey",
        ),
        (
            "Thought: x\nAction: click(start_box='<|box_start|>(147, 69)<|box_end|>')",
            "click",
        ),
        (
            "Thought: x\nAction: scroll(direction='down')",
            "scroll",
        ),
        (
            "Thought: x\nAction: drag(start_box='335, 362', end_box='401, 487')\n\nNext I will open Format.",
            "drag",
        ),
    ],
)
def test_parse_holo_samples(response: str, expected_kind: str) -> None:
    action = parse_holo_response(response, RH, RW, SH, SW)
    assert action.kind == expected_kind
    holo_to_pyautogui(action)


def test_parse_unclosed_quote_repair() -> None:
    response = "Thought: x\nAction: click(start_box='150, 69)"
    action = parse_holo_response(response, RH, RW, SH, SW)
    assert action.kind == "click"
    assert action.x is not None


@pytest.mark.skipif(not SMOKE_TRAJ.is_file(), reason="smoke traj not present")
def test_parse_smoke_traj_all_steps() -> None:
    for line in SMOKE_TRAJ.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        response = row.get("response")
        if not response or str(response).startswith("client error"):
            continue
        action = parse_holo_response(response, RH, RW, SH, SW)
        holo_to_pyautogui(action)
