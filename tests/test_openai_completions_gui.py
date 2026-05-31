from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mm_agents.coact.computer_exec import (
    actions_from_computer_call,
    build_computer_function_tool,
    execute_computer_tool_call,
)
from mm_agents.coact.cua_agent import run_cua


def test_build_computer_function_tool_schema() -> None:
    tool = build_computer_function_tool()
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "computer"
    params = tool["function"]["parameters"]["properties"]
    assert "actions" in params
    assert "action" in params


def test_actions_from_computer_call_round_trip() -> None:
    payload = {"actions": [{"type": "click", "x": 10, "y": 20, "button": "left"}]}
    actions = actions_from_computer_call(payload)
    assert len(actions) == 1
    assert actions[0]["type"] == "click"


def test_execute_computer_tool_call_respects_step_budget(tmp_path: Path) -> None:
    screenshot = b"fakepng"
    env = MagicMock()

    def execute_actions(actions):
        del actions
        return screenshot

    state = {"step_index": 1, "step_count": 0}
    result = execute_computer_tool_call(
        {"actions": [{"type": "click", "x": 1, "y": 2}]},
        save_path=str(tmp_path),
        step_state=state,
        max_steps=1,
        execute_actions=execute_actions,
    )
    assert result.error is False
    assert result.step_state == {"step_index": 2, "step_count": 1}

    exhausted = execute_computer_tool_call(
        {"actions": [{"type": "click", "x": 1, "y": 2}]},
        save_path=str(tmp_path),
        step_state=result.step_state or {},
        max_steps=1,
        execute_actions=execute_actions,
    )
    assert exhausted.error is True
    assert "Step budget exhausted" in exhausted.summary


def test_run_cua_routes_to_completions_when_flag_true() -> None:
    env = MagicMock()
    with patch("mm_agents.coact.cua_agent._run_cua_openai_completions") as mock_completions:
        mock_completions.return_value = ([], "TERMINATE", [], [], 0)
        run_cua(
            env,
            "do task",
            max_steps=3,
            save_path="/tmp/test",
            gui_protocol="openai",
            openai_force_completions_api=True,
        )
        mock_completions.assert_called_once()


def test_run_cua_routes_to_responses_when_flag_false(tmp_path: Path) -> None:
    env = MagicMock()
    env.controller.get_screenshot.return_value = b"png"
    env.step.return_value = ({"screenshot": b"png"}, 0.0, False, {})
    client = MagicMock()
    response = SimpleNamespace(
        id="resp_1",
        model="gpt-5.5",
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=1,
            model_dump=lambda mode="json": {"input_tokens": 1, "output_tokens": 1},
        ),
        output=[
            SimpleNamespace(
                type="message",
                model_dump=lambda mode="json": {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "done TERMINATE"}],
                },
            )
        ],
    )
    client.responses.create.return_value = response
    with patch("mm_agents.coact.cua_agent._build_openai_client", return_value=client):
        with patch("mm_agents.coact.cua_agent._run_cua_openai_completions") as mock_completions:
            run_cua(
                env,
                "do task",
                max_steps=3,
                save_path=str(tmp_path),
                gui_protocol="openai",
                openai_force_completions_api=False,
            )
            mock_completions.assert_not_called()
            client.responses.create.assert_called()
