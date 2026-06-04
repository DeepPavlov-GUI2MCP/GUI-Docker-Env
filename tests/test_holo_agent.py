"""Unit tests for doc-aligned Holo3 agent-loop harness."""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

GUI_DOCKER_ENV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUI_DOCKER_ENV))

from mm_agents.holo.executor import execute_step, scale_point, tool_call_to_pyautogui
from mm_agents.holo.messages import (
    EVICTED_SCREENSHOT_TEXT,
    observation_message,
    prepare_screenshot,
    system_message,
    tool_output_message,
    trim_to_last_n_images,
)
from mm_agents.holo.schema import (
    AnswerArgs,
    ClickArgs,
    FailArgs,
    Step,
    WaitArgs,
    WriteArgs,
    build_system_prompt,
    step_json_schema,
)
from mm_agents.holo_agent import HoloAgent


def _blank_png_bytes(width: int = 1920, height: int = 1080) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_step_json_schema_has_tool_union() -> None:
    schema = step_json_schema()
    assert "tool_call" in schema["properties"]
    note_schema = schema["properties"]["note"]
    if "anyOf" in note_schema:
        types = {entry.get("type") for entry in note_schema["anyOf"]}
        assert "string" in types
        assert "null" in types
    else:
        assert note_schema.get("type") in ("string", "null")


def test_build_system_prompt_includes_instruction_and_schema() -> None:
    prompt = build_system_prompt("Open Settings")
    assert "Open Settings" in prompt
    assert "<output_format>" in prompt
    assert '"tool_name"' in prompt


@pytest.mark.parametrize(
    "tool_name,args,expected_substring",
    [
        ("click", {"element": "Save", "x": 500, "y": 500}, "pyautogui.click(960, 540"),
        ("double_click", {"element": "Icon", "x": 100, "y": 200}, "pyautogui.doubleClick(192, 216"),
        ("write", {"content": "hello"}, "pyperclip.copy('hello')"),
        ("hotkey", {"keys": "ctrl a"}, "pyautogui.hotkey('ctrl', 'a')"),
        ("press", {"key": "enter"}, "pyautogui.press('enter')"),
        ("scroll", {"direction": "down"}, "pyautogui.scroll(-5)"),
        (
            "drag",
            {"start_x": 100, "start_y": 100, "end_x": 200, "end_y": 300},
            "pyautogui.dragTo(384, 324",
        ),
    ],
)
def test_executor_generates_pyautogui(tool_name: str, args: dict, expected_substring: str) -> None:
    payload = {"note": None, "thought": "do it", "tool_call": {"tool_name": tool_name, **args}}
    step = Step.model_validate(payload)
    code = tool_call_to_pyautogui(step, screen_width=1920, screen_height=1080, input_swap=True)
    assert expected_substring in code


def test_scale_point_center() -> None:
    assert scale_point(500, 500, 1920, 1080) == (960, 540)


def test_execute_step_terminal_tools() -> None:
    for tool_name, expected in [("answer", "DONE"), ("wait", "WAIT"), ("fail", "FAIL")]:
        if tool_name == "answer":
            tool_call = AnswerArgs(tool_name="answer", content="done")
        elif tool_name == "wait":
            tool_call = WaitArgs(tool_name="wait")
        else:
            tool_call = FailArgs(tool_name="fail", reason="impossible")
        step = Step(note=None, thought="finish", tool_call=tool_call)
        _response, actions = execute_step(step, screen_width=1920, screen_height=1080)
        assert actions == [expected]


def test_observation_message_shape() -> None:
    image = Image.new("RGB", (100, 100), color="red")
    msg = observation_message(image)
    assert msg["role"] == "user"
    assert msg["content"][0]["text"] == "<observation>\n"
    assert msg["content"][1]["type"] == "image_url"
    assert msg["content"][2]["text"] == "\n</observation>"


def test_tool_output_message_shape() -> None:
    msg = tool_output_message("click")
    assert '<tool_output tool="click">' in msg["content"]


def test_trim_to_last_n_images_evicts_old_screenshots() -> None:
    image = Image.new("RGB", (10, 10))
    messages = [
        system_message("sys"),
        observation_message(image),
        {"role": "assistant", "content": "{}"},
        observation_message(image),
        {"role": "assistant", "content": "{}"},
        observation_message(image),
        {"role": "assistant", "content": "{}"},
        observation_message(image),
    ]
    trim_to_last_n_images(messages, n=3)
    image_chunks = [
        chunk
        for msg in messages
        if isinstance(msg.get("content"), list)
        for chunk in msg["content"]
        if chunk.get("type") == "image_url" or chunk.get("text") == EVICTED_SCREENSHOT_TEXT
    ]
    evicted = sum(1 for chunk in image_chunks if chunk.get("text") == EVICTED_SCREENSHOT_TEXT)
    remaining = sum(1 for chunk in image_chunks if chunk.get("type") == "image_url")
    assert evicted == 1
    assert remaining == 3


def test_prepare_screenshot_tracks_screen_dimensions() -> None:
    png = _blank_png_bytes(1920, 1080)
    _image, sent_w, sent_h, screen_w, screen_h = prepare_screenshot(
        png,
        max_pixels=16384 * 28 * 28,
        min_pixels=100 * 28 * 28,
    )
    assert screen_w == 1920
    assert screen_h == 1080
    assert sent_w > 0 and sent_h > 0


def test_holo_agent_predict_builds_messages_and_calls_api() -> None:
    click_step = Step(
        note=None,
        thought="click save",
        tool_call=ClickArgs(tool_name="click", element="Save button", x=500, y=500),
    )
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = click_step.model_dump_json()
    mock_response.choices[0].message.reasoning_content = "planning"
    mock_response.usage = None

    agent = HoloAgent(
        model="holo3-35b-a3b",
        runtime_conf={
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 256,
            "input_swap": True,
            "max_pixels": 16384 * 28 * 28,
            "min_pixels": 100 * 28 * 28,
            "history_n": 3,
        },
        base_url="http://127.0.0.1:8010",
        api_key="empty",
    )
    agent.reset()

    with patch.object(agent.vlm.chat.completions, "create", return_value=mock_response) as mock_create:
        response, actions = agent.predict("Save the file", {"screenshot": _blank_png_bytes()})

    assert mock_create.call_count == 1
    call_kwargs = mock_create.call_args.kwargs
    assert "structured_outputs" in call_kwargs["extra_body"]
    assert call_kwargs["messages"][0]["role"] == "system"
    assert "Save the file" in call_kwargs["messages"][0]["content"]
    assert call_kwargs["messages"][1]["role"] == "user"
    assert json.loads(response)["tool_call"]["tool_name"] == "click"
    assert "pyautogui.click(960, 540" in actions[0]

    with patch.object(agent.vlm.chat.completions, "create", return_value=mock_response) as mock_create:
        agent.predict("Save the file", {"screenshot": _blank_png_bytes()})

    messages = mock_create.call_args.kwargs["messages"]
    assert any(
        isinstance(msg.get("content"), str) and msg["content"].startswith('<tool_output tool="click">')
        for msg in messages
    )
