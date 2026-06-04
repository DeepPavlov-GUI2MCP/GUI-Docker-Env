"""Convert Holo3 Step JSON to OSWorld pyautogui actions."""

from __future__ import annotations

import re
from typing import List, Tuple

from mm_agents.holo.schema import (
    AnswerArgs,
    ClickArgs,
    DoubleClickArgs,
    DragArgs,
    FailArgs,
    HotkeyArgs,
    PressArgs,
    ScrollArgs,
    Step,
    WaitArgs,
    WriteArgs,
)


def scale_coord(value: int, screen_dim: int) -> int:
    return int(value / 1000 * screen_dim)


def scale_point(x: int, y: int, screen_width: int, screen_height: int) -> Tuple[int, int]:
    return scale_coord(x, screen_width), scale_coord(y, screen_height)


def _escape_single_quotes(text: str) -> str:
    return re.sub(r"(?<!\\)'", r"\\'", text)


def _normalize_hotkey_keys(raw: str) -> List[str]:
    keys = [part for part in re.split(r"[+,]+|\s+", raw.strip()) if part]
    mapping = {
        "arrowleft": "left",
        "arrowright": "right",
        "arrowup": "up",
        "arrowdown": "down",
        "space": " ",
    }
    return [mapping.get(key.lower(), key.lower()) for key in keys]


def _code_header(thought: str) -> str:
    header = "import pyautogui\nimport time\n"
    cleaned = (thought or "").strip()
    if cleaned:
        header += f"'''\nThought:\n{cleaned}\n'''\n"
    return header


def _write_body(content: str, *, input_swap: bool, press_enter: bool) -> str:
    escaped = _escape_single_quotes(content)
    stripped = escaped.rstrip("\\n").rstrip("\n")
    if input_swap:
        code = (
            f"\nimport pyperclip\npyperclip.copy('{stripped}')\n"
            f"pyautogui.hotkey('ctrl', 'v')\ntime.sleep(0.5)\n"
        )
    else:
        code = f"\npyautogui.write('{stripped}', interval=0.1)\ntime.sleep(0.5)\n"
    if press_enter or content.endswith("\n") or content.endswith("\\n"):
        code += "\npyautogui.press('enter')"
    return code


def tool_call_to_pyautogui(
    step: Step,
    *,
    screen_width: int,
    screen_height: int,
    input_swap: bool = True,
) -> str:
    tool = step.tool_call
    header = _code_header(step.thought)

    if isinstance(tool, AnswerArgs):
        return "DONE"
    if isinstance(tool, WaitArgs):
        return "WAIT"
    if isinstance(tool, FailArgs):
        return "FAIL"

    if isinstance(tool, ClickArgs):
        x, y = scale_point(tool.x, tool.y, screen_width, screen_height)
        if tool.button == "right":
            return header + f"\npyautogui.click({x}, {y}, button='right')"
        if tool.button == "middle":
            return header + f"\npyautogui.click({x}, {y}, button='middle')"
        return header + f"\npyautogui.click({x}, {y}, button='left')"

    if isinstance(tool, DoubleClickArgs):
        x, y = scale_point(tool.x, tool.y, screen_width, screen_height)
        return header + f"\npyautogui.doubleClick({x}, {y}, button='left')"

    if isinstance(tool, HotkeyArgs):
        keys = _normalize_hotkey_keys(tool.keys)
        key_args = ", ".join(repr(key) for key in keys)
        return header + f"\npyautogui.hotkey({key_args})"

    if isinstance(tool, PressArgs):
        keys = _normalize_hotkey_keys(tool.key)
        key = keys[0] if keys else tool.key
        return header + f"\npyautogui.press({repr(key)})"

    if isinstance(tool, WriteArgs):
        code = header
        if tool.overwrite:
            code += "\npyautogui.hotkey('ctrl', 'a')\n"
        code += _write_body(tool.content, input_swap=input_swap, press_enter=tool.press_enter)
        return code

    if isinstance(tool, DragArgs):
        sx, sy = scale_point(tool.start_x, tool.start_y, screen_width, screen_height)
        ex, ey = scale_point(tool.end_x, tool.end_y, screen_width, screen_height)
        return (
            header
            + f"\npyautogui.moveTo({sx}, {sy})\n"
            + f"pyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
        )

    if isinstance(tool, ScrollArgs):
        direction = tool.direction.lower()
        scroll_amount = 5 if "up" in direction else -5
        if tool.x is not None and tool.y is not None:
            x, y = scale_point(tool.x, tool.y, screen_width, screen_height)
            return header + f"\npyautogui.scroll({scroll_amount}, x={x}, y={y})"
        return header + f"\npyautogui.scroll({scroll_amount})"

    raise ValueError(f"Unsupported tool call: {tool}")


def execute_step(
    step: Step,
    *,
    screen_width: int,
    screen_height: int,
    input_swap: bool = True,
) -> Tuple[str, List[str]]:
    response = step.model_dump_json()
    py_cmd = tool_call_to_pyautogui(
        step,
        screen_width=screen_width,
        screen_height=screen_height,
        input_swap=input_swap,
    )
    return response, [py_cmd]
