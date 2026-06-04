"""Pydantic schema for Holo3 agent-loop structured output."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

MouseButton = Literal["left", "right", "middle"]
ScrollDirection = Literal["up", "down", "left", "right"]


class ClickArgs(BaseModel):
    """Click at (x, y) coordinates."""

    tool_name: Literal["click"]
    element: str = Field(description="Detailed description of the target UI element to click on")
    x: int = Field(ge=0, le=1000, description="X coordinate as integer in [0, 1000]")
    y: int = Field(ge=0, le=1000, description="Y coordinate as integer in [0, 1000]")
    button: MouseButton = Field(default="left", description="Mouse button to click")


class DoubleClickArgs(BaseModel):
    """Double-click at (x, y) coordinates."""

    tool_name: Literal["double_click"]
    element: str = Field(description="Detailed description of the target UI element to double-click")
    x: int = Field(ge=0, le=1000, description="X coordinate as integer in [0, 1000]")
    y: int = Field(ge=0, le=1000, description="Y coordinate as integer in [0, 1000]")


class WriteArgs(BaseModel):
    """Type text into the currently focused element without clicking first."""

    tool_name: Literal["write"]
    content: str = Field(description="Content to write")
    press_enter: bool = Field(default=False, description="Whether to press Enter after typing")
    overwrite: bool = Field(default=False, description="Whether to clear existing text before typing")


class HotkeyArgs(BaseModel):
    """Press a key combination."""

    tool_name: Literal["hotkey"]
    keys: str = Field(description="Key combination, e.g. 'ctrl a' or 'ctrl shift s'")


class PressArgs(BaseModel):
    """Press and release a single key."""

    tool_name: Literal["press"]
    key: str = Field(description="Key to press, e.g. 'enter' or 'tab'")


class ScrollArgs(BaseModel):
    """Scroll the mouse wheel."""

    tool_name: Literal["scroll"]
    direction: ScrollDirection = Field(description="Scroll direction")
    element: str | None = Field(default=None, description="Optional description of scroll target")
    x: int | None = Field(default=None, ge=0, le=1000, description="Optional X coordinate in [0, 1000]")
    y: int | None = Field(default=None, ge=0, le=1000, description="Optional Y coordinate in [0, 1000]")


class DragArgs(BaseModel):
    """Drag from start to end coordinates."""

    tool_name: Literal["drag"]
    element: str | None = Field(default=None, description="Optional description of drag target")
    start_x: int = Field(ge=0, le=1000, description="Start X coordinate in [0, 1000]")
    start_y: int = Field(ge=0, le=1000, description="Start Y coordinate in [0, 1000]")
    end_x: int = Field(ge=0, le=1000, description="End X coordinate in [0, 1000]")
    end_y: int = Field(ge=0, le=1000, description="End Y coordinate in [0, 1000]")


class WaitArgs(BaseModel):
    """Wait for the UI to update."""

    tool_name: Literal["wait"]


class AnswerArgs(BaseModel):
    """Provide a final answer and finish the task."""

    tool_name: Literal["answer"]
    content: str = Field(description="The answer content")


class FailArgs(BaseModel):
    """Indicate the task cannot be completed."""

    tool_name: Literal["fail"]
    reason: str = Field(description="Why the task cannot be completed")


ToolCall = (
    ClickArgs
    | DoubleClickArgs
    | WriteArgs
    | HotkeyArgs
    | PressArgs
    | ScrollArgs
    | DragArgs
    | WaitArgs
    | AnswerArgs
    | FailArgs
)


class Step(BaseModel):
    """One agent-loop step emitted by Holo3."""

    note: str | None = Field(
        default=None,
        description="Task-relevant information extracted from the previous observation. Keep empty if no new info.",
    )
    thought: str = Field(description="Reasoning about next steps")
    tool_call: ToolCall


def step_json_schema() -> dict:
    return Step.model_json_schema()


def build_system_prompt(instruction: str) -> str:
    schema = step_json_schema()
    schema_block = json.dumps(schema, indent=2)
    return (
        "You are a GUI agent controlling an Ubuntu desktop. "
        "You receive screenshots of the current desktop state and must choose exactly one tool per step.\n\n"
        "Rules:\n"
        "- Coordinates are integers in [0, 1000], normalized to the screenshot you receive.\n"
        "- Origin is top-left.\n"
        "- Use `note` to record durable facts (URLs, IDs, intermediate answers) the model will need later.\n"
        "- Use `thought` for a one-line plan for the next action.\n"
        "- When the task is complete, call `answer`.\n"
        "- If the task is infeasible, call `fail`.\n"
        "- If you must wait for the UI to update, call `wait`.\n\n"
        f"User Instruction:\n{instruction}\n\n"
        f"<output_format>\n```json\n{schema_block}\n```\n</output_format>"
    )
