from __future__ import annotations

import base64
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent

from mm_agents.coact.computer_exec import (
    action_step_cost,
    actions_from_computer_call,
    append_tool_event,
    execute_cua_actions_http,
    load_mcp_state,
    save_mcp_state,
    save_step_screenshot,
)

mcp = FastMCP("desktop")


def _env_required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _save_path() -> str:
    return os.path.abspath(_env_required("SAVE_PATH"))


def _http_server() -> str:
    return _env_required("OSWORLD_SERVER_URL").rstrip("/")


def _max_steps() -> int:
    return int(os.environ.get("MAX_STEPS", "25"))


def _sleep_after_execution() -> float:
    return float(os.environ.get("SLEEP_AFTER_EXECUTION", "0.3"))


@mcp.tool()
def computer(
    actions: Optional[List[Dict[str, Any]]] = None,
    action: Optional[Dict[str, Any]] = None,
) -> CallToolResult:
    """Interact with the desktop using OpenAI computer-use style actions."""
    payload: Dict[str, Any] = {}
    if actions is not None:
        payload["actions"] = actions
    if action is not None:
        payload["action"] = action
    parsed_actions = actions_from_computer_call(payload)
    if not parsed_actions:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: provide a non-empty actions array.")]
        )

    save_path = _save_path()
    state = load_mcp_state(save_path)
    step_count = int(state.get("step_count", 0))
    step_index = int(state.get("step_index", 1))

    batch_cost = sum(action_step_cost(a) for a in parsed_actions)
    if step_count + batch_cost > _max_steps():
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"Step budget exhausted ({step_count}/{_max_steps()}). "
                        "Reply with IDK if you cannot finish."
                    ),
                )
            ]
        )

    try:
        screenshot = execute_cua_actions_http(
            _http_server(),
            parsed_actions,
            sleep_after_execution=_sleep_after_execution(),
        )
    except Exception as exc:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error executing actions: {exc}")]
        )

    step_count += batch_cost
    step_index += 1
    save_step_screenshot(save_path, step_index, screenshot)
    call_id = str(uuid.uuid4())
    append_tool_event(
        save_path,
        {
            "type": "computer_call",
            "call_id": call_id,
            "step_index": step_index,
            "actions": parsed_actions,
        },
    )
    save_mcp_state(save_path, {"step_index": step_index, "step_count": step_count})

    remaining = _max_steps() - step_count
    summary = (
        f"Executed {len(parsed_actions)} action(s). step_index={step_index} "
        f"step_count={step_count}/{_max_steps()} remaining={remaining}."
    )
    return CallToolResult(
        content=[
            TextContent(type="text", text=summary),
            ImageContent(
                type="image",
                data=base64.b64encode(screenshot).decode("utf-8"),
                mimeType="image/png",
            ),
        ]
    )


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
