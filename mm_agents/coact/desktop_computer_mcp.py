from __future__ import annotations

import base64
import os
import sys
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent

from mm_agents.coact.computer_exec import (
    execute_computer_tool_call,
    execute_cua_actions_http,
    load_mcp_state,
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

    save_path = _save_path()
    state = load_mcp_state(save_path)
    result = execute_computer_tool_call(
        payload,
        save_path=save_path,
        step_state=state,
        max_steps=_max_steps(),
        sleep_after_execution=_sleep_after_execution(),
        execute_actions=lambda parsed: execute_cua_actions_http(
            _http_server(),
            parsed,
            sleep_after_execution=_sleep_after_execution(),
        ),
    )
    if result.error or result.screenshot is None:
        return CallToolResult(content=[TextContent(type="text", text=result.summary)])

    return CallToolResult(
        content=[
            TextContent(type="text", text=result.summary),
            ImageContent(
                type="image",
                data=base64.b64encode(result.screenshot).decode("utf-8"),
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
