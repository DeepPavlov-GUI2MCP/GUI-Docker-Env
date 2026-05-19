from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Union

import requests

PYAUTOGUI_PREFIX = (
    "import pyautogui; import time; pyautogui.FAILSAFE = False; {command}"
)


def _normalize_key(key: str) -> str:
    normalized = str(key).strip().lower().replace("arrow", "")
    aliases = {
        "ctrl": "ctrl",
        "control": "ctrl",
        "alt": "alt",
        "option": "alt",
        "shift": "shift",
        "meta": "win",
        "cmd": "win",
        "command": "win",
        "super": "win",
        "windows": "win",
        "return": "enter",
        "esc": "escape",
        "pgup": "pageup",
        "pgdn": "pagedown",
        "spacebar": "space",
        "del": "delete",
    }
    return aliases.get(normalized, normalized)


def _wrap_with_modifiers(command: str, keys: Optional[List[str]]) -> str:
    if not keys:
        return command
    normalized_keys = [_normalize_key(key) for key in keys if str(key).strip()]
    if not normalized_keys:
        return command
    keydowns = [f"pyautogui.keyDown('{key}')" for key in normalized_keys]
    keyups = [f"pyautogui.keyUp('{key}')" for key in reversed(normalized_keys)]
    return "; ".join(keydowns + [command] + keyups)


def cua_to_pyautogui(action: Any) -> str:
    def fld(key: str, default: Any = None) -> Any:
        return action.get(key, default) if isinstance(action, dict) else getattr(action, key, default)

    act_type = fld("type")
    if not isinstance(act_type, str):
        act_type = str(act_type).split(".")[-1]
    act_type = act_type.lower()

    if act_type in ["click", "double_click"]:
        button = fld("button", "left")
        if button == 1 or button == "left":
            button = "left"
        elif button == 2 or button == "middle":
            button = "middle"
        elif button == 3 or button == "right":
            button = "right"
        elif button == "wheel":
            button = "middle"

        if act_type == "click":
            return _wrap_with_modifiers(
                f"pyautogui.click({fld('x')}, {fld('y')}, button='{button}')",
                fld("keys"),
            )
        if act_type == "double_click":
            return _wrap_with_modifiers(
                f"pyautogui.doubleClick({fld('x')}, {fld('y')}, button='{button}')",
                fld("keys"),
            )

    if act_type == "scroll":
        scroll_y = int(round(-fld("scroll_y", 0) / 100))
        scroll_x = int(round(fld("scroll_x", 0) / 100))
        commands: List[str] = []
        if scroll_y:
            commands.append(f"pyautogui.scroll({scroll_y}, x={fld('x', 0)}, y={fld('y', 0)})")
        if scroll_x:
            commands.append(f"pyautogui.hscroll({scroll_x}, x={fld('x', 0)}, y={fld('y', 0)})")
        return _wrap_with_modifiers("; ".join(commands) if commands else "WAIT", fld("keys"))

    if act_type == "drag":
        path = fld("path", [{"x": 0, "y": 0}, {"x": 0, "y": 0}])
        cmd = f"pyautogui.moveTo({path[0]['x']}, {path[0]['y']}, _pause=False); "
        cmd += f"pyautogui.dragTo({path[-1]['x']}, {path[-1]['y']}, duration=0.5, button='left')"
        return _wrap_with_modifiers(cmd, fld("keys"))

    if act_type == "move":
        return _wrap_with_modifiers(f"pyautogui.moveTo({fld('x')}, {fld('y')})", fld("keys"))

    if act_type == "keypress":
        keys = [_normalize_key(key) for key in (fld("keys", []) or [fld("key")]) if key]
        if len(keys) == 1:
            return f"pyautogui.press('{keys[0].lower()}')"
        return f"pyautogui.hotkey({', '.join(repr(key) for key in keys)})"

    if act_type == "type":
        text = str(fld("text", ""))
        return "pyautogui.typewrite({:})".format(repr(text))

    if act_type == "wait":
        return "WAIT"

    return "WAIT"


def item_to_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")
    raise TypeError(f"Unsupported item type: {type(item)!r}")


def actions_from_computer_call(action_call: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions = action_call.get("actions")
    if actions:
        return [item_to_dict(action) for action in actions]
    action = action_call.get("action")
    if action:
        return [item_to_dict(action)]
    return []


def action_step_cost(action: Dict[str, Any]) -> int:
    return 0 if action.get("type") == "screenshot" else 1


def save_step_screenshot(save_path: str, step_index: int, screenshot: bytes) -> str:
    path = os.path.abspath(os.path.join(save_path, f"step_{step_index:04d}.png"))
    with open(path, "wb") as f:
        f.write(screenshot)
    return path


def fetch_screenshot(http_server: str, *, retry_times: int = 3, retry_interval: float = 5.0) -> bytes:
    for _ in range(retry_times):
        try:
            response = requests.get(f"{http_server.rstrip('/')}/screenshot", timeout=10)
            if response.status_code == 200 and response.content:
                content_type = response.headers.get("Content-Type", "")
                data = response.content
                if _is_valid_image(data, content_type):
                    return data
        except Exception:
            pass
        time.sleep(retry_interval)
    raise RuntimeError(f"Failed to fetch screenshot from {http_server}")


def _is_valid_image(data: bytes, content_type: str) -> bool:
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return True
    if content_type and "image/" in content_type:
        return True
    return False


def execute_pyautogui_command(http_server: str, py_cmd: str, *, timeout: int = 90) -> None:
    if py_cmd == "WAIT":
        return
    command_list = ["python", "-c", PYAUTOGUI_PREFIX.format(command=py_cmd)]
    payload = json.dumps({"command": command_list, "shell": False})
    response = requests.post(
        f"{http_server.rstrip('/')}/execute",
        headers={"Content-Type": "application/json"},
        data=payload,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Execute failed ({response.status_code}): {response.text[:500]}")


def execute_cua_actions_http(
    http_server: str,
    actions: List[Dict[str, Any]],
    *,
    sleep_after_execution: float = 0.3,
) -> bytes:
    latest_screenshot: Optional[bytes] = None
    for action in actions:
        if action.get("type") == "screenshot":
            latest_screenshot = fetch_screenshot(http_server)
            continue
        py_cmd = cua_to_pyautogui(action)
        if py_cmd == "WAIT":
            time.sleep(sleep_after_execution)
        else:
            execute_pyautogui_command(http_server, py_cmd)
            time.sleep(sleep_after_execution)
        latest_screenshot = fetch_screenshot(http_server)
    if latest_screenshot is None:
        latest_screenshot = fetch_screenshot(http_server)
    return latest_screenshot


def execute_cua_actions_env(env: Any, actions: List[Dict[str, Any]], *, sleep_after_execution: float = 0.3) -> bytes:
    latest_screenshot: Optional[bytes] = None
    for action in actions:
        if action.get("type") == "screenshot":
            shot = env.controller.get_screenshot()
            latest_screenshot = shot["screenshot"] if isinstance(shot, dict) else shot
            continue
        py_cmd = cua_to_pyautogui(action)
        obs, *_ = env.step(py_cmd, sleep_after_execution)
        latest_screenshot = obs["screenshot"]
    if latest_screenshot is None:
        shot = env.controller.get_screenshot()
        latest_screenshot = shot["screenshot"] if isinstance(shot, dict) else shot
    return latest_screenshot


def load_mcp_state(save_path: str) -> Dict[str, Any]:
    path = os.path.join(save_path, "mcp_state.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"step_index": 1, "step_count": 0}


def save_mcp_state(save_path: str, state: Dict[str, Any]) -> None:
    path = os.path.join(save_path, "mcp_state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def append_tool_event(save_path: str, event: Dict[str, Any]) -> None:
    path = os.path.join(save_path, "tool_events.json")
    payload: Dict[str, Any] = {"events": []}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    payload.setdefault("events", []).append(event)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_tool_events(save_path: str) -> List[Dict[str, Any]]:
    path = os.path.join(save_path, "tool_events.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.get("events", []))
