from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
GUI_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = GUI_ROOT.parent
SKILL_DIR = REPO_ROOT / ".cursor" / "skills" / "live-emulator"
MIN_SETTINGS_PATH = SKILL_DIR / "min-settings.json"

if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))


DEFAULT_DESKTOP_SERVER_URL = os.getenv("LIVE_EMULATOR_SERVER_URL", "http://localhost:50003")
DEFAULT_TOKEN = os.getenv("OSWORLD_TOKEN") or os.getenv("DART_USER_TOKEN") or "dart"


class LiveEmulatorError(RuntimeError):
    pass


@dataclass
class EmulatorInfo:
    emulator_id: str
    server_port: int
    vnc_port: int
    chromium_port: int
    vlc_port: int
    token: str | None = None
    start_time: float | None = None
    duration_minutes: int | None = None
    container_id: str | None = None
    desktop_server_url: str = DEFAULT_DESKTOP_SERVER_URL

    @property
    def server_url(self) -> str:
        return f"http://localhost:{self.server_port}"

    @property
    def vnc_url(self) -> str:
        return f"http://localhost:{self.vnc_port}"

    @property
    def chromium_debug_url(self) -> str:
        return f"http://localhost:{self.chromium_port}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "emulator_id": self.emulator_id,
            "server_port": self.server_port,
            "server_url": self.server_url,
            "vnc_port": self.vnc_port,
            "vnc_url": self.vnc_url,
            "chromium_port": self.chromium_port,
            "chromium_debug_url": self.chromium_debug_url,
            "vlc_port": self.vlc_port,
            "token": self.token,
            "start_time": self.start_time,
            "duration_minutes": self.duration_minutes,
            "container_id": self.container_id,
            "desktop_server_url": self.desktop_server_url,
        }


@dataclass
class ScreenSize:
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return {"width": self.width, "height": self.height}


@dataclass
class WindowGeometry:
    window_id: str
    title: str
    wm_class: list[str]
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "title": self.title,
            "wm_class": self.wm_class,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class GeometryThreshold:
    source: str
    selected_key: str
    app: str | None
    task_id: str | None
    min_width: int
    min_height: int
    margins: dict[str, int]
    coordinate_policy: dict[str, Any]
    notes: str | None = None
    screen_size: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source,
            "selected_key": self.selected_key,
            "app": self.app,
            "task_id": self.task_id,
            "min_width": self.min_width,
            "min_height": self.min_height,
            "margins": dict(self.margins),
            "coordinate_policy": dict(self.coordinate_policy),
            "notes": self.notes,
        }
        if self.screen_size is not None:
            payload["screen_size"] = list(self.screen_size)
        return payload


@dataclass
class GeometryBounds:
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    min_width: int
    max_width: int
    min_height: int
    max_height: int

    def to_dict(self) -> dict[str, int]:
        return {
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "min_width": self.min_width,
            "max_width": self.max_width,
            "min_height": self.min_height,
            "max_height": self.max_height,
        }


def _auth_headers(token: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request(
    method: str,
    url: str,
    *,
    token: str | None = None,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
    stream: bool = False,
) -> requests.Response:
    try:
        response = requests.request(
            method,
            url,
            headers=_auth_headers(token),
            json=json_body,
            params=params,
            timeout=timeout,
            stream=stream,
        )
    except requests.exceptions.RequestException as exc:
        raise LiveEmulatorError(f"{method} {url} failed: {exc}") from exc
    if response.status_code >= 400:
        detail = response.text.strip()
        raise LiveEmulatorError(f"{method} {url} failed with {response.status_code}: {detail}")
    return response


def start_emulator(server_url: str, token: str) -> EmulatorInfo:
    response = _request("POST", f"{server_url}/start_emulator", token=token, json_body={"token": token}, timeout=60)
    payload = response.json()
    data = payload.get("data") or {}
    return normalize_emulator_info(data, server_url=server_url, token=token)


def stop_emulator(server_url: str, emulator_id: str, token: str | None) -> dict[str, Any]:
    response = _request(
        "DELETE",
        f"{server_url}/emulators/{emulator_id}",
        token=token,
        timeout=30,
    )
    return response.json()


def list_emulators(server_url: str, token: str | None, *, filter_by_token: bool = True) -> list[EmulatorInfo]:
    params = {"token": token} if token and filter_by_token else None
    response = _request("GET", f"{server_url}/emulators", token=token, params=params, timeout=30)
    payload = response.json()
    if not isinstance(payload, list):
        raise LiveEmulatorError(f"Unexpected /emulators payload: {payload!r}")
    return [normalize_emulator_info(item, server_url=server_url, token=item.get("token")) for item in payload]


def get_emulator(server_url: str, emulator_id: str, token: str | None) -> EmulatorInfo:
    for item in list_emulators(server_url, token, filter_by_token=False):
        if item.emulator_id == emulator_id:
            return item
    raise LiveEmulatorError(f"Emulator not found: {emulator_id}")


def normalize_emulator_info(item: dict[str, Any], *, server_url: str, token: str | None) -> EmulatorInfo:
    try:
        return EmulatorInfo(
            emulator_id=str(item["emulator_id"]),
            server_port=int(item["server_port"]),
            vnc_port=int(item["vnc_port"]),
            chromium_port=int(item["chromium_port"]),
            vlc_port=int(item["vlc_port"]),
            token=(str(item["token"]) if item.get("token") is not None else token),
            start_time=item.get("start_time"),
            duration_minutes=item.get("duration_minutes"),
            container_id=item.get("container_id"),
            desktop_server_url=server_url,
        )
    except KeyError as exc:
        raise LiveEmulatorError(f"Missing emulator field {exc!s}: {item!r}") from exc


def resolve_task_config(task: str | None, task_json: str | None) -> tuple[dict[str, Any], Path, str | None]:
    if bool(task) == bool(task_json):
        raise LiveEmulatorError("Provide exactly one of --task or --task-json.")

    if task_json:
        path = resolve_input_path(task_json)
        payload = load_json(path)
        return payload, path, _task_domain_from_path(path)

    assert task is not None
    domain, task_id = parse_task_ref(task)
    path = GUI_ROOT / "evaluation_examples" / "examples" / domain / f"{task_id}.json"
    if not path.is_file():
        raise LiveEmulatorError(f"Task config not found: {path}")
    payload = load_json(path)
    return payload, path, domain


def parse_task_ref(task: str) -> tuple[str, str]:
    cleaned = task.strip().strip("/")
    if "/" not in cleaned:
        raise LiveEmulatorError("Task must look like domain/task_id.")
    domain, task_id = cleaned.split("/", 1)
    if not domain or not task_id:
        raise LiveEmulatorError("Task must look like domain/task_id.")
    return normalize_app_name(domain), task_id


def normalize_app_name(app: str | None) -> str | None:
    return app.lower().strip() if app else None


def resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                REPO_ROOT / path,
                GUI_ROOT / path,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise LiveEmulatorError(f"File not found: {raw_path}")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LiveEmulatorError(f"Expected JSON object in {path}")
    return payload


def _task_domain_from_path(path: Path) -> str | None:
    parts = path.parts
    if "examples" in parts:
        idx = parts.index("examples")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def default_cache_dir(task_id: str) -> Path:
    path = GUI_ROOT / "cache" / "live_emulator" / task_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_task(
    emulator: EmulatorInfo,
    task_config: dict[str, Any],
    *,
    enable_task_proxy: bool = False,
    client_password: str = "",
) -> dict[str, Any]:
    from desktop_env.controllers.setup import SetupController

    task_id = str(task_config.get("id") or "unknown_task")
    cache_dir = default_cache_dir(task_id)
    controller = SetupController(
        vm_ip="localhost",
        server_port=emulator.server_port,
        chromium_port=emulator.chromium_port,
        vlc_port=emulator.vlc_port,
        cache_dir=str(cache_dir),
        client_password=client_password,
    )
    use_proxy = bool(enable_task_proxy and task_config.get("proxy", False))
    if use_proxy:
        controller._proxy_setup(client_password)
    config = task_config.get("config", [])
    if not isinstance(config, list):
        raise LiveEmulatorError("Task config must contain a list under 'config'.")
    success = controller.setup(config, use_proxy=use_proxy)
    if not success:
        raise LiveEmulatorError(f"Task setup failed for {task_id}")
    return {
        "task_id": task_id,
        "instruction": task_config.get("instruction"),
        "task_proxy_requested": bool(task_config.get("proxy", False)),
        "task_proxy_enabled": use_proxy,
        "cache_dir": str(cache_dir),
    }


def build_python_command(code: str, *, wrap_pyautogui: bool) -> list[str]:
    snippet = code.strip()
    if wrap_pyautogui:
        snippet = f"import pyautogui; import time; pyautogui.FAILSAFE = False; {snippet}"
    return ["python", "-c", snippet]


def execute_python(server_port: int, command: list[str], *, timeout: int = 90) -> dict[str, Any]:
    response = _request(
        "POST",
        f"http://localhost:{server_port}/execute",
        json_body={"command": command, "shell": False},
        timeout=timeout,
    )
    payload = response.json()
    if not isinstance(payload, dict):
        raise LiveEmulatorError(f"Unexpected execute payload: {payload!r}")
    return payload


def execute_python_json(server_port: int, code: str, *, timeout: int = 90) -> dict[str, Any]:
    payload = execute_python(server_port, build_python_command(code, wrap_pyautogui=False), timeout=timeout)
    output = str(payload.get("output", "")).strip()
    error = str(payload.get("error", "")).strip()
    returncode = int(payload.get("returncode", -1))

    parsed: dict[str, Any] | None = None
    for line in reversed([line.strip() for line in output.splitlines() if line.strip()]):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
            break

    if returncode != 0:
        if parsed and "error" in parsed:
            raise LiveEmulatorError(str(parsed["error"]))
        raise LiveEmulatorError(error or output or "Guest Python execution failed.")

    if parsed is None:
        raise LiveEmulatorError(f"Expected JSON output from guest Python, got: {output!r}")
    return parsed


def execute_step_action(server_port: int, action: dict[str, Any], *, timeout: int = 90, sleep_after: float = 0.5) -> dict[str, Any]:
    action_type = str(action.get("type", "")).strip().lower()
    if action_type == "click":
        x = int(action["x"])
        y = int(action["y"])
        button = str(action.get("button") or "left")
        code = f"import pyautogui, time; pyautogui.FAILSAFE=False; pyautogui.click({x}, {y}, button={button!r}); time.sleep({sleep_after})"
        return execute_python(server_port, build_python_command(code, wrap_pyautogui=False), timeout=timeout)
    if action_type == "type":
        text = str(action.get("text", ""))
        code = f"import pyautogui, time; pyautogui.FAILSAFE=False; pyautogui.typewrite({text!r}); time.sleep({sleep_after})"
        return execute_python(server_port, build_python_command(code, wrap_pyautogui=False), timeout=timeout)
    if action_type == "keypress":
        keys = [str(key).lower() for key in (action.get("keys") or []) if str(key).strip()]
        if not keys:
            raise LiveEmulatorError("Keypress action missing keys.")
        if len(keys) == 1:
            snippet = f"pyautogui.press({keys[0]!r})"
        else:
            joined = ", ".join(repr(key) for key in keys)
            snippet = f"pyautogui.hotkey({joined})"
        code = f"import pyautogui, time; pyautogui.FAILSAFE=False; {snippet}; time.sleep({sleep_after})"
        return execute_python(server_port, build_python_command(code, wrap_pyautogui=False), timeout=timeout)
    if action_type == "screenshot":
        return {"status": "success", "returncode": 0, "output": "", "error": ""}
    raise LiveEmulatorError(f"Unsupported action type: {action_type}")


def fetch_screenshot(server_port: int, *, attempts: int = 5, delay: float = 0.5) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = _request("GET", f"http://localhost:{server_port}/screenshot", timeout=20, stream=True)
            content = response.content
            if content.startswith(b"\x89PNG\r\n\x1a\n"):
                return content
            raise LiveEmulatorError(f"Screenshot response was not a PNG: {content[:80]!r}")
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(delay * (attempt + 1))
    raise LiveEmulatorError(f"Failed to fetch screenshot after {attempts} attempts: {last_error}") from last_error


def fetch_accessibility_tree(server_port: int) -> str:
    response = _request("GET", f"http://localhost:{server_port}/accessibility", timeout=30)
    payload = response.json()
    if not isinstance(payload, dict) or "AT" not in payload:
        raise LiveEmulatorError(f"Unexpected accessibility payload: {payload!r}")
    return str(payload["AT"])


def fetch_screen_size(server_port: int) -> ScreenSize:
    response = _request("POST", f"http://localhost:{server_port}/screen_size", timeout=30)
    payload = response.json()
    if not isinstance(payload, dict):
        raise LiveEmulatorError(f"Unexpected /screen_size payload: {payload!r}")
    return ScreenSize(width=int(payload["width"]), height=int(payload["height"]))


def normalize_margins(raw: dict[str, Any] | None, fallback: dict[str, int] | None = None) -> dict[str, int]:
    fallback = fallback or {"left": 0, "top": 0, "right": 0, "bottom": 0}
    raw = raw or {}
    return {
        "left": int(raw.get("left", fallback["left"])),
        "top": int(raw.get("top", fallback["top"])),
        "right": int(raw.get("right", fallback["right"])),
        "bottom": int(raw.get("bottom", fallback["bottom"])),
    }


def load_min_settings() -> dict[str, Any]:
    if not MIN_SETTINGS_PATH.is_file():
        raise LiveEmulatorError(f"Missing geometry settings file: {MIN_SETTINGS_PATH}")
    return load_json(MIN_SETTINGS_PATH)


def resolve_geometry_threshold(*, task: str | None = None, app: str | None = None) -> GeometryThreshold:
    settings = load_min_settings()
    defaults = settings.get("defaults", {})
    entries = settings.get("entries", [])
    coordinate_policy = settings.get("coordinate_policy", {})
    if not isinstance(entries, list):
        raise LiveEmulatorError("Geometry settings 'entries' must be a list.")

    requested_app = normalize_app_name(app)
    requested_task_id: str | None = None
    if task:
        task_app, task_id = parse_task_ref(task)
        requested_app = requested_app or task_app
        requested_task_id = task_id

    default_margins = normalize_margins(defaults.get("margins"))
    selected_entry: dict[str, Any] | None = None
    source = "defaults"
    selected_key = "defaults"

    if requested_task_id is not None:
        for entry in entries:
            if (
                str(entry.get("task_id", "")).strip() == requested_task_id
                and normalize_app_name(entry.get("app")) == requested_app
            ):
                selected_entry = entry
                source = "task"
                selected_key = f"task:{requested_app}/{requested_task_id}"
                break

    if selected_entry is None and requested_app is not None:
        for entry in entries:
            if normalize_app_name(entry.get("app")) == requested_app and not entry.get("task_id"):
                selected_entry = entry
                source = "app"
                selected_key = f"app:{requested_app}"
                break

    if selected_entry is None and entries:
        selected_entry = entries[0]
        source = "example"
        selected_key = "example:entries[0]"

    entry = selected_entry or {}
    margins = normalize_margins(entry.get("margins"), default_margins)
    resolved_app = normalize_app_name(entry.get("app")) or requested_app
    resolved_task_id = str(entry.get("task_id")).strip() if entry.get("task_id") else requested_task_id

    threshold = GeometryThreshold(
        source=source,
        selected_key=selected_key,
        app=resolved_app,
        task_id=resolved_task_id,
        min_width=int(entry.get("min_width", defaults.get("min_width", 900))),
        min_height=int(entry.get("min_height", defaults.get("min_height", 700))),
        margins=margins,
        coordinate_policy=coordinate_policy if isinstance(coordinate_policy, dict) else {},
        notes=str(entry.get("notes")).strip() if entry.get("notes") else None,
        screen_size=tuple(entry.get("screen_size")) if entry.get("screen_size") else None,
    )
    return threshold


def resolve_window_target(window_name: str | None, by_class: bool) -> tuple[str, str | None]:
    if not window_name:
        return "active", None
    return ("class" if by_class else "title"), window_name


def query_window_geometry(
    server_port: int,
    *,
    window_name: str | None = None,
    by_class: bool = False,
    timeout: int = 60,
) -> WindowGeometry:
    target_mode, target_value = resolve_window_target(window_name, by_class)
    script = build_window_query_script(target_mode, target_value)
    payload = execute_python_json(server_port, script, timeout=timeout)
    return normalize_window_geometry(payload)


def normalize_window_geometry(payload: dict[str, Any]) -> WindowGeometry:
    return WindowGeometry(
        window_id=str(payload["window_id"]),
        title=str(payload.get("title", "")),
        wm_class=[str(item) for item in payload.get("wm_class", [])],
        x=int(payload["x"]),
        y=int(payload["y"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
    )


def build_window_query_script(target_mode: str, target_value: str | None) -> str:
    target_value_literal = "None" if target_value is None else json.dumps(target_value)
    return f"""
import json
import subprocess
from Xlib import X, display

TARGET_MODE = {json.dumps(target_mode)}
TARGET_VALUE = {target_value_literal}

d = display.Display()
root = d.screen().root
window = None
active_window_id = None

if TARGET_MODE == "active":
    active_atom = d.intern_atom("_NET_ACTIVE_WINDOW")
    prop = root.get_full_property(active_atom, X.AnyPropertyType)
    if prop and len(prop.value):
        active_window_id = hex(int(prop.value[0])).lower()

result = subprocess.run(["wmctrl", "-lGx"], capture_output=True, text=True, check=True)
for raw_line in result.stdout.splitlines():
    line = raw_line.strip()
    if not line:
        continue
    parts = line.split(None, 8)
    if len(parts) < 8:
        continue
    entry = {{
        "window_id": parts[0].lower(),
        "window_id_int": int(parts[0], 16),
        "desktop": int(parts[1]),
        "x": int(parts[2]),
        "y": int(parts[3]),
        "width": int(parts[4]),
        "height": int(parts[5]),
        "wm_class": parts[6],
        "host": parts[7],
        "title": parts[8] if len(parts) > 8 else "",
    }}
    if TARGET_MODE == "active" and active_window_id and entry["window_id_int"] == int(active_window_id, 16):
        window = entry
        break
    if TARGET_MODE == "title" and TARGET_VALUE and TARGET_VALUE.lower() in entry["title"].lower():
        window = entry
        break
    if TARGET_MODE == "class" and TARGET_VALUE:
        class_parts = [item.lower() for item in entry["wm_class"].split(".") if item]
        if TARGET_VALUE.lower() in class_parts or TARGET_VALUE.lower() in entry["wm_class"].lower():
            window = entry
            break

if window is None:
    print(json.dumps({{"error": "Window not found"}}))
    raise SystemExit(1)

window["wm_class"] = [item for item in window["wm_class"].split(".") if item]
window.pop("window_id_int", None)
window.pop("desktop", None)
window.pop("host", None)
print(json.dumps(window))
""".strip()


def compute_size_limits(screen: ScreenSize, threshold: GeometryThreshold) -> GeometryBounds:
    max_width = screen.width - threshold.margins["left"] - threshold.margins["right"]
    max_height = screen.height - threshold.margins["top"] - threshold.margins["bottom"]
    if max_width < threshold.min_width or max_height < threshold.min_height:
        raise LiveEmulatorError(
            "Configured minimum window size exceeds the available screen size after margins."
        )
    return GeometryBounds(
        min_x=threshold.margins["left"],
        max_x=threshold.margins["left"],
        min_y=threshold.margins["top"],
        max_y=threshold.margins["top"],
        min_width=threshold.min_width,
        max_width=max_width,
        min_height=threshold.min_height,
        max_height=max_height,
    )


def clamp_window_size(
    width: int,
    height: int,
    screen: ScreenSize,
    threshold: GeometryThreshold,
    *,
    allow_below_min: bool = False,
) -> tuple[int, int, dict[str, bool]]:
    limits = compute_size_limits(screen, threshold)
    requested_width = int(width)
    requested_height = int(height)
    applied_width = requested_width
    applied_height = requested_height
    clamped = {"width_to_min": False, "height_to_min": False, "width_to_screen": False, "height_to_screen": False}

    if not allow_below_min and applied_width < limits.min_width:
        applied_width = limits.min_width
        clamped["width_to_min"] = True
    if not allow_below_min and applied_height < limits.min_height:
        applied_height = limits.min_height
        clamped["height_to_min"] = True
    if applied_width > limits.max_width:
        applied_width = limits.max_width
        clamped["width_to_screen"] = True
    if applied_height > limits.max_height:
        applied_height = limits.max_height
        clamped["height_to_screen"] = True

    if applied_width <= 0 or applied_height <= 0:
        raise LiveEmulatorError("Window size must remain positive after clamping.")
    return applied_width, applied_height, clamped


def compute_coordinate_limits(
    screen: ScreenSize,
    threshold: GeometryThreshold,
    *,
    width: int,
    height: int,
) -> GeometryBounds:
    size_limits = compute_size_limits(screen, threshold)
    max_x = screen.width - int(width) - threshold.margins["right"]
    max_y = screen.height - int(height) - threshold.margins["bottom"]
    min_x = threshold.margins["left"]
    min_y = threshold.margins["top"]
    if max_x < min_x or max_y < min_y:
        raise LiveEmulatorError("Requested window size leaves no valid on-screen coordinates.")
    return GeometryBounds(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        min_width=size_limits.min_width,
        max_width=size_limits.max_width,
        min_height=size_limits.min_height,
        max_height=size_limits.max_height,
    )


def clamp_window_position(
    x: int,
    y: int,
    bounds: GeometryBounds,
    *,
    allow_offscreen: bool = False,
) -> tuple[int, int, dict[str, bool]]:
    requested_x = int(x)
    requested_y = int(y)
    applied_x = requested_x
    applied_y = requested_y
    clamped = {"x": False, "y": False}

    if not allow_offscreen:
        if applied_x < bounds.min_x:
            applied_x = bounds.min_x
            clamped["x"] = True
        if applied_x > bounds.max_x:
            applied_x = bounds.max_x
            clamped["x"] = True
        if applied_y < bounds.min_y:
            applied_y = bounds.min_y
            clamped["y"] = True
        if applied_y > bounds.max_y:
            applied_y = bounds.max_y
            clamped["y"] = True
    return applied_x, applied_y, clamped


def apply_window_geometry(
    server_port: int,
    window: WindowGeometry,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    timeout: int = 60,
) -> WindowGeometry:
    geometry_spec = f"0,{int(x)},{int(y)},{int(width)},{int(height)}"
    script = f"""
import json
import subprocess
import time

WINDOW_ID = {json.dumps(window.window_id)}
GEOMETRY_SPEC = {json.dumps(geometry_spec)}

subprocess.run(["wmctrl", "-i", "-r", WINDOW_ID, "-b", "remove,maximized_vert,maximized_horz"], check=False)
subprocess.run(["wmctrl", "-i", "-r", WINDOW_ID, "-b", "remove,fullscreen"], check=False)
subprocess.run(["wmctrl", "-i", "-r", WINDOW_ID, "-e", GEOMETRY_SPEC], check=True)
time.sleep(0.2)

result = subprocess.run(["wmctrl", "-lGx"], capture_output=True, text=True, check=True)
for raw_line in result.stdout.splitlines():
    parts = raw_line.split(None, 8)
    if len(parts) < 8:
        continue
    if parts[0].lower() != WINDOW_ID.lower():
        continue
    print(json.dumps({{
        "window_id": parts[0].lower(),
        "title": parts[8] if len(parts) > 8 else "",
        "wm_class": [item for item in parts[6].split(".") if item],
        "x": int(parts[2]),
        "y": int(parts[3]),
        "width": int(parts[4]),
        "height": int(parts[5])
    }}))
    break
else:
    print(json.dumps({{"error": "Window disappeared after wmctrl update"}}))
    raise SystemExit(1)
""".strip()
    payload = execute_python_json(server_port, script, timeout=timeout)
    return normalize_window_geometry(payload)


def write_output(payload: Any, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if isinstance(payload, list):
        for item in payload:
            write_output(item, output_format)
        return

    if isinstance(payload, (EmulatorInfo, ScreenSize, WindowGeometry, GeometryThreshold, GeometryBounds)):
        payload = payload.to_dict()

    if not isinstance(payload, dict):
        print(payload)
        return

    for key, value in payload.items():
        print(f"{key}: {value}")


def timestamp_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{int(time.time())}{suffix}"
