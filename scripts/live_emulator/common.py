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
    response = requests.request(
        method,
        url,
        headers=_auth_headers(token),
        json=json_body,
        params=params,
        timeout=timeout,
        stream=stream,
    )
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
    task = task.strip().strip("/")
    if "/" not in task:
        raise LiveEmulatorError("Task must look like domain/task_id.")
    domain, task_id = task.split("/", 1)
    path = GUI_ROOT / "evaluation_examples" / "examples" / domain / f"{task_id}.json"
    if not path.is_file():
        raise LiveEmulatorError(f"Task config not found: {path}")
    payload = load_json(path)
    return payload, path, domain


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


def fetch_screenshot(server_port: int) -> bytes:
    response = _request("GET", f"http://localhost:{server_port}/screenshot", timeout=20, stream=True)
    return response.content


def fetch_accessibility_tree(server_port: int) -> str:
    response = _request("GET", f"http://localhost:{server_port}/accessibility", timeout=30)
    payload = response.json()
    if not isinstance(payload, dict) or "AT" not in payload:
        raise LiveEmulatorError(f"Unexpected accessibility payload: {payload!r}")
    return str(payload["AT"])


def write_output(payload: Any, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if isinstance(payload, list):
        for item in payload:
            write_output(item, output_format)
        return

    if isinstance(payload, EmulatorInfo):
        payload = payload.to_dict()

    if not isinstance(payload, dict):
        print(payload)
        return

    for key, value in payload.items():
        print(f"{key}: {value}")


def timestamp_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{int(time.time())}{suffix}"
