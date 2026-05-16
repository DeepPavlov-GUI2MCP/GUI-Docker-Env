from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_MODE = "default"
SUPPORTED_MODES = (DEFAULT_MODE, "search-first", "inspect-source-first")
DEFAULT_GUI_PROTOCOL = "openai"
SUPPORTED_GUI_PROTOCOLS = (DEFAULT_GUI_PROTOCOL, "vllm")


@dataclass(frozen=True)
class BackendSettings:
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    protocol: str = DEFAULT_GUI_PROTOCOL

    def as_llm_config_entry(self) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "api_type": "openai",
            "model": self.model,
        }
        if self.base_url:
            entry["base_url"] = self.base_url
        if self.api_key:
            entry["api_key"] = self.api_key
        return entry

    def as_openai_client_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        elif self.base_url:
            kwargs["api_key"] = "EMPTY"
        return kwargs


@dataclass(frozen=True)
class ResolvedRunConfig:
    config_mode: bool
    config_path: Optional[str]
    provider_name: str
    path_to_vm: Optional[str]
    screen_width: int
    screen_height: int
    sleep_after_execution: float
    region: str
    client_password: str
    orchestrator: BackendSettings
    gui: BackendSettings
    coding: BackendSettings
    enable_coding_agent: bool
    enable_web_search: bool
    mode: str
    orchestrator_max_steps: int
    coding_max_steps: int
    cua_max_steps: int
    cut_off_steps: int
    domain: str
    task_id: Optional[str]
    test_all_meta_path: str
    test_config_base_dir: str
    result_dir: str
    num_envs: int
    log_level: str

    def as_metadata_args(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["orchestrator"] = asdict(self.orchestrator)
        payload["gui"] = asdict(self.gui)
        payload["coding"] = asdict(self.coding)
        return payload


def load_config_file(path: str) -> ResolvedRunConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a YAML object: {path}")

    root = payload.get("coact", payload)
    if not isinstance(root, dict):
        raise ValueError(f"`coact` config must be a YAML object: {path}")

    runtime = _get_mapping(root, "runtime")
    tasks = _get_mapping(root, "tasks")

    mode = _get_optional_str(root, "mode", DEFAULT_MODE)
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode `{mode}` in {path}. Expected one of {SUPPORTED_MODES}.")

    resolved = ResolvedRunConfig(
        config_mode=True,
        config_path=str(config_path),
        provider_name=_get_optional_str(runtime, "provider_name", "aws"),
        path_to_vm=_get_optional_str(runtime, "path_to_vm"),
        screen_width=_get_optional_int(runtime, "screen_width", 1920),
        screen_height=_get_optional_int(runtime, "screen_height", 1080),
        sleep_after_execution=_get_optional_float(runtime, "sleep_after_execution", 0.5),
        region=_get_optional_str(runtime, "region", "us-east-1"),
        client_password=_get_optional_str(runtime, "client_password", "osworld-public-evaluation"),
        orchestrator=_parse_backend(root, "orchestrator"),
        gui=_parse_backend(root, "gui"),
        coding=_parse_backend(root, "coding"),
        enable_coding_agent=_get_optional_bool(root, "enable_coding_agent", False),
        enable_web_search=_get_optional_bool(root, "enable_web_search", False),
        mode=mode,
        orchestrator_max_steps=_get_optional_int(runtime, "orchestrator_max_steps", 15),
        coding_max_steps=_get_optional_int(runtime, "coding_max_steps", 20),
        cua_max_steps=_get_optional_int(runtime, "cua_max_steps", 25),
        cut_off_steps=_get_optional_int(runtime, "cut_off_steps", 200),
        domain=_get_optional_str(tasks, "domain", "all"),
        task_id=_get_optional_str(tasks, "task_id"),
        test_all_meta_path=_get_optional_str(tasks, "test_all_meta_path", "evaluation_examples/test_all.json"),
        test_config_base_dir=_get_optional_str(tasks, "test_config_base_dir", "evaluation_examples/examples"),
        result_dir=_get_optional_str(runtime, "result_dir", "./results_coact"),
        num_envs=_get_optional_int(runtime, "num_envs", 1),
        log_level=_get_optional_str(runtime, "log_level", "INFO").upper(),
    )
    return resolved


def _parse_backend(root: Dict[str, Any], key: str, *, default_protocol: str = DEFAULT_GUI_PROTOCOL) -> BackendSettings:
    backend = _get_mapping(root, key)
    model = _get_required_str(backend, "model", context=key)
    protocol = _get_optional_str(backend, "protocol", default_protocol) or default_protocol
    if protocol not in SUPPORTED_GUI_PROTOCOLS:
        raise ValueError(
            f"Unsupported `{key}.protocol` value `{protocol}`. Expected one of {SUPPORTED_GUI_PROTOCOLS}."
        )
    return BackendSettings(
        model=model,
        base_url=_get_optional_str(backend, "base_url"),
        api_key=_get_optional_str(backend, "api_key"),
        protocol=protocol,
    )


def _get_mapping(root: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = root.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{key}` must be a YAML object.")
    return value


def _get_required_str(root: Dict[str, Any], key: str, *, context: str) -> str:
    value = root.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{context}.{key}` must be a non-empty string.")
    return value.strip()


def _get_optional_str(root: Dict[str, Any], key: str, default: Optional[str] = None) -> Optional[str]:
    value = root.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"`{key}` must be a string.")
    stripped = value.strip()
    return stripped if stripped else default if default is not None else None


def _get_optional_int(root: Dict[str, Any], key: str, default: int) -> int:
    value = root.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"`{key}` must be an integer.")
    return value


def _get_optional_float(root: Dict[str, Any], key: str, default: float) -> float:
    value = root.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"`{key}` must be a number.")
    return float(value)


def _get_optional_bool(root: Dict[str, Any], key: str, default: bool) -> bool:
    value = root.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"`{key}` must be a boolean.")
    return value
