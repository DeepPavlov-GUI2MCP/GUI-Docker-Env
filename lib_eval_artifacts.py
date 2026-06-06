"""Persist per-step evaluation artifacts (screenshots and a11y XML dumps)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("desktopenv.experiment")

def step_screenshot_filename(step_num: int, action_timestamp: str) -> str:
    return f"step_{step_num}_{action_timestamp}.png"


def step_a11y_filename(step_num: int, action_timestamp: str) -> str:
    return f"step_{step_num}_{action_timestamp}.xml"


def resolve_a11y_tree(obs: Optional[Dict[str, Any]], env: Any = None) -> Optional[str]:
    tree = None
    if isinstance(obs, dict):
        tree = obs.get("accessibility_tree")
    if tree:
        return tree
    if env is not None and getattr(env, "controller", None) is not None:
        return env.controller.get_accessibility_tree()
    return None


def save_a11y_xml(example_result_dir: str, filename: str, tree: Optional[str]) -> Optional[str]:
    if not tree:
        logger.warning("Skipping a11y dump %s: accessibility tree unavailable", filename)
        return None
    os.makedirs(example_result_dir, exist_ok=True)
    path = os.path.join(example_result_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(tree)
    return filename


def save_step_screenshot(example_result_dir: str, step_num: int, action_timestamp: str, obs: Dict[str, Any]) -> Optional[str]:
    screenshot = obs.get("screenshot")
    if not screenshot:
        logger.warning("Skipping screenshot for step %d: screenshot unavailable", step_num)
        return None
    filename = step_screenshot_filename(step_num, action_timestamp)
    path = os.path.join(example_result_dir, filename)
    with open(path, "wb") as handle:
        handle.write(screenshot)
    return filename


def save_step_a11y_dump(
    example_result_dir: str,
    step_num: int,
    action_timestamp: str,
    obs: Dict[str, Any],
    env: Any = None,
) -> Optional[str]:
    tree = resolve_a11y_tree(obs, env)
    return save_a11y_xml(example_result_dir, step_a11y_filename(step_num, action_timestamp), tree)


def record_eval_step(
    example_result_dir: str,
    step_num: int,
    action_timestamp: str,
    obs: Dict[str, Any],
    env: Any = None,
    *,
    action: Any = None,
    response: Any = None,
    reward: float = 0,
    done: bool = False,
    info: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    screenshot_file = save_step_screenshot(example_result_dir, step_num, action_timestamp, obs)
    a11y_file = save_step_a11y_dump(example_result_dir, step_num, action_timestamp, obs, env)
    with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "step_num": step_num,
            "action_timestamp": action_timestamp,
            "action": action,
            "response": response,
            "reward": reward,
            "done": done,
            "info": info or {},
            "screenshot_file": screenshot_file,
            "a11y_file": a11y_file,
        }))
        handle.write("\n")
    return screenshot_file, a11y_file


def split_a11y_preflight_config(
    config: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return setup config without a11y_preflight and deferred preflight parameters."""
    base_config: List[Dict[str, Any]] = []
    preflight_params: Optional[Dict[str, Any]] = None
    for cfg in config:
        cfg_type = cfg.get("type")
        if cfg_type == "a11y_preflight":
            preflight_params = cfg.get("parameters") or {}
            continue
        if cfg_type == "act":
            params = cfg.get("parameters") or {}
            if isinstance(params.get("steps"), list) and params["steps"]:
                preflight_params = params
                continue
        base_config.append(cfg)
    return base_config, preflight_params


def example_for_reset_with_deferred_preflight(example: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    config = example.get("config") or []
    base_config, preflight_params = split_a11y_preflight_config(config)
    if preflight_params is None:
        return example, None
    reset_example = dict(example)
    reset_example["config"] = base_config
    return reset_example, preflight_params
