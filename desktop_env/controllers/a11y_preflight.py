"""Accessibility-guided preflight setup for OSWorld tasks."""

from __future__ import annotations

import ast
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import lxml.etree
import requests

from desktop_env.evaluators.metrics.general import check_accessibility_tree

logger = logging.getLogger("desktopenv.a11y_preflight")

A11Y_NS = {
    "st": "https://accessibility.ubuntu.example.org/ns/state",
    "attr": "https://accessibility.ubuntu.example.org/ns/attributes",
    "cp": "https://accessibility.ubuntu.example.org/ns/component",
}


def _normalize_role(role: str) -> str:
    return role.strip().lower().replace(" ", "-") if role else ""


@dataclass(frozen=True)
class MatchedNode:
    element: Any
    center: Tuple[int, int]


class A11yTreeIndex:
    def __init__(self, xml_text: str) -> None:
        payload = xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text
        self.root = lxml.etree.fromstring(payload)

    def iter_elements(self):
        for element in self.root.iter():
            if element is self.root:
                continue
            yield element

    @staticmethod
    def element_role(element: Any) -> str:
        tag = element.tag
        if isinstance(tag, str) and "}" in tag:
            tag = tag.split("}", 1)[1]
        return _normalize_role(tag or "")

    @staticmethod
    def element_name(element: Any) -> str:
        return (element.get("name") or "").strip()

    @staticmethod
    def element_center(element: Any) -> Optional[Tuple[int, int]]:
        coord_raw = element.get(f"{{{A11Y_NS['cp']}}}screencoord")
        size_raw = element.get(f"{{{A11Y_NS['cp']}}}size")
        if not coord_raw or not size_raw:
            return None
        try:
            x, y = ast.literal_eval(coord_raw)
            width, height = ast.literal_eval(size_raw)
        except (ValueError, SyntaxError):
            return None
        if width <= 0 or height <= 0:
            return None
        return int(x + width / 2), int(y + height / 2)

    @staticmethod
    def element_state(element: Any, state_name: str) -> Optional[bool]:
        key = f"{{{A11Y_NS['st']}}}{state_name}"
        raw = element.get(key)
        if raw is None:
            return None
        return raw.lower() == "true"

    def match_selector(self, selector: Dict[str, Any]) -> List[MatchedNode]:
        matches: List[MatchedNode] = []
        want_role = _normalize_role(selector.get("role", "")) if selector.get("role") else ""
        want_name = selector.get("name")
        name_contains = selector.get("name_contains")
        states = selector.get("states") or {}
        require_bbox = bool(selector.get("require_bbox", True))

        for element in self.iter_elements():
            role = self.element_role(element)
            name = self.element_name(element)
            if want_role and role != want_role:
                continue
            if isinstance(want_name, str) and want_name and name != want_name:
                continue
            if isinstance(name_contains, str) and name_contains:
                if name_contains.lower() not in name.lower():
                    continue
            state_ok = True
            for state_key, expected in states.items():
                actual = self.element_state(element, state_key)
                if actual is None or actual != bool(expected):
                    state_ok = False
                    break
            if not state_ok:
                continue
            center = self.element_center(element)
            if center is None and require_bbox:
                continue
            matches.append(MatchedNode(element=element, center=center or (0, 0)))
        return matches

    def assert_rules(self, rules: List[Dict[str, Any]]) -> bool:
        xml_text = lxml.etree.tostring(self.root, encoding="unicode")
        return check_accessibility_tree(xml_text, rules, osname="ubuntu") >= 1.0


def bbox_center(bbox: Sequence[int]) -> Tuple[int, int]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 integers, got {bbox!r}")
    x, y, width, height = [int(value) for value in bbox]
    return int(x + width / 2), int(y + height / 2)


def pyautogui_click_command(x: int, y: int) -> str:
    return (
        "import pyautogui; import time; "
        f"pyautogui.click({x}, {y}); time.sleep(0.2)"
    )


def pyautogui_move_command(x: int, y: int) -> str:
    return (
        "import pyautogui; import time; "
        f"pyautogui.moveTo({x}, {y}); time.sleep(0.2)"
    )


def pyautogui_hotkey_command(keys: Sequence[str]) -> str:
    joined = ", ".join(json.dumps(key) for key in keys)
    return f"import pyautogui; import time; pyautogui.hotkey({joined}); time.sleep(0.2)"


def pyautogui_press_command(key: str) -> str:
    return f"import pyautogui; import time; pyautogui.press({json.dumps(key)}); time.sleep(0.2)"


def pyautogui_type_command(text: str) -> str:
    return (
        "import pyautogui; import time; "
        f"pyautogui.typewrite({json.dumps(text)}); time.sleep(0.2)"
    )


class A11yPreflightExecutor:
    def __init__(
        self,
        http_server: str,
        *,
        cache_dir: str = "",
        poll_interval: float = 0.5,
        osname: str = "ubuntu",
    ) -> None:
        self.http_server = http_server.rstrip("/")
        self.cache_dir = cache_dir
        self.poll_interval = poll_interval
        self.osname = osname

    def fetch_accessibility_tree(self) -> str:
        response = requests.get(f"{self.http_server}/accessibility", timeout=30)
        response.raise_for_status()
        payload = response.json()
        tree = payload.get("AT")
        if not isinstance(tree, str) or not tree.strip():
            raise RuntimeError("accessibility endpoint returned empty tree")
        return tree

    def execute_python(self, command: str) -> None:
        payload = json.dumps({"command": ["python", "-c", command], "shell": False})
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{self.http_server}/setup/execute",
            headers=headers,
            data=payload,
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"setup execute failed with HTTP {response.status_code}: {response.text[:500]}"
            )

    def _match_step_selector(self, tree: A11yTreeIndex, step: Dict[str, Any]) -> List[MatchedNode]:
        selector = step.get("selector")
        if not isinstance(selector, dict):
            raise ValueError(f"step {step!r} requires selector")
        return tree.match_selector(selector)

    def _execute_step(self, tree: A11yTreeIndex, step: Dict[str, Any]) -> None:
        op = step.get("op")
        if not isinstance(op, str) or not op:
            raise ValueError(f"invalid preflight step: {step!r}")

        if op == "sleep":
            seconds = float(step.get("seconds", 0.5))
            time.sleep(seconds)
            return

        if op in {"wait_for", "assert", "click", "move"}:
            rules = step.get("rules")
            if isinstance(rules, list) and rules:
                if op == "assert" and not tree.assert_rules(rules):
                    raise RuntimeError(f"a11y assert rules failed: {rules!r}")
                if op == "wait_for":
                    raise ValueError("wait_for with rules is not supported; use selector")
                return

            matches = self._match_step_selector(tree, step)
            if op in {"wait_for", "assert"} and not matches:
                raise RuntimeError(f"a11y {op} found no matches for selector {step.get('selector')!r}")
            if op == "click":
                if not matches:
                    raise RuntimeError(f"a11y click found no matches for selector {step.get('selector')!r}")
                x, y = matches[0].center
                self.execute_python(pyautogui_click_command(x, y))
            if op == "move":
                if not matches:
                    raise RuntimeError(f"a11y move found no matches for selector {step.get('selector')!r}")
                x, y = matches[0].center
                self.execute_python(pyautogui_move_command(x, y))
            return

        if op in {"click_bbox", "move_bbox"}:
            bbox = step.get("bbox")
            if not isinstance(bbox, list):
                raise ValueError(f"{op} requires bbox list, got {bbox!r}")
            x, y = bbox_center(bbox)
            if op == "click_bbox":
                self.execute_python(pyautogui_click_command(x, y))
            else:
                self.execute_python(pyautogui_move_command(x, y))
            return

        if op == "hotkey":
            keys = step.get("keys")
            if not isinstance(keys, list) or not keys:
                raise ValueError(f"hotkey requires non-empty keys list, got {keys!r}")
            self.execute_python(pyautogui_hotkey_command(keys))
            return

        if op == "press":
            key = step.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError(f"press requires key string, got {key!r}")
            self.execute_python(pyautogui_press_command(key))
            return

        if op == "type":
            text = step.get("text")
            if not isinstance(text, str):
                raise ValueError(f"type requires text string, got {text!r}")
            self.execute_python(pyautogui_type_command(text))
            return

        raise ValueError(f"unsupported a11y preflight op: {op}")

    def run(
        self,
        *,
        steps: List[Dict[str, Any]],
        timeout_seconds: float = 20.0,
        screenshot_on_failure: bool = False,
        on_step_complete: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> None:
        deadline = time.time() + max(0.0, float(timeout_seconds))
        step_index = 0
        while step_index < len(steps):
            step = steps[step_index]
            if not isinstance(step, dict):
                raise ValueError(f"preflight step must be object, got {step!r}")

            op = step.get("op")
            if op == "wait_for":
                last_error: Optional[Exception] = None
                while time.time() <= deadline:
                    try:
                        tree = A11yTreeIndex(self.fetch_accessibility_tree())
                        self._execute_step(tree, step)
                        last_error = None
                        break
                    except RuntimeError as exc:
                        last_error = exc
                        time.sleep(self.poll_interval)
                if last_error is not None:
                    if screenshot_on_failure:
                        self._capture_failure_screenshot(step_index)
                    raise RuntimeError(
                        f"a11y wait_for timed out after {timeout_seconds}s: {last_error}"
                    ) from last_error
                if on_step_complete is not None:
                    on_step_complete(step_index, step)
                step_index += 1
                continue

            tree = A11yTreeIndex(self.fetch_accessibility_tree())
            try:
                self._execute_step(tree, step)
            except Exception:
                if screenshot_on_failure:
                    self._capture_failure_screenshot(step_index)
                raise
            if on_step_complete is not None:
                on_step_complete(step_index, step)
            step_index += 1

    def _capture_failure_screenshot(self, step_index: int) -> None:
        if not self.cache_dir:
            return
        try:
            response = requests.get(f"{self.http_server}/screenshot", timeout=30)
            if response.status_code != 200:
                return
            from pathlib import Path

            path = Path(self.cache_dir) / f"a11y_preflight_failure_step_{step_index:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            logger.info("Saved a11y preflight failure screenshot to %s", path)
        except Exception as exc:
            logger.warning("Failed to save a11y preflight failure screenshot: %s", exc)
