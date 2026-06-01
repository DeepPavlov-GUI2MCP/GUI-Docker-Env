"""Holo3 GUI agent: native Thought/Action parsing and pyautogui codegen."""

from __future__ import annotations

import logging
import math
import os
import random
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai
from PIL import Image

from mm_agents.env_loader import load_mm_agents_env

load_mm_agents_env()

from mm_agents.gui_image import pil_to_base64

logger = logging.getLogger("desktopenv.agent.holo")

FINISH_WORD = "finished"
WAIT_WORD = "wait"
ENV_FAIL_WORD = "error_env"
CALL_USER = "call_user"

HOLO_ACTION_SPACE = """
click(start_box='x, y')
left_double(start_box='x, y')
right_single(start_box='x, y')
drag(start_box='x1, y1', end_box='x2, y2')
hotkey(key='ctrl a')
press(key='enter')
type(content='text')
scroll(direction='down')
scroll(start_box='x, y', direction='down')
wait()
finished()
call_user()
"""

HOLO_PROMPT = """You are a GUI agent. You are given a task and screenshots from the current desktop state.

Return exactly:
Thought: <brief next-step reasoning>
Action: <one action>

Allowed actions:
{action_space}

Rules:
- Use absolute pixel coordinates from the screenshot for mouse actions, written as `x, y` inside `start_box='x, y'`.
- Never use <|box_start|>, <|box_end|>, UI-TARS templates, or markdown code fences.
- Output exactly one `Action:` line with a single action call. Do not add any text after the action call.
- After optional </think>, output only `Thought:` then `Action:` (no duplicate Action blocks).
- For hotkeys, join keys in one string like `ctrl a`, `ctrl shift s`, or `alt f4`.
- If the task is complete, return `finished()`.
- If you must wait for the UI to update, return `wait()`.

User Instruction:
{instruction}
"""

ACTION_NAMES = (
    "click",
    "left_double",
    "right_single",
    "left_single",
    "hover",
    "drag",
    "select",
    "hotkey",
    "press",
    "type",
    "scroll",
    "wait",
    "finished",
    "call_user",
)
ACTION_START_RE = re.compile(r"\b(" + "|".join(ACTION_NAMES) + r")\s*\(", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class HoloAction:
    kind: str
    thought: Optional[str] = None
    raw: str = ""
    x: Optional[float] = None
    y: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    keys: Optional[List[str]] = None
    key: Optional[str] = None
    content: Optional[str] = None
    direction: Optional[str] = None


def _escape_single_quotes(text: str) -> str:
    return re.sub(r"(?<!\\)'", r"\\'", text)


def _extract_thought(text: str) -> Optional[str]:
    match = re.search(r"Thought:\s*(.+?)(?=\n\s*Action:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if "</think>" in text:
        tail = text.split("</think>")[-1].strip()
        action_match = re.search(r"Thought:\s*(.+?)(?=\n\s*Action:|\Z)", tail, re.DOTALL | re.IGNORECASE)
        if action_match:
            return action_match.group(1).strip()
    return None


def _extract_balanced_call(text: str, start_idx: int) -> str:
    """Return one action call starting at start_idx, through its closing parenthesis."""
    paren = text.find("(", start_idx)
    if paren < 0:
        return text[start_idx:].strip()
    depth = 0
    for idx in range(paren, len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start_idx : idx + 1].strip()
    return text[start_idx:].strip()


def _truncate_segment_to_first_action(segment: str) -> str:
    segment = segment.strip().strip("`")
    segment = re.sub(r"^```[a-zA-Z]*\n?", "", segment)
    segment = re.sub(r"\n?```\s*$", "", segment)
    match = ACTION_START_RE.search(segment)
    if not match:
        return segment
    return _extract_balanced_call(segment, match.start())


def _extract_action_line(text: str) -> Optional[str]:
    segments = _split_action_segments(text)
    if not segments:
        return None
    for segment in reversed(segments):
        truncated = _truncate_segment_to_first_action(segment)
        if ACTION_START_RE.search(truncated):
            return truncated
    return None


def compact_holo_response(text: str, max_thought_chars: int = 400) -> str:
    """Store compact assistant turns in history to avoid context blow-up."""
    thought = _extract_thought(text)
    action_line = _extract_action_line(text)
    if thought and len(thought) > max_thought_chars:
        thought = thought[:max_thought_chars].rstrip() + "..."
    if action_line:
        if thought:
            return f"Thought: {thought}\nAction: {action_line}"
        return f"Action: {action_line}"
    return text[: max_thought_chars * 2]


def _split_action_segments(text: str) -> List[str]:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    matches = list(re.finditer(r"Action:\s*", cleaned, re.IGNORECASE))
    if matches:
        segments: List[str] = []
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
            segment = _truncate_segment_to_first_action(cleaned[start:end])
            if segment:
                segments.append(segment)
        if segments:
            return segments
    return [cleaned.strip()] if cleaned.strip() else []


def _repair_action_text(text: str) -> str:
    """Normalize Holo action syntax (including legacy box-token emissions)."""
    repaired = text.strip().strip("`")
    repaired = repaired.replace("<|box_start|>", "").replace("<|box_end|>", "")
    repaired = re.sub(
        r"<\|box_start\|>\s*\(([^)]+)\)\s*<\|box_end\|>",
        r"(\1)",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(r"\b(left_single|right_single)\b", lambda m: m.group(1), repaired, flags=re.I)
    repaired = re.sub(r"hotkey\s*\(\s*key\s*=\s*'([^']+)'\s*\)", r"hotkey(key='\1')", repaired, flags=re.I)
    repaired = re.sub(r"click\s*\(\s*start_box\s*=\s*'([^']+)'\s*\)", r"click(start_box='\1')", repaired, flags=re.I)
    repaired = re.sub(
        r"(start_box|end_box|key|content|direction)\s*=\s*'([^']*)\)'\s*\)",
        r"\1='\2')",
        repaired,
        flags=re.I,
    )
    repaired = re.sub(
        r"(start_box|end_box)\s*=\s*'\s*([0-9\t\s.,+-]+)\s*'\s*\)",
        r"\1='\2')",
        repaired,
        flags=re.I,
    )
    repaired = re.sub(
        r"(start_box|end_box)\s*=\s*'([^']*)\)(?!')",
        r"\1='\2'",
        repaired,
        flags=re.I,
    )
    return repaired.strip()


def _split_action_calls(segment: str) -> List[str]:
    normalized = _truncate_segment_to_first_action(_repair_action_text(segment))
    if ACTION_START_RE.search(normalized):
        return [normalized]
    return [normalized] if normalized else []


def _client_error_message(prediction: Optional[str], exc: Optional[Exception]) -> str:
    if exc is None:
        return "client error"
    if prediction:
        preview = prediction[:500].replace("\n", " ")
        return f"client error: {exc} | response[:500]={preview}"
    return f"client error: {exc}"


def _parse_coord_pair(raw: str) -> Tuple[float, float]:
    cleaned = raw.strip().strip("'\"").replace("\t", " ")
    numbers = [float(v) for v in NUMBER_RE.findall(cleaned)]
    if len(numbers) < 2:
        raise ValueError(f"Need two coordinates, got {raw!r}")
    return numbers[0], numbers[1]


def _extract_box_pair(action_text: str, param_name: str) -> Optional[Tuple[float, float]]:
    patterns = [
        rf"{param_name}\s*=\s*<\|box_start\|>\s*\(([^)]+)\)\s*<\|box_end\|>",
        rf"{param_name}\s*=\s*['\"]([^'\"]*)['\"]",
        rf"{param_name}\s*=\s*\(([^)]+)\)",
        rf"{param_name}\s*=\s*['\"]?\s*([0-9\t\s.,+-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, action_text, re.IGNORECASE)
        if not match:
            continue
        try:
            return _parse_coord_pair(match.group(1))
        except ValueError:
            continue
    return None


def _to_screen_coord(
    value: float,
    resized_dim: int,
    screen_dim: int,
) -> float:
    if resized_dim <= 0 or screen_dim <= 0:
        return 0.0
    if 0.0 <= value <= 1.0:
        pixel = value * resized_dim
    else:
        pixel = value
    return round(pixel * screen_dim / resized_dim, 3)


def _screen_point(
    x: float,
    y: float,
    resized_width: int,
    resized_height: int,
    screen_width: int,
    screen_height: int,
) -> Tuple[float, float]:
    return (
        _to_screen_coord(x, resized_width, screen_width),
        _to_screen_coord(y, resized_height, screen_height),
    )


def _extract_quoted_param(action_text: str, param_name: str) -> Optional[str]:
    patterns = (
        rf"{param_name}\s*=\s*'((?:\\.|[^'])*)'\s*\)",
        rf'{param_name}\s*=\s*"((?:\\.|[^"])*)"\s*\)',
    )
    for pattern in patterns:
        match = re.search(pattern, action_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).encode("utf-8").decode("unicode_escape")
    loose = re.search(rf"{param_name}\s*=\s*([^,\)]+)", action_text, re.IGNORECASE)
    if loose:
        return loose.group(1).strip().strip("'\"")
    return None


def _normalize_hotkey_keys(raw: str) -> List[str]:
    keys = [part for part in re.split(r"[+,]+|\s+", raw.strip()) if part]
    mapping = {
        "arrowleft": "left",
        "arrowright": "right",
        "arrowup": "up",
        "arrowdown": "down",
        "space": " ",
    }
    return [mapping.get(k.lower(), k.lower()) for k in keys]


def _parse_action_call(
    action_text: str,
    thought: Optional[str],
    full_text: str,
    resized_height: int,
    resized_width: int,
    screen_height: int,
    screen_width: int,
) -> HoloAction:
    stripped = _repair_action_text(action_text)
    lowered = stripped.lower()

    if re.match(r"finished\s*\(", lowered) or lowered == "finished()":
        return HoloAction(kind=FINISH_WORD, thought=thought, raw=stripped)
    if re.match(r"wait\s*\(", lowered) or lowered == "wait()":
        return HoloAction(kind=WAIT_WORD, thought=thought, raw=stripped)
    if re.match(r"call_user\s*\(", lowered) or lowered == "call_user()":
        return HoloAction(kind=CALL_USER, thought=thought, raw=stripped)

    name_match = ACTION_START_RE.search(stripped)
    if not name_match:
        raise ValueError(f"Could not find action name in: {stripped}")
    kind = name_match.group(1).lower()

    if kind in {"click", "left_double", "right_single", "left_single", "hover", "select"}:
        pair = _extract_box_pair(stripped, "start_box")
        if pair is None:
            numbers = [float(v) for v in NUMBER_RE.findall(stripped)]
            if len(numbers) < 2:
                raise ValueError(f"Could not find click coordinates in: {stripped}")
            pair = (numbers[0], numbers[1])
        x, y = _screen_point(pair[0], pair[1], resized_width, resized_height, screen_width, screen_height)
        if kind == "left_double":
            click_kind = "left_double"
        elif kind == "right_single":
            click_kind = "right_single"
        elif kind == "hover":
            click_kind = "hover"
        else:
            click_kind = "click"
        return HoloAction(kind=click_kind, thought=thought, raw=stripped, x=x, y=y)

    if kind == "drag":
        start = _extract_box_pair(stripped, "start_box")
        end = _extract_box_pair(stripped, "end_box")
        if start is None or end is None:
            numbers = [float(v) for v in NUMBER_RE.findall(stripped)]
            if len(numbers) < 4:
                raise ValueError(f"Could not find drag coordinates in: {stripped}")
            start = (numbers[0], numbers[1])
            end = (numbers[2], numbers[3])
        sx, sy = _screen_point(start[0], start[1], resized_width, resized_height, screen_width, screen_height)
        ex, ey = _screen_point(end[0], end[1], resized_width, resized_height, screen_width, screen_height)
        return HoloAction(kind="drag", thought=thought, raw=stripped, x=sx, y=sy, x2=ex, y2=ey)

    if kind == "hotkey":
        raw_key = _extract_quoted_param(stripped, "key") or _extract_quoted_param(stripped, "hotkey")
        if not raw_key:
            raise ValueError(f"Could not parse hotkey from: {stripped}")
        return HoloAction(kind="hotkey", thought=thought, raw=stripped, keys=_normalize_hotkey_keys(raw_key))

    if kind == "press":
        raw_key = _extract_quoted_param(stripped, "key")
        if not raw_key:
            raise ValueError(f"Could not parse press key from: {stripped}")
        keys = _normalize_hotkey_keys(raw_key)
        return HoloAction(kind="press", thought=thought, raw=stripped, key=keys[0] if keys else raw_key)

    if kind == "type":
        content = _extract_quoted_param(stripped, "content")
        if content is None:
            raise ValueError(f"Could not parse type content from: {stripped}")
        return HoloAction(kind="type", thought=thought, raw=stripped, content=content)

    if kind == "scroll":
        direction = _extract_quoted_param(stripped, "direction")
        if direction is None:
            fallback = re.search(r"\b(up|down|left|right)\b", lowered)
            direction = fallback.group(1) if fallback else None
        if direction is None:
            raise ValueError(f"Could not parse scroll direction from: {stripped}")
        pair = _extract_box_pair(stripped, "start_box")
        x = y = None
        if pair is not None:
            x, y = _screen_point(pair[0], pair[1], resized_width, resized_height, screen_width, screen_height)
        return HoloAction(kind="scroll", thought=thought, raw=stripped, x=x, y=y, direction=direction)

    raise ValueError(f"Unsupported Holo action: {stripped}")


def parse_holo_response(
    text: str,
    resized_height: int,
    resized_width: int,
    screen_height: int,
    screen_width: int,
) -> HoloAction:
    text = text.strip()
    if not text:
        raise ValueError("Empty model response")

    thought = _extract_thought(text)
    candidates: List[str] = []
    for segment in _split_action_segments(text):
        candidates.extend(_split_action_calls(segment))
    if not candidates:
        candidates.extend(_split_action_calls(_repair_action_text(text)))

    errors: List[str] = []
    for call in reversed(candidates):
        try:
            return _parse_action_call(
                call,
                thought,
                text,
                resized_height,
                resized_width,
                screen_height,
                screen_width,
            )
        except Exception as exc:
            errors.append(f"{call!r}: {exc}")

    raise ValueError(f"No parseable Holo action found ({len(errors)} attempts): {errors[-1] if errors else text[:200]}")


def holo_to_pyautogui(
    action: HoloAction,
    input_swap: bool = True,
) -> str:
    """Convert a parsed HoloAction directly to executable pyautogui code."""
    if action.kind == FINISH_WORD:
        return "DONE"
    if action.kind == WAIT_WORD:
        return "WAIT"
    if action.kind == ENV_FAIL_WORD:
        return "FAIL"

    thought = (action.thought or "").strip()
    header = "import pyautogui\nimport time\n"
    if thought:
        header += f"'''\nThought:\n{thought}\n'''\n"

    if action.kind == "hotkey" and action.keys:
        key_args = ", ".join(repr(k) for k in action.keys)
        return header + f"\npyautogui.hotkey({key_args})"

    if action.kind == "press" and action.key is not None:
        return header + f"\npyautogui.press({repr(action.key)})"

    if action.kind == "type" and action.content is not None:
        content = action.content
        escaped = _escape_single_quotes(content)
        stripped = escaped.rstrip("\\n").rstrip("\n")
        code = header
        if input_swap:
            code += f"\nimport pyperclip\npyperclip.copy('{stripped}')\npyautogui.hotkey('ctrl', 'v')\ntime.sleep(0.5)\n"
        else:
            code += f"\npyautogui.write('{stripped}', interval=0.1)\ntime.sleep(0.5)\n"
        if content.endswith("\n") or content.endswith("\\n"):
            code += "\npyautogui.press('enter')"
        return code

    if action.kind == "drag" and None not in (action.x, action.y, action.x2, action.y2):
        return (
            header
            + f"\npyautogui.moveTo({action.x}, {action.y})\n"
            + f"pyautogui.dragTo({action.x2}, {action.y2}, duration=1.0)\n"
        )

    if action.kind == "scroll" and action.direction:
        direction = action.direction.lower()
        if action.x is not None and action.y is not None:
            if "up" in direction:
                return header + f"\npyautogui.scroll(5, x={action.x}, y={action.y})"
            if "down" in direction:
                return header + f"\npyautogui.scroll(-5, x={action.x}, y={action.y})"
        if "up" in direction:
            return header + "\npyautogui.scroll(5)"
        if "down" in direction:
            return header + "\npyautogui.scroll(-5)"
        return header + f"\n# Unrecognized scroll direction: {action.direction}"

    if action.x is not None and action.y is not None:
        if action.kind == "left_double":
            return header + f"\npyautogui.doubleClick({action.x}, {action.y}, button='left')"
        if action.kind == "right_single":
            return header + f"\npyautogui.click({action.x}, {action.y}, button='right')"
        if action.kind == "hover":
            return header + f"\npyautogui.moveTo({action.x}, {action.y})"
        return header + f"\npyautogui.click({action.x}, {action.y}, button='left')"

    raise ValueError(f"Cannot codegen Holo action: {action}")


class HoloAgent:
    def __init__(
        self,
        model: str,
        runtime_conf: Dict[str, Any],
        platform: str = "ubuntu",
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        max_trajectory_length: int = 50,
        a11y_tree_max_tokens: int = 10000,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.platform = platform
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.runtime_conf = runtime_conf
        self.temperature = self.runtime_conf["temperature"]
        self.top_p = self.runtime_conf["top_p"]
        self.max_tokens = self.runtime_conf["max_tokens"]
        self.input_swap = self.runtime_conf["input_swap"]
        self.language = self.runtime_conf["language"]
        self.max_pixels = self.runtime_conf["max_pixels"]
        self.min_pixels = self.runtime_conf["min_pixels"]
        self.callusr_tolerance = self.runtime_conf["callusr_tolerance"]
        self.history_n = self.runtime_conf.get("history_n", 2)
        self.max_thought_chars = int(self.runtime_conf.get("max_thought_chars", 400))

        def _ev(*names: str) -> str:
            for name in names:
                value = (os.environ.get(name) or "").strip()
                if value:
                    return value
            return ""

        configured_base_urls = kwargs.get("base_urls") or kwargs.get("base_url")
        if isinstance(configured_base_urls, (list, tuple)):
            raw_bases = ",".join(str(part).strip() for part in configured_base_urls if str(part).strip())
        elif configured_base_urls is None:
            raw_bases = _ev("HOLO_OPENAI_BASE_URL", "HOLO_OPENAI_BASE_URLS", "OPENAI_BASE_URL")
        else:
            raw_bases = str(configured_base_urls).strip()

        baseurl_list: List[str] = []
        if raw_bases:
            for part in raw_bases.split(","):
                url = part.strip().rstrip("/")
                if not url:
                    continue
                if not url.endswith("/v1"):
                    url = f"{url}/v1"
                baseurl_list.append(url)
            random.shuffle(baseurl_list)
        if not baseurl_list:
            baseurl_list = ["http://127.0.0.1:8010/v1"]

        api_key = kwargs.get("api_key") or _ev("HOLO_OPENAI_API_KEY", "OPENAI_API_KEY") or "empty"
        self.vlm = openai.OpenAI(api_key=api_key, base_url=baseurl_list[0])
        self._api_model_override = (kwargs.get("api_model") or "").strip()

        self.thoughts: List[str] = []
        self.actions: List[List[str]] = []
        self.observations: List[Dict[str, Any]] = []
        self.history_images: List[bytes] = []
        self.history_responses: List[str] = []
        self.last_usage: Any = None
        self.cur_callusr_count = 0

    def _resolve_model_name(self, model: str) -> str:
        if self._api_model_override:
            return self._api_model_override
        configured = (
            os.environ.get("HOLO_OPENAI_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or ""
        ).strip()
        if configured:
            return configured
        return model

    def reset(self, runtime_logger: Any = None) -> None:
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        self.last_usage = None
        self.cur_callusr_count = 0

    def predict(self, instruction: str, obs: Dict[str, Any], last_action_after_obs: Dict[str, Any] = None) -> List[Any]:
        assert len(self.observations) == len(self.actions) == len(self.thoughts)

        self.history_images.append(obs["screenshot"])

        if self.observation_type not in ["screenshot", "screenshot_a11y_tree"]:
            raise ValueError("Invalid observation_type type: " + self.observation_type)

        linearized_accessibility_tree = None
        if self.observation_type == "screenshot_a11y_tree":
            try:
                from mm_agents.agent import (  # lazy: agent.py has heavy optional deps
                    linearize_accessibility_tree,
                    trim_accessibility_tree,
                )

                linearized_accessibility_tree = linearize_accessibility_tree(
                    accessibility_tree=obs["accessibility_tree"],
                    platform=self.platform,
                )
                if linearized_accessibility_tree:
                    linearized_accessibility_tree = trim_accessibility_tree(
                        linearized_accessibility_tree, self.a11y_tree_max_tokens
                    )
            except Exception:
                linearized_accessibility_tree = None

        self.observations.append(
            {"screenshot": obs["screenshot"], "accessibility_tree": linearized_accessibility_tree}
        )

        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n :]
        if len(self.history_responses) > self.history_n:
            self.history_responses = self.history_responses[-self.history_n :]

        user_prompt = HOLO_PROMPT.format(
            instruction=instruction,
            action_space=HOLO_ACTION_SPACE.strip(),
        )

        images: List[Image.Image] = []
        if isinstance(self.history_images, bytes):
            image_bytes_list = [self.history_images]
        elif isinstance(self.history_images, np.ndarray):
            image_bytes_list = list(self.history_images)
        else:
            image_bytes_list = self.history_images

        for image_bytes in image_bytes_list:
            image = Image.open(BytesIO(image_bytes))
            if image.width * image.height > self.max_pixels:
                resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
                image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))
            if image.width * image.height < self.min_pixels:
                resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
                image = image.resize(
                    (math.ceil(image.width * resize_factor), math.ceil(image.height * resize_factor))
                )
            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append(image)

        screen_image = Image.open(BytesIO(self.history_images[-1]))
        resized_image = images[-1]
        screen_height, screen_width = screen_image.height, screen_image.width
        resized_height, resized_width = resized_image.height, resized_image.width

        try_times = 3
        prediction: Optional[str] = None
        parsed_action: Optional[HoloAction] = None
        last_parse_error: Optional[Exception] = None
        model_name = self._resolve_model_name(self.model)
        history_window = min(max(1, self.history_n), len(images))

        def build_messages(window: int) -> List[Dict[str, Any]]:
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ]
            prompt_images = images[-window:]
            prompt_responses = self.history_responses[-max(0, window - 1) :]
            image_num = 0
            if prompt_responses:
                for history_response in prompt_responses:
                    encoded_string = pil_to_base64(prompt_images[image_num])
                    messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}],
                        }
                    )
                    image_num += 1
                    messages.append(
                        {"role": "assistant", "content": [{"type": "text", "text": history_response}]}
                    )
            encoded_string = pil_to_base64(prompt_images[image_num])
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}],
                }
            )
            return messages

        while True:
            if try_times <= 0:
                logger.warning("Holo agent exhausted parse/API retries")
                return _client_error_message(prediction, last_parse_error), ["DONE"]
            try:
                messages = build_messages(history_window)
                response = self.vlm.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    frequency_penalty=1,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                self.last_usage = getattr(response, "usage", None)
                prediction = (response.choices[0].message.content or "").strip()
            except Exception as exc:
                err_text = str(exc).lower()
                if "maximum context length" in err_text and history_window > 1:
                    history_window -= 1
                    logger.warning("Reducing Holo history window to %s after context overflow", history_window)
                    continue
                logger.exception("Error when fetching response from Holo client: %s", exc)
                try_times -= 1
                continue

            try:
                parsed_action = parse_holo_response(
                    prediction,
                    resized_height=resized_height,
                    resized_width=resized_width,
                    screen_height=screen_height,
                    screen_width=screen_width,
                )
                break
            except Exception as exc:
                last_parse_error = exc
                logger.warning("Holo parse failed (retries left %s): %s", try_times - 1, exc)
                try_times -= 1

        if prediction is None or parsed_action is None:
            return _client_error_message(prediction, last_parse_error), ["DONE"]

        self.history_responses.append(compact_holo_response(prediction, self.max_thought_chars))
        self.thoughts.append(prediction)

        if parsed_action.kind == FINISH_WORD:
            self.actions.append([])
            return prediction, ["DONE"]
        if parsed_action.kind == WAIT_WORD:
            self.actions.append([])
            return prediction, ["WAIT"]
        if parsed_action.kind == CALL_USER:
            if self.callusr_tolerance > self.cur_callusr_count:
                self.cur_callusr_count += 1
                self.actions.append([])
                return prediction, ["WAIT"]
            self.actions.append([])
            return prediction, ["FAIL"]

        try:
            pyautogui_code = holo_to_pyautogui(parsed_action, input_swap=self.input_swap)
        except Exception as exc:
            logger.warning("Holo codegen failed: %s", exc)
            return f"Parsing action error: {prediction}, with error:\n{exc}", ["DONE"]

        actions = [pyautogui_code]
        self.actions.append(actions)
        if len(self.history_responses) >= self.max_trajectory_length:
            actions = ["FAIL"]
        return prediction, actions
