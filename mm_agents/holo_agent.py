import logging
import math
import os
import random
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import openai
from PIL import Image

from mm_agents.env_loader import load_mm_agents_env

load_mm_agents_env()

from mm_agents.uitars15_v1 import (
    CALL_USER,
    ENV_FAIL_WORD,
    FINISH_WORD,
    WAIT_WORD,
    linearize_accessibility_tree,
    parsing_response_to_pyautogui_code,
    pil_to_base64,
    trim_accessibility_tree,
)


logger = logging.getLogger("desktopenv.agent.holo")


HOLO_ACTION_SPACE = """
click(start_box='x, y')
left_double(start_box='x, y')
right_single(start_box='x, y')
drag(start_box='x1, y1', end_box='x2, y2')
hotkey(key='ctrl a')
press(key='enter')
type(content='text')
scroll(direction='down')
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
- Use absolute pixel coordinates from the screenshot for mouse actions, written as `x, y`.
- For hotkeys, join keys in one string like `ctrl a`, `ctrl shift s`, or `alt f4`.
- Return only one action for the next step.
- If the task is complete, return `finished()`.
- If you must wait for the UI to update, return `wait()`.
- Do not wrap the answer in markdown fences.

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


def _extract_thought(text: str) -> Optional[str]:
    match = re.search(r"Thought:\s*(.+?)(?=\n\s*Action:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if "</think>" in text:
        tail = text.split("</think>")[-1].strip()
        if tail and "Action:" not in tail:
            return tail
    return None


def _split_action_segments(text: str) -> List[str]:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("</think>", "\n")
    matches = list(re.finditer(r"Action:\s*", cleaned, re.IGNORECASE))
    if matches:
        segments: List[str] = []
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
            segment = cleaned[start:end].strip().strip("`")
            if segment:
                segments.append(segment)
        if segments:
            return segments
    return [cleaned.strip()]


def _split_action_calls(segment: str) -> List[str]:
    matches = list(ACTION_START_RE.finditer(segment))
    if not matches:
        return [segment.strip()]
    calls: List[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(segment)
        call = segment[start:end].strip()
        if call:
            calls.append(call)
    return calls


def _extract_quoted_param(action_text: str, param_name: str) -> Optional[str]:
    patterns = (
        rf"{param_name}\s*=\s*'((?:\\.|[^'])*)'",
        rf'{param_name}\s*=\s*"((?:\\.|[^"])*)"',
    )
    for pattern in patterns:
        match = re.search(pattern, action_text, re.DOTALL)
        if match:
            return bytes(match.group(1), "utf-8").decode("unicode_escape")
    return None


def _extract_numbers(action_text: str) -> List[float]:
    return [float(value) for value in NUMBER_RE.findall(action_text)]


def _normalize_axis(value: float, total: int) -> float:
    if total <= 0:
        return 0.0
    if 0.0 <= value <= 1.0:
        normalized = value
    else:
        normalized = value / float(total)
    return max(0.0, min(1.0, normalized))


def _box_from_coords(coords: List[float], image_height: int, image_width: int) -> str:
    if len(coords) >= 4:
        x1, y1, x2, y2 = coords[:4]
    elif len(coords) >= 2:
        x1, y1 = coords[:2]
        x2, y2 = x1, y1
    else:
        raise ValueError(f"Need at least two coordinates, got: {coords}")
    box = [
        _normalize_axis(x1, image_width),
        _normalize_axis(y1, image_height),
        _normalize_axis(x2, image_width),
        _normalize_axis(y2, image_height),
    ]
    return str(box)


def _parse_single_action(
    action_text: str,
    thought: Optional[str],
    full_text: str,
    image_height: int,
    image_width: int,
) -> Dict[str, Any]:
    stripped = action_text.strip().strip("`")
    lowered = stripped.lower()

    if lowered.startswith("finished(") or lowered == "finished()":
        return {
            "reflection": None,
            "thought": thought,
            "action_type": FINISH_WORD,
            "action_inputs": {},
            "text": full_text,
        }
    if lowered.startswith("wait(") or lowered == "wait()":
        return {
            "reflection": None,
            "thought": thought,
            "action_type": WAIT_WORD,
            "action_inputs": {},
            "text": full_text,
        }
    if lowered.startswith("call_user(") or lowered == "call_user()":
        return {
            "reflection": None,
            "thought": thought,
            "action_type": CALL_USER,
            "action_inputs": {},
            "text": full_text,
        }

    name_match = ACTION_START_RE.search(stripped)
    if not name_match:
        raise ValueError(f"Could not find action name in: {stripped}")
    action_type = name_match.group(1).lower()

    if action_type in {"click", "left_double", "right_single", "left_single", "hover", "select"}:
        coords = _extract_numbers(stripped)
        if len(coords) < 2:
            raise ValueError(f"Could not find click coordinates in: {stripped}")
        return {
            "reflection": None,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": {"start_box": _box_from_coords(coords[:2], image_height, image_width)},
            "text": full_text,
        }

    if action_type == "drag":
        coords = _extract_numbers(stripped)
        if len(coords) < 4:
            raise ValueError(f"Could not find drag coordinates in: {stripped}")
        return {
            "reflection": None,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": {
                "start_box": _box_from_coords(coords[:2], image_height, image_width),
                "end_box": _box_from_coords(coords[2:4], image_height, image_width),
            },
            "text": full_text,
        }

    if action_type == "hotkey":
        key = _extract_quoted_param(stripped, "key") or _extract_quoted_param(stripped, "hotkey")
        if not key:
            raise ValueError(f"Could not parse hotkey string from: {stripped}")
        key = " ".join(part for part in re.split(r"[+,]+|\s+", key.strip()) if part)
        return {
            "reflection": None,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": {"key": key},
            "text": full_text,
        }

    if action_type == "press":
        key = _extract_quoted_param(stripped, "key")
        if not key:
            raise ValueError(f"Could not parse press key from: {stripped}")
        return {
            "reflection": None,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": {"key": key.strip()},
            "text": full_text,
        }

    if action_type == "type":
        content = _extract_quoted_param(stripped, "content")
        if content is None:
            raise ValueError(f"Could not parse type content from: {stripped}")
        return {
            "reflection": None,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": {"content": content},
            "text": full_text,
        }

    if action_type == "scroll":
        direction = _extract_quoted_param(stripped, "direction")
        if direction is None:
            fallback = re.search(r"\b(up|down|left|right)\b", lowered)
            if fallback:
                direction = fallback.group(1)
        if direction is None:
            raise ValueError(f"Could not parse scroll direction from: {stripped}")
        action_inputs: Dict[str, Any] = {"direction": direction}
        coords = _extract_numbers(stripped)
        if len(coords) >= 2:
            action_inputs["start_box"] = _box_from_coords(coords[:2], image_height, image_width)
        return {
            "reflection": None,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": full_text,
        }

    raise ValueError(f"Unsupported Holo action: {stripped}")


def parse_holo_action_to_structure_output(
    text: str,
    image_height: int,
    image_width: int,
) -> List[Dict[str, Any]]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model response")

    thought = _extract_thought(text)
    parsed_actions: List[Dict[str, Any]] = []
    for segment in _split_action_segments(text):
        for call in _split_action_calls(segment):
            parsed_actions.append(
                _parse_single_action(
                    action_text=call,
                    thought=thought,
                    full_text=text,
                    image_height=image_height,
                    image_width=image_width,
                )
            )

    if not parsed_actions:
        raise ValueError(f"No parseable actions found in response: {text}")
    return parsed_actions


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
        self.history_n = self.runtime_conf.get("history_n", 5)

        def _ev(name: str) -> str:
            return (os.environ.get(name) or "").strip()

        configured_base_urls = kwargs.get("base_urls") or kwargs.get("base_url")
        if isinstance(configured_base_urls, (list, tuple)):
            raw_bases = ",".join(str(part).strip() for part in configured_base_urls if str(part).strip())
        elif configured_base_urls is None:
            raw_bases = _ev("OPENAI_BASE_URL") or _ev("UITARS_OPENAI_BASE_URL") or _ev("UITARS_OPENAI_BASE_URLS")
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

        api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY") or os.environ.get("UITARS_OPENAI_API_KEY") or "empty"
        self.vlm = openai.OpenAI(api_key=api_key, base_url=baseurl_list[0])

        self.thoughts: List[str] = []
        self.actions: List[List[str]] = []
        self.observations: List[Dict[str, Any]] = []
        self.history_images: List[bytes] = []
        self.history_responses: List[str] = []
        self.last_usage: Any = None
        self.cur_callusr_count = 0

    def _resolve_model_name(self, model: str) -> str:
        configured = (os.environ.get("OPENAI_MODEL") or os.environ.get("UITARS_OPENAI_MODEL") or "").strip()
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
        assert len(self.observations) == len(self.actions) == len(self.thoughts), (
            "The number of observations and actions should be the same."
        )

        self.history_images.append(obs["screenshot"])

        if self.observation_type not in ["screenshot", "screenshot_a11y_tree"]:
            raise ValueError("Invalid observation_type type: " + self.observation_type)

        try:
            linearized_accessibility_tree = (
                linearize_accessibility_tree(
                    accessibility_tree=obs["accessibility_tree"],
                    platform=self.platform,
                )
                if self.observation_type == "screenshot_a11y_tree"
                else None
            )
        except Exception:
            linearized_accessibility_tree = None

        if linearized_accessibility_tree:
            linearized_accessibility_tree = trim_accessibility_tree(
                linearized_accessibility_tree, self.a11y_tree_max_tokens
            )

        self.observations.append(
            {
                "screenshot": obs["screenshot"],
                "accessibility_tree": linearized_accessibility_tree,
            }
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
        elif isinstance(self.history_images, list):
            image_bytes_list = self.history_images
        else:
            raise TypeError(f"Unidentified images type: {type(self.history_images)}")

        for image_bytes in image_bytes_list:
            image = Image.open(BytesIO(image_bytes))
            if image.width * image.height > self.max_pixels:
                resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
                width = int(image.width * resize_factor)
                height = int(image.height * resize_factor)
                image = image.resize((width, height))
            if image.width * image.height < self.min_pixels:
                resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
                width = math.ceil(image.width * resize_factor)
                height = math.ceil(image.height * resize_factor)
                image = image.resize((width, height))
            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append(image)

        try_times = 3
        prediction: Optional[str] = None
        model_name = self._resolve_model_name(self.model)
        history_window = min(max(1, self.history_n), len(images))

        def build_messages(window: int) -> List[Dict[str, Any]]:
            messages: List[Dict[str, Any]] = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                },
            ]

            prompt_images = images[-window:]
            prompt_responses = self.history_responses[-max(0, window - 1) :]
            image_num = 0
            if prompt_responses:
                for history_response in prompt_responses:
                    cur_image = prompt_images[image_num]
                    encoded_string = pil_to_base64(cur_image)
                    messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}],
                        }
                    )
                    image_num += 1
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": history_response}],
                        }
                    )
            cur_image = prompt_images[image_num]
            encoded_string = pil_to_base64(cur_image)
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}],
                }
            )
            return messages

        while True:
            if try_times <= 0:
                print("Reach max retry times to fetch response from client, as error flag.")
                return "client error", ["DONE"]
            try:
                messages = build_messages(history_window)
                origin_resized_height = images[-1].height
                origin_resized_width = images[-1].width
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
                    logger.warning("Reducing Holo prompt history window to %s after context overflow", history_window)
                    continue
                logger.exception("Error when fetching response from client: %s", exc)
                prediction = None
                try_times -= 1
                continue

            try:
                parsed_responses = parse_holo_action_to_structure_output(
                    prediction,
                    image_height=origin_resized_height,
                    image_width=origin_resized_width,
                )
                break
            except Exception as exc:
                print(f"Error when parsing response from client: {exc}")
                prediction = None
                try_times -= 1

        if prediction is None:
            return "client error", ["DONE"]

        self.history_responses.append(prediction)
        self.thoughts.append(prediction)

        actions: List[str] = []
        last_image = Image.open(BytesIO(self.history_images[-1]))
        obs_image_height = last_image.height
        obs_image_width = last_image.width
        for parsed_response in parsed_responses:
            action_type = parsed_response.get("action_type")
            if action_type == FINISH_WORD:
                self.actions.append(actions)
                return prediction, ["DONE"]
            if action_type == WAIT_WORD:
                self.actions.append(actions)
                return prediction, ["WAIT"]
            if action_type == ENV_FAIL_WORD:
                self.actions.append(actions)
                return prediction, ["FAIL"]
            if action_type == CALL_USER:
                if self.callusr_tolerance > self.cur_callusr_count:
                    self.actions.append(actions)
                    self.cur_callusr_count += 1
                    return prediction, ["WAIT"]
                self.actions.append(actions)
                return prediction, ["FAIL"]

            pyautogui_code = parsing_response_to_pyautogui_code(
                parsed_response,
                obs_image_height,
                obs_image_width,
                self.input_swap,
            )
            actions.append(pyautogui_code)

        self.actions.append(actions)
        if len(self.history_responses) >= self.max_trajectory_length:
            actions = ["FAIL"]
        return prediction, actions
