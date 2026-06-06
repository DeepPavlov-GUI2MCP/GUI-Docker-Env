"""Holo3 GUI agent: structured JSON agent-loop harness."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import openai

from mm_agents.env_loader import load_mm_agents_env
from mm_agents.holo.executor import execute_step
from mm_agents.holo.messages import (
    observation_message,
    prepare_screenshot,
    system_message,
    tool_output_message,
    trim_to_last_n_images,
)
from mm_agents.holo.schema import Step, build_system_prompt, step_json_schema

load_mm_agents_env()

logger = logging.getLogger("desktopenv.agent.holo")


def _client_error_message(prediction: Optional[str], exc: Optional[Exception]) -> str:
    if exc is None:
        return "client error"
    if prediction:
        preview = prediction[:500].replace("\n", " ")
        return f"client error: {exc} | response[:500]={preview}"
    return f"client error: {exc}"


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
        self.top_p = self.runtime_conf.get("top_p", 0.9)
        self.max_tokens = self.runtime_conf["max_tokens"]
        self.input_swap = self.runtime_conf["input_swap"]
        self.max_pixels = self.runtime_conf["max_pixels"]
        self.min_pixels = self.runtime_conf["min_pixels"]
        self.image_budget = int(self.runtime_conf.get("history_n", 3))

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
        if not baseurl_list:
            baseurl_list = ["http://127.0.0.1:8010/v1"]

        api_key = kwargs.get("api_key") or _ev("HOLO_OPENAI_API_KEY", "OPENAI_API_KEY") or "empty"
        base_url_index = int(kwargs.get("base_url_index", 0))
        selected_base_url = baseurl_list[base_url_index % len(baseurl_list)]
        self.vlm = openai.OpenAI(api_key=api_key, base_url=selected_base_url)
        self._base_url = selected_base_url
        logger.info(
            "Holo OpenAI base_url=%s (index=%d of %d)",
            selected_base_url,
            base_url_index % len(baseurl_list),
            len(baseurl_list),
        )
        self._api_model_override = (kwargs.get("api_model") or "").strip()

        self.messages: List[Dict[str, Any]] = []
        self._step_index = 0
        self._last_tool_name: Optional[str] = None
        self._instruction: Optional[str] = None
        self._screen_width = 0
        self._screen_height = 0
        self.last_usage: Any = None
        self._schema = step_json_schema()

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
        self.messages = []
        self._step_index = 0
        self._last_tool_name = None
        self._instruction = None
        self._screen_width = 0
        self._screen_height = 0
        self.last_usage = None

    def predict(
        self,
        instruction: str,
        obs: Dict[str, Any],
        last_action_after_obs: Dict[str, Any] = None,
    ) -> List[Any]:
        if self.observation_type not in ["screenshot", "screenshot_a11y_tree"]:
            raise ValueError("Invalid observation_type type: " + self.observation_type)

        if not self.messages:
            self._instruction = instruction
            self.messages.append(system_message(build_system_prompt(instruction)))

        if self._step_index > 0 and self._last_tool_name:
            self.messages.append(tool_output_message(self._last_tool_name))

        image, _sent_w, _sent_h, screen_width, screen_height = prepare_screenshot(
            obs["screenshot"],
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
        )
        self._screen_width = screen_width
        self._screen_height = screen_height
        self.messages.append(observation_message(image))
        trim_to_last_n_images(self.messages, n=self.image_budget)

        try_times = 3
        image_budget = self.image_budget
        prediction: Optional[str] = None
        last_error: Optional[Exception] = None
        model_name = self._resolve_model_name(self.model)

        while try_times > 0:
            try:
                response = self.vlm.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    extra_body={"structured_outputs": {"json": self._schema}},
                )
                self.last_usage = getattr(response, "usage", None)
                message = response.choices[0].message
                reasoning = getattr(message, "reasoning_content", None)
                if reasoning:
                    logger.debug("Holo reasoning: %s", str(reasoning)[:500])
                prediction = (message.content or "").strip()
                step = Step.model_validate_json(prediction)
                self.messages.append({"role": "assistant", "content": step.model_dump_json()})
                self._last_tool_name = step.tool_call.tool_name
                self._step_index += 1

                response_json, actions = execute_step(
                    step,
                    screen_width=self._screen_width,
                    screen_height=self._screen_height,
                    input_swap=self.input_swap,
                )
                if self._step_index >= self.max_trajectory_length and actions != ["DONE"]:
                    return response_json, ["FAIL"]
                return response_json, actions
            except Exception as exc:
                last_error = exc
                err_text = str(exc).lower()
                if "maximum context length" in err_text and image_budget > 1:
                    image_budget -= 1
                    trim_to_last_n_images(self.messages, n=image_budget)
                    logger.warning("Reducing Holo image budget to %s after context overflow", image_budget)
                    continue
                logger.warning("Holo agent error (retries left %s): %s", try_times - 1, exc)
                try_times -= 1

        logger.warning("Holo agent exhausted parse/API retries")
        return _client_error_message(prediction, last_error), ["DONE"]
