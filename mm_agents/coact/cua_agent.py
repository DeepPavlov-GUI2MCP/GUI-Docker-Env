import base64
import json
import logging
import os
import re
import time
from html import unescape
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

import openai
import requests
import trafilatura
from desktop_env.desktop_env import DesktopEnv
from openai import OpenAI
from mm_agents.env_loader import load_mm_agents_env

load_mm_agents_env()

logger = logging.getLogger("desktopenv")

GPT4O_INPUT_PRICE_PER_1M_TOKENS = 3.00
GPT4O_OUTPUT_PRICE_PER_1M_TOKENS = 12.00
DEFAULT_CUA_MODEL = "gpt-5.5"
DEFAULT_GUI_PROTOCOL = "openai"
SUPPORTED_GUI_PROTOCOLS = (DEFAULT_GUI_PROTOCOL, "vllm")
DEFAULT_PROMPT_MODE = "default"

BASE_PROMPT_TEMPLATE = """# Task
{instruction}

# Hints
- Sudo password is "{CLIENT_PASSWORD}".
- Keep the windows/applications opened at the end of the task.
- Do not use shortcut to reload the application except for the browser, just close and reopen.
- If "The document has been changed by others" pops out, you should click "cancel" and reopen the file.
{mode_hints}
- If you have completed the user task, reply with the information you want the user to know along with 'TERMINATE'.
- If you don't know how to continue the task, reply your concern or question along with 'IDK'.
""".strip()
DEFAULT_REPLY = "Please continue the user task. If you have completed the user task, reply with the information you want the user to know along with 'TERMINATE'."
SEARCH_TOOL_TYPE = "web_search"
READ_WEBPAGE_TOOL_NAME = "read_webpage"
SUPPORTED_CUA_MODEL_PREFIXES = ("gpt-5",)
LEGACY_CUA_MODELS = {"computer-use-preview"}
MAX_RESPONSE_MULTIPLIER = 3
DEFAULT_WEBPAGE_MAX_CHARS = 12000
MAX_WEBPAGE_MAX_CHARS = 20000
REQUEST_TIMEOUT_SECONDS = 20
DEFAULT_WEBPAGE_BACKEND = "trafilatura"
FALLBACK_WEBPAGE_BACKEND = "html"
SUPPORTED_WEBPAGE_BACKENDS = {DEFAULT_WEBPAGE_BACKEND, FALLBACK_WEBPAGE_BACKEND}

READ_WEBPAGE_TOOL = {
    "type": "function",
    "name": READ_WEBPAGE_TOOL_NAME,
    "description": "Fetch a webpage by URL and return extracted text content for planning.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The absolute URL of the page to inspect.",
            },
            "backend": {
                "type": "string",
                "description": "Extraction backend to use. Defaults to trafilatura for concise page text.",
                "default": DEFAULT_WEBPAGE_BACKEND,
                "enum": sorted(SUPPORTED_WEBPAGE_BACKENDS),
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum number of extracted characters to return.",
                "default": DEFAULT_WEBPAGE_MAX_CHARS,
                "minimum": 500,
                "maximum": MAX_WEBPAGE_MAX_CHARS,
            },
        },
        "required": ["url"],
    },
}


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag in {"p", "div", "section", "article", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in {"p", "div", "section", "article", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "br"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data)

    def get_text(self) -> str:
        text = unescape(" ".join(self.parts))
        text = re.sub(r"\r", "\n", text)
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()


def _mode_hints(prompt_mode: str) -> str:
    if prompt_mode == "search-first":
        return (
            "- The orchestrator may provide research-backed instructions for this task.\n"
            "- Focus on carrying out the delegated GUI actions precisely and efficiently."
        )
    if prompt_mode == "inspect-source-first":
        return (
            "- The orchestrator may provide source-guided instructions for this task.\n"
            "- Focus on carrying out the delegated GUI actions precisely and efficiently."
        )
    return "- Focus on carrying out the delegated GUI actions precisely and efficiently."


def _build_prompt(
    instruction: str,
    client_password: str,
    prompt_mode: str = DEFAULT_PROMPT_MODE,
    task_source: Optional[str] = None,
) -> str:
    del task_source
    return BASE_PROMPT_TEMPLATE.format(
        instruction=instruction,
        CLIENT_PASSWORD=client_password,
        mode_hints=_mode_hints(prompt_mode),
    )


def validate_cua_model(cua_model: str) -> None:
    if not cua_model:
        raise ValueError("Missing GUI model. Set `OPENAI_CUA_MODEL` or pass `--cua_model`.")
    if cua_model in LEGACY_CUA_MODELS:
        raise ValueError(f"Unsupported legacy GUI model `{cua_model}`.")


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


def _cua_to_pyautogui(action) -> str:
    def fld(key: str, default: Any = None) -> Any:
        return action.get(key, default) if isinstance(action, dict) else getattr(action, key, default)

    act_type = fld("type")
    if not isinstance(act_type, str):
        act_type = str(act_type).split(".")[-1]
    act_type = act_type.lower()

    if act_type in ["click", "double_click"]:
        button = fld('button', 'left')
        if button == 1 or button == 'left':
            button = 'left'
        elif button == 2 or button == 'middle':
            button = 'middle'
        elif button == 3 or button == 'right':
            button = 'right'
        elif button == 'wheel':
            button = 'middle'

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
        path = fld('path', [{"x": 0, "y": 0}, {"x": 0, "y": 0}])
        cmd = f"pyautogui.moveTo({path[0]['x']}, {path[0]['y']}, _pause=False); "
        cmd += f"pyautogui.dragTo({path[-1]['x']}, {path[-1]['y']}, duration=0.5, button='left')"
        return _wrap_with_modifiers(cmd, fld("keys"))

    if act_type == 'move':
        return _wrap_with_modifiers(f"pyautogui.moveTo({fld('x')}, {fld('y')})", fld("keys"))

    if act_type == "keypress":
        keys = [_normalize_key(key) for key in (fld("keys", []) or [fld("key")]) if key]
        if len(keys) == 1:
            return f"pyautogui.press('{keys[0].lower()}')"
        else:
            return f"pyautogui.hotkey({', '.join(repr(key) for key in keys)})"
        
    if act_type == "type":
        text = str(fld("text", ""))
        return "pyautogui.typewrite({:})".format(repr(text))
    
    if act_type == "wait":
        return "WAIT"
    
    return "WAIT"  # fallback


def _item_to_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")
    raise TypeError(f"Unsupported response item type: {type(item)!r}")


def _sanitize_images(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for key, item_value in value.items():
            if key == "image_url" and isinstance(item_value, str) and item_value.startswith("data:image/"):
                sanitized[key] = "<image>"
            else:
                sanitized[key] = _sanitize_images(item_value)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_images(item) for item in value]
    return value


def _extract_message_text(message_item: Dict[str, Any]) -> str:
    texts: List[str] = []
    for content_item in message_item.get("content", []):
        if content_item.get("type") == "output_text" and content_item.get("text"):
            texts.append(content_item["text"])
    return "\n".join(texts).strip()


def _extract_reasoning_text(reasoning_item: Dict[str, Any]) -> str:
    texts = [summary.get("text", "").strip() for summary in reasoning_item.get("summary", []) if summary.get("text")]
    return "\n".join(text for text in texts if text)


def _response_to_transcript_entry(response: Any) -> Dict[str, Any]:
    usage = None
    if hasattr(response, "usage") and response.usage:
        usage = response.usage.model_dump(mode="json") if hasattr(response.usage, "model_dump") else response.usage
    return _sanitize_images(
        {
            "response_id": getattr(response, "id", None),
            "output": [_item_to_dict(item) for item in getattr(response, "output", [])],
            "usage": usage,
        }
    )


def _estimate_cost(cua_model: str, response: Any) -> float:
    if not response or not getattr(response, "usage", None):
        return 0.0
    if cua_model != "gpt-4o":
        return 0.0
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    input_cost = (input_tokens / 1_000_000) * GPT4O_INPUT_PRICE_PER_1M_TOKENS
    output_cost = (output_tokens / 1_000_000) * GPT4O_OUTPUT_PRICE_PER_1M_TOKENS
    return input_cost + output_cost


def _build_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    api_key = api_key if api_key is not None else (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY_CUA"))
    base_url = base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL")
    kwargs: Dict[str, Any] = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    elif base_url:
        kwargs["api_key"] = "EMPTY"
    return OpenAI(**kwargs)


def _build_cua_tools() -> List[Dict[str, Any]]:
    return [{"type": "computer"}]


def _parse_function_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _bounded_max_chars(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_WEBPAGE_MAX_CHARS
    return max(500, min(parsed, MAX_WEBPAGE_MAX_CHARS))


def _extract_title(html: str) -> Optional[str]:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return unescape(re.sub(r"\s+", " ", match.group(1))).strip() or None


def _extract_html_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.get_text()


def read_webpage(
    url: str,
    max_chars: int = DEFAULT_WEBPAGE_MAX_CHARS,
    backend: str = DEFAULT_WEBPAGE_BACKEND,
) -> Dict[str, Any]:
    cleaned_url = str(url).strip()
    if not cleaned_url:
        return {"url": cleaned_url, "title": None, "content": "", "truncated": False, "error": "Missing URL."}
    if not cleaned_url.startswith(("http://", "https://")):
        return {
            "url": cleaned_url,
            "title": None,
            "content": "",
            "truncated": False,
            "error": "Only http:// and https:// URLs are supported.",
        }

    normalized_backend = str(backend or DEFAULT_WEBPAGE_BACKEND).strip().lower()
    if normalized_backend not in SUPPORTED_WEBPAGE_BACKENDS:
        return {
            "url": cleaned_url,
            "title": None,
            "content": "",
            "truncated": False,
            "error": (
                f"Unsupported backend: {normalized_backend}. "
                f"Supported backends: {', '.join(sorted(SUPPORTED_WEBPAGE_BACKENDS))}."
            ),
        }

    try:
        response = requests.get(
            cleaned_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                )
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return {
            "url": cleaned_url,
            "title": None,
            "content": "",
            "truncated": False,
            "error": f"Failed to fetch page: {exc}",
        }

    text_content = ""
    effective_backend = normalized_backend
    if normalized_backend == DEFAULT_WEBPAGE_BACKEND:
        text_content = (
            trafilatura.extract(
                response.text,
                url=response.url,
                include_comments=False,
                include_tables=False,
                include_links=False,
                favor_precision=True,
            )
            or ""
        ).strip()
        if not text_content:
            effective_backend = FALLBACK_WEBPAGE_BACKEND

    if effective_backend == FALLBACK_WEBPAGE_BACKEND:
        text_content = _extract_html_text(response.text)

    bounded_chars = _bounded_max_chars(max_chars)
    truncated = len(text_content) > bounded_chars
    return {
        "url": response.url,
        "title": _extract_title(response.text),
        "content": text_content[:bounded_chars],
        "truncated": truncated,
        "error": None if text_content else "Could not extract readable page text.",
        "backend": effective_backend,
    }


def _read_webpage(url: str, max_chars: int = DEFAULT_WEBPAGE_MAX_CHARS) -> Dict[str, Any]:
    return read_webpage(url=url, max_chars=max_chars, backend=DEFAULT_WEBPAGE_BACKEND)


def _execute_function_call(item: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    function_name = item.get("name")
    arguments = _parse_function_arguments(item.get("arguments"))
    if function_name != READ_WEBPAGE_TOOL_NAME:
        result = {"error": f"Unsupported function tool: {function_name}"}
    else:
        result = read_webpage(
            url=arguments.get("url", ""),
            max_chars=arguments.get("max_chars", DEFAULT_WEBPAGE_MAX_CHARS),
            backend=arguments.get("backend", DEFAULT_WEBPAGE_BACKEND),
        )

    output_item = {
        "type": "function_call_output",
        "call_id": item["call_id"],
        "output": json.dumps(result),
    }
    tool_event = {
        "type": "function_call",
        "tool_name": function_name,
        "call_id": item.get("call_id"),
        "arguments": arguments,
        "output_preview": {
            "url": result.get("url"),
            "title": result.get("title"),
            "truncated": result.get("truncated"),
            "error": result.get("error"),
            "content_chars": len(result.get("content", "")),
        },
    }
    return output_item, tool_event


def _actions_from_computer_call(action_call: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions = action_call.get("actions")
    if actions:
        return [_item_to_dict(action) for action in actions]
    action = action_call.get("action")
    if action:
        return [_item_to_dict(action)]
    return []


def _action_step_cost(action: Dict[str, Any]) -> int:
    return 0 if action.get("type") == "screenshot" else 1


def _capture_screenshot(env: DesktopEnv) -> bytes:
    screenshot = env.controller.get_screenshot()
    return screenshot["screenshot"] if isinstance(screenshot, dict) else screenshot


def _execute_computer_call(
    env: DesktopEnv,
    action_call: Dict[str, Any],
    action_index: int,
    save_path: str,
    sleep_after_execution: float,
) -> Dict[str, Any]:
    actions = _actions_from_computer_call(action_call)
    latest_screenshot: Optional[bytes] = None

    for action in actions:
        if action.get("type") == "screenshot":
            latest_screenshot = _capture_screenshot(env)
            continue
        py_cmd = _cua_to_pyautogui(action)
        obs, *_ = env.step(py_cmd, sleep_after_execution)
        latest_screenshot = obs["screenshot"]

    if latest_screenshot is None:
        latest_screenshot = _capture_screenshot(env)

    screenshot_b64 = base64.b64encode(latest_screenshot).decode("utf-8")
    with open(os.path.join(save_path, f"step_{action_index}.png"), "wb") as f:
        f.write(latest_screenshot)

    output_item: Dict[str, Any] = {
        "type": "computer_call_output",
        "call_id": action_call["call_id"],
        "output": {
            "type": "computer_screenshot",
            "image_url": f"data:image/png;base64,{screenshot_b64}",
        },
    }
    pending_safety_checks = action_call.get("pending_safety_checks", [])
    if pending_safety_checks:
        output_item["acknowledged_safety_checks"] = [
            {
                "id": safety_check["id"],
                "code": safety_check.get("code"),
                "message": safety_check.get("message") or "Proceed with the acknowledged safety check.",
            }
            for safety_check in pending_safety_checks
        ]
    return output_item


def call_openai_cua(client: OpenAI,
                    response_input: list,
                    cua_model: str,
                    previous_response_id: Optional[str] = None) -> Tuple[Any, float]:
    retry = 0
    response = None
    last_error: Optional[Exception] = None
    validate_cua_model(cua_model)
    while retry < 3:
        try:
            response = client.responses.create(
                model=cua_model,
                tools=_build_cua_tools(),
                input=response_input,
                previous_response_id=previous_response_id,
                parallel_tool_calls=False,
                reasoning={"summary": "concise"},
            )
            break
        except (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.BadRequestError,
            openai.InternalServerError,
            openai.RateLimitError,
        ) as e:
            retry += 1
            last_error = e
            logger.error(f"Error in response.create: {e}")
            time.sleep(0.5)
    if response is None:
        if last_error is not None:
            raise RuntimeError(f"OpenAI Responses call failed after {retry} attempts: {last_error}") from last_error
        raise RuntimeError("OpenAI Responses call failed without returning a response.")

    return response, _estimate_cost(cua_model=cua_model, response=response)


def _build_vllm_runtime_conf(
    *,
    temperature: float = 0.0,
    top_p: float = 0.95,
) -> Dict[str, Any]:
    return {
        "infer_mode": "qwen25vl_normal",
        "prompt_style": "qwen25vl_normal",
        "input_swap": False,
        "language": "English",
        "history_n": 5,
        "max_pixels": 16384 * 28 * 28,
        "min_pixels": 100 * 28 * 28,
        "callusr_tolerance": 3,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": -1,
        "max_tokens": 2048,
    }


def _build_vllm_instruction(
    instruction: str,
    client_password: str,
    prompt_mode: str = DEFAULT_PROMPT_MODE,
    task_source: Optional[str] = None,
) -> str:
    lines = [instruction.strip()]
    if client_password:
        lines.append(f'Sudo password: "{client_password}".')
    lines.extend(
        [
            "Keep the windows/applications opened at the end of the task.",
            "Do not reload applications with shortcuts except for browsers.",
            'If a dialog says "The document has been changed by others", choose "cancel" and reopen the file.',
            "If the UI needs time to update, use wait().",
            "If the environment is broken or the task is impossible, use error_env() or call_user().",
        ]
    )
    if prompt_mode in {"search-first", "inspect-source-first"} and task_source:
        lines.append(f"Task source URL: {task_source}")
    return "\n".join(line for line in lines if line).strip()


def _is_vllm_error_prediction(prediction: str) -> bool:
    normalized = (prediction or "").strip().lower()
    return (
        normalized.startswith("client error")
        or normalized.startswith("parsing action error")
        or normalized.startswith("error when parsing response")
    )


def _run_cua_vllm(
    env: DesktopEnv,
    instruction: str,
    max_steps: int,
    save_path: str = "./",
    cua_model: str = "ui-tars-1.5",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    sleep_after_execution: float = 0.3,
    client_password: str = "",
    prompt_mode: str = DEFAULT_PROMPT_MODE,
    task_source: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 0.95,
) -> Tuple[List[Dict[str, Any]], str, float, List[Dict[str, Any]], List[Dict[str, Any]], int]:
    from mm_agents.uitars15_v1 import UITARSAgent

    vllm_instruction = _build_vllm_instruction(
        instruction=instruction,
        client_password=client_password,
        prompt_mode=prompt_mode,
        task_source=task_source,
    )
    agent = UITARSAgent(
        model=cua_model,
        action_space="pyautogui",
        observation_type="screenshot",
        max_trajectory_length=max(max_steps, 1),
        model_type="qwen25vl",
        runtime_conf=_build_vllm_runtime_conf(temperature=temperature, top_p=top_p),
        base_url=base_url,
        api_key=api_key or "empty",
    )
    agent.reset()

    logger.info(f"Instruction: {instruction}")
    obs = env._get_obs()
    initial_screenshot = obs["screenshot"]
    screenshot_b64 = base64.b64encode(initial_screenshot).decode("utf-8")
    with open(os.path.join(save_path, "initial_screenshot.png"), "wb") as f:
        f.write(initial_screenshot)

    history_inputs: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": vllm_instruction},
                {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}", "detail": "original"},
            ],
        }
    ]
    raw_transcript: List[Dict[str, Any]] = []
    tool_events: List[Dict[str, Any]] = []
    total_cost = 0.0
    final_result = ""
    step_count = 0
    action_index = 0

    while step_count < max_steps:
        prediction, actions = agent.predict(vllm_instruction, obs)
        raw_transcript.append(
            {
                "protocol": "vllm",
                "step_index": action_index + 1,
                "prediction": prediction,
                "actions": list(actions),
            }
        )
        history_inputs.append(
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": prediction}],
            }
        )

        if not actions:
            final_result = "IDK The GUI loop stopped before producing an action."
            break

        for action in actions:
            if step_count >= max_steps:
                final_result = "IDK Step budget exhausted before the GUI task was completed."
                break

            action_index += 1
            step_count += 1
            obs, reward, done, info = env.step(action, sleep_after_execution)
            screenshot = obs["screenshot"]
            screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")
            with open(os.path.join(save_path, f"step_{action_index}.png"), "wb") as f:
                f.write(screenshot)

            tool_events.append(
                {
                    "type": "vllm_action",
                    "step_index": action_index,
                    "prediction": prediction,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": info,
                }
            )
            history_inputs.append(
                {
                    "role": "tool",
                    "content": [
                        {"type": "tool_result", "text": str(action)},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}", "detail": "original"},
                    ],
                }
            )

            if action == "DONE":
                if _is_vllm_error_prediction(prediction):
                    final_result = f"IDK {prediction}".strip()
                else:
                    final_result = prediction if "TERMINATE" in prediction else f"{prediction}\nTERMINATE"
                break
            if action == "FAIL":
                final_result = f"IDK {prediction}".strip()
                break

        if final_result:
            break

    if not final_result:
        final_result = "IDK Step budget exhausted before the GUI task was completed."

    return _sanitize_images(history_inputs), final_result, total_cost, raw_transcript, tool_events, step_count


def run_cua(
    env: DesktopEnv,
    instruction: str,
    max_steps: int,
    save_path: str = './',
    cua_model: str = os.environ.get("OPENAI_CUA_MODEL", DEFAULT_CUA_MODEL),
    gui_protocol: str = DEFAULT_GUI_PROTOCOL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    screen_width: int = 1920,
    screen_height: int = 1080,
    sleep_after_execution: float = 0.3,
    truncate_history_inputs: int = 100,
    client_password: str = "",
    prompt_mode: str = DEFAULT_PROMPT_MODE,
    task_source: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], str, float, List[Dict[str, Any]], List[Dict[str, Any]], int]:
    if gui_protocol not in SUPPORTED_GUI_PROTOCOLS:
        raise ValueError(f"Unsupported GUI protocol `{gui_protocol}`. Expected one of {SUPPORTED_GUI_PROTOCOLS}.")
    if gui_protocol == "vllm":
        del screen_width, screen_height, truncate_history_inputs
        return _run_cua_vllm(
            env=env,
            instruction=instruction,
            max_steps=max_steps,
            save_path=save_path,
            cua_model=cua_model,
            base_url=base_url,
            api_key=api_key,
            sleep_after_execution=sleep_after_execution,
            client_password=client_password,
            prompt_mode=prompt_mode,
            task_source=task_source,
            temperature=0.0 if temperature is None else temperature,
            top_p=0.95 if top_p is None else top_p,
        )

    client = _build_openai_client(api_key=api_key, base_url=base_url)
    validate_cua_model(cua_model)

    logger.info(f"Instruction: {instruction}")
    del screen_width, screen_height, truncate_history_inputs
    obs = _capture_screenshot(env)
    screenshot_b64 = base64.b64encode(obs).decode("utf-8")
    with open(os.path.join(save_path, "initial_screenshot.png"), "wb") as f:
        f.write(obs)
    history_inputs = [{
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": _build_prompt(
                    instruction=instruction,
                    client_password=client_password,
                    prompt_mode=prompt_mode,
                    task_source=task_source,
                ),
            },
            {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}", "detail": "original"},
        ],
    }]

    response, cost = call_openai_cua(
        client,
        history_inputs,
        cua_model=cua_model,
    )
    total_cost = cost
    logger.info(f"Cost: ${cost:.6f} | Total Cost: ${total_cost:.6f}")
    step_count = 0
    response_count = 0
    action_index = 0
    reasoning_list: List[str] = []
    raw_transcript = [_response_to_transcript_entry(response)]
    tool_events: List[Dict[str, Any]] = []
    final_result = ""

    while response_count < max(max_steps * MAX_RESPONSE_MULTIPLIER, 1):
        response_count += 1
        follow_up_inputs: List[Dict[str, Any]] = []
        should_continue = False

        for output_item in response.output:
            item = _item_to_dict(output_item)
            item_type = item.get("type")

            if item_type == "reasoning":
                reasoning_text = _extract_reasoning_text(item)
                if reasoning_text:
                    reasoning_list.append(reasoning_text)
                    logger.info(f"[Reasoning]: {reasoning_text}")
                continue

            if item_type == "web_search_call":
                tool_events.append(
                    {
                        "type": "web_search_call",
                        "response_id": response.id,
                        "status": item.get("status"),
                        "action": item.get("action"),
                    }
                )
                continue

            if item_type == "function_call":
                follow_up_item, tool_event = _execute_function_call(item)
                tool_events.append(tool_event)
                history_inputs.append(_sanitize_images(follow_up_item))
                follow_up_inputs.append(follow_up_item)
                should_continue = True
                continue

            if item_type == "computer_call":
                actions = _actions_from_computer_call(item)
                action_cost = sum(_action_step_cost(action) for action in actions)
                if step_count + action_cost > max_steps:
                    final_result = "IDK Step budget exhausted before the GUI task was completed."
                    break

                step_count += action_cost
                action_index += 1
                tool_events.append(
                    {
                        "type": "computer_call",
                        "response_id": response.id,
                        "call_id": item.get("call_id"),
                        "actions": actions,
                        "pending_safety_checks": item.get("pending_safety_checks", []),
                        "step_index": action_index,
                    }
                )
                follow_up_item = _execute_computer_call(
                    env=env,
                    action_call=item,
                    action_index=action_index,
                    save_path=save_path,
                    sleep_after_execution=sleep_after_execution,
                )
                history_inputs.append(_sanitize_images(follow_up_item))
                follow_up_inputs.append(follow_up_item)
                should_continue = True
                continue

            if item_type == "message":
                message_text = _extract_message_text(item)
                if message_text:
                    logger.info(f"[Message]: {message_text}")
                    reasoning_list.append(message_text)
                    if "TERMINATE" in message_text or "IDK" in message_text:
                        final_result = message_text
                        break

        if final_result:
            break

        if not should_continue:
            follow_up_message = {
                "role": "user",
                "content": [{"type": "input_text", "text": DEFAULT_REPLY}],
            }
            history_inputs.append(_sanitize_images(follow_up_message))
            follow_up_inputs.append(follow_up_message)

        response, cost = call_openai_cua(
            client,
            response_input=follow_up_inputs,
            cua_model=cua_model,
            previous_response_id=response.id,
        )
        total_cost += cost
        raw_transcript.append(_response_to_transcript_entry(response))
        logger.info(f"Cost: ${cost:.6f} | Total Cost: ${total_cost:.6f}")

    if not final_result:
        final_result = "IDK The GUI loop stopped before producing a terminal answer."

    logger.info(f"Total cost for the task: ${total_cost:.4f}")
    return _sanitize_images(history_inputs), final_result, total_cost, raw_transcript, tool_events, step_count

