"""Holo3 agent-loop chat message builders."""

from __future__ import annotations

import math
from io import BytesIO
from typing import Any, Dict, List, Tuple

from PIL import Image

from mm_agents.gui_image import pil_to_base64

EVICTED_SCREENSHOT_TEXT = "[screenshot evicted]"


def prepare_screenshot(
    screenshot_bytes: bytes,
    *,
    max_pixels: float,
    min_pixels: float,
) -> Tuple[Image.Image, int, int, int, int]:
    """Resize screenshot for the model and return sent/screen dimensions."""
    screen_image = Image.open(BytesIO(screenshot_bytes))
    if screen_image.mode != "RGB":
        screen_image = screen_image.convert("RGB")
    screen_width, screen_height = screen_image.size

    image = screen_image.copy()
    if image.width * image.height > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))
    if image.width * image.height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        image = image.resize(
            (math.ceil(image.width * resize_factor), math.ceil(image.height * resize_factor))
        )

    return image, image.width, image.height, screen_width, screen_height


def observation_message(image: Image.Image) -> Dict[str, Any]:
    encoded = pil_to_base64(image)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "<observation>\n"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            {"type": "text", "text": "\n</observation>"},
        ],
    }


def tool_output_message(tool_name: str, result: str = "Action executed successfully.") -> Dict[str, Any]:
    return {
        "role": "user",
        "content": f'<tool_output tool="{tool_name}">\n{result}\n</tool_output>',
    }


def system_message(text: str) -> Dict[str, Any]:
    return {"role": "system", "content": text}


def trim_to_last_n_images(messages: List[Dict[str, Any]], n: int = 3) -> None:
    """Replace image chunks beyond the last n screenshots with text placeholders."""
    seen = 0
    for msg in reversed(messages):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for chunk in msg["content"]:
            if chunk.get("type") != "image_url":
                continue
            seen += 1
            if seen > n:
                chunk["type"] = "text"
                chunk["text"] = EVICTED_SCREENSHOT_TEXT
                chunk.pop("image_url", None)
