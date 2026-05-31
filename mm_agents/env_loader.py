from pathlib import Path
from typing import Optional
import os

_ROOT: Optional[Path] = None
_LOADED = False


def gui_docker_env_root() -> Path:
    global _ROOT
    if _ROOT is None:
        _ROOT = Path(__file__).resolve().parent.parent
    return _ROOT


def load_mm_agents_env() -> None:
    global _LOADED
    if _LOADED:
        return
    root = gui_docker_env_root()
    try:
        from dotenv import load_dotenv
    except ImportError:
        _LOADED = True
        return
    default = root / ".env-default"
    local = root / ".env"
    if default.is_file():
        load_dotenv(default, override=False)
    if local.is_file():
        load_dotenv(local, override=True)
    for key in (
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENROUTER_BASE_URL",
        "UITARS_OPENAI_BASE_URL",
        "UITARS_OPENAI_BASE_URLS",
        "AZURE_OPENAI_API_BASE",
        "AZURE_OPENAI_ENDPOINT",
    ):
        if os.environ.get(key, "").strip() == "":
            os.environ.pop(key, None)
    _LOADED = True
