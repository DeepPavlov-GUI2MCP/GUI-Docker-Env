import argparse
import base64
import datetime
import glob
import json
import logging
import os
import shutil
import sys
import time
import traceback
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from desktop_env.desktop_env import DesktopEnv
from mm_agents.coact.autogen import LLMConfig
from mm_agents.coact.cua_agent import DEFAULT_CUA_MODEL, run_cua, validate_cua_model
from mm_agents.coact.operator_agent import OrchestratorAgent, OrchestratorUserProxyAgent
from mm_agents.coact.run_config import (
    DEFAULT_GUI_PROTOCOL,
    DEFAULT_MODE,
    DEFAULT_SCRIPT_MODE,
    SUPPORTED_GUI_PROTOCOLS,
    SUPPORTED_MODES,
    SUPPORTED_SCRIPT_MODES,
    BackendSettings,
    MultiRolloutSettings,
    ResolvedRunConfig,
    load_config_file,
)
from mm_agents.env_loader import load_mm_agents_env

load_mm_agents_env()

DEFAULT_OAI_CONFIG_PATH = str((Path(__file__).resolve().parent / "mm_agents" / "coact" / "OAI_CONFIG_LIST").resolve())
datetime_str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
logger = logging.getLogger("desktopenv.expeiment")

BASE_TASK_DESCRIPTION = """# Your role
You are a task solver, you need to complete a computer-using task step-by-step.
1. Describe the screenshot.
2. Provide a detailed plan, including a list of user requirements like specific file name, file path, etc.
3. Follow the following instructions and complete the task with your skills.
    - If you think the task is impossible to complete (no file, wrong environment, etc.), reply with "INFEASIBLE" to end the conversation.
    - **Do not** do (or let a helper agent do) anything else out of the user's instruction like change the file name. This will make the task fail.
    - Check every screenshot carefully and see if it fulfills the task requirement.
    - When research or a source page contains actionable details, preserve all of them in your plan instead of reducing them to a minimal subset.
    - Treat concrete settings, values, value types, restart requirements, verification requirements, and related follow-up changes as actionable details that must be tracked explicitly.
    - Before delegating, identify which discovered details are required and which are optional. Do not omit or downgrade a detail unless the evidence clearly says it is optional or irrelevant.
    - When you delegate, copy every required detail into the delegated task so the helper does not need to reconstruct the plan from memory.
4. Verify the result and see if it fulfills the user's requirement.
"""

CODING_AGENT_SECTION = """
## Programmer
Let a programmer solve a subtask you assign.
The Programmer can write python or bash code to modify files, apps, and system settings.
It requires an environment description and a detailed task description.
Can use any python package you instruct.
Will return a summary with the output of the code.
When letting the coding agent modify a spreadsheet, you MUST verify the result in the GUI afterward.
""".strip()


def _setup_logging(log_level_name: str) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    log_level = getattr(logging, log_level_name.upper())
    root_logger.setLevel(log_level)

    file_handler = logging.FileHandler(os.path.join("logs", f"normal-{datetime_str}.log"), encoding="utf-8")
    debug_handler = logging.FileHandler(os.path.join("logs", f"debug-{datetime_str}.log"), encoding="utf-8")
    stdout_handler = logging.StreamHandler(sys.stdout)

    file_handler.setLevel(logging.INFO)
    debug_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="[1;33m[%(asctime)s [31m%(levelname)s [32m%(module)s/%(lineno)d-%(processName)s[1;33m] [0m%(message)s"
    )
    file_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(logging.Filter("desktopenv"))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(debug_handler)
    root_logger.addHandler(stdout_handler)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation on the benchmark")
    parser.add_argument("--config", type=str, default=None, help="Path to a full YAML config. Cannot be combined with other flags.")

    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--provider_name", type=str, default="aws")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.5)
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--client_password", type=str, default="osworld-public-evaluation")

    parser.add_argument("--oai_config_path", type=str, default=DEFAULT_OAI_CONFIG_PATH)
    parser.add_argument("--orchestrator_model", type=str, default="o3")
    parser.add_argument("--orchestrator_base_url", type=str, default=None)
    parser.add_argument("--orchestrator_api_key", type=str, default=None)
    parser.add_argument("--coding_model", type=str, default="o4-mini")
    parser.add_argument("--coding_base_url", type=str, default=None)
    parser.add_argument("--coding_api_key", type=str, default=None)
    parser.add_argument("--gui_model", "--cua_model", dest="gui_model", type=str, default=os.environ.get("OPENAI_CUA_MODEL", DEFAULT_CUA_MODEL))
    parser.add_argument("--gui_base_url", type=str, default=None)
    parser.add_argument("--gui_api_key", type=str, default=None)
    parser.add_argument("--gui_protocol", type=str, choices=SUPPORTED_GUI_PROTOCOLS, default=DEFAULT_GUI_PROTOCOL)
    parser.add_argument("--enable_coding_agent", action="store_true", default=False, help="Expose the coding agent tool to the CoAct orchestrator.")
    parser.add_argument("--mode", type=str, choices=SUPPORTED_MODES, default=DEFAULT_MODE, help="Prompt strategy for the orchestrator research path.")
    parser.add_argument("--script_mode", type=str, choices=SUPPORTED_SCRIPT_MODES, default=DEFAULT_SCRIPT_MODE, help="Script-level execution strategy.")
    parser.add_argument(
        "--enable_web_search",
        "--enable_duckduckgo_search",
        dest="enable_web_search",
        action="store_true",
        default=(
            os.environ.get("COACT_ENABLE_WEB_SEARCH", "").lower() in {"1", "true", "yes", "on"}
            or os.environ.get("COACT_ENABLE_DUCKDUCKGO_SEARCH", "").lower() in {"1", "true", "yes", "on"}
        ),
        help="Enable orchestrator research tools in default mode.",
    )
    parser.add_argument("--orchestrator_max_steps", type=int, default=15)
    parser.add_argument("--coding_max_steps", type=int, default=20)
    parser.add_argument("--cua_max_steps", type=int, default=25)
    parser.add_argument("--cut_off_steps", type=int, default=200)
    parser.add_argument("--multi_rollout_k_per_grid_point", type=int, default=1)
    parser.add_argument("--multi_rollout_temperatures", type=float, nargs="+", default=[0.0])
    parser.add_argument("--multi_rollout_top_ps", type=float, nargs="+", default=[0.95])
    parser.add_argument("--multi_rollout_cua_max_steps", type=int, default=None)

    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--task_id",
        type=str,
        default=None,
        help="Run exactly one task. Accepts either a bare task id or a direct path to a task json file.",
    )
    parser.add_argument("--test_all_meta_path", type=str, default="evaluation_examples/test_all.json")
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples/examples")

    parser.add_argument("--result_dir", type=str, default="./results_coact")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel")
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    return parser


def _reject_mixed_config_usage(parser: argparse.ArgumentParser, argv: Sequence[str], args: argparse.Namespace) -> None:
    if not args.config:
        return
    option_names = [token.split("=", 1)[0] for token in argv if token.startswith("--")]
    disallowed = [name for name in option_names if name != "--config"]
    if disallowed:
        parser.error("`--config` cannot be combined with any other CLI options. Use either config mode or args mode.")


def _build_task_description(
    enable_coding_agent: bool,
    enable_gui_agent: bool,
    enable_web_search_tool: bool,
    enable_read_webpage_tool: bool,
) -> str:
    sections = [BASE_TASK_DESCRIPTION.strip(), "# Your helpers"]
    helper_lines = ["You can only call one helper agent or research tool per reply."]

    if enable_web_search_tool:
        helper_lines.extend(
            [
                "",
                "## Web Search",
                "Use web_search to gather reliable public instructions when research is required.",
                "Prefer official documentation and strong Stack Exchange answers when relevant.",
                "Summarize research faithfully and preserve all actionable details needed for execution and verification.",
            ]
        )

    if enable_read_webpage_tool:
        helper_lines.extend(
            [
                "",
                "## Read Webpage",
                "Use read_webpage to inspect a specific source URL and extract actionable instructions before planning or verification.",
            ]
        )

    if enable_coding_agent:
        helper_lines.extend(["", CODING_AGENT_SECTION])

    if enable_gui_agent:
        helper_lines.extend(
            [
                "",
                "## GUI Operator",
                "Let a GUI agent solve a subtask you assign.",
                "The GUI agent can operate the computer by clicking and typing, but it is not accurate in dense UI.",
                "Require a detailed task description.",
                "When you call the GUI agent, it only has a **20-step** budget to complete your task.",
            ]
        )

    sections.append("\\n".join(helper_lines))
    if enable_coding_agent:
        sections.append(
            "If you let GUI Operator check a coding result, you MUST let it close and reopen the file because programmer changes will not automatically appear on screen."
        )
    return "\\n\\n".join(section for section in sections if section)


def _build_task_instruction(task_config: Dict[str, object], mode: str) -> str:
    instruction = str(task_config["instruction"]).strip()
    source = str(task_config.get("source", "")).strip()

    if mode == "search-first":
        lines = [
            instruction,
            "",
            "# Research policy",
            "- Before planning or taking GUI actions, you must first call the custom web_search tool.",
            "- Use read_webpage to inspect promising documentation pages or source URLs when that helps convert search results into exact instructions.",
            "- Prioritize official documentation for the application involved in the task.",
            "- Also look for strong Stack Exchange family answers such as Super User, Stack Overflow, or Ask Ubuntu when relevant.",
            "- Do not declare the task infeasible until you have used web_search and reviewed the returned evidence.",
            "- Preserve all actionable details from the research, including settings, values, value types, restart requirements, verification requirements, and related follow-up changes.",
            "- If the research contains multiple actionable details, decide explicitly which are required and which are optional, and keep every required detail in the execution plan.",
            "- When delegating, restate every required detail directly in the delegated task instead of relying on a shortened summary.",
            "- Only after collecting relevant instructions should you form a plan and execute the task.",
        ]
        if source:
            lines.extend(["", f"Provided source URL: {source}", "- You may inspect the provided source URL with read_webpage after the mandatory web_search step if it is useful."])
        return "\\n".join(lines)

    if mode == "inspect-source-first":
        lines = [
            instruction,
            "",
            "# Source-aware policy",
            "- The read_webpage and web_search tools are available for planning and verification.",
            "- If a source URL is provided, you may inspect it with read_webpage before or during planning when it seems useful.",
            "- You may also use web_search when additional context or troubleshooting guidance would help.",
            "- Preserve all actionable details from any research you use, including settings, values, value types, restart requirements, verification requirements, and related follow-up changes.",
            "- When delegating, restate every required detail directly in the delegated task instead of relying on a shortened summary.",
        ]
        if source:
            lines.extend(["", f"Provided source URL: {source}"])
        return "\\n".join(lines)

    return instruction


def _build_initial_message(task_instruction: str) -> str:
    return (
        f"{task_instruction}\n"
        "Check my computer screenshot and describe it first. If this task is possible to complete, please complete it on my computer. "
        'If not, reply with "INFEASIBLE" to end the conversation.\n'
        "I will not provide further information to you."
    )


def _load_oai_config_list(config_path: str) -> List[Dict[str, Any]]:
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"OAI config file not found: {config_path}")

    with open(config_file, encoding="utf-8") as f:
        config_list = json.load(f)

    if not isinstance(config_list, list):
        raise ValueError(f"OAI config file must contain a JSON list: {config_path}")
    return [dict(entry) for entry in config_list if isinstance(entry, dict)]


def _lookup_oai_entry(config_list: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    for entry in config_list:
        if entry.get("model") == model:
            return dict(entry)
    return {}


def _resolve_args_backend(
    *,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    protocol: str = DEFAULT_GUI_PROTOCOL,
    env_base_url: Optional[str],
    env_api_key: Optional[str],
    legacy_entry: Optional[Dict[str, Any]] = None,
) -> BackendSettings:
    entry = legacy_entry or {}
    resolved_base_url = base_url if base_url is not None else entry.get("base_url") or env_base_url
    resolved_api_key = api_key if api_key is not None else entry.get("api_key") or env_api_key
    if resolved_api_key == "KEY":
        resolved_api_key = env_api_key
    return BackendSettings(model=model, base_url=resolved_base_url, api_key=resolved_api_key, protocol=protocol)


def _resolve_run_config_from_args(args: argparse.Namespace) -> ResolvedRunConfig:
    legacy_config_list = _load_oai_config_list(args.oai_config_path) if args.oai_config_path else []
    env_base_url = os.environ.get("OPENAI_BASE_URL")
    env_api_key = os.environ.get("OPENAI_API_KEY")
    env_gui_api_key = os.environ.get("OPENAI_API_KEY_CUA") or env_api_key
    multi_rollout = MultiRolloutSettings(
        k_per_grid_point=args.multi_rollout_k_per_grid_point,
        temperatures=tuple(float(value) for value in args.multi_rollout_temperatures),
        top_ps=tuple(float(value) for value in args.multi_rollout_top_ps),
        cua_max_steps=args.multi_rollout_cua_max_steps,
    )

    orchestrator = _resolve_args_backend(
        model=args.orchestrator_model,
        base_url=args.orchestrator_base_url,
        api_key=args.orchestrator_api_key,
        protocol=DEFAULT_GUI_PROTOCOL,
        env_base_url=env_base_url,
        env_api_key=env_api_key,
        legacy_entry=_lookup_oai_entry(legacy_config_list, args.orchestrator_model),
    )
    coding = _resolve_args_backend(
        model=args.coding_model,
        base_url=args.coding_base_url,
        api_key=args.coding_api_key,
        protocol=DEFAULT_GUI_PROTOCOL,
        env_base_url=env_base_url,
        env_api_key=env_api_key,
        legacy_entry=_lookup_oai_entry(legacy_config_list, args.coding_model),
    )
    gui = _resolve_args_backend(
        model=args.gui_model,
        base_url=args.gui_base_url,
        api_key=args.gui_api_key,
        protocol=args.gui_protocol,
        env_base_url=env_base_url,
        env_api_key=env_gui_api_key,
    )

    return ResolvedRunConfig(
        config_mode=False,
        config_path=None,
        script_mode=args.script_mode,
        provider_name=args.provider_name,
        path_to_vm=args.path_to_vm,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        sleep_after_execution=args.sleep_after_execution,
        region=args.region,
        client_password=args.client_password,
        orchestrator=orchestrator,
        gui=gui,
        coding=coding,
        enable_coding_agent=args.enable_coding_agent,
        enable_web_search=args.enable_web_search,
        mode=args.mode,
        orchestrator_max_steps=args.orchestrator_max_steps,
        coding_max_steps=args.coding_max_steps,
        cua_max_steps=args.cua_max_steps,
        cut_off_steps=args.cut_off_steps,
        domain=args.domain,
        task_id=args.task_id,
        test_all_meta_path=args.test_all_meta_path,
        test_config_base_dir=args.test_config_base_dir,
        result_dir=args.result_dir,
        num_envs=args.num_envs,
        log_level=args.log_level.upper(),
        multi_rollout=multi_rollout,
    )


def _resolve_task_path(test_config_base_dir: str, task_id: str, domain: str) -> tuple[str, str, str]:
    task_arg = Path(task_id)
    if task_arg.is_file():
        cfg_path = task_arg.resolve()
        resolved_task_id = cfg_path.stem
        resolved_domain = domain if domain != "all" else cfg_path.parent.name or "custom"
        return resolved_domain, resolved_task_id, str(cfg_path)

    base_dir = Path(test_config_base_dir)
    candidate_paths: List[Path] = []

    if domain != "all":
        domain_candidate = base_dir / domain / f"{task_id}.json"
        if domain_candidate.is_file():
            candidate_paths.append(domain_candidate)

    if not candidate_paths:
        candidate_paths = sorted(base_dir.glob(f"**/{task_id}.json"))

    if not candidate_paths:
        raise FileNotFoundError(
            f"Could not resolve task '{task_id}'. Pass a valid task id under '{test_config_base_dir}' or a direct path to a task json file."
        )

    if len(candidate_paths) > 1:
        matches = ", ".join(str(path) for path in candidate_paths[:5])
        raise ValueError(
            f"Task id '{task_id}' matched multiple json files. Please disambiguate with --domain or pass a direct path. Matches: {matches}"
        )

    cfg_path = candidate_paths[0].resolve()
    resolved_domain = domain if domain != "all" else cfg_path.parent.name or "custom"
    return resolved_domain, task_id, str(cfg_path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _redact_sensitive(value: Any, key: str = "") -> Any:
    lowered_key = key.lower()
    if any(marker in lowered_key for marker in ("api_key", "password", "secret", "token", "credential")):
        return "<redacted>" if value else value
    if isinstance(value, dict):
        return {str(k): _redact_sensitive(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_sensitive(item, key) for item in value]
    return _json_safe(value)


def _resolve_metadata_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return str(Path(path).expanduser().resolve())


def _write_json_file(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)


def _build_run_metadata(
    run_config: ResolvedRunConfig,
    tasks: List[Tuple[str, str, str]],
    selected_tasks: List[Tuple[str, str, str]],
    test_all_meta: Dict[str, List[str]],
) -> Dict[str, Any]:
    return {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_id": datetime_str,
        "entrypoint": str(Path(__file__).resolve()),
        "args": _redact_sensitive(run_config.as_metadata_args()),
        "mode": "config" if run_config.config_mode else "args",
        "config_path": _resolve_metadata_path(run_config.config_path),
        "models": {
            "orchestrator_model": run_config.orchestrator.model,
            "coding_model": run_config.coding.model,
            "gui_model": run_config.gui.model,
        },
        "backends": _redact_sensitive(
            {
                "orchestrator": run_config.orchestrator.as_llm_config_entry(),
                "coding": run_config.coding.as_llm_config_entry(),
                "gui": run_config.gui.as_openai_client_kwargs() | {"model": run_config.gui.model, "protocol": run_config.gui.protocol},
            }
        ),
        "features": {
            "mode": run_config.mode,
            "script_mode": run_config.script_mode,
            "enable_web_search": run_config.enable_web_search,
            "enable_coding_agent": run_config.enable_coding_agent,
            "gui_protocol": run_config.gui.protocol,
        },
        "multi_rollout": {
            "k_per_grid_point": run_config.multi_rollout.k_per_grid_point,
            "temperatures": list(run_config.multi_rollout.temperatures),
            "top_ps": list(run_config.multi_rollout.top_ps),
            "cua_max_steps": run_config.multi_rollout.cua_max_steps,
        },
        "files": {
            "test_all_meta_path": _resolve_metadata_path(run_config.test_all_meta_path),
            "test_config_base_dir": _resolve_metadata_path(run_config.test_config_base_dir),
            "result_dir": _resolve_metadata_path(run_config.result_dir),
        },
        "tasks": [
            {
                "domain": domain,
                "task_id": ex_id,
                "config_path": _resolve_metadata_path(cfg),
            }
            for domain, ex_id, cfg in tasks
        ],
        "selected_tasks": [
            {
                "domain": domain,
                "task_id": ex_id,
                "config_path": _resolve_metadata_path(cfg),
                "will_process": (domain, ex_id, cfg) in tasks,
            }
            for domain, ex_id, cfg in selected_tasks
        ],
        "selected_task_ids": test_all_meta,
        "environment": {
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL"),
            "OPENAI_CUA_MODEL": os.environ.get("OPENAI_CUA_MODEL"),
            "COACT_ENABLE_WEB_SEARCH": os.environ.get("COACT_ENABLE_WEB_SEARCH"),
            "COACT_ENABLE_DUCKDUCKGO_SEARCH": os.environ.get("COACT_ENABLE_DUCKDUCKGO_SEARCH"),
            "OPENAI_API_KEY_set": bool(os.environ.get("OPENAI_API_KEY")),
            "OPENAI_API_KEY_CUA_set": bool(os.environ.get("OPENAI_API_KEY_CUA")),
        },
    }


def _build_task_metadata(domain: str, ex_id: str, cfg: str, run_config: ResolvedRunConfig) -> Dict[str, Any]:
    return {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_id": datetime_str,
        "domain": domain,
        "task_id": ex_id,
        "task_config_path": _resolve_metadata_path(cfg),
        "settings": _redact_sensitive(run_config.as_metadata_args()),
        "backends": _redact_sensitive(
            {
                "orchestrator": run_config.orchestrator.as_llm_config_entry(),
                "coding": run_config.coding.as_llm_config_entry(),
                "gui": run_config.gui.as_openai_client_kwargs() | {"model": run_config.gui.model, "protocol": run_config.gui.protocol},
            }
        ),
    }


def _resolve_research_tool_flags(run_config: ResolvedRunConfig) -> Tuple[bool, bool]:
    if run_config.mode in {"search-first", "inspect-source-first"}:
        return True, True
    if run_config.enable_web_search:
        return True, True
    return False, False


def _multi_rollout_cua_max_steps(run_config: ResolvedRunConfig) -> int:
    return run_config.multi_rollout.cua_max_steps or run_config.cua_max_steps


def _task_result_path(run_config: ResolvedRunConfig, domain: str, ex_id: str) -> str:
    base_dir = os.path.join(run_config.result_dir, "coact", f"{domain}/{ex_id}")
    if run_config.script_mode == "multi-rollout":
        return os.path.join(base_dir, "results_aggregated.json")
    return os.path.join(base_dir, "result.txt")


def _load_task_config(cfg: str) -> Dict[str, Any]:
    with open(cfg, encoding="utf-8") as f:
        return json.load(f)


def _sanitize_chat_history(chat_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chat_history = []
    for item in chat_messages:
        item = dict(item)
        item.pop("tool_responses", None)
        if item.get("role") in ["tool", "assistant"] and item.get("content"):
            for msg in item["content"]:
                if msg.get("type") == "image_url":
                    msg["image_url"] = "<image>"
        chat_history.append(item)
    return chat_history


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def _extract_last_assistant_text(chat_history: List[Dict[str, Any]]) -> str:
    for item in reversed(chat_history):
        if item.get("role") != "assistant":
            continue
        text = _extract_text_from_content(item.get("content"))
        if text:
            return text
    return ""


def _extract_planning_result(chat_history: List[Dict[str, Any]]) -> str:
    latest_plan = ""
    for item in chat_history:
        if item.get("name") != "orchestrator":
            continue
        text = _extract_text_from_content(item.get("content"))
        if not text:
            continue
        normalized = text.strip()
        if normalized == "INFEASIBLE":
            return "INFEASIBLE"
        if normalized == "TERMINATE":
            continue
        latest_plan = normalized
    return latest_plan


def _build_planning_message(task_instruction: str) -> str:
    return (
        f"{task_instruction}\n"
        "Check my computer screenshot and describe it first. Then produce a detailed step-by-step execution plan only. "
        "Do not call the GUI Operator or Programmer. Do not execute the task. "
        'If the task is impossible, reply with "INFEASIBLE".\n'
        "I will not provide further information to you."
    )


def _create_desktop_env(run_config: ResolvedRunConfig) -> DesktopEnv:
    screen_size = (run_config.screen_width, run_config.screen_height)
    snapshot_name = "init_state"
    if run_config.provider_name == "aws":
        from desktop_env.providers.aws.manager import IMAGE_ID_MAP

        snapshot_name = IMAGE_ID_MAP[run_config.region].get(screen_size, IMAGE_ID_MAP[run_config.region][(1920, 1080)])

    return DesktopEnv(
        path_to_vm=run_config.path_to_vm,
        action_space="pyautogui",
        provider_name=run_config.provider_name,
        os_type="Ubuntu",
        region=run_config.region,
        snapshot_name=snapshot_name,
        screen_size=screen_size,
        headless=True,
        require_a11y_tree=False,
        enable_proxy=True,
        client_password=run_config.client_password,
    )


def _rollout_result_category(result: str) -> str:
    if "IDK" in result:
        return "IDK"
    if "TERMINATE" in result:
        return "TERMINATE"
    return "OTHER"


def _canonicalize_action(action: Any) -> str:
    if isinstance(action, dict):
        return json.dumps(action, sort_keys=True)
    if isinstance(action, list):
        return json.dumps(action, sort_keys=True)
    return str(action)


def _trajectory_signature(tool_events: List[Dict[str, Any]]) -> str:
    actions: List[str] = []
    for event in tool_events:
        if event.get("type") == "computer_call":
            actions.extend(_canonicalize_action(action) for action in event.get("actions", []))
        elif event.get("type") == "vllm_action":
            actions.append(_canonicalize_action(event.get("action")))
    return "\n".join(actions)


def _build_standard_process_context(
    task_info: Tuple[str, str, str],
    run_config: ResolvedRunConfig,
) -> Tuple[str, str, str, str, Dict[str, Any], str, LLMConfig, LLMConfig, bool, bool]:
    domain, ex_id, cfg = task_info
    if run_config.gui.protocol == DEFAULT_GUI_PROTOCOL:
        validate_cua_model(run_config.gui.model)
    orchestrator_llm_config = LLMConfig(config_list=[run_config.orchestrator.as_llm_config_entry()])
    coding_llm_config = LLMConfig(config_list=[run_config.coding.as_llm_config_entry()])
    enable_web_search_tool, enable_read_webpage_tool = _resolve_research_tool_flags(run_config)
    history_save_dir = os.path.join(run_config.result_dir, "coact", f"{domain}/{ex_id}")
    os.makedirs(history_save_dir, exist_ok=True)
    task_metadata = _build_task_metadata(domain, ex_id, cfg, run_config)
    _write_json_file(os.path.join(history_save_dir, "metadata.json"), task_metadata)
    task_config = _load_task_config(cfg)
    task_instruction = _build_task_instruction(task_config, run_config.mode)
    return (
        domain,
        ex_id,
        cfg,
        history_save_dir,
        task_config,
        task_instruction,
        orchestrator_llm_config,
        coding_llm_config,
        enable_web_search_tool,
        enable_read_webpage_tool,
    )


def process_task(task_info: Tuple[str, str, str], run_config: ResolvedRunConfig):
    (
        domain,
        ex_id,
        cfg,
        history_save_dir,
        task_config,
        task_instruction,
        orchestrator_llm_config,
        coding_llm_config,
        enable_web_search_tool,
        enable_read_webpage_tool,
    ) = _build_standard_process_context(task_info, run_config)
    task_metadata = _build_task_metadata(domain, ex_id, cfg, run_config)
    retry = 0
    orchestrator_proxy = None

    while True:
        try:
            with orchestrator_llm_config:
                orchestrator = OrchestratorAgent(
                    name="orchestrator",
                    system_message=_build_task_description(
                        enable_coding_agent=run_config.enable_coding_agent,
                        enable_gui_agent=True,
                        enable_web_search_tool=enable_web_search_tool,
                        enable_read_webpage_tool=enable_read_webpage_tool,
                    ),
                    enable_coding_agent=run_config.enable_coding_agent,
                    enable_gui_agent_tool=True,
                    enable_web_search_tool=enable_web_search_tool,
                    enable_read_webpage_tool=enable_read_webpage_tool,
                )
                orchestrator_proxy = OrchestratorUserProxyAgent(
                    name="orchestrator_proxy",
                    is_termination_msg=lambda x: x.get("content", "") and (x.get("content", "")[0]["text"].lower() == "terminate" or x.get("content", "")[0]["text"].lower() == "infeasible"),
                    human_input_mode="NEVER",
                    provider_name=run_config.provider_name,
                    path_to_vm=run_config.path_to_vm,
                    screen_width=run_config.screen_width,
                    screen_height=run_config.screen_height,
                    sleep_after_execution=run_config.sleep_after_execution,
                    code_execution_config=False,
                    history_save_dir=history_save_dir,
                    llm_model=run_config.coding.model,
                    gui_model=run_config.gui.model,
                    orchestrator_model=run_config.orchestrator.model,
                    truncate_history_inputs=run_config.cua_max_steps + 1,
                    cua_max_steps=run_config.cua_max_steps,
                    coding_max_steps=run_config.coding_max_steps,
                    enable_web_search_tool=enable_web_search_tool,
                    enable_read_webpage_tool=enable_read_webpage_tool,
                    enable_coding_agent=run_config.enable_coding_agent,
                    region=run_config.region,
                    client_password=run_config.client_password,
                    user_instruction=task_instruction,
                    prompt_mode=run_config.mode,
                    task_source=str(task_config.get("source", "")).strip() or None,
                    orchestrator_client_kwargs=run_config.orchestrator.as_openai_client_kwargs(),
                    orchestrator_backend_config=run_config.orchestrator.as_llm_config_entry(),
                    gui_client_kwargs=run_config.gui.as_openai_client_kwargs(),
                    gui_protocol=run_config.gui.protocol,
                    coding_llm_config=coding_llm_config,
                )

            orchestrator_proxy.reset(task_config=task_config)
            time.sleep(60)
            screenshot = orchestrator_proxy.env.controller.get_screenshot()

            with open(os.path.join(history_save_dir, "initial_screenshot_orchestrator.png"), "wb") as f:
                f.write(screenshot)

            orchestrator_proxy.initiate_chat(
                recipient=orchestrator,
                message=_build_initial_message(task_instruction) + "<img data:image/png;base64," + base64.b64encode(screenshot).decode("utf-8") + ">",
                max_turns=run_config.orchestrator_max_steps,
            )

            key = list(orchestrator_proxy.chat_messages.keys())[0]
            chat_history = _sanitize_chat_history(orchestrator_proxy.chat_messages[key])

            with open(os.path.join(history_save_dir, "chat_history.json"), "w") as f:
                json.dump(chat_history, f)

            if chat_history[-1]["role"] == "user" and "INFEASIBLE" in chat_history[-1]["content"][0]["text"]:
                orchestrator_proxy.env.action_history.append("FAIL")

            cua_steps = len(glob.glob(f"{history_save_dir}/cua_output*/step_*.png"))
            coding_paths = glob.glob(f"{history_save_dir}/coding_output*/chat_history.json")
            coding_steps = 0
            for hist in coding_paths:
                with open(hist, "r", encoding="utf-8") as f:
                    hist = json.dumps(json.load(f))
                    coding_steps += hist.count("exitcode:")
            if cua_steps + coding_steps > run_config.cut_off_steps:
                score = 0.0
            else:
                score = orchestrator_proxy.env.evaluate()
            print(f"Score: {score}")

            with open(os.path.join(history_save_dir, "result.txt"), "w", encoding="utf-8") as f:
                f.write(str(score))
            break

        except Exception as e:
            retry += 1
            if retry < 3:
                shutil.rmtree(history_save_dir)
                os.makedirs(history_save_dir)
                _write_json_file(os.path.join(history_save_dir, "metadata.json"), task_metadata)
                print(f"Retry {retry} times, error: {str(e)}")
                traceback.print_exc()
                continue

            print(f"Error processing task {domain}/{ex_id}")
            traceback.print_exc()
            score = 0.0
            with open(os.path.join(history_save_dir, "result.txt"), "w", encoding="utf-8") as f:
                f.write(str(score))
            with open(os.path.join(history_save_dir, "err_reason.txt"), "w", encoding="utf-8") as f:
                f.write(f"Fatal error: {str(e)}")
        finally:
            if orchestrator_proxy is not None and orchestrator_proxy.env is not None:
                orchestrator_proxy.env.close()

    return domain, score


def _plan_task_once(
    *,
    run_config: ResolvedRunConfig,
    task_config: Dict[str, Any],
    task_instruction: str,
    history_save_dir: str,
    orchestrator_llm_config: LLMConfig,
    coding_llm_config: LLMConfig,
    enable_web_search_tool: bool,
    enable_read_webpage_tool: bool,
) -> str:
    planning_dir = os.path.join(history_save_dir, "planning")
    os.makedirs(planning_dir, exist_ok=True)
    orchestrator_proxy = None

    try:
        with orchestrator_llm_config:
            orchestrator = OrchestratorAgent(
                name="orchestrator",
                system_message=_build_task_description(
                    enable_coding_agent=run_config.enable_coding_agent,
                    enable_gui_agent=False,
                    enable_web_search_tool=enable_web_search_tool,
                    enable_read_webpage_tool=enable_read_webpage_tool,
                ),
                enable_coding_agent=run_config.enable_coding_agent,
                enable_gui_agent_tool=False,
                enable_web_search_tool=enable_web_search_tool,
                enable_read_webpage_tool=enable_read_webpage_tool,
            )
            orchestrator_proxy = OrchestratorUserProxyAgent(
                name="orchestrator_proxy",
                is_termination_msg=lambda x: x.get("content", "") and (x.get("content", "")[0]["text"].lower() == "terminate" or x.get("content", "")[0]["text"].lower() == "infeasible"),
                human_input_mode="NEVER",
                provider_name=run_config.provider_name,
                path_to_vm=run_config.path_to_vm,
                screen_width=run_config.screen_width,
                screen_height=run_config.screen_height,
                sleep_after_execution=run_config.sleep_after_execution,
                code_execution_config=False,
                history_save_dir=planning_dir,
                llm_model=run_config.coding.model,
                gui_model=run_config.gui.model,
                orchestrator_model=run_config.orchestrator.model,
                truncate_history_inputs=run_config.cua_max_steps + 1,
                cua_max_steps=run_config.cua_max_steps,
                coding_max_steps=run_config.coding_max_steps,
                enable_web_search_tool=enable_web_search_tool,
                enable_read_webpage_tool=enable_read_webpage_tool,
                enable_coding_agent=run_config.enable_coding_agent,
                region=run_config.region,
                client_password=run_config.client_password,
                user_instruction=task_instruction,
                prompt_mode=run_config.mode,
                task_source=str(task_config.get("source", "")).strip() or None,
                orchestrator_client_kwargs=run_config.orchestrator.as_openai_client_kwargs(),
                orchestrator_backend_config=run_config.orchestrator.as_llm_config_entry(),
                gui_client_kwargs=run_config.gui.as_openai_client_kwargs(),
                gui_protocol=run_config.gui.protocol,
                coding_llm_config=coding_llm_config,
            )

        orchestrator_proxy.reset(task_config=task_config)
        time.sleep(60)
        screenshot = orchestrator_proxy.env.controller.get_screenshot()

        with open(os.path.join(planning_dir, "initial_screenshot_orchestrator.png"), "wb") as f:
            f.write(screenshot)

        orchestrator_proxy.initiate_chat(
            recipient=orchestrator,
            message=_build_planning_message(task_instruction) + "<img data:image/png;base64," + base64.b64encode(screenshot).decode("utf-8") + ">",
            max_turns=run_config.orchestrator_max_steps,
        )

        key = list(orchestrator_proxy.chat_messages.keys())[0]
        chat_history = _sanitize_chat_history(orchestrator_proxy.chat_messages[key])
        _write_json_file(os.path.join(planning_dir, "chat_history.json"), {"messages": chat_history})

        plan_text = _extract_planning_result(chat_history)
        if not plan_text:
            raise RuntimeError("Planning phase did not produce a plan.")

        with open(os.path.join(planning_dir, "plan.txt"), "w", encoding="utf-8") as f:
            f.write(plan_text)
        return plan_text
    finally:
        if orchestrator_proxy is not None and orchestrator_proxy.env is not None:
            orchestrator_proxy.env.close()


def _run_single_multi_rollout_attempt(
    *,
    run_config: ResolvedRunConfig,
    task_config: Dict[str, Any],
    plan_text: str,
    domain: str,
    ex_id: str,
    attempt_dir: str,
    rollout_index: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    os.makedirs(attempt_dir, exist_ok=True)
    env = None
    started_at = time.time()
    score = 0.0
    result = "IDK Rollout failed before the GUI agent started."
    step_count = 0
    tool_events: List[Dict[str, Any]] = []

    instruction = (
        f"{_build_task_instruction(task_config, run_config.mode)}\n\n"
        "# Execution plan\n"
        f"{plan_text.strip()}\n\n"
        "Follow the execution plan exactly. Do not ask the orchestrator for more help."
    ).strip()

    try:
        env = _create_desktop_env(run_config)
        env.reset(task_config=task_config)
        history_inputs, result, cost, raw_transcript, tool_events, step_count = run_cua(
            env=env,
            instruction=instruction,
            max_steps=_multi_rollout_cua_max_steps(run_config),
            save_path=attempt_dir,
            cua_model=run_config.gui.model,
            gui_protocol=run_config.gui.protocol,
            base_url=run_config.gui.base_url,
            api_key=run_config.gui.api_key,
            screen_width=run_config.screen_width,
            screen_height=run_config.screen_height,
            sleep_after_execution=run_config.sleep_after_execution,
            truncate_history_inputs=_multi_rollout_cua_max_steps(run_config) + 1,
            client_password=run_config.client_password,
            prompt_mode=run_config.mode,
            task_source=str(task_config.get("source", "")).strip() or None,
            temperature=temperature,
            top_p=top_p,
        )

        _write_json_file(os.path.join(attempt_dir, "history_inputs.json"), {"messages": history_inputs})
        _write_json_file(os.path.join(attempt_dir, "responses.json"), {"responses": raw_transcript})
        _write_json_file(os.path.join(attempt_dir, "tool_events.json"), {"events": tool_events})
        with open(os.path.join(attempt_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(result)
        with open(os.path.join(attempt_dir, "cost.txt"), "w", encoding="utf-8") as f:
            f.write(str(cost))

        if step_count <= run_config.cut_off_steps:
            score = float(env.evaluate())
    except Exception as exc:
        with open(os.path.join(attempt_dir, "err_reason.txt"), "w", encoding="utf-8") as f:
            f.write(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    finally:
        if env is not None:
            env.close()

    elapsed_time_seconds = time.time() - started_at
    attempt_record = {
        "domain": domain,
        "task_id": ex_id,
        "rollout_index": rollout_index,
        "temperature": temperature,
        "top_p": top_p,
        "attempt_dir": _resolve_metadata_path(attempt_dir),
        "binary_result": int(score > 0),
        "score": score,
        "reported_result_category": _rollout_result_category(result),
        "reported_result": result,
        "steps_used": step_count,
        "elapsed_time_seconds": elapsed_time_seconds,
        "trajectory_signature": _trajectory_signature(tool_events),
    }
    _write_json_file(os.path.join(attempt_dir, "metadata.json"), attempt_record)
    return attempt_record


def process_task_multi_rollout(task_info: Tuple[str, str, str], run_config: ResolvedRunConfig):
    (
        domain,
        ex_id,
        cfg,
        history_save_dir,
        task_config,
        task_instruction,
        orchestrator_llm_config,
        coding_llm_config,
        enable_web_search_tool,
        enable_read_webpage_tool,
    ) = _build_standard_process_context(task_info, run_config)
    task_metadata = _build_task_metadata(domain, ex_id, cfg, run_config)

    retry = 0
    while True:
        try:
            plan_text = _plan_task_once(
                run_config=run_config,
                task_config=task_config,
                task_instruction=task_instruction,
                history_save_dir=history_save_dir,
                orchestrator_llm_config=orchestrator_llm_config,
                coding_llm_config=coding_llm_config,
                enable_web_search_tool=enable_web_search_tool,
                enable_read_webpage_tool=enable_read_webpage_tool,
            )
            if plan_text == "INFEASIBLE":
                aggregated = {
                    "domain": domain,
                    "task_id": ex_id,
                    "pass_at_1": False,
                    "pass_at_k": False,
                    "unique_trajectory_count": 0,
                    "total_attempts": 0,
                    "plan_result": "INFEASIBLE",
                }
                _write_json_file(os.path.join(history_save_dir, "results_aggregated.json"), aggregated)
                _write_json_file(os.path.join(history_save_dir, "results_per_configuration.json"), {"configurations": []})
                return domain, 0.0

            attempts: List[Dict[str, Any]] = []
            per_configuration: List[Dict[str, Any]] = []
            rollout_counter = 0

            for temperature in run_config.multi_rollout.temperatures:
                for top_p in run_config.multi_rollout.top_ps:
                    config_attempts: List[Dict[str, Any]] = []
                    config_dir = os.path.join(history_save_dir, "multi_rollout", f"temp_{temperature}_top_p_{top_p}")
                    for rollout_index in range(run_config.multi_rollout.k_per_grid_point):
                        attempt_dir = os.path.join(config_dir, f"rollout_{rollout_index:03d}")
                        attempt_record = _run_single_multi_rollout_attempt(
                            run_config=run_config,
                            task_config=task_config,
                            plan_text=plan_text,
                            domain=domain,
                            ex_id=ex_id,
                            attempt_dir=attempt_dir,
                            rollout_index=rollout_counter,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        rollout_counter += 1
                        config_attempts.append(attempt_record)
                        attempts.append(attempt_record)

                    per_configuration.append(
                        {
                            "temperature": temperature,
                            "top_p": top_p,
                            "attempt_count": len(config_attempts),
                            "pass_at_1": bool(config_attempts and config_attempts[0]["binary_result"]),
                            "pass_at_k": any(attempt["binary_result"] for attempt in config_attempts),
                            "unique_trajectory_count": len({attempt["trajectory_signature"] for attempt in config_attempts}),
                            "attempts": config_attempts,
                        }
                    )

            aggregated = {
                "domain": domain,
                "task_id": ex_id,
                "plan_path": _resolve_metadata_path(os.path.join(history_save_dir, "planning", "plan.txt")),
                "script_mode": run_config.script_mode,
                "k_per_grid_point": run_config.multi_rollout.k_per_grid_point,
                "temperatures": list(run_config.multi_rollout.temperatures),
                "top_ps": list(run_config.multi_rollout.top_ps),
                "total_attempts": len(attempts),
                "pass_at_1": bool(attempts and attempts[0]["binary_result"]),
                "pass_at_k": any(attempt["binary_result"] for attempt in attempts),
                "unique_trajectory_count": len({attempt["trajectory_signature"] for attempt in attempts}),
                "successful_attempts": sum(attempt["binary_result"] for attempt in attempts),
            }
            per_configuration_payload = {
                "domain": domain,
                "task_id": ex_id,
                "script_mode": run_config.script_mode,
                "configurations": per_configuration,
            }

            _write_json_file(os.path.join(history_save_dir, "results_aggregated.json"), aggregated)
            _write_json_file(os.path.join(history_save_dir, "results_per_configuration.json"), per_configuration_payload)
            _write_json_file(os.path.join(history_save_dir, "metadata.json"), task_metadata | {"plan_path": aggregated["plan_path"]})
            return domain, float(aggregated["pass_at_k"])
        except Exception as e:
            retry += 1
            if retry < 3:
                shutil.rmtree(history_save_dir)
                os.makedirs(history_save_dir)
                _write_json_file(os.path.join(history_save_dir, "metadata.json"), task_metadata)
                print(f"Retry {retry} times, error: {str(e)}")
                traceback.print_exc()
                continue

            print(f"Error processing task {domain}/{ex_id}")
            traceback.print_exc()
            _write_json_file(
                os.path.join(history_save_dir, "results_aggregated.json"),
                {
                    "domain": domain,
                    "task_id": ex_id,
                    "pass_at_1": False,
                    "pass_at_k": False,
                    "unique_trajectory_count": 0,
                    "total_attempts": 0,
                    "error": str(e),
                },
            )
            with open(os.path.join(history_save_dir, "err_reason.txt"), "w", encoding="utf-8") as f:
                f.write(f"Fatal error: {str(e)}")
            return domain, 0.0


def _read_existing_task_score(result_path: str, script_mode: str) -> float:
    if script_mode == "multi-rollout":
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return float(bool(payload.get("pass_at_k")))
    with open(result_path, "r", encoding="utf-8") as f:
        return float(f.read())


if __name__ == "__main__":
    parser = _build_parser()
    raw_argv = sys.argv[1:]
    args = parser.parse_args(raw_argv)
    _reject_mixed_config_usage(parser, raw_argv, args)
    run_config = load_config_file(args.config) if args.config else _resolve_run_config_from_args(args)
    validate_cua_model(run_config.gui.model)
    _setup_logging(run_config.log_level)

    tasks: List[Tuple[str, str, str]] = []
    selected_tasks: List[Tuple[str, str, str]] = []
    scores: Dict[str, List[float]] = {}
    if run_config.task_id:
        domain, ex_id, cfg = _resolve_task_path(run_config.test_config_base_dir, run_config.task_id, run_config.domain)
        selected_tasks.append((domain, ex_id, cfg))
        scores[domain] = []
        result_path = _task_result_path(run_config, domain, ex_id)
        if os.path.exists(result_path):
            if run_config.script_mode == "multi-rollout":
                with open(result_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
            else:
                with open(result_path, "r", encoding="utf-8") as f:
                    result = f.read()
            print(f"Results already exist in {domain}/{ex_id}, result: {result}")
        else:
            tasks.append((domain, ex_id, cfg))
        test_all_meta = {domain: [ex_id]}
    else:
        with open(run_config.test_all_meta_path, encoding="utf-8") as f:
            test_all_meta = json.load(f)
        if run_config.domain != "all":
            test_all_meta = {run_config.domain: test_all_meta[run_config.domain]}

        for domain in test_all_meta:
            scores[domain] = []
            for ex_id in test_all_meta[domain]:
                cfg = os.path.join(run_config.test_config_base_dir, f"{domain}/{ex_id}.json")
                selected_tasks.append((domain, ex_id, cfg))
                score_path = _task_result_path(run_config, domain, ex_id)
                if os.path.exists(score_path):
                    if run_config.script_mode == "multi-rollout":
                        with open(score_path, "r", encoding="utf-8") as f:
                            result = json.load(f)
                    else:
                        with open(score_path, "r", encoding="utf-8") as f:
                            result = f.read()
                    print(f"Results already exist in {domain}/{ex_id}, result: {result}")
                    continue
                tasks.append((domain, ex_id, cfg))

    run_metadata = _build_run_metadata(run_config, tasks, selected_tasks, test_all_meta)
    metadata_dir = Path(run_config.result_dir) / "coact"
    _write_json_file(metadata_dir / f"run_metadata_{datetime_str}.json", run_metadata)
    _write_json_file(metadata_dir / "run_metadata_latest.json", run_metadata)

    if not tasks:
        print("No tasks to process. All tasks have already been completed.")
        print("\n=== Summary of Existing Results ===")
        for domain in test_all_meta:
            domain_scores = []
            for ex_id in test_all_meta[domain]:
                score_file = _task_result_path(run_config, domain, ex_id)
                if os.path.exists(score_file):
                    domain_scores.append(_read_existing_task_score(score_file, run_config.script_mode))
            if domain_scores:
                avg_score = sum(domain_scores) / len(domain_scores)
                print(f"{domain}: {len(domain_scores)} tasks, average score: {avg_score:.2f}")
    else:
        num_workers = max(1, min(cpu_count() // 2, run_config.num_envs))
        print(f"Processing {len(tasks)} tasks with {num_workers} workers...")
        task_processor = process_task_multi_rollout if run_config.script_mode == "multi-rollout" else process_task
        process_func = partial(task_processor, run_config=run_config)
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_func, tasks)

        for domain, score in results:
            scores[domain].append(score)

        print("\n=== Task Processing Complete ===")
        for domain in scores:
            if scores[domain]:
                avg_score = sum(scores[domain]) / len(scores[domain])
                print(f"{domain}: {len(scores[domain])} tasks, average score: {avg_score:.2f}")
