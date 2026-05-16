import argparse
import base64
import glob
import datetime
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time
import os
from mm_agents.coact.operator_agent import OrchestratorAgent, OrchestratorUserProxyAgent
from mm_agents.coact.cua_agent import DEFAULT_CUA_MODEL, validate_cua_model
from mm_agents.coact.autogen import LLMConfig
from mm_agents.env_loader import load_mm_agents_env
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import sys

load_mm_agents_env()

DEFAULT_OAI_CONFIG_PATH = str((Path(__file__).resolve().parent / "mm_agents" / "coact" / "OAI_CONFIG_LIST").resolve())
DEFAULT_MODE = "default"
SUPPORTED_MODES = (DEFAULT_MODE, "search-first", "inspect-source-first")


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
    - When you delegate to a helper agent, copy every required detail into the delegated task so the helper does not need to reconstruct the plan from memory.
4. Verify the result and see if it fulfills the user's requirement.

# Your helpers
You can use the following tools to solve the task. You can call the web search tool for research, and you can only call one helper agent per reply:

## Web Search
Use web_search to gather reliable public instructions before you plan or declare a task infeasible.
Prefer official documentation and strong Stack Exchange answers when relevant.
If the task includes a source URL or a research policy, you must follow it instead of relying on memory.
Summarize research faithfully and preserve all actionable details needed for execution and verification.

## GUI Operator
Let a GUI agent to solve a subtask you assigned. 
GUI agent can operate the computer by clicking and typing (but not accurate). 
When web search is enabled, the GUI agent can also look up app documentation or help pages before taking GUI actions.
Require a detailed task description.
When you call GUI agent, it will only have a **20-step** budget to complete your task. Each step is a one-time interaction with OS like mouse click or keyboard typing. Please take this into account when you plan the actions.
"""

CODING_AGENT_SECTION = """

## Programmer
Let a programmer to solve a subtask you assigned. 
The Programmer can write python or bash code to modify almost everything in the computer, like files, apps, system settings, etc. 
It requires a environment description and a detailed task description. As detailed as possible.
Can use any python package you instructed.
Will return a summary with the output of the code.
When letting coding agent to modify the spreadsheet, after the task completed, you MUST make sure EVERY modified value in the spreadsheet is in the desired position (e.g., filled in the expected cell) by a GUI Operator.
After that, if anything is wrong, tell the programmer to modify it.
"""


def _build_task_description(enable_coding_agent: bool, prompt_mode: str = DEFAULT_MODE) -> str:
    task_description = BASE_TASK_DESCRIPTION
    if prompt_mode == "inspect-source-first":
        task_description = task_description.replace(
            "You can use the following tools to solve the task. You can call the web search tool for research, and you can only call one helper agent per reply:",
            "You can use the following tools to solve the task. In source-guided mode you should inspect the provided page with read_webpage before delegating, and you can only call one helper agent per reply:",
        )
        task_description = task_description.replace(
            "## Web Search\nUse web_search to gather reliable public instructions before you plan or declare a task infeasible.\nPrefer official documentation and strong Stack Exchange answers when relevant.\nIf the task includes a source URL or a research policy, you must follow it instead of relying on memory.\nSummarize research faithfully and preserve all actionable details needed for execution and verification.\n\n",
            "## Source Page Inspector\nUse read_webpage to inspect the provided source URL directly before you plan or declare a task infeasible.\nTreat the source page as primary evidence and preserve all actionable details needed for execution and verification.\nOnly rely on helper-agent web search if the source page is missing information required to complete or verify the task.\n\n",
        )
    if enable_coding_agent:
        task_description = task_description.replace(
            "4. Verify the result and see if it fulfills the user's requirement.",
            "    - You MUST try the Coding Agent first for file operation tasks like spreadsheet modification.\n"
            "4. Verify the result and see if it fulfills the user's requirement.",
        )
        task_description = task_description.replace(
            "## GUI Operator",
            f"{CODING_AGENT_SECTION}\n## GUI Operator",
        )
        task_description += (
            "\nIf you let GUI Operator to check the result, you MUST let it close and reopen the file "
            "because programmer's result will NOT be updated to the screen.\n"
        )
    return task_description


def _build_task_instruction(task_config: Dict[str, object], mode: str) -> str:
    instruction = str(task_config["instruction"]).strip()
    source = str(task_config.get("source", "")).strip()

    if mode == "search-first":
        return (
            f"{instruction}\n\n"
            "# Research policy\n"
            "- Before planning or taking GUI actions, you must first call the custom web_search tool to look for relevant instructions on the internet.\n"
            "- Prioritize official documentation for the application involved in the task.\n"
            "- Also look for strong Stack Exchange family answers such as Super User, Stack Overflow, or Ask Ubuntu when they are relevant.\n"
            "- Do not declare the task infeasible until you have used the web_search tool and reviewed the returned evidence.\n"
            "- Preserve all actionable details from the research, including settings, values, value types, restart requirements, verification requirements, and related follow-up changes.\n"
            "- If the research contains multiple actionable details, decide explicitly which are required and which are optional, and keep every required detail in the execution plan.\n"
            "- When delegating, restate every required detail directly in the delegated task instead of relying on a shortened summary.\n"
            "- Only after collecting relevant instructions should you form a plan and execute the task.\n"
            "- Do not rely on any hidden task source URL in this mode. Work only from your own search results and the observed UI."
        )

    if mode == "inspect-source-first":
        if not source:
            raise ValueError("Mode 'inspect-source-first' requires the task JSON to include a non-empty 'source' URL.")
        return (
            f"{instruction}\n\n"
            "# Source-guided policy\n"
            f"Source URL: {source}\n"
            "- Before planning or taking GUI actions, you must first call the custom read_webpage tool on the provided source URL.\n"
            "- Use the source page as the primary guidance for the task.\n"
            "- Do not declare the task infeasible until you have inspected the source page and reviewed the returned evidence.\n"
            "- Preserve all actionable details from the source-guided lookup, including settings, values, value types, restart requirements, verification requirements, and related follow-up changes.\n"
            "- If the source-guided lookup contains multiple actionable details, decide explicitly which are required and which are optional, and keep every required detail in the execution plan.\n"
            "- When delegating, restate every required detail directly in the delegated task instead of relying on a shortened summary.\n"
            "- Form your plan according to the information on the source page, then execute it in the UI.\n"
            "- If the source page does not contain enough information to complete the task or verify a missing step, delegate the missing research to a helper that has web-search capability."
        )

    return instruction


def _build_initial_message(task_instruction: str) -> str:
    return (
        f"{task_instruction}\n"
        "Check my computer screenshot and describe it first. If this task is possible to complete, please complete it on my computer. "
        'If not, reply with "INFEASIBLE" to end the conversation.\n'
        "I will not provide further information to you."
    )


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--provider_name", type=str, default="aws")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.5)
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--client_password", type=str, default="osworld-public-evaluation")

    # agent config
    parser.add_argument("--oai_config_path", type=str, default=DEFAULT_OAI_CONFIG_PATH)
    parser.add_argument("--orchestrator_model", type=str, default="o3")
    parser.add_argument("--coding_model", type=str, default="o4-mini")
    parser.add_argument("--cua_model", type=str, default=os.environ.get("OPENAI_CUA_MODEL", DEFAULT_CUA_MODEL))
    parser.add_argument(
        "--enable_coding_agent",
        action="store_true",
        default=False,
        help="Expose the coding agent tool to the CoAct orchestrator.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=SUPPORTED_MODES,
        default=DEFAULT_MODE,
        help="Prompt strategy for the GUI search path.",
    )
    parser.add_argument(
        "--enable_web_search",
        "--enable_duckduckgo_search",
        dest="enable_web_search",
        action="store_true",
        default=(
            os.environ.get("COACT_ENABLE_WEB_SEARCH", "").lower() in {"1", "true", "yes", "on"}
            or os.environ.get("COACT_ENABLE_DUCKDUCKGO_SEARCH", "").lower() in {"1", "true", "yes", "on"}
        ),
        help="Enable hosted OpenAI web search in the GUI agent path",
    )
    parser.add_argument("--orchestrator_max_steps", type=int, default=15)
    parser.add_argument("--coding_max_steps", type=int, default=20)
    parser.add_argument("--cua_max_steps", type=int, default=25)
    parser.add_argument("--cut_off_steps", type=int, default=200)

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--task_id",
        type=str,
        default=None,
        help="Run exactly one task. Accepts either a bare task id or a direct path to a task json file.",
    )
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_examples/test_all.json"
    )
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples/examples"
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results_coact")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel")
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help="Set the logging level")

    args = parser.parse_args()
    return args

args = config()

logger = logging.getLogger()

log_level = getattr(logging, args.log_level.upper())
logger.setLevel(log_level)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(log_level)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
#  }}} Logger Configs #

logger = logging.getLogger("desktopenv.expeiment")


def _load_oai_config_list(config_path: str) -> List[Dict[str, object]]:
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"OAI config file not found: {config_path}")

    with open(config_file, encoding="utf-8") as f:
        config_list = json.load(f)

    if not isinstance(config_list, list):
        raise ValueError(f"OAI config file must contain a JSON list: {config_path}")

    env_api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY_CUA")
    resolved_config_list: List[Dict[str, object]] = []
    for entry in config_list:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid OAI config entry in {config_path}: expected object, got {type(entry).__name__}")
        resolved_entry = dict(entry)
        if resolved_entry.get("api_key") == "KEY":
            if not env_api_key:
                raise ValueError(
                    "The selected OAI config uses placeholder `api_key` values. "
                    "Set `OPENAI_API_KEY`/`OPENAI_API_KEY_CUA` or pass a populated `--oai_config_path`."
                )
            resolved_entry["api_key"] = env_api_key
        resolved_config_list.append(resolved_entry)
    return resolved_config_list


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
            f"Could not resolve task '{task_id}'. Pass a valid task id under '{test_config_base_dir}' "
            f"or a direct path to a task json file."
        )

    if len(candidate_paths) > 1:
        matches = ", ".join(str(path) for path in candidate_paths[:5])
        raise ValueError(
            f"Task id '{task_id}' matched multiple json files. "
            f"Please disambiguate with --domain or pass a direct path. Matches: {matches}"
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
    args: argparse.Namespace,
    tasks: List[tuple[str, str, str]],
    selected_tasks: List[tuple[str, str, str]],
    test_all_meta: Dict[str, List[str]],
    oai_config_list: List[Dict[str, object]],
) -> Dict[str, Any]:
    return {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_id": datetime_str,
        "entrypoint": str(Path(__file__).resolve()),
        "args": _redact_sensitive(vars(args)),
        "models": {
            "orchestrator_model": args.orchestrator_model,
            "coding_model": args.coding_model,
            "cua_model": args.cua_model,
        },
        "features": {
            "mode": args.mode,
            "enable_web_search": args.enable_web_search,
            "enable_coding_agent": args.enable_coding_agent,
        },
        "files": {
            "oai_config_path": _resolve_metadata_path(args.oai_config_path),
            "test_all_meta_path": _resolve_metadata_path(args.test_all_meta_path),
            "test_config_base_dir": _resolve_metadata_path(args.test_config_base_dir),
            "result_dir": _resolve_metadata_path(args.result_dir),
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
        "oai_config_list": _redact_sensitive(oai_config_list),
        "environment": {
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL"),
            "OPENAI_CUA_MODEL": os.environ.get("OPENAI_CUA_MODEL"),
            "COACT_ENABLE_WEB_SEARCH": os.environ.get("COACT_ENABLE_WEB_SEARCH"),
            "COACT_ENABLE_DUCKDUCKGO_SEARCH": os.environ.get("COACT_ENABLE_DUCKDUCKGO_SEARCH"),
            "OPENAI_API_KEY_set": bool(os.environ.get("OPENAI_API_KEY")),
            "OPENAI_API_KEY_CUA_set": bool(os.environ.get("OPENAI_API_KEY_CUA")),
        },
    }


def _build_task_metadata(
    domain: str,
    ex_id: str,
    cfg: str,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_id": datetime_str,
        "domain": domain,
        "task_id": ex_id,
        "task_config_path": _resolve_metadata_path(cfg),
        "settings": _redact_sensitive(settings),
    }


def process_task(task_info, 
                provider_name,
                path_to_vm,
                orchestrator_model="o3",
                coding_model='o4-mini',
                cua_model=DEFAULT_CUA_MODEL,
                mode=DEFAULT_MODE,
                enable_web_search=False,
                enable_coding_agent=False,
                save_dir='results',
                orchestrator_max_steps=15,
                cua_max_steps=25,
                coding_max_steps=20,
                cut_off_steps=150,
                screen_width=1920,
                screen_height=1080,
                sleep_after_execution=0.5,
                oai_config_list=None,
                region="us-east-1",
                client_password="",
                ):
    """Worker function to process a single task"""
    domain, ex_id, cfg = task_info
    validate_cua_model(cua_model)
    
    # Recreate llm_config inside the worker process
    llm_config = LLMConfig(config_list=oai_config_list).where(model=orchestrator_model)
    
    history_save_dir = os.path.join(save_dir, "coact", f"{domain}/{ex_id}")
    if not os.path.exists(history_save_dir):
        os.makedirs(history_save_dir)
    task_metadata = _build_task_metadata(
        domain,
        ex_id,
        cfg,
        {
            "provider_name": provider_name,
            "path_to_vm": path_to_vm,
            "orchestrator_model": orchestrator_model,
            "coding_model": coding_model,
            "cua_model": cua_model,
            "mode": mode,
            "enable_web_search": enable_web_search,
            "enable_coding_agent": enable_coding_agent,
            "save_dir": save_dir,
            "orchestrator_max_steps": orchestrator_max_steps,
            "cua_max_steps": cua_max_steps,
            "coding_max_steps": coding_max_steps,
            "cut_off_steps": cut_off_steps,
            "screen_width": screen_width,
            "screen_height": screen_height,
            "sleep_after_execution": sleep_after_execution,
            "region": region,
            "client_password": client_password,
            "oai_config_list": oai_config_list,
        },
    )
    _write_json_file(os.path.join(history_save_dir, "metadata.json"), task_metadata)
    
    task_config = json.load(open(cfg))
    task_instruction = _build_task_instruction(task_config, mode)
    retry = 0
    orchestrator_proxy = None

    while True:
        try:
            with llm_config:
                orchestrator = OrchestratorAgent(
                    name="orchestrator",
                    system_message=_build_task_description(enable_coding_agent, mode),
                    enable_coding_agent=enable_coding_agent,
                    prompt_mode=mode,
                )
                orchestrator_proxy = OrchestratorUserProxyAgent(
                    name="orchestrator_proxy",
                    is_termination_msg=lambda x: x.get("content", "") and (x.get("content", "")[0]["text"].lower() == "terminate" or x.get("content", "")[0]["text"].lower() == "infeasible"),
                    human_input_mode="NEVER",
                    provider_name=provider_name,
                    path_to_vm=path_to_vm,
                    screen_width=screen_width,
                    screen_height=screen_height,
                    sleep_after_execution=sleep_after_execution,
                    code_execution_config=False,
                    history_save_dir=history_save_dir,
                    llm_model=coding_model,
                    gui_model=cua_model,
                    truncate_history_inputs=cua_max_steps + 1,
                    cua_max_steps=cua_max_steps,
                    coding_max_steps=coding_max_steps,
                    enable_web_search=enable_web_search,
                    enable_coding_agent=enable_coding_agent,
                    region=region,
                    client_password=client_password,
                    user_instruction=task_instruction,
                    prompt_mode=mode,
                    task_source=(
                        str(task_config.get("source", "")).strip() or None
                        if mode == "inspect-source-first"
                        else None
                    ),
                )

            orchestrator_proxy.reset(task_config=task_config)
            time.sleep(60)
            screenshot = orchestrator_proxy.env.controller.get_screenshot()

            with open(os.path.join(history_save_dir, f'initial_screenshot_orchestrator.png'), "wb") as f:
                f.write(screenshot)
                
            orchestrator_proxy.initiate_chat(
                recipient=orchestrator,
                message=_build_initial_message(task_instruction) + "<img data:image/png;base64," + base64.b64encode(screenshot).decode("utf-8") + ">",
                max_turns=orchestrator_max_steps
            )
            
            chat_history = []
            key = list(orchestrator_proxy.chat_messages.keys())[0]
            chat_messages = orchestrator_proxy.chat_messages[key]
            for item in chat_messages:
                item.pop('tool_responses', None)
                if item.get('role', None) in ['tool', 'assistant'] and item.get('content', None):
                    for msg in item['content']:
                        if msg.get('type', None) == 'image_url':
                            msg['image_url'] = "<image>"
                chat_history.append(item)
            
            with open(os.path.join(history_save_dir, f'chat_history.json'), "w") as f:
                json.dump(chat_history, f)

            if chat_history[-1]['role'] == 'user' and 'INFEASIBLE' in chat_history[-1]['content'][0]['text']:
                orchestrator_proxy.env.action_history.append("FAIL")

            cua_steps = len(glob.glob(f"{history_save_dir}/cua_output*/step_*.png"))
            coding_paths = glob.glob(f"{history_save_dir}/coding_output*/chat_history.json")
            coding_steps = 0
            for hist in coding_paths:
                with open(hist, 'r') as f:
                    hist = json.dumps(json.load(f))
                    coding_steps += hist.count('exitcode:')
            if cua_steps + coding_steps > cut_off_steps:
                score = 0.0
            else:
                score = orchestrator_proxy.env.evaluate()
            print(f"Score: {score}")
            
            with open(os.path.join(history_save_dir, f'result.txt'), "w") as f:
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
            with open(os.path.join(history_save_dir, f'result.txt'), "w") as f:
                f.write(str(score))
            with open(os.path.join(history_save_dir, f'err_reason.txt'), "w") as f:
                f.write(f"Fatal error: {str(e)}")
        finally:
            if orchestrator_proxy is not None and orchestrator_proxy.env is not None:
                orchestrator_proxy.env.close()
    
    return domain, score


if __name__ == "__main__":
    args = config()
    validate_cua_model(args.cua_model)
    if args.mode != DEFAULT_MODE:
        args.enable_web_search = True
    oai_config_list = _load_oai_config_list(args.oai_config_path)

    tasks = []
    selected_tasks = []
    scores: Dict[str, List[float]] = {}
    if args.task_id:
        domain, ex_id, cfg = _resolve_task_path(args.test_config_base_dir, args.task_id, args.domain)
        selected_tasks.append((domain, ex_id, cfg))
        scores[domain] = []
        result_path = os.path.join(args.result_dir, 'coact', f"{domain}/{ex_id}/result.txt")
        if os.path.exists(result_path):
            result = open(result_path, "r").read()
            print(f"Results already exist in {domain}/{ex_id}, result: {result}")
        else:
            tasks.append((domain, ex_id, cfg))
        test_all_meta = {domain: [ex_id]}
    else:
        with open(args.test_all_meta_path, encoding="utf-8") as f:
            test_all_meta = json.load(f)
        if args.domain != "all":
            test_all_meta = {args.domain: test_all_meta[args.domain]}

        for domain in test_all_meta:
            scores[domain] = []
            for ex_id in test_all_meta[domain]:
                cfg = os.path.join(args.test_config_base_dir, f"{domain}/{ex_id}.json")
                selected_tasks.append((domain, ex_id, cfg))
                if os.path.exists(os.path.join(args.result_dir, 'coact', f"{domain}/{ex_id}/result.txt")):
                    result = open(os.path.join(args.result_dir, 'coact', f"{domain}/{ex_id}/result.txt"), "r").read()
                    print(f"Results already exist in {domain}/{ex_id}, result: {result}")
                    continue
                tasks.append((domain, ex_id, cfg))
    run_metadata = _build_run_metadata(args, tasks, selected_tasks, test_all_meta, oai_config_list)
    metadata_dir = Path(args.result_dir) / "coact"
    _write_json_file(metadata_dir / f"run_metadata_{datetime_str}.json", run_metadata)
    _write_json_file(metadata_dir / "run_metadata_latest.json", run_metadata)
    # Check if there are any tasks to process
    if not tasks:
        print("No tasks to process. All tasks have already been completed.")
        # Print summary of existing results
        print("\n=== Summary of Existing Results ===")
        for domain in test_all_meta:
            domain_scores = []
            for ex_id in test_all_meta[domain]:
                score_file = os.path.join(args.result_dir, 'coact', f"{domain}/{ex_id}/result.txt")
                if os.path.exists(score_file):
                    with open(score_file, "r") as f:
                        domain_scores.append(float(f.read()))
            if domain_scores:
                avg_score = sum(domain_scores) / len(domain_scores)
                print(f"{domain}: {len(domain_scores)} tasks, average score: {avg_score:.2f}")
    else:
        # Use multiprocessing to process tasks in parallel
        # Determine number of workers (you can adjust this based on your system)
        num_workers = min(cpu_count() // 2, args.num_envs)  # Use half of CPU cores, max 4
        print(f"Processing {len(tasks)} tasks with {num_workers} workers...")

        # Create a partial function with fixed config_path, model and debug
        process_func = partial(process_task, 
                               provider_name=args.provider_name,
                               path_to_vm=args.path_to_vm,
                               save_dir=args.result_dir,
                               coding_model=args.coding_model,
                               cua_model=args.cua_model,
                               mode=args.mode,
                               enable_web_search=args.enable_web_search,
                               enable_coding_agent=args.enable_coding_agent,
                               orchestrator_model=args.orchestrator_model,
                               oai_config_list=oai_config_list,
                               orchestrator_max_steps=args.orchestrator_max_steps,
                               cua_max_steps=args.cua_max_steps,
                               coding_max_steps=args.coding_max_steps,
                               cut_off_steps=args.cut_off_steps,
                               screen_width=args.screen_width,
                               screen_height=args.screen_height,
                               sleep_after_execution=args.sleep_after_execution,
                               region=args.region,
                               client_password=args.client_password
                               )

        # Process tasks in parallel
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_func, tasks)

        # Collect scores from results
        for domain, score in results:
            scores[domain].append(score)

        # Print summary
        print("\n=== Task Processing Complete ===")
        for domain in scores:
            if scores[domain]:
                avg_score = sum(scores[domain]) / len(scores[domain])
                print(f"{domain}: {len(scores[domain])} tasks, average score: {avg_score:.2f}")
