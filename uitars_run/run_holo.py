"""Run OSWorld evaluations with a Holo-specific action parser."""

import argparse
import datetime
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from mm_agents.env_loader import load_mm_agents_env

load_mm_agents_env()

from mm_agents.holo_agent import HoloAgent
import lib_run_single
from desktop_env.desktop_env import DesktopEnv


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

file_handler = logging.FileHandler(
    os.path.join("logs", f"normal-{datetime_str}.log"), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", f"debug-{datetime_str}.log"), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(
    os.path.join("logs", f"sdebug-{datetime_str}.log"), encoding="utf-8"
)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
sdebug_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)

logger = logging.getLogger("desktopenv.experiment")


def ping(base_url: str) -> bool:
    url = f"{base_url.rstrip('/')}/ping"
    try:
        response = requests.get(url, timeout=10)
        print(f"[info] ping: {response.status_code} {getattr(response, 'text', '')[:200]}")
        return response.ok
    except Exception as exc:
        print(f"[error] ping failed: {exc}")
        return False


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Holo end-to-end evaluation on OSWorld")

    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OSWORLD_BASE_URL") or "http://127.0.0.1:50003",
        help="Base URL for docker_server API",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("OSWORLD_TOKEN") or "dart",
        help="Token for docker_server",
    )
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum concurrent tasks (VMs)")
    parser.add_argument("--os-type", type=str, default="Ubuntu", help="OS type to pass to DesktopEnv")
    parser.add_argument(
        "--enable-proxy",
        action="store_true",
        help="Honor task proxy settings when supported by the task config",
    )

    parser.add_argument("--path_to_vm", "--path-to-vm", type=str, default=None)
    parser.add_argument("--headless", action="store_true", help="Run in headless machine")
    parser.add_argument("--action_space", "--action-space", type=str, default="pyautogui", help="Action type")
    parser.add_argument(
        "--observation_type",
        "--observation-type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument("--screen_width", "--screen-width", type=int, default=1920)
    parser.add_argument("--screen_height", "--screen-height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", "--sleep-after-execution", type=float, default=2.0)
    parser.add_argument("--max_steps", "--max-steps", type=int, default=15)

    parser.add_argument("--max_trajectory_length", "--max-trajectory-length", type=int, default=50)
    parser.add_argument("--test_config_base_dir", "--test-config-base-dir", type=str, default="evaluation_examples")

    parser.add_argument("--model", type=str, default="Hcompany/Holo3-35B-A3B")
    parser.add_argument("--input_swap", "--input-swap", action="store_true", help="Use copy/paste typing")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--max_pixels", "--max-pixels", type=float, default=16384 * 28 * 28)
    parser.add_argument("--min_pixels", "--min-pixels", type=float, default=100 * 28 * 28)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", "--top-p", type=float, default=0.9)
    parser.add_argument("--history_n", "--history-n", type=int, default=2)
    parser.add_argument("--callusr_tolerance", "--callusr-tolerance", type=int, default=3)
    parser.add_argument("--max_tokens", "--max-tokens", type=int, default=1024)

    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path",
        "--test-all-meta-path",
        type=str,
        default="evaluation_examples/test_nogdrive.json",
    )

    parser.add_argument("--result_dir", "--result-dir", type=str, default="./results_holo")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run all tasks in the meta file and clear existing result dirs",
    )
    return parser.parse_args()


def run_one_example(args: argparse.Namespace, domain: str, example_id: str) -> Dict[str, Any]:
    example_result_dir = os.path.join(
        args.result_dir,
        args.action_space,
        args.observation_type,
        args.model,
        domain,
        example_id,
    )
    if args.overwrite and os.path.isdir(example_result_dir):
        shutil.rmtree(example_result_dir)
    os.makedirs(example_result_dir, exist_ok=True)

    config_file = os.path.join(args.test_config_base_dir, f"examples/{domain}/{example_id}.json")
    with open(config_file, "r", encoding="utf-8") as file_obj:
        example = json.load(file_obj)

    instruction = example.get("instruction", "")
    logger.info(f"[Domain]: {domain}")
    logger.info(f"[Example ID]: {example_id}")
    logger.info(f"[Instruction]: {instruction}")

    agent = HoloAgent(
        model=args.model,
        action_space=args.action_space,
        observation_type=args.observation_type,
        max_trajectory_length=args.max_trajectory_length,
        runtime_conf={
            "input_swap": args.input_swap,
            "language": args.language,
            "history_n": args.history_n,
            "max_pixels": args.max_pixels,
            "min_pixels": args.min_pixels,
            "callusr_tolerance": args.callusr_tolerance,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
    )

    env = None
    try:
        env = DesktopEnv(
            action_space="pyautogui",
            provider_name="docker_server",
            os_type=args.os_type,
            enable_proxy=args.enable_proxy,
        )
        scores_local: List[float] = []
        try:
            lib_run_single.run_single_example(
                agent,
                env,
                example,
                args.max_steps,
                instruction,
                args,
                example_result_dir,
                scores_local,
            )
            status = "success"
        except Exception as exc:
            logger.error(f"Exception in {domain}/{example_id}: {exc}")
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps({"Error": f"Exception in {domain}/{example_id}: {str(exc)}"}))
                file_obj.write("\n")
            status = "failed"
        return {"task_id": example.get("id", example_id), "status": status}
    finally:
        try:
            if env:
                env.close()
        except Exception as exc:
            logger.warning(f"env.close() error: {exc}")


def test(args: argparse.Namespace, test_all_meta: Dict[str, List[str]]) -> None:
    if not ping(args.base_url):
        print("[error] docker_server not reachable. Start it via:")
        print("  PYTHONPATH=. python -m desktop_env.docker_server.server")
        return

    tasks: List[Dict[str, str]] = []
    for domain in sorted(test_all_meta.keys()):
        for example_id in test_all_meta[domain]:
            tasks.append({"domain": domain, "example_id": example_id})

    successes: List[str] = []
    failures: List[str] = []

    print(f"[info] Dispatching {len(tasks)} tasks with max_workers={args.max_workers}")
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = {pool.submit(run_one_example, args, task["domain"], task["example_id"]): task for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                logger.warning(f"runner exception: {exc}")
                continue
            task_id = result.get("task_id")
            if result.get("status") == "success":
                successes.append(task_id)
                print(f"[info] {task_id}: success")
            else:
                failures.append(task_id)
                print(f"[info] {task_id}: failed")

    print(f"[info] Completed. success={len(successes)} failed={len(failures)}")


def get_unfinished(action_space: str, use_model: str, observation_type: str, result_dir: str, total_file_json: Dict[str, List[str]]) -> Dict[str, List[str]]:
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        return total_file_json

    finished: Dict[str, List[str]] = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            continue
        for example_id in os.listdir(domain_path):
            if example_id == "onboard":
                continue
            example_path = os.path.join(domain_path, example_id)
            if not os.path.isdir(example_path):
                continue
            if "result.txt" not in os.listdir(example_path):
                for file_name in os.listdir(example_path):
                    os.remove(os.path.join(example_path, file_name))
            else:
                finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [task_id for task_id in total_file_json[domain] if task_id not in examples]
    return total_file_json


def get_result(action_space: str, use_model: str, observation_type: str, result_dir: str, total_file_json: Dict[str, List[str]]) -> Optional[List[float]]:
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result: List[float] = []
    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            continue
        for example_id in os.listdir(domain_path):
            example_path = os.path.join(domain_path, example_id)
            if not os.path.isdir(example_path):
                continue
            if "result.txt" in os.listdir(example_path):
                try:
                    with open(os.path.join(example_path, "result.txt"), "r", encoding="utf-8") as file_obj:
                        all_result.append(float(file_obj.read()))
                except Exception:
                    all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None

    print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
    return all_result


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    os.environ["OSWORLD_TOKEN"] = args.token
    os.environ["OSWORLD_BASE_URL"] = args.base_url

    with open(args.test_all_meta_path, "r", encoding="utf-8") as file_obj:
        test_all_meta = json.load(file_obj)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    if args.overwrite:
        test_file_list = {key: list(value) for key, value in test_all_meta.items()}
    else:
        test_file_list = get_unfinished(
            args.action_space,
            args.model,
            args.observation_type,
            args.result_dir,
            test_all_meta,
        )

    left_info = ""
    for domain in test_file_list:
        left_info += f"{domain}: {len(test_file_list[domain])}\n"
    logger.info(f"Left tasks:\n{left_info}")

    get_result(
        args.action_space,
        args.model,
        args.observation_type,
        args.result_dir,
        test_all_meta,
    )
    test(args, test_file_list)
