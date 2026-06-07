import datetime
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
from wrapt_timeout_decorator import *

from desktop_env.controllers.a11y_preflight import A11yPreflightExecutor

from lib_eval_artifacts import (
    example_for_reset_with_deferred_preflight,
    record_eval_step,
)

logger = logging.getLogger("desktopenv.experiment")

DEFAULT_SETTLE_TIMEOUT_SECONDS = float(os.environ.get("OSWORLD_SETTLE_TIMEOUT_SECONDS", "45"))
DEFAULT_SETTLE_POLL_INTERVAL_SECONDS = float(os.environ.get("OSWORLD_SETTLE_POLL_INTERVAL_SECONDS", "0.5"))
DEFAULT_SETTLE_STABLE_POLLS = int(os.environ.get("OSWORLD_SETTLE_STABLE_POLLS", "2"))
MIN_A11Y_TREE_CHARS = 1000
GENERIC_A11Y_TREE_CHARS = 5000


def _settle_needles_from_example(example: Dict[str, Any]) -> List[str]:
    needles: List[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value and value not in needles:
            needles.append(value)

    for block in example.get("config") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "activate_window":
            add((block.get("parameters") or {}).get("window_name"))
        if block.get("type") == "open":
            path = (block.get("parameters") or {}).get("path")
            if isinstance(path, str) and path.endswith(".docx"):
                add(f"{os.path.basename(path)} - LibreOffice Writer")

    evaluator = example.get("evaluator") or {}
    for block in evaluator.get("postconfig") or []:
        if isinstance(block, dict) and block.get("type") == "activate_window":
            add((block.get("parameters") or {}).get("window_name"))

    provenance = example.get("provenance") or {}
    add(provenance.get("window_name"))

    snapshot = str(example.get("snapshot") or "").lower()
    if "writer" in snapshot:
        add("LibreOffice Writer")

    return needles


def _a11y_tree_ready(tree: Optional[str], needles: List[str], example: Dict[str, Any]) -> bool:
    if not tree or len(tree) < MIN_A11Y_TREE_CHARS:
        return False
    if "writer" in str(example.get("snapshot") or "").lower():
        if re.search(r"\.docx - LibreOffice Writer", tree):
            return True
    if not needles:
        return len(tree) >= GENERIC_A11Y_TREE_CHARS
    return any(needle in tree for needle in needles)


def wait_for_post_reset_settle(
    env,
    example: Dict[str, Any],
    *,
    timeout_seconds: float = DEFAULT_SETTLE_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_SETTLE_POLL_INTERVAL_SECONDS,
    stable_polls: int = DEFAULT_SETTLE_STABLE_POLLS,
) -> float:
    needles = _settle_needles_from_example(example)
    logger.info(
        "Post-reset settle polling starting (timeout=%.0fs, needles=%s)",
        timeout_seconds,
        needles[:4] if needles else ["<generic>"],
    )
    started_at = time.monotonic()
    deadline = started_at + max(0.0, float(timeout_seconds))
    stable = 0
    while time.monotonic() < deadline:
        obs = env._get_obs()
        tree = obs.get("accessibility_tree") if isinstance(obs, dict) else None
        if _a11y_tree_ready(tree, needles, example):
            stable += 1
            if stable >= max(1, stable_polls):
                elapsed = time.monotonic() - started_at
                logger.info("Post-reset settle ready after %.1fs", elapsed)
                return elapsed
        else:
            stable = 0
        time.sleep(max(0.1, poll_interval))
    elapsed = time.monotonic() - started_at
    logger.warning("Post-reset settle timed out after %.1fs", elapsed)
    return elapsed


def _record_desktop_host(env, example_result_dir: str) -> None:
    provider = getattr(env, "provider", None)
    if provider is None:
        return
    server_ip = getattr(provider, "remote_docker_server_ip", None)
    server_port = getattr(provider, "remote_docker_server_port", None)
    if not server_ip or not server_port:
        return
    host_path = os.path.join(example_result_dir, "host.txt")
    with open(host_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"{server_ip}:{server_port}\n")


def _run_deferred_a11y_preflight(
    env,
    example_result_dir: str,
    preflight_params: dict,
    *,
    artifact_step: int,
) -> int:
    executor = A11yPreflightExecutor(
        env.setup_controller.http_server,
        cache_dir=env.cache_dir,
    )

    def on_preflight_step(step_index: int, step: dict) -> None:
        nonlocal artifact_step
        obs = env._get_obs()
        artifact_step += 1
        timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
        record_eval_step(
            example_result_dir,
            artifact_step,
            timestamp,
            obs,
            env,
            action={
                "phase": "a11y_preflight",
                "preflight_step_index": step_index,
                "op": step.get("op"),
                "selector": step.get("selector"),
            },
        )
        logger.info("Preflight artifact step %d: op=%s", artifact_step, step.get("op"))

    executor.run(
        steps=preflight_params["steps"],
        timeout_seconds=preflight_params.get("timeout_seconds", 20.0),
        screenshot_on_failure=preflight_params.get("screenshot_on_failure", False),
        on_step_complete=on_preflight_step,
    )
    return artifact_step


def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    try:
        agent.reset(runtime_logger)
    except Exception as e:
        agent.reset()

    reset_example, preflight_params = example_for_reset_with_deferred_preflight(example)
    logger.info("Task reset starting")
    env.reset(task_config=reset_example)
    _record_desktop_host(env, example_result_dir)
    logger.info("Task reset finished")

    logger.info("Initial environment settle wait starting")
    wait_for_post_reset_settle(env, example)
    logger.info("Initial environment settle wait finished")
    logger.info("Initial observation capture starting")
    obs = env._get_obs()
    logger.info(
        "Initial observation capture finished (screenshot=%s, a11y=%s)",
        bool(obs.get("screenshot")),
        bool(obs.get("accessibility_tree")),
    )
    initial_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
    logger.info("Initial artifact recording starting")
    record_eval_step(
        example_result_dir,
        1,
        initial_timestamp,
        obs,
        env,
        action=None,
    )
    logger.info("Initial artifact recording finished")
    artifact_step = 1

    if preflight_params is not None:
        logger.info("Deferred a11y preflight starting (%d steps)", len(preflight_params.get("steps", [])))
        artifact_step = _run_deferred_a11y_preflight(
            env,
            example_result_dir,
            preflight_params,
            artifact_step=artifact_step,
        )
        logger.info("Deferred a11y preflight finished at artifact step %d", artifact_step)
        logger.info("Post-preflight observation capture starting")
        obs = env._get_obs()
        logger.info(
            "Post-preflight observation capture finished (screenshot=%s, a11y=%s)",
            bool(obs.get("screenshot")),
            bool(obs.get("accessibility_tree")),
        )

    done = False
    step_idx = 0
    # env.controller.start_recording()
    while not done and step_idx < max_steps:
        logger.info("Agent predict starting at loop step %d", step_idx + 1)
        response, actions = agent.predict(
            instruction,
            obs
        )
        logger.info("Agent predict finished at loop step %d with %d action(s)", step_idx + 1, len(actions))
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Environment step starting for artifact step %d", artifact_step + 1)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)
            logger.info("Environment step finished for artifact step %d", artifact_step + 1)
            artifact_step += 1
            logger.info("Step %d: %s", artifact_step, action)
            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            record_eval_step(
                example_result_dir,
                artifact_step,
                action_timestamp,
                obs,
                env,
                action=action,
                response=response,
                reward=reward,
                done=done,
                info=info,
            )
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    # env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(logging.FileHandler(os.path.join(example_result_dir, "runtime.log")))
    return runtime_logger

def run_single_example_human(env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    
    # Save initial screenshot
    with open(os.path.join(example_result_dir, "initial_state.png"), "wb") as _f:
        _f.write(obs['screenshot'])
    
    # Save trajectory information
    with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
        f.write(json.dumps({
            "instruction": instruction,
            "initial_state": "initial_state.png"
        }))
        f.write("\n")
    
    # Evaluate the result
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")



def run_single_example_openaicua(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset(runtime_logger)
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    env.controller.start_recording()
    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs
        )

        done = not response.get('state_correct', False)

        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info, step_info = agent.step(action)

            if not done:
                if not response.get('state_correct', False):
                    done = True

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])

            # Remove pending checks if they exist which will cause issues with json serialization
            if action.get('pending_checks', None):
                del action['pending_checks']

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))

def run_single_example_opencua(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset(runtime_logger)
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    env.controller.start_recording()
    while not done and step_idx < max_steps:
        response, actions, info_dict = agent.predict(instruction, obs)

        logger.info(f"Got Action: {actions}")
        # Breack if no actions
        if not actions or len(actions)==0 or actions[0]=="" or actions[0].lower().startswith("error"): 
            break

        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info(f"Action {action} executed, reward: {reward}, done: {done}")
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))

def run_single_example_autoglm(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    try:
        agent.reset(runtime_logger)
    except Exception as e:
        agent.reset()

    env.reset(task_config=example)
    
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    env.controller.start_recording()
    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs
        )
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
                
            if done:
                logger.info("The episode is done.")
                break
        
        # Invalid Action
        if not actions:
            obs = env._get_obs() # update observation
            
        step_idx += 1
    
    if not done: # not completed the task yet
        env.action_history.append('FAIL')
    
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))

def run_single_example_mano(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset(runtime_logger)
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    env.controller.start_recording()
    
    with open(os.path.join(example_result_dir, f"step_0.png"),
      "wb") as _f:
        _f.write(obs['screenshot'])
    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs
        )
        if len(actions) > 1:
            if (("pyautogui.hotkey('shift')" in actions[0] or "pyautogui.hotkey('ctrl')" in actions[0]) 
                and "pyautogui.click" in actions[1]):
                hotkey_type = 'shift' if "shift" in actions[0] else 'ctrl'
                action = f"pyautogui.keyDown('{hotkey_type}')\n{actions[1]}\npyautogui.keyUp('{hotkey_type}')"
                actions = [action]  
                
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png",
                    "response":response
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
    
def run_single_example_uipath(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    try:
        agent.reset(runtime_logger)
    except Exception as e:
        agent.reset()

    env.reset(task_config=example)

    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    env.controller.start_recording()
    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs,
            args,
            step_idx
        )
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
