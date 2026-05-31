from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mm_agents.coact.computer_exec import load_tool_events, save_mcp_state

CLI_MCP_INSTRUCTION = """
- Use only the MCP `computer` tool for all GUI interaction. Do not run shell commands.
- You may include multiple actions in one `computer` call (OpenAI computer-use JSON action objects).
- After each tool result, inspect the screenshot and continue until the task is done.
- Finish all required GUI work first: set and verify every required value on screen before ending.
- When the task is fully complete and verified, send one final agent message that contains only TERMINATE and do not call `computer` again.
- Do not put TERMINATE in the same turn as GUI actions you still need to perform.
- If stuck, send one final agent message that contains only IDK and why, then do not call `computer` again.
""".strip()


def require_codex_cli() -> str:
    codex_path = shutil.which("codex")
    if not codex_path:
        raise RuntimeError("Codex CLI not found on PATH. Install codex and ensure `codex` is available.")
    return codex_path


def build_cli_agent_prompt(
    *,
    instruction: str,
    client_password: str,
    mode_hints: str = "",
) -> str:
    hints = f'- Sudo password is "{client_password}".\n' if client_password else ""
    extra = f"{mode_hints}\n" if mode_hints else ""
    return (
        f"{instruction.strip()}\n\n"
        f"{hints}"
        "- Keep windows/applications open at the end of the task.\n"
        f"{extra}"
        f"{CLI_MCP_INSTRUCTION}"
    )


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _write_codex_config(
    codex_home: Path,
    *,
    python_executable: str,
    gui_docker_env_root: str,
    http_server: str,
    save_path: str,
    max_steps: int,
    sleep_after_execution: float,
) -> None:
    codex_home.mkdir(parents=True, exist_ok=True)
    env_entries = {
        "OSWORLD_SERVER_URL": http_server,
        "SAVE_PATH": save_path,
        "MAX_STEPS": str(max_steps),
        "SLEEP_AFTER_EXECUTION": str(sleep_after_execution),
        "PYTHONPATH": gui_docker_env_root,
    }
    env_toml = ", ".join(f"{key} = {_toml_string(value)}" for key, value in env_entries.items())
    config = (
        f'approval_policy = "never"\n\n'
        f"[mcp_servers.desktop]\n"
        f"command = {_toml_string(python_executable)}\n"
        f'args = [{_toml_string("-m")}, {_toml_string("mm_agents.coact.desktop_computer_mcp")}]\n'
        f"env = {{ {env_toml} }}\n"
        f"enabled = true\n"
        f"startup_timeout_sec = 30\n"
        f"tool_timeout_sec = 120\n"
        f'default_tools_approval_mode = "auto"\n\n'
        f"[mcp_servers.desktop.tools.computer]\n"
        f'approval_mode = "auto"\n'
    )
    (codex_home / "config.toml").write_text(config, encoding="utf-8")


def _link_codex_auth(codex_home: Path) -> None:
    user_codex = Path.home() / ".codex"
    for name in ("auth.json", "credentials.json"):
        src = user_codex / name
        if src.is_file():
            dst = codex_home / name
            if not dst.exists():
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy2(src, dst)


def _session_id_from_json_line(line: str) -> Optional[str]:
    line = line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("session_id", "conversation_id", "thread_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = payload.get("session")
    if isinstance(nested, dict):
        for key in ("id", "session_id"):
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_assistant_text(raw_text: str, json_events_path: Optional[Path]) -> str:
    if raw_text.strip():
        return raw_text.strip()
    if json_events_path and json_events_path.is_file():
        texts: List[str] = []
        for line in json_events_path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            item = payload.get("item")
            if isinstance(item, dict) and item.get("type") == "message":
                for part in item.get("content", []):
                    if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                        text = part.get("text", "")
                        if text:
                            texts.append(text)
            message = payload.get("message")
            if isinstance(message, dict):
                for part in message.get("content", []):
                    if isinstance(part, dict) and part.get("text"):
                        texts.append(part["text"])
        if texts:
            return "\n".join(texts).strip()
    return ""


def _final_result_from_text(text: str) -> str:
    if not text:
        return "IDK The GUI loop stopped before producing a terminal answer."
    if "IDK" in text:
        return text if text.upper().startswith("IDK") else f"IDK {text}"
    if "TERMINATE" in text:
        return text
    return f"{text}\nTERMINATE"


def _is_terminal_only_message(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    upper = normalized.upper()
    if upper in {"TERMINATE", "IDK"}:
        return True
    if upper.startswith("IDK"):
        return True
    if "TERMINATE" in upper and "ACTION:" not in upper and len(normalized) <= 80:
        return True
    return False


def _iter_jsonl_completed_events(json_events_path: Path) -> List[Dict[str, Any]]:
    if not json_events_path.is_file():
        return []
    events: List[Dict[str, Any]] = []
    for line in json_events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("type") != "item.completed":
            continue
        item = payload.get("item")
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "agent_message":
            events.append(
                {
                    "kind": "agent_message",
                    "text": str(item.get("text", "")).strip(),
                }
            )
        elif item_type == "mcp_tool_call" and item.get("tool") == "computer":
            arguments = item.get("arguments") or {}
            actions = arguments.get("actions") if isinstance(arguments, dict) else None
            events.append(
                {
                    "kind": "computer",
                    "actions": list(actions or []),
                }
            )
    return events


def build_cli_turn_transcript(
    *,
    tool_events: List[Dict[str, Any]],
    json_events_path: Path,
    session_id: Optional[str],
    raw_text: str,
    assistant_text: str,
) -> List[Dict[str, Any]]:
    computer_events = [event for event in tool_events if event.get("type") == "computer_call"]
    computer_iter = iter(computer_events)
    jsonl_events = _iter_jsonl_completed_events(json_events_path)

    turns: List[Dict[str, Any]] = []
    terminal_messages: List[str] = []
    pending_message = ""

    for event in jsonl_events:
        if event["kind"] == "agent_message":
            text = str(event.get("text", "")).strip()
            if not text:
                continue
            if _is_terminal_only_message(text):
                terminal_messages.append(text)
                pending_message = ""
            else:
                pending_message = text
            continue

        try:
            tool_event = next(computer_iter)
        except StopIteration:
            tool_event = {}

        agent_message = pending_message
        pending_message = ""
        turns.append(
            {
                "protocol": "cli_mcp",
                "turn_index": len(turns) + 1,
                "step_index": tool_event.get("step_index"),
                "agent_message": agent_message,
                "prediction": agent_message,
                "actions": list(event.get("actions") or tool_event.get("actions") or []),
                "call_id": tool_event.get("call_id"),
            }
        )

    for tool_event in computer_iter:
        turns.append(
            {
                "protocol": "cli_mcp",
                "turn_index": len(turns) + 1,
                "step_index": tool_event.get("step_index"),
                "agent_message": "",
                "prediction": "",
                "actions": list(tool_event.get("actions") or []),
                "call_id": tool_event.get("call_id"),
            }
        )

    final_message = assistant_text.strip()
    if terminal_messages:
        final_message = terminal_messages[-1]
    elif not final_message:
        non_terminal = [
            turn["agent_message"]
            for turn in turns
            if str(turn.get("agent_message", "")).strip()
        ]
        if non_terminal:
            final_message = non_terminal[-1]

    turns.append(
        {
            "protocol": "cli_mcp",
            "type": "final",
            "session_id": session_id,
            "raw_text": raw_text,
            "assistant_text": final_message,
            "agent_message": final_message,
            "prediction": final_message,
            "terminal_messages": terminal_messages,
        }
    )
    return turns


def cli_final_assistant_text(raw_transcript: List[Dict[str, Any]]) -> str:
    for entry in reversed(raw_transcript):
        if entry.get("type") == "final":
            return str(entry.get("assistant_text", "")).strip()
    for entry in reversed(raw_transcript):
        text = str(entry.get("assistant_text", "")).strip()
        if text:
            return text
    return ""


def _extract_usage_from_jsonl(json_events_path: Path) -> Optional[Dict[str, Any]]:
    if not json_events_path.is_file():
        return None
    usage: Optional[Dict[str, Any]] = None
    try:
        for line in json_events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                candidate = payload.get("usage")
                if isinstance(candidate, dict):
                    usage = candidate
                response = payload.get("response")
                if isinstance(response, dict) and isinstance(response.get("usage"), dict):
                    usage = response["usage"]
    except OSError:
        return None
    return usage


@dataclass
class CliMcpSession:
    save_path: str
    model: str
    http_server: str
    max_steps: int
    sleep_after_execution: float = 0.3
    timeout_seconds: int = 600
    extra_args: Sequence[str] = field(default_factory=list)
    _session_id: Optional[str] = None
    last_usage: Optional[Dict[str, Any]] = field(default=None, init=False)

    def run(self, *, prompt: str, initial_screenshot_path: str) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        self.save_path = os.path.abspath(self.save_path)
        os.makedirs(self.save_path, exist_ok=True)
        codex_home = Path(self.save_path) / ".codex"
        gui_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        _write_codex_config(
            codex_home,
            python_executable=sys.executable,
            gui_docker_env_root=gui_root,
            http_server=self.http_server,
            save_path=self.save_path,
            max_steps=self.max_steps,
            sleep_after_execution=self.sleep_after_execution,
        )
        _link_codex_auth(codex_home)
        save_mcp_state(self.save_path, {"step_index": 1, "step_count": 0})
        tool_events_path = Path(self.save_path) / "tool_events.json"
        tool_events_path.write_text(json.dumps({"events": []}, indent=2), encoding="utf-8")
        (Path(self.save_path) / ".rules").write_text(
            "reject command_execution\n",
            encoding="utf-8",
        )

        output_path = Path(self.save_path) / "cli_turn_0001.txt"
        json_events_path = Path(self.save_path) / "cli_turn_0001.jsonl"
        cmd = self._build_command(
            output_path=str(output_path),
            screenshot_path=initial_screenshot_path,
            json_events_path=str(json_events_path),
        )
        env = os.environ.copy()
        env["CODEX_HOME"] = str(codex_home)

        console_log = open(Path(self.save_path) / "cli_console.log", "a", encoding="utf-8")
        try:
            completed = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                cwd=self.save_path,
                env=env,
            )
            console_log.write("\n--- turn 1 ---\n")
            console_log.write(" ".join(cmd) + "\n")
            if completed.stdout:
                console_log.write(completed.stdout)
                json_events_path.write_text(completed.stdout, encoding="utf-8")
            if completed.stderr:
                console_log.write(completed.stderr)
            console_log.flush()

            if completed.returncode != 0:
                raise RuntimeError(
                    f"Codex CLI failed (code {completed.returncode}): "
                    f"{completed.stderr.strip() or completed.stdout.strip()}"
                )

            self._session_id = self._extract_session_id(json_events_path, completed.stdout)
            session_meta = {
                "provider": "codex_mcp",
                "model": self.model,
                "save_path": self.save_path,
                "session_id": self._session_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "http_server": self.http_server,
            }
            (Path(self.save_path) / "cli_session.json").write_text(
                json.dumps(session_meta, indent=2), encoding="utf-8"
            )

            raw_text = output_path.read_text(encoding="utf-8") if output_path.is_file() else ""
            assistant_text = _extract_assistant_text(raw_text, json_events_path)
            final_result = _final_result_from_text(assistant_text)

            tool_events = load_tool_events(self.save_path)
            raw_transcript = build_cli_turn_transcript(
                tool_events=tool_events,
                json_events_path=json_events_path,
                session_id=self._session_id,
                raw_text=raw_text,
                assistant_text=assistant_text,
            )
            self.last_usage = _extract_usage_from_jsonl(json_events_path)
            return final_result, raw_transcript, tool_events
        finally:
            console_log.close()

    def _build_command(
        self,
        *,
        output_path: str,
        screenshot_path: str,
        json_events_path: str,
    ) -> List[str]:
        codex_bin = require_codex_cli()
        cmd = [
            codex_bin,
            "-a",
            "never",
            "exec",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "-m",
            self.model,
            "-C",
            os.path.abspath(self.save_path),
            "-i",
            os.path.abspath(screenshot_path),
            "-o",
            os.path.abspath(output_path),
            "--json",
        ]
        cmd.extend(self.extra_args)
        cmd.extend(["-c", 'approval_policy="never"'])
        cmd.append("-")
        return cmd

    def _extract_session_id(self, json_events_path: Path, stdout: str) -> Optional[str]:
        candidates: List[str] = []
        if json_events_path.is_file():
            for line in json_events_path.read_text(encoding="utf-8").splitlines():
                session_id = _session_id_from_json_line(line)
                if session_id:
                    candidates.append(session_id)
        for line in stdout.splitlines():
            session_id = _session_id_from_json_line(line)
            if session_id:
                candidates.append(session_id)
        return candidates[-1] if candidates else None
