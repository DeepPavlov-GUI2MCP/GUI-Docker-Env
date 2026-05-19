import json
from pathlib import Path

from mm_agents.coact.cli_mcp_session import build_cli_turn_transcript


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_terminal_message_becomes_final_turn_only(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "cli_turn_0001.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "Open settings"}},
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "tool": "computer",
                    "arguments": {"actions": [{"type": "click", "button": "left", "x": 1, "y": 2}]},
                },
            },
            {"type": "item.completed", "item": {"type": "agent_message", "text": "TERMINATE"}},
        ],
    )
    tool_events = [
        {
            "type": "computer_call",
            "step_index": 1,
            "actions": [{"type": "click", "button": "left", "x": 1, "y": 2}],
            "call_id": "call_1",
        }
    ]
    turns = build_cli_turn_transcript(
        tool_events=tool_events,
        json_events_path=jsonl_path,
        session_id="sess",
        raw_text="TERMINATE",
        assistant_text="TERMINATE",
    )
    assert turns[0]["agent_message"] == "Open settings"
    assert turns[0].get("type") != "final"
    assert turns[-1]["type"] == "final"
    assert turns[-1]["agent_message"] == "TERMINATE"
    assert "TERMINATE" not in turns[0]["agent_message"]


def test_back_to_back_computer_without_message(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "cli_turn_0001.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "First"}},
            {
                "type": "item.completed",
                "item": {"type": "mcp_tool_call", "tool": "computer", "arguments": {"actions": [{"type": "click"}]}},
            },
            {
                "type": "item.completed",
                "item": {"type": "mcp_tool_call", "tool": "computer", "arguments": {"actions": [{"type": "key", "keys": ["enter"]}]}},
            },
        ],
    )
    tool_events = [
        {"type": "computer_call", "step_index": 1, "actions": [{"type": "click"}], "call_id": "c1"},
        {"type": "computer_call", "step_index": 2, "actions": [{"type": "key", "keys": ["enter"]}], "call_id": "c2"},
    ]
    turns = build_cli_turn_transcript(
        tool_events=tool_events,
        json_events_path=jsonl_path,
        session_id=None,
        raw_text="",
        assistant_text="",
    )
    assert len([turn for turn in turns if turn.get("type") != "final"]) == 2
    assert turns[0]["agent_message"] == "First"
    assert turns[1]["agent_message"] == ""
