from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_agents.coact.spending import (
    SpendingSession,
    aggregate_spending_payloads,
    compute_cost,
    is_local_vllm_endpoint,
    merge_spending_into_metadata,
    read_spending_from_metadata,
    resolve_billing_mode,
    resolve_default_credentials,
)


def _sample_payload(*, role: str = "gui", model: str = "gpt-5.4", tokens: int = 1000, cost: float = 0.01) -> dict:
    return {
        "spending_per_step": [
            {"num_tokens": tokens, "cost": cost, "model": model, "role": role},
        ],
        "spending_by_model": [{"model": model, "num_tokens": tokens, "cost": cost}],
        "spending_by_role": [{"role": role, "num_tokens": tokens, "cost": cost}],
        "spending_total": {"num_tokens": tokens, "cost": cost},
    }


def test_attempt_to_task_aggregation_includes_spending_by_attempt() -> None:
    children = [
        ({"attempt_index": 0, "rollout_index": 0}, _sample_payload(tokens=1000, cost=0.01)),
        ({"attempt_index": 1, "rollout_index": 1}, _sample_payload(tokens=2000, cost=0.02)),
    ]
    payload = aggregate_spending_payloads(
        children,
        group_totals_key="spending_by_attempt",
        annotate_step=lambda meta, step: {**step, "attempt_index": meta["attempt_index"]},
    )
    assert len(payload["spending_by_attempt"]) == 2
    assert payload["spending_total"]["num_tokens"] == 3000
    assert payload["spending_total"]["cost"] == pytest.approx(0.03)
    assert all("attempt_index" in step for step in payload["spending_per_step"])


def test_single_attempt_omits_spending_by_attempt() -> None:
    payload = aggregate_spending_payloads(
        [( {"attempt_index": 0}, _sample_payload())],
        group_totals_key="spending_by_attempt",
    )
    assert "spending_by_attempt" not in payload


def test_task_to_run_aggregation_includes_spending_by_task() -> None:
    children = [
        ({"task_index": 0, "domain": "writer", "task_id": "a"}, _sample_payload(tokens=1000, cost=0.01)),
        ({"task_index": 1, "domain": "writer", "task_id": "b"}, _sample_payload(tokens=2000, cost=0.02)),
    ]
    payload = aggregate_spending_payloads(
        children,
        group_totals_key="spending_by_task",
        annotate_step=lambda meta, step: {
            **step,
            "domain": meta["domain"],
            "task_id": meta["task_id"],
            "task_index": meta["task_index"],
        },
    )
    assert len(payload["spending_by_task"]) == 2
    assert payload["spending_total"]["num_tokens"] == 3000
    assert payload["spending_per_step"][0]["domain"] == "writer"


def test_partial_rerun_run_total_reflects_updated_task() -> None:
    unchanged = _sample_payload(tokens=1000, cost=0.01)
    updated = _sample_payload(tokens=5000, cost=0.05)
    run_payload = aggregate_spending_payloads(
        [
            ({"task_index": 0, "domain": "d", "task_id": "keep"}, unchanged),
            ({"task_index": 1, "domain": "d", "task_id": "overwrite"}, updated),
        ],
        group_totals_key="spending_by_task",
        annotate_step=lambda meta, step: {**step, **meta},
    )
    assert run_payload["spending_total"]["num_tokens"] == 6000
    assert run_payload["spending_total"]["cost"] == pytest.approx(0.06)


def test_merge_spending_into_metadata_preserves_config_keys() -> None:
    metadata = {"domain": "writer", "task_id": "abc", "models": {"gui": "gpt-5.4"}}
    merged = merge_spending_into_metadata(metadata, _sample_payload())
    assert merged["domain"] == "writer"
    assert merged["models"]["gui"] == "gpt-5.4"
    assert "spending" in merged


def test_read_spending_from_metadata_roundtrip(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    spending = _sample_payload()
    metadata_path.write_text(json.dumps({"task_id": "x", "spending": spending}), encoding="utf-8")
    assert read_spending_from_metadata(metadata_path) == spending


def test_vllm_local_endpoint_has_zero_cost() -> None:
    assert is_local_vllm_endpoint("http://127.0.0.1:8010/v1")
    cost = compute_cost(
        "ui-tars-1.5",
        {"prompt_tokens": 1000, "completion_tokens": 500},
        base_url="http://127.0.0.1:8010/v1",
    )
    assert cost == 0.0


def test_spending_session_records_and_aggregates() -> None:
    session = SpendingSession()
    session.record(
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        model="gpt-5.4",
        role="gui",
        base_url="http://31.56.222.86:8002/api/providers/openai/v1",
    )
    payload = session.to_spending_payload()
    assert payload["spending_total"]["num_tokens"] == 150
    assert payload["spending_total"]["cost"] >= 0.0


def test_resolve_billing_mode() -> None:
    assert resolve_billing_mode("https://openrouter.ai/api/v1") == "openrouter"
    assert resolve_billing_mode("http://31.56.222.86:8002/api/providers/openai/v1") == "openai"


def test_resolve_default_credentials_prefers_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    creds = resolve_default_credentials()
    assert creds["billing_mode"] == "openrouter"
    assert creds["api_key"] == "or-key"
