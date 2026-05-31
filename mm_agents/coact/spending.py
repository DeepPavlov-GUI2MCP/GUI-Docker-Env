from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from mm_agents.coact.autogen.oai.openai_utils import calculate_oai_model_cost, resolve_oai_price_1k

logger = logging.getLogger("desktopenv")

# Per-1K token prices (input, output) in USD — OpenRouter rates may differ from OpenAI direct.
OPENROUTER_PRICE1K: Dict[str, tuple[float, float]] = {
    "gpt-5.5": (0.005, 0.03),
    "gpt-5.4": (0.0025, 0.015),
    "gpt-5.4-mini": (0.00075, 0.0045),
    "gpt-5.4-nano": (0.0002, 0.00125),
    "o3": (0.002, 0.008),
    "o4-mini": (0.0011, 0.0044),
    "o3-mini": (0.0011, 0.0044),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
}

LOCAL_VLLM_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class SpendingStep:
    num_tokens: int
    cost: float
    model: str
    role: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_tokens": self.num_tokens,
            "cost": round(self.cost, 6),
            "model": self.model,
            "role": self.role,
        }


def resolve_default_credentials() -> Dict[str, Optional[str]]:
    openrouter_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if openrouter_key:
        return {
            "api_key": openrouter_key,
            "base_url": (os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip(),
            "billing_mode": "openrouter",
        }
    return {
        "api_key": os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY_CUA"),
        "base_url": os.environ.get("OPENAI_BASE_URL"),
        "billing_mode": "openai",
    }


def resolve_billing_mode(base_url: Optional[str]) -> str:
    if not base_url:
        return "openai"
    host = (urlparse(base_url).hostname or "").lower()
    if "openrouter.ai" in host:
        return "openrouter"
    return "openai"


def is_local_vllm_endpoint(base_url: Optional[str]) -> bool:
    if not base_url:
        return False
    host = (urlparse(base_url).hostname or "").lower()
    return host in LOCAL_VLLM_HOSTS


def normalize_usage(usage: Any) -> Tuple[int, int, int]:
    if usage is None:
        return 0, 0, 0
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(mode="json")
    if not isinstance(usage, dict):
        return 0, 0, 0

    input_tokens = int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("prompt_token_count")
        or 0
    )
    output_tokens = int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("candidates_token_count")
        or usage.get("completion_token_count")
        or 0
    )
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    return input_tokens, output_tokens, total_tokens


def _resolve_openrouter_price_1k(model: Optional[str]) -> Optional[tuple[float, float]]:
    if not model:
        return None
    normalized = model.strip()
    candidates = [normalized]
    if "/" in normalized:
        candidates.append(normalized.split("/", 1)[1])
    for candidate in candidates:
        price = OPENROUTER_PRICE1K.get(candidate)
        if price is not None:
            return price
    return None


def compute_cost(
    model: Optional[str],
    usage: Any,
    *,
    billing_mode: str = "openai",
    base_url: Optional[str] = None,
    force_zero_cost: bool = False,
) -> float:
    if force_zero_cost or is_local_vllm_endpoint(base_url):
        return 0.0
    input_tokens, output_tokens, _total = normalize_usage(usage)
    if billing_mode == "openrouter":
        price_1k = _resolve_openrouter_price_1k(model)
        if price_1k is None:
            logger.warning("No OpenRouter price for model %s; falling back to OpenAI table.", model)
            cost = calculate_oai_model_cost(model, input_tokens, output_tokens)
            return 0.0 if cost is None else cost
        return (price_1k[0] * input_tokens + price_1k[1] * output_tokens) / 1000
    cost = calculate_oai_model_cost(model, input_tokens, output_tokens)
    return 0.0 if cost is None else cost


class SpendingSession:
    def __init__(self) -> None:
        self._steps: List[SpendingStep] = []

    @property
    def steps(self) -> List[SpendingStep]:
        return list(self._steps)

    def record(
        self,
        *,
        usage: Any,
        model: str,
        role: str,
        base_url: Optional[str] = None,
        billing_mode: Optional[str] = None,
        force_zero_cost: bool = False,
    ) -> SpendingStep:
        _input, _output, total_tokens = normalize_usage(usage)
        mode = billing_mode or resolve_billing_mode(base_url)
        cost = compute_cost(
            model,
            usage,
            billing_mode=mode,
            base_url=base_url,
            force_zero_cost=force_zero_cost,
        )
        step = SpendingStep(
            num_tokens=total_tokens,
            cost=cost,
            model=model or "unknown",
            role=role,
        )
        self._steps.append(step)
        self.log_step(step)
        return step

    def log_step(self, step: SpendingStep) -> None:
        logger.info(
            "Spending step role=%s model=%s tokens=%s cost=$%.6f",
            step.role,
            step.model,
            step.num_tokens,
            step.cost,
        )

    def log_total(self, label: str = "session") -> None:
        payload = self.to_spending_payload()
        total = payload["spending_total"]
        logger.info(
            "Spending %s total tokens=%s cost=$%.6f",
            label,
            total["num_tokens"],
            total["cost"],
        )

    def to_spending_payload(self) -> Dict[str, Any]:
        return _build_spending_payload(self._steps, group_totals_key=None)

    def extend(self, other: SpendingSession) -> None:
        self._steps.extend(other._steps)


def _sum_grouped(steps: Sequence[SpendingStep], key: str) -> List[Dict[str, Any]]:
    totals: Dict[str, Dict[str, Any]] = {}
    for step in steps:
        group_key = getattr(step, key, None) if hasattr(step, key) else None
        if group_key is None:
            continue
        bucket = totals.setdefault(str(group_key), {"num_tokens": 0, "cost": 0.0, key: group_key})
        bucket["num_tokens"] += step.num_tokens
        bucket["cost"] += step.cost
    result = []
    for bucket in totals.values():
        bucket["cost"] = round(bucket["cost"], 6)
        result.append(bucket)
    return result


def _build_spending_payload(
    steps: Sequence[SpendingStep],
    *,
    group_totals_key: Optional[str],
    group_meta_list: Optional[Sequence[Dict[str, Any]]] = None,
    annotated_steps: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if annotated_steps is None:
        annotated_steps = [step.to_dict() for step in steps]

    by_model: Dict[str, Dict[str, Any]] = {}
    by_role: Dict[str, Dict[str, Any]] = {}
    total_tokens = 0
    total_cost = 0.0
    for step in steps:
        total_tokens += step.num_tokens
        total_cost += step.cost
        model_bucket = by_model.setdefault(step.model, {"model": step.model, "num_tokens": 0, "cost": 0.0})
        model_bucket["num_tokens"] += step.num_tokens
        model_bucket["cost"] += step.cost
        role_bucket = by_role.setdefault(step.role, {"role": step.role, "num_tokens": 0, "cost": 0.0})
        role_bucket["num_tokens"] += step.num_tokens
        role_bucket["cost"] += step.cost

    for bucket in list(by_model.values()) + list(by_role.values()):
        bucket["cost"] = round(bucket["cost"], 6)

    payload: Dict[str, Any] = {
        "spending_per_step": annotated_steps,
        "spending_by_model": list(by_model.values()),
        "spending_by_role": list(by_role.values()),
        "spending_total": {"num_tokens": total_tokens, "cost": round(total_cost, 6)},
    }
    if group_totals_key and group_meta_list:
        payload[group_totals_key] = group_meta_list
    return payload


def aggregate_spending_payloads(
    children: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    *,
    group_totals_key: Optional[str] = None,
    annotate_step: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    all_steps: List[SpendingStep] = []
    annotated_steps: List[Dict[str, Any]] = []
    group_meta_list: List[Dict[str, Any]] = []

    for child_meta, spending in children:
        child_steps = spending.get("spending_per_step") or []
        child_total = spending.get("spending_total") or {"num_tokens": 0, "cost": 0.0}
        if group_totals_key:
            group_entry = {
                "num_tokens": int(child_total.get("num_tokens") or 0),
                "cost": round(float(child_total.get("cost") or 0.0), 6),
            }
            group_entry.update({k: v for k, v in child_meta.items() if k not in group_entry})
            group_meta_list.append(group_entry)

        for step_dict in child_steps:
            step = SpendingStep(
                num_tokens=int(step_dict.get("num_tokens") or 0),
                cost=float(step_dict.get("cost") or 0.0),
                model=str(step_dict.get("model") or "unknown"),
                role=str(step_dict.get("role") or "unknown"),
            )
            all_steps.append(step)
            if annotate_step:
                annotated_steps.append(annotate_step(child_meta, step.to_dict()))
            else:
                annotated_steps.append(step.to_dict())

    effective_group_key = group_totals_key if len(children) > 1 else None
    return _build_spending_payload(
        all_steps,
        group_totals_key=effective_group_key,
        group_meta_list=group_meta_list if effective_group_key else None,
        annotated_steps=annotated_steps,
    )


def merge_spending_into_metadata(metadata: Dict[str, Any], spending: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(metadata)
    payload["spending"] = spending
    return payload


def read_spending_from_metadata(path: Path | str) -> Optional[Dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return None
    try:
        with open(target, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    spending = payload.get("spending")
    if not isinstance(spending, dict):
        return None
    if not spending.get("spending_per_step") and not spending.get("spending_total"):
        return None
    return spending


def install_autogen_spending_hook(agent: Any, session: SpendingSession, role: str, base_url: Optional[str] = None) -> None:
    client = getattr(agent, "client", None)
    if client is None:
        return
    if getattr(client, "_spending_hook_installed", False):
        return

    original_create = client.create

    def wrapped_create(**config: Any) -> Any:
        response = original_create(**config)
        try:
            usage = client.get_usage(response)
            model = str(usage.get("model") or config.get("model") or "unknown")
            session.record(
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                model=model,
                role=role,
                base_url=base_url,
            )
        except Exception as exc:
            logger.debug("Failed to record autogen spending for role=%s: %s", role, exc)
        return response

    client.create = wrapped_create  # type: ignore[method-assign]
    client._spending_hook_installed = True  # type: ignore[attr-defined]
