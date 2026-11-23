"""Check source coverage and citations before publishing a draft."""

import asyncio
import json
import re
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.context.conversation import checked_messages
from app.agents.context.selection import evidence_prompt
from app.agents.evidence import citations, claims_from_draft
from app.agents.models import Budget, Evidence, Review, Subtask, TerminationReason
from app.agents.state import TaskState
from app.core.config import settings

_ANY_CITATION = re.compile(r"\[E-[^\]]+\]")


async def review_task(state: TaskState, model: BaseChatModel) -> dict[str, Any]:
    evidence = [Evidence.model_validate(item) for item in state.get("evidence", [])]
    ids = {item.id for item in evidence}
    plan = [Subtask.model_validate(item) for item in state.get("plan", [])]
    draft = state.get("draft", "")
    found = citations(draft)
    claims = claims_from_draft(draft)
    invalid = [item for item in _ANY_CITATION.findall(draft) if item[1:-1] not in ids]
    if draft.strip() and not found:
        invalid.append("missing_citations")
    for claim in claims:
        if not claim.evidence_ids and not claim.text.startswith(("未查到", "无法确认", "证据不足")):
            invalid.append("uncited_claim")
            break
    missing = [item.id for item in plan if not item.evidence_ids]
    uncited = [item.id for item in plan if item.evidence_ids and not set(item.evidence_ids).intersection(found)]
    budget = Budget.model_validate(state.get("budget", {}))
    unsupported: list[str] = []
    reason = "证据覆盖或引用不足" if not evidence or not found or invalid or missing or uncited else ""
    if len(draft) > settings.AGENT_MAX_DRAFT_CHARS:
        reason = "草稿超出原型完整审查长度，请缩短回答后重试"
    if not reason:
        if budget.model_calls >= budget.max_model_calls or budget.elapsed_seconds >= budget.max_seconds:
            reason = "审查模型预算已用尽"
        else:
            by_id = {item.id: item for item in evidence}
            cited_evidence = [by_id[item] for item in dict.fromkeys(found) if item in by_id]
            started = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    model.ainvoke(
                        checked_messages(
                            [
                                SystemMessage(
                                    content="你是证据审查器。逐条核对草稿的事实性断言是否被所引证据支持。"
                                    '仅输出 JSON：{"approved":true|false,"unsupported_claims":["原句"],"reason":"简述"}。'
                                    "来源仅有标题或相关词、不直接支持断言时判 false。用户偏好不是证据。"
                                ),
                                HumanMessage(
                                    content=f"草稿：\n{draft}\n\n所引证据：\n{evidence_prompt(cited_evidence, max_chars=None)}"
                                ),
                            ]
                        )
                    ),
                    timeout=max(0.1, budget.max_seconds - budget.elapsed_seconds),
                )
                budget.usage.append({"purpose": "review", "usage": response.usage_metadata})
                match = re.search(r"\{.*\}", str(response.content), re.DOTALL)
                parsed = json.loads(match.group() if match else "")
                if (
                    not isinstance(parsed, dict)
                    or parsed.get("approved") is not True
                    or parsed.get("unsupported_claims")
                ):
                    reason = (
                        str(parsed.get("reason") or "审查模型未批准") if isinstance(parsed, dict) else "审查结果无效"
                    )
                    unsupported = (
                        [str(item) for item in parsed.get("unsupported_claims", [])][:3]
                        if isinstance(parsed, dict)
                        else []
                    )
            except Exception as exc:
                reason = f"审查失败：{type(exc).__name__}"
            budget.model_calls += 1
            budget.elapsed_seconds += time.monotonic() - started
    review = Review(
        approved=not reason,
        missing_subtasks=[*missing, *uncited],
        invalid_citations=invalid,
        unsupported_claims=unsupported,
        reason=reason,
    )
    trace = [
        *state.get("trace", []),
        {
            "stage": "review",
            "approved": review.approved,
            "missing": [*missing, *uncited],
            "unsupported": len(unsupported),
        },
    ]
    result: dict[str, Any] = {
        "review": review.model_dump(),
        "claims": [item.model_dump() for item in claims],
        "trace": trace,
        "next_step": "finalize",
        "budget": budget.model_dump(),
    }
    if review.approved:
        result["termination"] = TerminationReason.COMPLETED.value
    elif budget.review_retries < budget.max_review_retries:
        budget.review_retries += 1
        result["budget"] = budget.model_dump()
        if missing and budget.model_calls < budget.max_model_calls and budget.tool_calls < budget.max_tool_calls:
            result["current_index"] = next(i for i, item in enumerate(plan) if item.id == missing[0])
            result["next_step"] = "execute"
        elif unsupported and budget.model_calls < budget.max_model_calls and budget.tool_calls < budget.max_tool_calls:
            plan.append(Subtask(id=f"S-R{budget.review_retries}", goal=f"核查：{unsupported[0][:300]}"))
            result["plan"] = [item.model_dump() for item in plan]
            result["current_index"] = len(plan) - 1
            result["next_step"] = "execute"
        elif evidence and budget.model_calls < budget.max_model_calls:
            result["next_step"] = "draft"
    else:
        result["termination"] = (
            TerminationReason.INSUFFICIENT_EVIDENCE.value if missing else TerminationReason.REVIEW_BUDGET.value
        )
    if not review.approved and result["next_step"] == "finalize":
        if budget.elapsed_seconds >= budget.max_seconds:
            result["termination"] = TerminationReason.TIME_BUDGET.value
        elif budget.model_calls >= budget.max_model_calls:
            result["termination"] = TerminationReason.MODEL_BUDGET.value
        elif budget.tool_calls >= budget.max_tool_calls and missing:
            result["termination"] = TerminationReason.TOOL_BUDGET.value
    return result
