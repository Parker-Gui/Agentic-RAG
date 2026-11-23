"""A safe final answer when review rejects a model draft."""

from typing import Any

from app.agents.evidence import source_location
from app.agents.models import Evidence, MemoryResult, Review, TerminationReason
from app.agents.state import TaskState


def finalize_task(state: TaskState) -> dict[str, Any]:
    if state.get("memory_only"):
        receipt = MemoryResult.model_validate(state["memory_result"])
        failed = receipt.reason in {"write_uncertain", "unavailable"}
        return {
            "answer": receipt.message,
            "termination": state.get("termination")
            or (TerminationReason.EXECUTION_ERROR.value if failed else TerminationReason.COMPLETED.value),
        }
    review = Review.model_validate(state["review"]) if state.get("review") else None
    if review and review.approved:
        return {"answer": state["draft"], "termination": TerminationReason.COMPLETED.value}
    evidence = [Evidence.model_validate(row) for row in state.get("evidence", [])]
    if not evidence:
        answer = "目前没有取得可引用的研究证据，无法可靠回答这个问题。"
    else:
        lines = ["已取得以下资料，但尚不足以形成经过完整审查的结论："]
        for item in evidence[:8]:
            lines.append(f"- [{item.id}] {item.title}：{item.content[:180]}（{source_location(item.model_dump())}）")
        answer = "\n".join(lines)
    return {
        "answer": answer,
        "termination": state.get("termination") or TerminationReason.INSUFFICIENT_EVIDENCE.value,
    }
