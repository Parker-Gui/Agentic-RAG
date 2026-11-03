"""端到端 faithfulness 评测：判"答案是否被检索到的上下文支撑"。

原设计复用 app/agents/audit.py，但该模块已随 ReAct 重构移除，故改为 **LLM-judge**：
对每题跑 agent 图 → 拿最终答案 + 检索到的上下文 → 裁判模型判有据性 → 聚合成 groundedness_rate。

judge 与 run_agent_fn 都可注入，便于用假实现单测（不连模型/图）。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

# run_agent_fn(question) -> {"answer": str, "contexts": list[str]}
RunAgentFn = Callable[[str], Awaitable[dict[str, Any]]]
# judge(answer, contexts) -> GroundednessVerdict
JudgeFn = Callable[[str, list[str]], Awaitable["GroundednessVerdict"]]

JUDGE_PROMPT = (
    "你是事实核查员。只依据<上下文>判断<答案>里的事实性陈述是否都能被支撑："
    "存在无法从上下文得到支撑的事实断言 → grounded=false，否则 true。"
    "语气、详略、排版不作为依据。"
)


class GroundednessVerdict(BaseModel):
    grounded: bool = Field(description="答案的事实性内容是否都被上下文支撑")
    reason: str = Field(default="", description="一句话依据")


def _default_judge() -> JudgeFn:
    """默认裁判：中立 provider/模型 + temperature=0（可复现）做结构化判定。

    RAG_EVAL_JUDGE_PROVIDER/MODEL 留空则退回生成用的 provider/默认模型；配置了就用中立裁判
    （如 bedrock + Claude），缓解"生成=评判同模型"的自评偏差。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from eval.judge_base import build_judge_model

    model = build_judge_model(GroundednessVerdict)

    async def judge(answer: str, contexts: list[str]) -> GroundednessVerdict:
        ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts)) or "（无上下文）"
        payload = f"<上下文>\n{ctx}\n</上下文>\n\n<答案>\n{answer}\n</答案>"
        result = await model.ainvoke([SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=payload)])
        return result if isinstance(result, GroundednessVerdict) else GroundednessVerdict.model_validate(result)

    return judge


async def evaluate_faithfulness(
    dataset: list[dict[str, Any]],
    *,
    run_agent_fn: RunAgentFn,
    judge: JudgeFn | None = None,
) -> dict[str, Any]:
    """对每题跑 agent → 裁判判有据性 → 返回 groundedness_rate 与逐题明细。"""
    judge = judge or _default_judge()
    if not dataset:
        return {"groundedness_rate": 0.0, "n": 0, "details": []}

    grounded = 0
    details: list[dict[str, Any]] = []
    for row in dataset:
        out = await run_agent_fn(row["question"])
        verdict = await judge(out.get("answer", ""), out.get("contexts", []))
        grounded += int(verdict.grounded)
        details.append({"question": row["question"], "grounded": verdict.grounded, "reason": verdict.reason})

    return {"groundedness_rate": grounded / len(dataset), "n": len(dataset), "details": details}
