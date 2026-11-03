"""RAG 评估三元组编排：一次 agent 运行喂三个 judge，聚合成三元组指标。

triad = Faithfulness（答案 vs 上下文）+ Context Relevance（上下文 vs 问题）+
Answer Relevance（答案 vs 问题）。agent 端到端很贵，所以每题只跑一次 agent，产出的
{answer, contexts} 同时喂三个 judge（三次评判并发）。judge 均可注入，便于单测。
"""

from __future__ import annotations

import asyncio
from typing import Any

from eval.faithfulness_eval import JudgeFn, RunAgentFn, _default_judge
from eval.relevance_eval import (
    AnswerJudgeFn,
    ContextJudgeFn,
    _default_answer_judge,
    _default_context_judge,
    context_relevance_score,
)

_EMPTY = {"faithfulness": 0.0, "context_relevance": 0.0, "answer_relevance": 0.0, "n": 0, "details": []}


async def evaluate_triad(
    dataset: list[dict[str, Any]],
    *,
    run_agent_fn: RunAgentFn,
    faith_judge: JudgeFn | None = None,
    context_judge: ContextJudgeFn | None = None,
    answer_judge: AnswerJudgeFn | None = None,
) -> dict[str, Any]:
    """对每题跑一次 agent → 三个 judge（并发）→ 三元组均值 + 逐题明细。"""
    if not dataset:
        return dict(_EMPTY)

    faith_judge = faith_judge or _default_judge()
    context_judge = context_judge or _default_context_judge()
    answer_judge = answer_judge or _default_answer_judge()

    tot_faith = tot_ctx = tot_ans = 0.0
    details: list[dict[str, Any]] = []

    for row in dataset:
        question = row["question"]
        out = await run_agent_fn(question)
        answer = out.get("answer", "")
        contexts = out.get("contexts", [])

        # 三个 judge 相互独立：每题只跑一次 agent，但三次评判并发
        faith_v, ctx_v, ans_v = await asyncio.gather(
            faith_judge(answer, contexts),
            context_judge(question, contexts),
            answer_judge(question, answer),
        )

        faith = float(faith_v.grounded)
        ctx = context_relevance_score(ctx_v, len(contexts))
        ans = float(ans_v.relevant)

        tot_faith += faith
        tot_ctx += ctx
        tot_ans += ans
        details.append(
            {
                "question": question,
                "faithfulness": faith,
                "context_relevance": round(ctx, 4),
                "answer_relevance": ans,
                "reasons": {
                    "faithfulness": faith_v.reason,
                    "context_relevance": ctx_v.reason,
                    "answer_relevance": ans_v.reason,
                },
            }
        )

    n = len(dataset)
    return {
        "faithfulness": tot_faith / n,
        "context_relevance": tot_ctx / n,
        "answer_relevance": tot_ans / n,
        "n": n,
        "details": details,
    }
