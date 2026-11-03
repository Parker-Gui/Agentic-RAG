"""相关性评测：Context Relevance + Answer Relevance（LLM-judge）。

评估三元组里除 faithfulness 外的两项：
- Context Relevance：检索到的上下文与问题相关吗（信噪比 = 相关条数 / 总条数）。
- Answer Relevance：答案切题回答了问题吗（≠有据性，那是 faithfulness）。

judge 都可注入，便于用假实现单测（不连模型）。中立裁判构造复用 judge_base.build_judge_model。
数据集聚合与 agent 运行不在这里——由 triad_eval 统一编排（一次 agent 运行喂三个 judge）。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from eval.judge_base import build_judge_model

# judge(question, contexts) -> ContextRelevanceVerdict
ContextJudgeFn = Callable[[str, list[str]], Awaitable["ContextRelevanceVerdict"]]
# judge(question, answer) -> AnswerRelevanceVerdict
AnswerJudgeFn = Callable[[str, str], Awaitable["AnswerRelevanceVerdict"]]


# ============ Context Relevance ============

CTX_JUDGE_PROMPT = (
    "你在评估检索质量。逐条判断<上下文>是否与回答<问题>相关（能提供有用信息），"
    "返回相关条目的编号列表。宁缺毋滥：只沾边、答不了问题的不算相关。"
)


class ContextRelevanceVerdict(BaseModel):
    relevant_indices: list[int] = Field(default_factory=list, description="与回答问题相关的上下文编号（从1开始）")
    reason: str = Field(default="", description="一句话依据")


def _default_context_judge() -> ContextJudgeFn:
    from langchain_core.messages import HumanMessage, SystemMessage

    model = build_judge_model(ContextRelevanceVerdict)

    async def judge(question: str, contexts: list[str]) -> ContextRelevanceVerdict:
        ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts)) or "（无上下文）"
        payload = f"<问题>\n{question}\n</问题>\n\n<上下文>\n{ctx}\n</上下文>"
        result = await model.ainvoke([SystemMessage(content=CTX_JUDGE_PROMPT), HumanMessage(content=payload)])
        return result if isinstance(result, ContextRelevanceVerdict) else ContextRelevanceVerdict.model_validate(result)

    return judge


def context_relevance_score(verdict: ContextRelevanceVerdict, n_contexts: int) -> float:
    """信噪比：相关条数 / 总条数。空上下文→0；过滤越界/幻觉编号，避免污染分数。"""
    if n_contexts <= 0:
        return 0.0
    valid = {i for i in verdict.relevant_indices if 1 <= i <= n_contexts}
    return len(valid) / n_contexts


# ============ Answer Relevance ============

ANS_JUDGE_PROMPT = (
    "你在评估答案是否切题。判断<答案>是否直接回答了<问题>（不跑题、不空泛、不答非所问）。"
    "只看切题与否——语气、详略、以及事实是否有据都不作为依据（有据性由另一环负责）。"
)


class AnswerRelevanceVerdict(BaseModel):
    relevant: bool = Field(description="答案是否直接、切题地回答了问题")
    reason: str = Field(default="", description="一句话依据")


def _default_answer_judge() -> AnswerJudgeFn:
    from langchain_core.messages import HumanMessage, SystemMessage

    model = build_judge_model(AnswerRelevanceVerdict)

    async def judge(question: str, answer: str) -> AnswerRelevanceVerdict:
        payload = f"<问题>\n{question}\n</问题>\n\n<答案>\n{answer}\n</答案>"
        result = await model.ainvoke([SystemMessage(content=ANS_JUDGE_PROMPT), HumanMessage(content=payload)])
        return result if isinstance(result, AnswerRelevanceVerdict) else AnswerRelevanceVerdict.model_validate(result)

    return judge
