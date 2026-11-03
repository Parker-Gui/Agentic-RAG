"""评估三元组三个 LLM-judge 共用的裁判模型构造器。

集中"中立裁判"逻辑：RAG_EVAL_JUDGE_PROVIDER/MODEL 指定裁判 provider/模型（留空则退回生成
用的 provider/默认模型），temperature=0 求判定可复现。faithfulness / context relevance /
answer relevance 三个 judge 都用它，避免样板重复、口径一致。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
    from pydantic import BaseModel


def build_judge_model(output_schema: type[BaseModel]) -> Runnable:
    """构造中立裁判 model：结构化输出（with_structured_output）+ temperature=0。"""
    from app.agents.providers import get_llm_provider
    from app.core.config import settings

    provider = get_llm_provider(settings.RAG_EVAL_JUDGE_PROVIDER)
    return provider.build_model(settings.RAG_EVAL_JUDGE_MODEL, temperature=0).with_structured_output(output_schema)
