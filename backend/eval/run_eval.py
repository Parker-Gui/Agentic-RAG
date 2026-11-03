"""评测入口：uv run python -m eval.run_eval

前提：DB 已迁移(alembic upgrade head) + 有已索引数据 + 模型可用 + eval/dataset.jsonl 标注好。
对 dense-only / hybrid / hybrid+rerank 三配置各跑检索指标，再跑一次端到端 faithfulness，打印对比。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATASET = Path(__file__).parent / "dataset.jsonl"


def load_dataset(path: Path = DATASET) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _query_fn(hybrid: bool, rerank: bool):
    """构造某配置下的 query_fn：临时置 settings 开关后走 RagService.query。"""

    async def qf(question: str, k: int) -> list[dict[str, Any]]:
        from app.core.config import settings
        from app.db.session import get_worker_db_context
        from app.services.rag.facade import RagService

        settings.RAG_HYBRID_ENABLED = hybrid
        settings.RAG_RERANK_ENABLED = rerank
        async with get_worker_db_context() as db:
            return await RagService(db).query(question, top_k=k)

    return qf


def _flatten_tool_message(raw: str) -> list[str]:
    """一条 ToolMessage 的 JSON → 一条条可读上下文（按 results[*] 拆）。

    不能把 ToolMessage.content 整串当一条上下文：context_relevance 的分母是
    len(contexts)，整串塞进去分母就变成"tool 调用次数"（通常是 1），裁判只能对一个
    JSON blob 二元表态，还极易得 1/1=1.000——而该指标本意是信噪比
    （相关片段数 / 召回片段数）。所以按片段拆条，并只留裁判要看的可读字段，
    丢掉 score/source_kind/article_id 这类对"是否相关"无用的噪声。

    非本项目 schema（如 tavily）或解析失败时退回原串，宁可粗一点也不丢上下文；
    results 为空（未命中/报错）则不产出上下文——"没检索到"就该记 0，不是满分。
    """
    # 用 raw_decode 而非 json.loads：tool_node 会在 ToolMessage 末尾追加
    # "\n\n[tool round n/N]" 预算戳，整串已不是合法 JSON。raw_decode 只解析前导
    # JSON 值、忽略尾部附加物，图以后再追加别的后缀也不会把这里打回 fallback。
    try:
        payload, _ = json.JSONDecoder().raw_decode(raw.lstrip())
    except (json.JSONDecodeError, TypeError):
        return [raw]
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return [raw]

    out: list[str] = []
    for rec in payload["results"]:
        if not isinstance(rec, dict):
            out.append(str(rec))
            continue
        head = " | ".join(str(rec[f]) for f in ("title", "published_at", "url") if rec.get(f))
        body = str(rec.get("content") or "").strip()
        if not body:
            continue
        out.append(f"{head}\n{body}" if head else body)
    return out


async def _run_agent(question: str, idx: int) -> dict[str, Any]:
    """跑整张 agent 图，抽出最终答案 + 检索到的上下文(ToolMessage)。"""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from app.agents.graph import graph

    out = await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": f"eval-{idx}"}},
    )
    msgs = out["messages"]
    answer = next((str(m.content) for m in reversed(msgs) if isinstance(m, AIMessage) and str(m.content).strip()), "")
    # 多轮 react 会重复召回同一片段，去重保序：否则重复项同时抬高分子和分母，虚化信噪比
    contexts: list[str] = []
    seen: set[str] = set()
    for m in msgs:
        if not isinstance(m, ToolMessage):
            continue
        for ctx in _flatten_tool_message(str(m.content)):
            if ctx not in seen:
                seen.add(ctx)
                contexts.append(ctx)
    return {"answer": answer, "contexts": contexts}


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from eval.report import build_report, save_report
    from eval.retrieval_eval import evaluate_retrieval
    from eval.triad_eval import evaluate_triad

    dataset = load_dataset()
    logger.info("评估集 %d 条", len(dataset))

    retrieval_results: dict[str, dict[str, Any]] = {}
    print("=== 检索指标 (recall@5 / MRR / nDCG@5) ===")
    for name, hybrid, rerank in [("dense", False, False), ("hybrid", True, False), ("hybrid+rerank", True, True)]:
        r = await evaluate_retrieval(dataset, query_fn=_query_fn(hybrid, rerank), k=5)
        retrieval_results[name] = r
        print(f"  {name:14} recall={r['recall@k']:.3f}  mrr={r['mrr']:.3f}  ndcg={r['ndcg@k']:.3f}  (n={r['n']})")

    print("=== 三元组 (agent 端到端): faithfulness / context_rel / answer_rel ===")
    counter = {"i": 0}

    async def run_agent_fn(q: str) -> dict[str, Any]:
        counter["i"] += 1
        return await _run_agent(q, counter["i"])

    t = await evaluate_triad(dataset, run_agent_fn=run_agent_fn)
    print(
        f"  faithfulness={t['faithfulness']:.3f}  "
        f"context_rel={t['context_relevance']:.3f}  "
        f"answer_rel={t['answer_relevance']:.3f}  (n={t['n']})"
    )

    json_path, md_path = save_report(build_report(retrieval_results, t))
    print(f"\n报告已写入:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    from app.db.session import run_compatible_async

    run_compatible_async(main())
