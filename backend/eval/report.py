"""评估报告产物化：把检索指标 + 三元组结果落成 JSON + Markdown，供追溯与展示。

否则 run_eval 只在终端打印汇总 rate，逐题明细与 judge 的 reason 算了就丢——指标低时
无法追溯是哪一题、裁判给了什么理由。

build_report / render_markdown 是纯函数（可单测）；save_report 做文件 IO。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORTS_DIR = Path(__file__).parent / "reports"

_TRIAD_KEYS = ("faithfulness", "context_relevance", "answer_relevance")


def build_report(retrieval: dict[str, dict[str, Any]], triad: dict[str, Any]) -> dict[str, Any]:
    """组装结构化报告：检索三配置指标 + 三元组均值 + 逐题明细（含 reason）。"""
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "n": triad.get("n", 0),
        "retrieval": retrieval,
        "triad": {k: triad[k] for k in _TRIAD_KEYS if k in triad},
        "details": triad.get("details", []),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """把报告渲染成人读的 Markdown（检索对比表 + 三元组 + 逐题 + 低分明细）。"""
    out: list[str] = ["# RAG 评估报告", ""]
    out.append(f"- 生成时间: {report.get('generated_at', '')}")
    out.append(f"- 评估集: {report.get('n', 0)} 条")
    out.append("")

    out.append("## 检索指标 (recall@5 / MRR / nDCG@5)")
    out.append("")
    out.append("| 配置 | recall@5 | MRR | nDCG@5 |")
    out.append("| --- | --- | --- | --- |")
    for name, m in report.get("retrieval", {}).items():
        out.append(f"| {name} | {m['recall@k']:.3f} | {m['mrr']:.3f} | {m['ndcg@k']:.3f} |")
    out.append("")

    triad = report.get("triad", {})
    out.append("## 三元组 (agent 端到端)")
    out.append("")
    out.append("| 指标 | 值 |")
    out.append("| --- | --- |")
    out.append(f"| Faithfulness | {triad.get('faithfulness', 0.0):.3f} |")
    out.append(f"| Context Relevance | {triad.get('context_relevance', 0.0):.3f} |")
    out.append(f"| Answer Relevance | {triad.get('answer_relevance', 0.0):.3f} |")
    out.append("")

    details = report.get("details", [])
    if details:
        out.append("## 逐题明细")
        out.append("")
        out.append("| # | 问题 | faith | ctx_rel | ans_rel |")
        out.append("| --- | --- | --- | --- | --- |")
        for i, d in enumerate(details, 1):
            q = str(d.get("question", "")).replace("|", "\\|")[:60]
            out.append(
                f"| {i} | {q} | {d['faithfulness']:.1f} | {d['context_relevance']:.2f} | {d['answer_relevance']:.1f} |"
            )
        out.append("")

        low = [d for d in details if d["faithfulness"] < 1 or d["context_relevance"] < 1 or d["answer_relevance"] < 1]
        if low:
            out.append("## 低分明细（可追溯 reason）")
            out.append("")
            for d in low:
                reasons = d.get("reasons", {})
                out.append(f"- **{d.get('question', '')}**")
                out.append(f"  - faithfulness={d['faithfulness']:.1f} — {reasons.get('faithfulness', '')}")
                out.append(
                    f"  - context_relevance={d['context_relevance']:.2f} — {reasons.get('context_relevance', '')}"
                )
                out.append(f"  - answer_relevance={d['answer_relevance']:.1f} — {reasons.get('answer_relevance', '')}")
            out.append("")

    return "\n".join(out)


def save_report(report: dict[str, Any], out_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    """把报告写成 report_<UTC时间戳>.json + .md，返回两个路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"report_{stamp}.json"
    md_path = out_dir / f"report_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
