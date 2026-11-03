"""eval 聚合逻辑测试（注入假 query_fn / judge / run_agent，不连库/模型）。"""

import pytest

from eval.faithfulness_eval import GroundednessVerdict, evaluate_faithfulness
from eval.relevance_eval import AnswerRelevanceVerdict, ContextRelevanceVerdict, context_relevance_score
from eval.report import build_report, render_markdown, save_report
from eval.retrieval_eval import evaluate_retrieval
from eval.triad_eval import evaluate_triad


class TestEvaluateRetrieval:
    @pytest.mark.anyio
    async def test_aggregates_mean_metrics(self):
        dataset = [
            {"question": "q1", "expected_article_ids": [1]},
            {"question": "q2", "expected_article_ids": [9]},
        ]
        results = {"q1": [{"article_id": 1}, {"article_id": 2}], "q2": [{"article_id": 3}]}

        async def qf(q, k):
            return results[q]

        r = await evaluate_retrieval(dataset, query_fn=qf, k=5)
        # q1: gold{1} 命中在第1位 → recall1/mrr1/ndcg1；q2: gold{9} 未命中 → 全0；均值 0.5
        assert r["recall@k"] == 0.5
        assert r["mrr"] == 0.5
        assert r["n"] == 2

    @pytest.mark.anyio
    async def test_dedups_article_ids_across_chunks(self):
        async def qf(q, k):
            return [{"article_id": 5}, {"article_id": 5}, {"article_id": 1}]  # 同文章多块

        r = await evaluate_retrieval([{"question": "q", "expected_article_ids": [1]}], query_fn=qf, k=5)
        assert r["recall@k"] == 1.0

    @pytest.mark.anyio
    async def test_empty_dataset(self):
        async def qf(q, k):
            return []

        r = await evaluate_retrieval([], query_fn=qf, k=5)
        assert r["n"] == 0


class TestEvaluateFaithfulness:
    @pytest.mark.anyio
    async def test_groundedness_rate_with_fake_judge(self):
        dataset = [{"question": "q1"}, {"question": "q2"}]

        async def run_agent(q):
            return {"answer": q, "contexts": ["ctx"]}

        async def judge(answer, contexts):
            return GroundednessVerdict(grounded=(answer == "q1"))  # 只有 q1 判有据

        f = await evaluate_faithfulness(dataset, run_agent_fn=run_agent, judge=judge)
        assert f["groundedness_rate"] == 0.5
        assert f["n"] == 2


class TestContextRelevanceScore:
    """信噪比纯函数：相关条数 / 总条数，含边界与幻觉编号防护。"""

    def test_normal_ratio(self):
        v = ContextRelevanceVerdict(relevant_indices=[1, 3])
        assert context_relevance_score(v, 4) == 0.5  # 2/4

    def test_empty_contexts_returns_zero(self):
        v = ContextRelevanceVerdict(relevant_indices=[1])
        assert context_relevance_score(v, 0) == 0.0

    def test_out_of_range_indices_are_filtered(self):
        """LLM 可能返回越界/幻觉编号（99/0/负数），不能污染分数——只有 1、3 有效。"""
        v = ContextRelevanceVerdict(relevant_indices=[1, 3, 99, 0, -1])
        assert context_relevance_score(v, 3) == pytest.approx(2 / 3)

    def test_duplicate_indices_counted_once(self):
        v = ContextRelevanceVerdict(relevant_indices=[2, 2, 2])
        assert context_relevance_score(v, 4) == 0.25  # 去重 → {2} → 1/4


class TestEvaluateTriad:
    @pytest.mark.anyio
    async def test_aggregates_three_metrics_with_fake_judges(self):
        dataset = [{"question": "q1"}, {"question": "q2"}]

        async def run_agent(q):
            return {"answer": f"ans-{q}", "contexts": ["c1", "c2"]}

        async def faith_judge(answer, contexts):
            return GroundednessVerdict(grounded=(answer == "ans-q1"))  # 只有 q1 有据 → 0.5

        async def context_judge(question, contexts):
            return ContextRelevanceVerdict(relevant_indices=[1])  # 每题 1/2 → 0.5

        async def answer_judge(question, answer):
            return AnswerRelevanceVerdict(relevant=True)  # 全切题 → 1.0

        r = await evaluate_triad(
            dataset,
            run_agent_fn=run_agent,
            faith_judge=faith_judge,
            context_judge=context_judge,
            answer_judge=answer_judge,
        )
        assert r["faithfulness"] == 0.5
        assert r["context_relevance"] == 0.5
        assert r["answer_relevance"] == 1.0
        assert r["n"] == 2

    @pytest.mark.anyio
    async def test_details_carry_per_question_scores_and_reasons(self):
        async def run_agent(q):
            return {"answer": "a", "contexts": ["c1"]}

        async def faith_judge(a, c):
            return GroundednessVerdict(grounded=True, reason="有据")

        async def context_judge(q, c):
            return ContextRelevanceVerdict(relevant_indices=[1], reason="相关")  # 1/1 → 1.0

        async def answer_judge(q, a):
            return AnswerRelevanceVerdict(relevant=False, reason="跑题")

        r = await evaluate_triad(
            [{"question": "q1"}],
            run_agent_fn=run_agent,
            faith_judge=faith_judge,
            context_judge=context_judge,
            answer_judge=answer_judge,
        )
        d = r["details"][0]
        assert d["question"] == "q1"
        assert (d["faithfulness"], d["context_relevance"], d["answer_relevance"]) == (1.0, 1.0, 0.0)
        assert d["reasons"] == {"faithfulness": "有据", "context_relevance": "相关", "answer_relevance": "跑题"}

    @pytest.mark.anyio
    async def test_empty_dataset_short_circuits_without_judges(self):
        """空集不构造裁判（不连模型），直接返回零值三元组。"""

        async def run_agent(q):
            return {"answer": "", "contexts": []}

        r = await evaluate_triad([], run_agent_fn=run_agent)
        assert r["n"] == 0
        assert r["faithfulness"] == 0.0
        assert r["details"] == []


class TestReport:
    @staticmethod
    def _sample() -> tuple[dict, dict]:
        retrieval = {
            "dense": {"recall@k": 0.5, "mrr": 0.5, "ndcg@k": 0.4, "n": 2, "k": 5},
            "hybrid+rerank": {"recall@k": 0.8, "mrr": 0.75, "ndcg@k": 0.7, "n": 2, "k": 5},
        }
        triad = {
            "faithfulness": 0.5,
            "context_relevance": 0.75,
            "answer_relevance": 1.0,
            "n": 2,
            "details": [
                {
                    "question": "q1",
                    "faithfulness": 1.0,
                    "context_relevance": 1.0,
                    "answer_relevance": 1.0,
                    "reasons": {"faithfulness": "", "context_relevance": "", "answer_relevance": ""},
                },
                {
                    "question": "q2",
                    "faithfulness": 0.0,
                    "context_relevance": 0.5,
                    "answer_relevance": 1.0,
                    "reasons": {
                        "faithfulness": "编了个数字",
                        "context_relevance": "一半是噪声",
                        "answer_relevance": "",
                    },
                },
            ],
        }
        return retrieval, triad

    def test_build_report_structure(self):
        retrieval, triad = self._sample()
        rep = build_report(retrieval, triad)
        assert rep["n"] == 2
        assert rep["retrieval"] == retrieval
        assert rep["triad"] == {"faithfulness": 0.5, "context_relevance": 0.75, "answer_relevance": 1.0}
        assert "generated_at" in rep
        assert len(rep["details"]) == 2

    def test_render_markdown_has_sections_and_traces_low_score_reason(self):
        retrieval, triad = self._sample()
        md = render_markdown(build_report(retrieval, triad))
        assert "## 检索指标" in md
        assert "## 三元组" in md
        assert "hybrid+rerank" in md
        assert "## 低分明细" in md
        assert "编了个数字" in md  # 低分题的 judge reason 被追溯进报告

    def test_save_report_writes_json_and_md(self, tmp_path):
        import json

        retrieval, triad = self._sample()
        json_path, md_path = save_report(build_report(retrieval, triad), out_dir=tmp_path)

        assert json_path.exists() and md_path.exists()
        assert json_path.suffix == ".json" and md_path.suffix == ".md"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["triad"]["faithfulness"] == 0.5


class TestFlattenToolMessage:
    """ToolMessage → 逐条上下文。

    这层是 context_relevance 的分母来源：整串 JSON 当一条会让分母退化成"tool 调用
    次数"（通常是 1），裁判对一个 blob 只能二元表态、还容易幻觉出不存在的条目编号，
    分数被过滤成 1/1=1.000 的假满分。所以拆条不是美化，是指标能否成立的前提。
    """

    def _blob(self, n: int) -> str:
        import json

        recs = [
            {
                "article_id": 100 + i,
                "title": f"标题{i}",
                "url": f"http://u{i}",
                "published_at": "2026-06-18",
                "content": f"正文{i}",
                "score": 0.9,
                "source_kind": "internal_kb",
            }
            for i in range(n)
        ]
        return json.dumps({"query": "q", "total": n, "results": recs}, ensure_ascii=False)

    def test_splits_each_hit_into_its_own_context(self):
        from eval.run_eval import _flatten_tool_message

        out = _flatten_tool_message(self._blob(5))

        assert len(out) == 5
        assert out[0].startswith("标题0 | 2026-06-18 | http://u0")
        assert "正文0" in out[0]
        assert "internal_kb" not in out[0]  # 对"是否相关"无用的字段不喂裁判

    def test_tolerates_tool_round_stamp_suffix(self):
        """tool_node 会在 ToolMessage 末尾追加 "[tool round n/N]"，整串已非合法 JSON。

        这里退回原串就等于指标失真，所以必须用 raw_decode 而非 json.loads。
        """
        from eval.run_eval import _flatten_tool_message

        stamped = self._blob(3) + "\n\n[tool round 1/5]"

        assert len(_flatten_tool_message(stamped)) == 3

    def test_empty_results_yields_no_context(self):
        import json

        from eval.run_eval import _flatten_tool_message

        # 未命中该记 0 分，不能靠 fallback 塞一条"上下文"进去把分母凑成 1
        assert _flatten_tool_message(json.dumps({"results": [], "note": "库内无相关"})) == []

    def test_falls_back_to_raw_for_unknown_shape(self):
        import json

        from eval.run_eval import _flatten_tool_message

        assert _flatten_tool_message("not json at all") == ["not json at all"]
        assert _flatten_tool_message(json.dumps({"results": "not-a-list"})) == ['{"results": "not-a-list"}']
