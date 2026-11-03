"""graph._budget_gate 的单测：工具轮次预算真的会让图收敛。

在接上这道门之前，MAX_TOOL_ROUNDS 只是被打印进 ToolMessage 的一句文本，没有任何
控制流读它 —— 模型可以无限 reasoning → action → tools 循环，最后撞 recursion_limit
抛异常（实测复现过：5 轮全部零命中，然后 GraphRecursionError）。

纯状态函数，不需要 LLM，也不需要跑图。
"""

import pytest

from app.agents.graph import _budget_gate
from app.agents.nodes import MAX_TOOL_ROUNDS


class TestBudgetGate:
    @pytest.mark.parametrize("rounds", [0, 1, MAX_TOOL_ROUNDS - 1])
    def test_under_budget_keeps_reasoning(self, rounds: int):
        assert _budget_gate({"messages": [], "tool_rounds": rounds}) == "reasoning"

    @pytest.mark.parametrize("rounds", [MAX_TOOL_ROUNDS, MAX_TOOL_ROUNDS + 1])
    def test_at_or_over_budget_goes_to_finalize(self, rounds: int):
        """用满就收口。>= 而非 ==：万一 tool_rounds 被并行分支多加过，也不能漏判。"""
        assert _budget_gate({"messages": [], "tool_rounds": rounds}) == "finalize"

    def test_missing_tool_rounds_counts_as_zero(self):
        """tool_rounds 是 NotRequired —— 首轮进来时 state 里没这个键。"""
        assert _budget_gate({"messages": []}) == "reasoning"


class TestGraphWiring:
    def test_tools_never_edges_straight_back_to_reasoning(self):
        """tools 的出边必须经过这道门；直连 reasoning 就等于没有预算。"""
        from app.agents.graph import build_graph

        edges = build_graph().get_graph().edges
        tool_targets = {e.target for e in edges if e.source == "tools"}

        assert tool_targets == {"reasoning", "finalize"}

    def test_finalize_terminates(self):
        from langgraph.graph import END

        from app.agents.graph import build_graph

        edges = build_graph().get_graph().edges
        finalize_targets = {e.target for e in edges if e.source == "finalize"}

        assert finalize_targets == {END}
