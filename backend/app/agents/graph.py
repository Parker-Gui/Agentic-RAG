from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from app.agents.nodes import (
    MAX_TOOL_ROUNDS,
    action_node,
    finalize_node,
    reasoning_node,
    tool_node,
)
from app.agents.state import ToolUseState

checkpointer = InMemorySaver()


def _budget_gate(state: ToolUseState) -> str:
    """工具轮次预算门：没用完继续想，用完了收口作答。

    这道门必须放在 tools 之后、而不是 action 之后：action 一旦跑完，那条带 tool_calls
    的 AIMessage 已经进了 state，此时改判去 finalize 会在历史里留下一个没有 ToolMessage
    应答的悬空 tool_call。放在 tools 之后，每个 tool_call 都拿到了结果，历史始终自洽 ——
    语义也更合理：允许用满 MAX_TOOL_ROUNDS 轮，最后一轮的结果照样能进最终答案。

    在这之前 MAX_TOOL_ROUNDS 只是被打印进 ToolMessage 的一句文本，没有任何控制流读它，
    所以模型可以无限 reasoning → action → tools 循环下去，直到撞 recursion_limit 抛错。
    """
    if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
        return "finalize"
    return "reasoning"


def build_graph():
    workflow = StateGraph(ToolUseState)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("action", action_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "reasoning")
    workflow.add_edge("reasoning", "action")
    # 模型自己决定不调工具时（tools_condition → END），说明它已经给出最终答案，直接收。
    workflow.add_conditional_edges("action", tools_condition, {"tools": "tools", END: END})
    workflow.add_conditional_edges(
        "tools",
        _budget_gate,
        {"reasoning": "reasoning", "finalize": "finalize"},
    )
    workflow.add_edge("finalize", END)
    return workflow.compile(checkpointer=checkpointer)


graph = build_graph()
