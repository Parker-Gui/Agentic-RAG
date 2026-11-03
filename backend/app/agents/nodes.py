import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode

from app.agents.agent_tools import tools
from app.agents.agent_tools.errors import tool_error_to_json
from app.agents.providers import get_llm_provider
from app.agents.state import ToolUseState

logger = logging.getLogger(__name__)

llm = get_llm_provider().build_model()
# 绑定与执行用同一套 tools（来自 agent_tools 聚合），避免错配
llm_with_tools = llm.bind_tools(tools)

# 工具调用轮次上限，由 graph._budget_gate 强制执行（用满则转 finalize 收口）。
#
# 调大它之前先算一下 recursion_limit：一轮工具吃掉 3 个 super-step
# （reasoning → action → tools），最后还要一步 finalize，所以图最坏需要
# 3 * MAX_TOOL_ROUNDS + 1 步。langgraph 默认 recursion_limit=25，即上限约 8；
# 超了会在 finalize 之前抛 GraphRecursionError —— 答案已经生成也照样丢掉。
MAX_TOOL_ROUNDS = 5

DEFAULT_SYSTEM_PROMPT = (
    "你是一名专业的研究型助手，负责通过多轮“思考（Thought）→ 行动（Action）”"
    "来完成用户的任务。"
    "\n\n"
    "工具使用原则："
    "1. 优先使用内部知识库工具获取信息。"
    "内部知识库是你的首选信息来源。"
    "2. 只有当内部知识库无法提供足够的信息、信息明显缺失，或者用户明确要求"
    "获取最新的互联网信息时，才使用联网搜索进行外部网页搜索。"
    "3. 如果内部知识库已经提供了足够的信息，不要重复调用搜索工具，也不要为了"
    "补充无关信息而进行额外搜索。"
    "4. 如果需要多个来源，应优先通过内部知识库获取；只有在内部知识库不足时，"
    "再使用外部网页搜索进行补充。"
    "\n\n"
    "回答原则："
    "1. 始终使用中文进行 Thought、Action 和最终回答。"
    "2. Thought 必须简洁、明确，并说明当前已经掌握的信息以及下一步需要做什么。"
    "3. 每一轮只能执行一个行动：调用一个工具，或者直接给出最终答案。"
    "4. 最终回答必须基于已经获取的证据，不得凭空编造事实。"
    "5. 如果使用了外部网页搜索，应在最终回答中提供对应的来源 URL。"
)

THINK_INSTRUCTION = (
    "请严格按照以下要求生成本轮 Thought："
    "1. 必须只输出一个简短的中文段落。"
    "2. 必须以“Thought:”开头。"
    "3. 总结当前从用户问题、历史对话以及已有工具结果中获得的关键信息。"
    "4. 明确指出当前还缺少什么信息，以及下一步应该采取什么行动。"
    "5. 如果需要检索，优先考虑使用内部知识库 search_rag_tool；"
    "只有内部知识库不足时才考虑联网搜索。"
    "6. 本步骤只能进行思考，不得调用任何工具，也不得直接给出最终答案。"
)

ACT_INSTRUCTION = (
    "根据你刚刚生成的 Thought，严格执行一个行动："
    "1. 如果 Thought 判断需要获取信息，则只能调用一个工具，"
    "并且工具参数必须直接实现 Thought 中描述的行动。"
    "2. 如果需要检索信息，优先调用 内部知识库检索工具；"
    "只有当内部知识库无法满足需求时，才调用联网搜索。"
    "3. 如果当前已经拥有足够可靠的证据，则不要调用工具，直接生成最终答案。"
    "4. 最终答案必须使用中文。"
    "5. 如果使用了联网搜索获取外部信息，最终答案必须提供相关来源 URL。"
    "6. 每一轮只能执行一个工具调用，不能同时调用多个工具。"
)


async def reasoning_node(state: ToolUseState) -> dict:
    # Inline the instructions WITHOUT persisting them to state (message history
    # stays clean — only the actual Thought is kept).
    msgs = [
        SystemMessage(content=DEFAULT_SYSTEM_PROMPT),
        *state["messages"],
        HumanMessage(content=THINK_INSTRUCTION),
    ]
    # No tools bound — the model cannot make a tool call here.
    response = await llm.ainvoke(msgs)
    thought = AIMessage(
        content=str(response.content),
        additional_kwargs={"react_step": "thought"},
    )
    return {"messages": [thought]}


async def action_node(state: ToolUseState) -> dict:
    msgs = [
        SystemMessage(content=DEFAULT_SYSTEM_PROMPT),
        *state["messages"],
        HumanMessage(content=ACT_INSTRUCTION),
    ]
    response = await llm_with_tools.ainvoke(msgs)
    return {"messages": [response]}


FINAL_INSTRUCTION = (
    "工具调用预算已用尽，本轮必须直接给出最终答案，不得再请求任何检索："
    "1. 只依据上文已经取得的工具结果作答，不得凭记忆或推测补足事实。"
    "2. 如果证据不足以完整回答，明确写出哪一部分没有查到，不要编造。"
    "3. 使用中文作答；引用了外部网页结果时给出来源 URL。"
)


# 执行同一套 tools（内部 RAG + tavily）。
# handle_tool_errors 换成自定义翻译：默认行为是把异常兜成一句裸报错文本，而参数校验
# 失败（模型编了个非法 group_type）正是走这条路——裸文本里没有「合法值是什么」，
# 模型看不出是自己参数错了，只会原样重试。见 agent_tools/errors.py。
_tool_node = ToolNode(tools, handle_tool_errors=tool_error_to_json)


async def tool_node(state: ToolUseState, config: RunnableConfig) -> dict:
    """Run the tools, then stamp the round budget onto each tool result.

    控制流不变，只在 ToolMessage 末尾追加当前轮次/上限，让模型看到还剩多少预算。
    """
    result = await _tool_node.ainvoke(state, config)
    rounds = state.get("tool_rounds", 0) + 1
    messages = result["messages"]
    for msg in messages:
        if isinstance(msg, ToolMessage):
            msg.content = f"{msg.content}\n\n[tool round {rounds}/{MAX_TOOL_ROUNDS}]"
    return {"messages": messages, "tool_rounds": rounds}


async def finalize_node(state: ToolUseState) -> dict:
    """预算耗尽时的收口：强制基于已有证据作答。

    没有这个节点，预算耗尽只能直接跳 END，而此时最后一条消息可能是带 tool_calls 的
    AIMessage —— 既没有最终答案交给用户，历史里还留着一个没有 ToolMessage 应答的
    悬空 tool_call（checkpointer 存下来后，下一轮追加 human message 会让部分 provider
    直接 400）。所以走这里出一段正常答案，图才算真的收敛。
    """
    msgs = [
        SystemMessage(content=DEFAULT_SYSTEM_PROMPT),
        *state["messages"],
        HumanMessage(content=FINAL_INSTRUCTION),
    ]
    # 不绑 tools：这一步只能作答，不能再发起调用。
    response = await llm.ainvoke(msgs)
    return {"messages": [AIMessage(content=str(response.content))]}
