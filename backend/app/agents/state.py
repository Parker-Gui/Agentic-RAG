from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_action: str
    iterations_count: int

class ToolUseState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_rounds: NotRequired[int]
