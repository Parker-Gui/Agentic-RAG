"""agent_tools 包：聚合 agent 可用的全部工具（内部 RAG + 外网 tavily）。

对外只暴露一个 `tools`——nodes.py 同时用它做 bind_tools 与 ToolNode，保证「模型被绑的工具」
和「实际执行的工具」是同一套，杜绝错配。以后加工具只需在这里往 `tools` 里追加。
"""

from app.agents.agent_tools.rag_tool import read_article_tool, search_rag_tool
from app.agents.agent_tools.tavily_tool import tavily_extract, tavily_search

# 内部情报库(rag) + 外网(tavily)
tools = [search_rag_tool, read_article_tool, tavily_search, tavily_extract]

__all__ = [
    "read_article_tool",
    "search_rag_tool",
    "tavily_extract",
    "tavily_search",
    "tools",
]
