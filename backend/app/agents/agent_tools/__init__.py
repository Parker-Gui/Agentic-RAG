"""agent_tools 包：聚合 agent 可用的全部工具（内部 RAG + 外网 tavily）。

对外只暴露一个 `tools`，由 executor.py 注册到 Deep Agents，并在装配后核对工具名单。
旧图保留在 legacy_nodes.py 供对比。
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
