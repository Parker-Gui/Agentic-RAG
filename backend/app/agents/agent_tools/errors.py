"""工具异常 → 与工具输出同形状的 JSON。

ToolNode 默认把异常兜成一句裸报错文本塞进 ToolMessage。那条消息接着会被
reasoning 节点当检索结果读、被 critic_tool._parse_tool_content 解析成「一条内容是
该报错文案的证据」。所以这里统一翻译成 {"results": [], "error": ...}——0 条结果，
和 rag_tool / tavily_tool 成功时的形状一致。

参数校验失败要额外给出「合法值是什么」。参数是模型采样出来的，它传错了以后唯一的
线索就是这条 ToolMessage；只说「invalid」它没法自纠，只会原样重试。
"""

import json
from typing import Any

from pydantic import ValidationError


def _param_problems(exc: ValidationError) -> list[dict[str, Any]]:
    """pydantic 报错 → 每个参数一条 {param, given, problem}。

    msg 本身就含合法值（literal_error 的 msg 是 "Input should be 'competitor' or
    'regulatory'"），直接透给模型，不另造措辞。
    """
    problems: list[dict[str, Any]] = []
    for err in exc.errors():
        problems.append(
            {
                "param": ".".join(str(part) for part in err["loc"]) or "(root)",
                "given": err.get("input"),
                "problem": err["msg"],
            }
        )
    return problems


def tool_error_to_json(exc: Exception) -> str:
    """ToolNode(handle_tool_errors=...) 用的翻译函数。"""
    if isinstance(exc, ValidationError):
        problems = _param_problems(exc)
        payload = {
            "results": [],
            "error": "工具参数不合法，本次调用没有执行。",
            "invalid_params": problems,
            "note": "请按 problem 里给出的合法取值改参数后重试；不要重复原样调用。",
        }
    else:
        payload = {
            "results": [],
            "error": f"工具执行失败：{type(exc).__name__}: {exc}",
            "note": "这是工具故障，不是检索结果；不要把这段文字当证据引用。",
        }
    return json.dumps(payload, ensure_ascii=False)
