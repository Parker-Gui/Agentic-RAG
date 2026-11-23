"""Conservative authorization for explicit response preferences, not general memory extraction."""

import re
from typing import Any

from app.agents.models import MemoryRequest, MemoryResult

_COMMAND = re.compile(r"^(?:请)?记住(?:我的(?:回答)?偏好)?[：:\s]*|^以后(?:回答我|回复我|回答|回复)时?请")
_MIXED = re.compile(
    r"(?:另外|此外|顺便|同时|并且|以及|然后|并)[，,：:\s]*(?:请)?(?:帮我)?(?:查|检索|研究|比较|分析|整理|生成|总结)"
)
_CANCEL = re.compile(r"(?:不要|别|不用|无需|不需要)(?:长期)?(?:保存|记住)|不保存")
_TEMPORARY = re.compile(r"这次|本次|暂时|仅此|只在今天")
# A small, deliberately closed vocabulary: recognized style words cannot smuggle
# a business assertion or an instruction about evidence/tool permissions into storage.
_STYLE_WORDS = (
    "先给结论",
    "先说结论",
    "结论优先",
    "先结论后分析",
    "先结论后解释",
    "再展开解释",
    "再给分析",
    "再解释",
    "先给摘要",
    "先摘要后正文",
    "分点回答",
    "使用列表",
    "分点列出",
    "用表格",
    "表格形式",
    "条目式",
    "简明扼要",
    "不要长篇大论",
    "简短一点",
    "详细一点",
    "尽量简短",
    "尽量详细",
    "简洁",
    "简短",
    "详细",
    "中文",
    "英文",
    "英语",
    "汉语",
    "简体中文",
    "繁体中文",
    "中英双语",
    "双语",
    "以后",
    "今后",
    "之后",
    "我的",
    "我",
    "给我",
    "回答",
    "回复",
    "答复",
    "简报",
    "报告",
    "输出",
    "请",
    "都",
    "默认",
    "尽量",
    "使用",
    "采用",
    "写成",
    "保持",
    "用",
    "时",
    "先",
    "再",
    "的",
    "以",
    "为主",
)
_STYLE = re.compile("|".join(re.escape(word) for word in sorted(_STYLE_WORDS, key=len, reverse=True)))
_STYLE_SIGNAL = re.compile(r"中文|英文|英语|汉语|双语|结论|摘要|分点|列表|表格|条目|简明|简洁|简短|详细|长篇大论")


def memory_only_hint(question: str) -> bool:
    return bool(_COMMAND.match(question.strip()) and not _MIXED.search(question))


def style_preference(text: str) -> bool:
    if not 1 <= len(text) <= 500 or not _STYLE_SIGNAL.search(text):
        return False
    remainder = _STYLE.sub("", text)
    return not re.sub(r"[\s，,。.!！；;、：:]+|并且|以及|同时|并|且|和", "", remainder)


def memory_decision(question: str, parsed: dict[str, Any]) -> dict[str, Any]:
    """The model proposes text; authority must be present in the current raw message."""
    raw = parsed.get("memory_request")
    goals = parsed.get("goals")
    has_goals = (
        raw is not None and isinstance(goals, list) and any(isinstance(item, str) and item.strip() for item in goals)
    )
    pure = memory_only_hint(question) and not has_goals
    if raw is None and not pure:
        return {"memory_only": False, "memory_request": None, "memory_result": None}
    result = MemoryResult(
        status="not_saved", message="未保存长期偏好，请明确写出希望长期记住的回答偏好。", reason="clarification"
    )
    try:
        request = MemoryRequest.model_validate(raw)
        command = _COMMAND.match(request.quote.strip())
        quote = request.quote.strip()
        text = request.text.strip()
        # Accept a direct command at the start, optionally after an explicit
        # additional request delimiter. Quoted/third-person prose is not authority.
        offset = question.find(quote)
        prefix = question[:offset].strip() if offset >= 0 else ""
        direct = offset >= 0 and (not prefix or bool(re.search(r"(?:另外|此外|顺便)[，,：:\s]*$", prefix)))
        if not command or not direct or (text not in quote[command.end() :] and text != quote):
            raise ValueError("No current-message authorization")
        body = _MIXED.split(quote[command.end() :], maxsplit=1)[0].strip(" ，,。.!！；;\n")
        if (
            _CANCEL.search(question)
            or _TEMPORARY.search(body)
            or not style_preference(body)
            or not style_preference(text)
            or parsed.get("memory_conflict") not in (False, None)
        ):
            raise ValueError("Unsupported or conflicting preference")
    except (ValueError, TypeError):
        return {"memory_only": pure, "memory_request": None, "memory_result": result.model_dump(mode="json")}
    return {
        "memory_only": pure,
        "memory_request": MemoryRequest(text=text, quote=quote).model_dump(),
        "memory_result": MemoryResult(status="pending", message="等待保存明确的个人偏好。").model_dump(mode="json"),
    }
