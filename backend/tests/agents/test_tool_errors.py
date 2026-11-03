"""agent_tools.errors 的单测：异常 → 与工具输出同形状的 JSON。

这层存在的理由是 ToolNode 的默认行为：异常被兜成一句裸报错文本塞进 ToolMessage，
而下游是把 ToolMessage 当检索结果读的。所以两件事都要钉住 ——
形状永远是 {"results": [], ...}，且参数校验失败必须带上「合法值是什么」，
否则模型看不出是自己传错了，只会原样重试。
"""

import json

import pytest
from pydantic import BaseModel, Field, ValidationError

from app.agents.agent_tools.errors import tool_error_to_json


class _Args(BaseModel):
    top_k: int = Field(ge=1, le=20)


def _validation_error() -> ValidationError:
    with pytest.raises(ValidationError) as exc_info:
        _Args(top_k=99)
    return exc_info.value


class TestValidationError:
    def test_reports_which_param_and_what_was_given(self):
        payload = json.loads(tool_error_to_json(_validation_error()))

        problem = payload["invalid_params"][0]
        assert problem["param"] == "top_k"
        assert problem["given"] == 99

    def test_message_carries_the_legal_bound(self):
        """pydantic 的 msg 自带合法区间，直接透给模型，别另造措辞。"""
        payload = json.loads(tool_error_to_json(_validation_error()))

        assert "less than or equal to 20" in payload["invalid_params"][0]["problem"]

    def test_says_the_call_did_not_run(self):
        """模型必须知道这次调用没执行，否则可能以为「查过了，库里没有」。"""
        payload = json.loads(tool_error_to_json(_validation_error()))

        assert "没有执行" in payload["error"]

    def test_literal_error_lists_the_allowed_values(self):
        from typing import Literal

        class _Filter(BaseModel):
            group_type: Literal["competitor", "regulatory"]

        with pytest.raises(ValidationError) as exc_info:
            _Filter(group_type="news")  # type: ignore[arg-type]
        payload = json.loads(tool_error_to_json(exc_info.value))

        problem = payload["invalid_params"][0]["problem"]
        assert "competitor" in problem
        assert "regulatory" in problem


class TestGenericError:
    def test_shape_matches_a_zero_hit_tool_result(self):
        payload = json.loads(tool_error_to_json(RuntimeError("connection refused")))

        assert payload["results"] == []
        assert "RuntimeError" in payload["error"]

    def test_warns_not_to_cite_the_error_text(self):
        payload = json.loads(tool_error_to_json(RuntimeError("boom")))

        assert "不要把这段文字当证据" in payload["note"]


class TestAlwaysUsable:
    @pytest.mark.parametrize(
        "exc",
        [RuntimeError("x"), ValueError("中文报错"), TimeoutError(), _validation_error()],
    )
    def test_output_is_always_json_with_empty_results(self, exc: Exception):
        payload = json.loads(tool_error_to_json(exc))

        assert payload["results"] == []
