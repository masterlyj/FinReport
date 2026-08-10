"""pytest 公共 fixture:key 检测与 mock 工具,供 network/live 测试复用。

  FakeModel — configurable_model 的替身。图节点链式调用
    configurable_model.bind_tools(...).with_retry(...).with_config(...)
    再 .ainvoke(...);FakeModel 把所有链式方法收成 self,ainvoke 吐预设响应,
    使节点逻辑可在不触真实 LLM 的前提下被测。
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def skip_without_deepseek_key() -> None:
    """无 DEEPSEEK_API_KEY 时跳过 live 测试。"""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")


@pytest.fixture
def skip_without_tavily_key() -> None:
    """无 TAVILY_API_KEY 时跳过 live 测试。"""
    if not os.environ.get("TAVILY_API_KEY"):
        pytest.skip("TAVILY_API_KEY not set")


class FakeModel:
    """configurable_model 的替身,供图节点单测注入。

    所有链式方法(bind_tools/with_structured_output/with_retry/with_config)
    返回 self;ainvoke 按 responses 队列依次返回或抛异常。

    Args:
        responses: ainvoke 依次返回/抛出的项;为 AIMessage 则返回,为 Exception
            实例则抛(用于 final_report 的 token 超限重试场景)。队列耗尽后
            返回 last(无则 None)。
        default: responses 为空时的默认返回。

    用法:
        monkeypatch.setattr(
            "company_report_kit.graph.nodes.configurable_model", FakeModel(resp)
        )
    """

    def __init__(
        self,
        responses: list | None = None,
        default: object | None = None,
    ) -> None:
        self._responses = list(responses) if responses else []
        self._default = default
        # 记录每次 ainvoke 收到的 messages,供断言 prompt 拼装
        self.invocations: list = []

    def bind_tools(self, _tools, **_kw):
        return self

    def with_structured_output(self, _schema, **_kw):
        return self

    def with_retry(self, **_kw):
        return self

    def with_config(self, _config):
        return self

    async def ainvoke(self, messages, **_kw):
        self.invocations.append(messages)
        if self._responses:
            item = self._responses.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        return self._default
