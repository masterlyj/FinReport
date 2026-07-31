"""DeepSeekSearcher._parse_message 的纯逻辑测试(用 SimpleNamespace mock Anthropic 响应)。

聚焦解析逻辑,不调真实 Anthropic API;重点验证 web_search_tool_result_error 过滤坑。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from company_report_kit.search_tools.deepseek_search import _parse_message


def _result(url: str, title: str = "t", page_age: str | None = None) -> SimpleNamespace:
    """构造单条 web_search_result block。"""
    return SimpleNamespace(
        type="web_search_result",
        url=url,
        title=title,
        page_age=page_age,
        encrypted_content="enc",
    )


def _error() -> SimpleNamespace:
    """构造无 url 的 error 条目(服务端搜索失败时混入)。"""
    return SimpleNamespace(type="web_search_tool_result_error")


def _text(text: str) -> SimpleNamespace:
    """构造 text block。"""
    return SimpleNamespace(type="text", text=text)


def _wsr(items: list[SimpleNamespace]) -> SimpleNamespace:
    """构造 web_search_tool_result block,内含命中条目(可混入 error)。"""
    return SimpleNamespace(type="web_search_tool_result", content=items)


def test_parse_extracts_last_text_as_answer() -> None:
    """多个 text block 时,最终答案取最后一个(过渡 text 被覆盖)。"""
    msg = SimpleNamespace(
        content=[_text("过渡"), _wsr([_result("https://a", "A")]), _text("最终答案")]
    )
    resp = _parse_message("q", msg)
    assert resp.answer == "最终答案"
    assert resp.query == "q"


def test_parse_filters_error_entries() -> None:
    """error 条目无 url,按 type 过滤,不进 sources。"""
    msg = SimpleNamespace(
        content=[_wsr([_result("https://a"), _error(), _result("https://b", "B")])]
    )
    resp = _parse_message("q", msg)
    # error 被过滤,只留两条真结果
    assert [s.url for s in resp.sources] == ["https://a", "https://b"]


def test_parse_maps_source_fields() -> None:
    """web_search_result 的 url/title/page_age 映射到 Source。"""
    msg = SimpleNamespace(content=[_wsr([_result("https://a", "A", page_age="2026-01-01")])])
    resp = _parse_message("q", msg)
    assert resp.sources[0].url == "https://a"
    assert resp.sources[0].title == "A"
    assert resp.sources[0].page_age == "2026-01-01"


def test_parse_no_text_answer_none() -> None:
    """无 text block 时 answer 为 None(只有来源无总结)。"""
    msg = SimpleNamespace(content=[_wsr([_result("https://a")])])
    resp = _parse_message("q", msg)
    assert resp.answer is None
    assert len(resp.sources) == 1


def _make_message() -> SimpleNamespace:
    """构造一条完整 Anthropic 响应:一个 web_search 结果 + 最终答案。"""
    return SimpleNamespace(
        content=[_wsr([_result("https://a", "A")]), _text("最终答案")]
    )


def test_search_with_mock_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """DeepSeekSearcher.search 走 mock Anthropic,验 tools 传入与 _parse_message 接线。"""
    from company_report_kit.search_tools import deepseek_search as mod

    mock_messages = MagicMock()
    mock_messages.create.return_value = _make_message()
    monkeypatch.setattr(mod.Anthropic, "__init__", lambda self, **_: None)
    monkeypatch.setattr(
        mod.Anthropic, "messages", SimpleNamespace(create=mock_messages.create)
    )
    searcher = mod.DeepSeekSearcher(api_key="sk-x")
    resp = searcher.search("query")
    # tools 透传 Anthropic 服务端 web_search 工具
    _, kwargs = mock_messages.create.call_args
    assert kwargs["tools"] == [mod._WEB_SEARCH_TOOL]
    assert kwargs["messages"] == [{"role": "user", "content": "query"}]
    # 结果走 _parse_message:答案与来源均映射
    assert resp.answer == "最终答案"
    assert resp.sources[0].url == "https://a"


def test_get_default_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """_get_default 返回模块级单例,重复调用同对象(复用连接池)。"""
    from company_report_kit.search_tools import deepseek_search as mod

    monkeypatch.setattr(mod.Anthropic, "__init__", lambda self, **_: None)
    monkeypatch.setattr(mod, "_default", None)
    a = mod._get_default()
    b = mod._get_default()
    assert a is b
