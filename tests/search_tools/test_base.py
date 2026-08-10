"""search_tools base 的纯逻辑测试:Source / SearchResponse / format_for_agent。"""

from __future__ import annotations

from company_report_kit.search_tools.base import (
    SearchResponse,
    Source,
    format_for_agent,
    normalize_url,
)


def test_source_defaults() -> None:
    """Source 仅 url 必填,其余字段默认 None。"""
    s = Source(url="https://a")
    assert s.url == "https://a"
    assert s.title is None and s.page_age is None
    assert s.content is None and s.raw_content is None


def test_normalize_url_strips_drift_differences() -> None:
    """归一化去除协议/www/尾斜杠/查询参数/fragment,微漂移 URL 归到同一键。"""
    # 协议 + www + 尾斜杠 + 查询参数 + fragment 差异
    assert normalize_url("https://www.example.com/a/b/?tab=2#x") == "example.com/a/b"
    assert normalize_url("http://example.com/a/b") == "example.com/a/b"
    # 大小写 host 归一
    assert normalize_url("https://EXAMPLE.com/A/B") == "example.com/A/B"
    # 尾斜杠去除
    assert normalize_url("https://example.com/path/") == "example.com/path"


def test_normalize_url_strips_trailing_punct() -> None:
    """去除 URL 后的中文/英文标点(句号/逗号/分号),防脚注正则捕获尾标点。"""
    assert normalize_url("https://example.com/cash。") == "example.com/cash"
    assert normalize_url("https://example.com/cash.") == "example.com/cash"


def test_normalize_url_keeps_path_parens_strips_markdown_close() -> None:
    """URL 内配对的括号保留(Wikipedia 消歧义),markdown 闭合 `)` 剥离。"""
    # Wikipedia 消歧义 `Film_(2024)`:括号是 URL 的一部分,保留
    assert normalize_url("https://en.wikipedia.org/wiki/Film_(2024)") == "en.wikipedia.org/wiki/Film_(2024)"
    # markdown 闭合 `)`(被 urlsplit 误解析进 host):剥离
    assert normalize_url("https://a.com)") == "a.com"


def test_format_for_agent_answer_and_content() -> None:
    """有 answer + content 时,输出总结、来源标题、摘要片段。"""
    resp = SearchResponse(
        query="天气",
        answer="晴",
        sources=[Source(url="https://a", title="A", content="片段")],
    )
    out = format_for_agent(resp)
    assert "晴" in out
    assert "来源:" in out
    assert "A — https://a" in out
    assert "片段" in out


def test_format_for_agent_no_answer() -> None:
    """answer 为空时,只输出来源列表(原始结果型工具)。"""
    resp = SearchResponse(
        query="x",
        sources=[Source(url="https://a", title="A")],
    )
    out = format_for_agent(resp)
    assert "来源:" in out and "A — https://a" in out
    # 原始结果型无总结,首行不应是答案文本
    assert not out.startswith("晴")


def test_format_for_agent_no_title_only_url() -> None:
    """来源无 title 时,只输出 url,避免行首多余分隔符。"""
    resp = SearchResponse(query="x", sources=[Source(url="https://a")])
    out = format_for_agent(resp)
    assert "https://a" in out
    assert " — " not in out  # 无 title 不应出现 " — " 分隔


def test_format_for_agent_prefers_raw_content() -> None:
    """有 raw_content 全文时优先输出全文,且不再输出 content 短片段。"""
    resp = SearchResponse(
        query="x",
        sources=[
            Source(
                url="https://a",
                title="A",
                content="短片段",
                raw_content="# 全文\n正文",
            )
        ],
    )
    out = format_for_agent(resp)
    assert "# 全文" in out
    assert "正文" in out
    # 有全文时不再附带短片段
    assert "短片段" not in out


def test_format_for_agent_empty() -> None:
    """无 answer 且无 sources 时返回空串。"""
    resp = SearchResponse(query="x")
    assert format_for_agent(resp) == ""
