"""网页搜索工具的统一接口定义,供各具体工具(deepseek、tavily 等)实现。

设计约束:SearchResponse 的 answer 可空,以同时容纳两类工具——
「LLM 总结型」(如 deepseek,返回答案文本)与「原始结果型」(如 tavily,
仅返回来源列表)。具体工具的原始响应结构不进此层,避免泄漏实现细节。

对外暴露:
  SearchResponse    — 搜索结果数据类
  Source            — 单条来源数据类
  WebSearcher       — 统一接口 Protocol
  format_for_agent  — 把 SearchResponse 格式化成给 agent 阅读的文本
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Source:
    """单条搜索来源。"""

    url: str
    title: str | None = None
    page_age: str | None = None


@dataclass
class SearchResponse:
    """搜索结果,answer 可为空(原始结果型工具不产生总结)。

    Attributes:
        query: 原始查询词。
        sources: 搜索命中的来源链接列表。
        answer: 模型基于搜索结果生成的总结;仅「LLM 总结型」工具填充。
    """

    query: str
    sources: list[Source] = field(default_factory=list)
    answer: str | None = None


class WebSearcher(Protocol):
    """网页搜索工具的统一接口。"""

    def search(self, query: str) -> SearchResponse:
        """对 query 发起搜索,返回统一结构的响应。"""
        ...


def format_for_agent(response: SearchResponse) -> str:
    """把搜索结果格式化成供 agent 阅读的文本(总结 + 逐条来源)。

    Args:
        response: 搜索工具返回的统一结果。

    Returns:
        总结文本在前,其后列出全部来源;无内容时返回空串。
    """
    parts: list[str] = []
    if response.answer:
        parts.append(response.answer)
    if response.sources:
        parts.append("")
        parts.append("来源:")
        for i, s in enumerate(response.sources, start=1):
            title = s.title or ""
            # "标题 — url",无标题时只留 url,避免行首多余分隔符。
            line = f"{title} — {s.url}" if title else s.url
            parts.append(f"{i}. {line}")
    return "\n".join(parts) if parts else ""
