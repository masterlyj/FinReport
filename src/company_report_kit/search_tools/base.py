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
from urllib.parse import urlsplit


def normalize_url(url: str) -> str:
    """把 URL 归一化为可稳定比对的键,去除比对时常见的漂移差异.

    搜索摘要里的 URL 与章节脚注/分组簇的 URL 常因协议前缀、www、尾斜杠、
    查询参数、fragment 的细微差异而对不上,导致按 URL 精确匹配的过滤/兜底
    失效(静默删条目或 fail-open 泄漏)。归一化统一去掉这些差异:

      https://www.example.com/a/b/?tab=2#x  →  example.com/a/b

    Args:
        url: 原始 URL.

    Returns:
        归一化后的 URL 键(小写 host + 路径,无协议/www/尾斜杠/查询/fragment)。
    """
    # 预剥离尾部标点(中英文句逗分号)——这些是真实噪音,不影响 URL 内括号。
    # 注意:不剥 `)]}`——`Film_(2024)` 的尾 `)` 是 URL 的一部分,urlsplit 前剥会切坏。
    parts = urlsplit(url.strip().rstrip(".,;。，；"))
    # hostname 可能被 markdown 闭合 `)` 污染(如 `[标题](https://a.com)` 的 `)` 被
    # urlsplit 误解析进 host);剥离尾随定界符。path 保留括号(如 Wikipedia 消歧义
    # `Film_(2024)` 的 `)` 是 URL 的一部分,不能剥)。
    host = (parts.hostname or "").lower().removeprefix("www.").rstrip(")]}")
    path = parts.path.rstrip("/")
    return f"{host}{path}"


@dataclass
class Source:
    """单条搜索来源。"""

    url: str
    title: str | None = None
    page_age: str | None = None
    content: str | None = None
    raw_content: str | None = None


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
    """把搜索结果格式化成供 agent 阅读的文本(总结 + 逐条来源,来源附全文或片段)。

    Args:
        response: 搜索工具返回的统一结果。

    Returns:
        总结文本在前,其后列出全部来源;每条来源优先取 raw_content 全文,
        无全文时退回 content 短片段;均无则只留标题与 url。
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
            if s.raw_content:
                # markdown 全文,保留换行/结构(供 agent 阅读原文)。
                parts.append(s.raw_content.rstrip())
            elif s.content:
                # 无全文时退回短片段,压缩多余空白/换行成单行。
                snippet = " ".join(s.content.split())
                parts.append(f"   {snippet}")
    return "\n".join(parts) if parts else ""
