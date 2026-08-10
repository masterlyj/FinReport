"""Tavily 网页搜索实现。

走 Tavily 官方 SDK(TavilyClient)直接调搜索 API,按 url 取来源列表。
不请求 Tavily 内置 answer,摘要交由自己的 LLM整理。
开启 include_raw_content 后,每条来源附带 markdown 全文(raw_content),供入库/按需取原文。

对外暴露:
  TavilySearcher    — 基于 Tavily SDK 的搜索器
  tavily_web_search — LangChain @tool,供 agent 自主调用
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient  # type: ignore # tavily 缺 py.typed

from .base import SearchResponse, Source, format_for_agent

load_dotenv()

DEFAULT_MAX_RESULTS = 5


class TavilySearcher:
    """基于 Tavily SDK 的搜索器。

    Args:
        api_key: Tavily API key,默认读 TAVILY_API_KEY 环境变量。
        max_results: 单次查询默认返回条数。
        include_raw_content: 是否附带每条结果的网页全文(raw_content)。
            False(默认)不拉全文,省带宽,@tool 场景只给来源片段;
            入库时传 "markdown",取清洗后的 markdown 全文(质量优于裸 HTML)。
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        include_raw_content: bool | Literal["markdown", "text"] = False,
    ) -> None:
        self._client = TavilyClient(
            api_key=api_key or os.environ.get("TAVILY_API_KEY"),
        )
        self._max_results = max_results
        self._include_raw_content = include_raw_content

    def search(self, query: str, max_results: int | None = None) -> SearchResponse:
        """对 query 发起搜索,返回来源列表(不含 Tavily 内置摘要)。

        摘要不取 Tavily answer,改由调用方用自己的 LLM 整理。

        Args:
            query: 搜索查询词。
            max_results: 覆盖默认返回条数;None 用实例默认值。
        """
        resp = self._client.search(
            query,
            max_results=max_results or self._max_results,
            topic="finance",
            include_raw_content=self._include_raw_content,
        )
        sources = [
            Source(
                url=item.get("url", ""),
                title=item.get("title"),
                content=item.get("content"),
                raw_content=item.get("raw_content"),
            )
            for item in resp.get("results", [])
        ]
        return SearchResponse(query=query, sources=sources, answer=resp.get("answer"))


@tool
def tavily_web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    """联网搜索:用 Tavily 对 query 检索,返回逐条来源链接与其 markdown 全文(交由 researcher 提炼 notes)。"""
    # @tool 默认拉 markdown 全文:content 片段会被 Tavily 按 query 相关性裁掉中后段细节
    # (海外资金/产能过剩/估值等),把原始语料完整交给 researcher 的 compress 节点去噪提炼。
    return format_for_agent(
        TavilySearcher(include_raw_content="markdown").search(query, max_results=max_results)
    )
