"""DuckDuckGo 网页搜索实现。

走 ddgs 库直接检索,无需 API key。
支持通过 backend 指定搜索引擎(默认 auto):duckduckgo / bing / google / brave /
yahoo / yandex / wikipedia / mojeek / startpage / annasarchive / grokipedia 等。
属于「原始结果型」工具:只返回来源列表,不自带总结,answer 为空。
ddgs 的 body(结果摘要)作为 Source.content 片段保留。

对外暴露:
  DuckDuckGoSearcher     — 基于 ddgs 的搜索器
  duckduckgo_web_search  — LangChain @tool,供 agent 自主调用
"""

from __future__ import annotations

from ddgs import DDGS
from langchain_core.tools import tool

from .base import SearchResponse, Source, format_for_agent

DEFAULT_MAX_RESULTS = 5
DEFAULT_BACKEND = "auto"


class DuckDuckGoSearcher:
    """基于 ddgs 库的搜索器,无需 API key。

    Args:
        max_results: 单次查询默认返回条数。
        backend: 搜索引擎,默认 auto(ddgs 自动选);可指定 duckduckgo/bing/google/
            brave/yahoo/yandex/wikipedia/mojeek/startpage 等,或逗号分隔多引擎。
    """

    def __init__(
        self,
        max_results: int = DEFAULT_MAX_RESULTS,
        backend: str = DEFAULT_BACKEND,
    ) -> None:
        self._max_results = max_results
        self._backend = backend

    def search(
        self,
        query: str,
        max_results: int | None = None,
        backend: str | None = None,
    ) -> SearchResponse:
        """对 query 发起 DuckDuckGo 搜索,返回来源链接(无总结)。

        Args:
            query: 搜索查询词。
            max_results: 覆盖默认返回条数;None 用实例默认值。
            backend: 覆盖默认引擎;None 用实例默认值。
        """
        results = DDGS().text(
            query,
            max_results=max_results or self._max_results,
            backend=backend or self._backend,
        )
        # ddgs 每条结果含 title/href/body;body 作为 content 片段保留。
        sources = [
            Source(
                url=item.get("href", ""),
                title=item.get("title"),
                content=item.get("body"),
            )
            for item in results
        ]
        return SearchResponse(query=query, sources=sources, answer=None)


@tool
def duckduckgo_web_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    backend: str = DEFAULT_BACKEND,
) -> str:
    """联网搜索:用 DuckDuckGo(ddgs,无需 key)对 query 检索,返回逐条来源链接(无总结)。

    backend 可选引擎(默认 auto):duckduckgo/bing/google/brave/yahoo/yandex/
    wikipedia/mojeek/startpage 等,一般不传用 auto 即可。
    """
    return format_for_agent(
        DuckDuckGoSearcher().search(query, max_results=max_results, backend=backend)
    )
