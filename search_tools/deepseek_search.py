"""DeepSeek(Anthropic 兼容端点)网页搜索实现。

走 Anthropic 服务端 web_search 工具:模型在服务端自动执行搜索、把结果
回灌后给出总结,一次 messages.create 即可拿到最终答案与来源链接,调用
方无需(也无法)手动回传 tool_result。返回的 message.content 为 block
序列,本模块从中提取最终答案(最后一个 text block)与全部来源
(web_search_tool_result 中的条目)。

对外暴露:
  DeepSeekSearcher — 基于 DeepSeek + web_search 工具的搜索器
"""

from __future__ import annotations

import os

from anthropic import Anthropic
from dotenv import load_dotenv

from search_tools.base import SearchResponse, Source

load_dotenv()

DEFAULT_BASE_URL = "https://api.deepseek.com/anthropic"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_TOKENS = 16384

# Anthropic 原生服务端搜索工具;经 DeepSeek 兼容端点透传执行。
_WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
}


class DeepSeekSearcher:
    """基于 DeepSeek + Anthropic web_search 工具的搜索器。

    Args:
        api_key: DeepSeek API key,默认读 DEEPSEEK_API_KEY 环境变量。
        base_url: Anthropic 兼容端点地址。
        model: 调用的模型名称。
        max_tokens: 单次回复上限,过小会截断最终答案。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._client = Anthropic(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=base_url,
        )
        self._model = model
        self._max_tokens = max_tokens

    def search(self, query: str) -> SearchResponse:
        """对 query 发起搜索,返回答案文本与来源链接。"""
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            tools=[_WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": query}],
        )
        return _parse_message(query, message)


def _parse_message(query: str, message) -> SearchResponse:
    """从 message.content 提取最终答案与来源。"""
    sources: list[Source] = []
    answer: str | None = None
    for block in message.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            # 多轮搜索时会出现多个 text block,最终答案在最后一个。
            answer = block.text
        elif btype == "web_search_tool_result":
            for item in block.content:
                # 该列表可能混入 web_search_tool_result_error 等非结果条目,
                # 仅取 web_search_result,其携带可用的 url/title。
                if getattr(item, "type", None) != "web_search_result":
                    continue
                sources.append(
                    Source(
                        url=item.url,
                        title=getattr(item, "title", None),
                        page_age=getattr(item, "page_age", None),
                    )
                )
    return SearchResponse(query=query, sources=sources, answer=answer)
