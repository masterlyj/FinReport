"""researcher 子图：单个研究主题的搜索-反思-压缩循环.

三节点：
  researcher        — LLM + bind_tools([duckduckgo_web_search, ddg_extract_url, think_tool])，产出工具调用
  researcher_tools  — 并行执行工具调用，循环回 researcher 或跳 compress
  compress_research — LLM 把研究消息按事件分组压缩成结构化 notes

researcher 搜索后若摘要信息不足，可自主调 ddg_extract_url 抓取重要来源全文；
compress_research 把全部来源(含抓回的全文)一次性交给 LLM 按事件分组，
每簇选代表 URL 作正文引用、其余为佐证 URL，产出 notes 回传 supervisor.
"""

import asyncio
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from company_report_kit.configuration import Configuration
from company_report_kit.graph.state import (
    ResearcherOutputState,
    ResearcherState,
    SourceGrouping,
    SourceGroupingBatch,
)
from company_report_kit.prompts import (
    group_sources_into_events_prompt,
    research_system_prompt,
)
from company_report_kit.search_tools import ddg_extract_url, duckduckgo_web_search
from company_report_kit.utils import RETRY_KWARGS, configurable_model, get_model_config, get_today_str, think_tool

_RESEARCHER_TOOLS = [duckduckgo_web_search, ddg_extract_url, think_tool]


def _log(node: str, msg: str) -> None:
    """打印节点日志."""
    print(f"[{node}] {msg}")


async def researcher(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher_tools"]]:
    """LLM 调用工具收集信息.

    bind duckduckgo_web_search + think_tool，按 research_topic 搜索.
    思考模式关闭以兼容工具调用（strict 限制同 supervisor）.
    """
    configurable = Configuration.from_runnable_config(config)
    model_config = get_model_config(
        configurable, configurable.research_model, configurable.research_model_max_tokens
    )
    research_model = (
        configurable_model
        .bind_tools(_RESEARCHER_TOOLS)
        .with_retry(**RETRY_KWARGS)
        .with_config(model_config)
    )
    system_prompt = research_system_prompt.format(mcp_prompt="", date=get_today_str())
    messages = [SystemMessage(content=system_prompt)] + state.get("researcher_messages", [])
    response = await research_model.ainvoke(messages)
    _log("researcher", f"topic={state.get('research_topic', '')[:60]}")
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
        },
    )


async def _execute_tool_safely(tool, args, config):
    """安全执行工具，异常时返回错误文本而非抛出."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"工具执行错误：{e}"


async def researcher_tools(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher", "compress_research"]]:
    """并行执行工具调用，判断循环或压缩.

    无工具调用直接跳压缩；超 max_react_tool_calls 或调 ResearchComplete 也跳压缩.
    """
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent = researcher_messages[-1]

    if not most_recent.tool_calls:
        _log("researcher_tools", "无工具调用，跳压缩")
        return Command(goto="compress_research")

    tools_by_name = {t.name: t for t in _RESEARCHER_TOOLS}
    tasks = [
        _execute_tool_safely(tools_by_name[tc["name"]], tc["args"], config)
        for tc in most_recent.tool_calls
    ]
    observations = await asyncio.gather(*tasks)
    tool_outputs = [
        ToolMessage(content=obs, name=tc["name"], tool_call_id=tc["id"])
        for obs, tc in zip(observations, most_recent.tool_calls)
    ]

    exceeded = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    if exceeded:
        _log("researcher_tools", f"超 max_react_tool_calls={configurable.max_react_tool_calls}，跳压缩")
        return Command(goto="compress_research", update={"researcher_messages": tool_outputs})
    return Command(goto="researcher", update={"researcher_messages": tool_outputs})


def _clusters_to_notes(clusters: list[SourceGrouping]) -> str:
    """把事件簇列表转成 notes 文本(事件简述 + 关键事实 + 正文引用 + 佐证 URL)."""
    parts: list[str] = []
    for c in clusters:
        lines = [f"事件: {c.event_summary}"]
        if c.key_facts:
            lines.append(f"关键事实: {c.key_facts}")
        lines.append(f"正文引用: {c.primary_url}")
        if c.supporting_urls:
            lines.append("佐证来源: " + ", ".join(c.supporting_urls))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


async def compress_research(
    state: ResearcherState, config: RunnableConfig
) -> dict:
    """对 researcher 收集的来源做事件簇分组,产出结构化 notes.

    搜索结果经 format_for_agent 格式化后作为 ToolMessage.content,本节点
    直接拼接这些文本一次性交给 LLM 按事件分组(同一事件/转载聚一簇,
    孤立单独成簇),每簇选代表 URL 作正文引用、其余为佐证 URL.
    """
    from langchain_core.messages import filter_messages

    researcher_messages = list(state.get("researcher_messages", []))
    raw_notes = "\n".join(
        str(m.content) for m in filter_messages(researcher_messages, include_types=["tool", "ai"])
    )

    # 搜索结果文本(每条已是"来源:..."格式),供 LLM 分组.
    search_texts = [str(m.content) for m in researcher_messages if getattr(m, "type", "") == "tool"]
    if not search_texts:
        _log("compress_research", f"topic={state.get('research_topic', '')[:60]} 无来源,降级")
        return {"compressed_research": raw_notes or "未找到可用来源", "raw_notes": [raw_notes]}

    configurable = Configuration.from_runnable_config(config)
    model_config = get_model_config(
        configurable, configurable.research_model, configurable.research_model_max_tokens
    )
    grouping_model = (
        configurable_model
        .with_structured_output(SourceGroupingBatch, strict=True)
        .with_retry(**RETRY_KWARGS)
        .with_config(model_config)
    )
    prompt_content = group_sources_into_events_prompt.format(sources="\n\n".join(search_texts))
    response: SourceGroupingBatch = await grouping_model.ainvoke(
        [HumanMessage(content=prompt_content)]
    )
    _log("compress_research", f"topic={state.get('research_topic', '')[:60]} 分组完成 {len(response.clusters)} 簇")

    return {
        "compressed_research": _clusters_to_notes(response.clusters),
        "raw_notes": [raw_notes],
    }


# researcher 子图编译
researcher_builder = StateGraph(
    ResearcherState,
    output_schema=ResearcherOutputState,
    context_schema=Configuration,
)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()
