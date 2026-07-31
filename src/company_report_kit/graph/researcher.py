"""researcher 子图：单个研究主题的搜索-反思-压缩循环.

三节点：
  researcher        — LLM + bind_tools([duckduckgo_web_search, think_tool])，产出工具调用
  researcher_tools  — 并行执行工具调用，循环回 researcher 或跳 compress
  compress_research — LLM 把研究消息压缩成结构化 notes

由 supervisor_tools 通过 Send 并行派发，每个 researcher 独立运行，
产出 compressed_research 回传 supervisor.
"""

import asyncio
from typing import Literal

from langchain_core.messages import SystemMessage, ToolMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from company_report_kit.configuration import Configuration
from company_report_kit.graph.state import ResearcherOutputState, ResearcherState
from company_report_kit.prompts import (
    compress_research_simple_human_message,
    compress_research_system_prompt,
    research_system_prompt,
)
from company_report_kit.search_tools import duckduckgo_web_search
from company_report_kit.utils import configurable_model, get_model_config, get_today_str, think_tool

_RESEARCHER_TOOLS = [duckduckgo_web_search, think_tool]


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
        .with_retry(stop_after_attempt=3)
        .with_config(model_config)
    )
    system_prompt = research_system_prompt.format(mcp_prompt="", date=get_today_str())
    messages = [SystemMessage(content=system_prompt)] + state.get("researcher_messages", [])
    _log("researcher", f"topic={state.get('research_topic', '')[:60]}")
    response = await research_model.ainvoke(messages)
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


async def compress_research(
    state: ResearcherState, config: RunnableConfig
) -> dict:
    """LLM 压缩研究消息成结构化 notes.

    保留所有相关来源与信息，原样重写非总结，供 final_report 引用追溯.
    """
    configurable = Configuration.from_runnable_config(config)
    model_config = get_model_config(
        configurable, configurable.research_model, configurable.research_model_max_tokens
    )
    synthesizer = configurable_model.with_config(model_config)
    researcher_messages = list(state.get("researcher_messages", []))
    researcher_messages.append({"role": "user", "content": compress_research_simple_human_message})
    system_prompt = compress_research_system_prompt.format(date=get_today_str())
    _log("compress_research", f"topic={state.get('research_topic', '')[:60]}")
    response = await synthesizer.ainvoke([SystemMessage(content=system_prompt)] + researcher_messages)

    from langchain_core.messages import filter_messages
    raw_notes = "\n".join(
        str(m.content) for m in filter_messages(researcher_messages, include_types=["tool", "ai"])
    )
    return {
        "compressed_research": str(response.content),
        "raw_notes": [raw_notes],
    }


# researcher 子图编译
researcher_builder = StateGraph(
    ResearcherState,
    output=ResearcherOutputState,
    config_schema=Configuration,
)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()
