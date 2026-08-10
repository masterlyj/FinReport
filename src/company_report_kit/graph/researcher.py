"""researcher 子图：单个研究主题的搜索-反思-分组-写章节循环.

四节点：
  researcher        — LLM + bind_tools([duckduckgo_web_search, extract_url, think_tool])，产出工具调用
  researcher_tools  — 并行执行工具调用，循环回 researcher 或跳 compress
  compress_research — LLM 把搜索结果按事件分组去重，产出 clusters
  write_section     — LLM 基于 clusters 写本维度报告章节(markdown 含脚注引用)

researcher 搜索后若摘要信息不足，可自主调 extract_url 抓取重要来源全文；
compress_research 把全部来源一次性交给 LLM 按事件分组去重；
write_section 基于分组结果写章节，产出 section_text 回传主 agent.
"""

import asyncio
import re
from typing import Literal, cast

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
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
from company_report_kit.logging_utils import get_logger
from company_report_kit.prompts import (
    group_sources_into_events_prompt,
    research_system_prompt,
    write_section_prompt,
)
from company_report_kit.search_tools import duckduckgo_web_search, extract_url
from company_report_kit.search_tools.base import normalize_url
from company_report_kit.utils import (
    RETRY_KWARGS,
    configurable_model,
    get_model_config,
    get_today_str,
    think_tool,
)

_RESEARCHER_TOOLS = [duckduckgo_web_search, extract_url, think_tool]

logger = get_logger("graph.researcher")


def _short_topic(topic: str, maxlen: int = 24) -> str:
    """把 research_topic 截成短标签(取开头,超长加省略号)。

    5 个 researcher 并行研究不同维度,日志若只打完整 topic 会互相交织且
    因截断无法区分。用 topic 开头的短标签(如"研究阶跃星辰的融资历史…")
    让每行日志一眼可辨是哪个维度。

    Args:
        topic: 研究主题原文.
        maxlen: 保留的字符数.

    Returns:
        短标签字符串(可能含省略号).
    """
    return topic[:maxlen] + ("…" if len(topic) > maxlen else "")


def _log(node: str, msg: str, topic: str = "") -> str:
    """记录节点日志,可选带研究主题短标签便于区分并行 researcher.

    Args:
        node: 节点名(researcher / researcher_tools / compress_research / write_section).
        msg: 日志正文.
        topic: 研究主题;非空时自动加"[短标签]"前缀.

    Returns:
        拼接后的短标签(供调用方复用,如单独打 topic).
    """
    label = f"[{_short_topic(topic)}] " if topic else ""
    logger.info("[%s] %s%s", node, label, msg)
    return label


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
    _log("researcher", "搜索中", topic=state.get("research_topic", ""))
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
    except Exception as e:  # noqa: BLE001 - 工具异常兜成错误文本回喂 LLM 自纠
        return f"工具执行错误：{e}"


async def researcher_tools(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher", "compress_research"]]:
    """并行执行工具调用，判断循环或压缩.

    无工具调用直接跳压缩；超 max_react_tool_calls 或调 ResearchComplete 也跳压缩.
    """
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    # 最近一条消息由 researcher 节点写入(AIMessage with tool_calls)。
    # researcher_messages 是 list[MessageLikeRepresentation](含 str/dict 联合)，
    # cast 收窄到 AIMessage 才能访问 .tool_calls。
    most_recent = cast(AIMessage, researcher_messages[-1])
    topic = state.get("research_topic", "")

    if not most_recent.tool_calls:
        _log("researcher_tools", "无工具调用，跳压缩", topic=topic)
        return Command(goto="compress_research")

    tools_by_name = {t.name: t for t in _RESEARCHER_TOOLS}
    # 用 .get() 而非 [ ]:模型可能幻觉出不在 _RESEARCHER_TOOLS 里的工具名,
    # [ ] 抛 KeyError 会中断列表推导、泄漏已建协程(never awaited 警告)。
    # .get() 返回 None,经 _execute_tool_safely 的 try/except 兜成错误文本回喂 LLM 自纠。
    tasks = [
        _execute_tool_safely(tools_by_name.get(tc["name"]), tc["args"], config)
        for tc in most_recent.tool_calls
    ]
    observations = await asyncio.gather(*tasks)
    tool_outputs = [
        ToolMessage(content=obs, name=tc["name"], tool_call_id=tc["id"])
        for obs, tc in zip(observations, most_recent.tool_calls)
    ]

    exceeded = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    if exceeded:
        _log("researcher_tools", f"超 max_react_tool_calls={configurable.max_react_tool_calls}，跳压缩", topic=topic)
        return Command(goto="compress_research", update={"researcher_messages": tool_outputs})
    return Command(goto="researcher", update={"researcher_messages": tool_outputs})


def _filter_raw_notes_by_urls(raw_notes: str, keep_urls: set[str]) -> str:
    """从原始笔记中按 URL 保留对应条目,丢弃其余来源原文.

    compress_research 的范围过滤只作用于 clusters 骨架,若不处理 raw_notes,
    被判定越界丢弃的来源原文仍原样进入 write_section 的<来源原文>段——写作
    端 LLM 会看到被过滤的越界内容(如财务 researcher 搜到的逐轮融资全文),
    把过滤从"双保险"退化成"写作 LLM 一条规则自省"。本函数按条目(标题—url
    + 缩进内容)过滤,只保留 clusters 实际使用 URL 对应的原文;非条目行
    (如"来源:"头、搜索总结)原样保留。URL 先归一化(去协议/www/尾斜杠/查询
    参数)再比对,避免微漂移导致漏删(泄漏)或误删(丢事实)。

    Args:
        raw_notes: compress_research 拼接的原始工具输出全文.
        keep_urls: 保留的 URL 集合(clusters 的 primary + supporting).

    Returns:
        过滤后的原文文本;保留条目不足时返回原文本(不误伤格式异常场景).
    """
    kept_lines: list[str] = []
    current_url: str | None = None
    current_block: list[str] = []
    kept_any = False
    # 归一化后的保留 URL 集合,匹配时用规范化键
    norm_keep = {normalize_url(u) for u in keep_urls}

    def _flush() -> None:
        nonlocal current_url, current_block, kept_any
        if current_url is not None and normalize_url(current_url) in norm_keep:
            kept_lines.extend(current_block)
            kept_any = True
        current_url = None
        current_block = []

    entry_re = re.compile(r"(?:^|\n)\d+\.\s+(?:.*?[-—]\s*)?(https?://[^\s]+)")
    for line in raw_notes.split("\n"):
        m = entry_re.match(line)
        if m:
            _flush()
            current_url = m.group(1)
            current_block = [line]
        elif current_url is not None:
            current_block.append(line)
        else:
            # 非条目行(来源:/总结/空行):原样保留
            kept_lines.append(line)
    _flush()
    return "\n".join(kept_lines) if kept_any else raw_notes


def _clusters_to_text(clusters: list[SourceGrouping]) -> str:
    """把事件簇列表格式化成供 write_section LLM 阅读的文本."""
    parts: list[str] = []
    for i, c in enumerate(clusters, start=1):
        lines = [f"事件 {i}: {c.event_summary}"]
        if c.key_facts:
            lines.append(f"  关键事实: {c.key_facts}")
        lines.append(f"  正文引用: {c.primary_url}")
        if c.supporting_urls:
            lines.append(f"  佐证来源: {', '.join(c.supporting_urls)}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


async def compress_research(
    state: ResearcherState, config: RunnableConfig
) -> dict:
    """对 researcher 收集的来源做事件簇分组,产出 clusters 供 write_section 使用.

    分组 LLM 拿到研究主题,自行判断每个来源是否支撑本研究范围——搜索常带回
    与其他维度/主题无关的来源(如研究财务时搜到逐轮融资新闻),由分组阶段
    丢弃,不进入写作上下文。维度边界由 LLM 按主题语义判断,而非代码硬编码。
    """
    from langchain_core.messages import filter_messages

    researcher_messages = list(state.get("researcher_messages", []))
    topic = state.get("research_topic", "")
    raw_notes = "\n".join(
        str(m.content) for m in filter_messages(researcher_messages, include_types=["tool", "ai"])
    )

    search_texts = [
        str(m.content)
        for m in filter_messages(researcher_messages, include_types=["tool"])
    ]
    if not search_texts:
        _log("compress_research", "无来源", topic=topic)
        return {"clusters": [], "raw_notes": [raw_notes]}

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
    prompt_content = group_sources_into_events_prompt.format(
        topic=topic, sources="\n\n".join(search_texts)
    )
    response: SourceGroupingBatch = cast(
        SourceGroupingBatch,
        await grouping_model.ainvoke([HumanMessage(content=prompt_content)]),
    )
    if response is None or not response.clusters:
        _log("compress_research", "分组返回空,降级", topic=topic)
        return {"clusters": [], "raw_notes": [raw_notes]}
    _log("compress_research", f"分组完成 {len(response.clusters)} 簇", topic=topic)

    # 范围过滤只保留本维度来源,并同步裁剪 raw_notes:clusters 是骨架,被丢弃
    # 来源的原文若仍留在 raw_notes,write_section 会在<来源原文>里看到越界
    # 内容(如财务 researcher 搜到的逐轮融资全文),过滤形同虚设。
    clusters = response.clusters
    keep_urls = {
        c.primary_url
        for c in clusters
    } | {u for c in clusters for u in c.supporting_urls}
    filtered_notes = _filter_raw_notes_by_urls(raw_notes, keep_urls)

    return {
        "clusters": clusters,
        "raw_notes": [filtered_notes],
    }


async def write_section_with_feedback(
    topic: str,
    clusters: list[SourceGrouping],
    raw_notes: str,
    review_issues: str,
    config: RunnableConfig,
) -> str:
    """基于事件分组与来源原文写章节;带审查意见时按意见修订措辞.

    抽出为独立函数,供 researcher 子图的 write_section 节点与快照审查修正
    共用——修正仍基于同一批 clusters + raw_notes(不重新搜索),只是把审查
    ReviewIssue 格式化成 review_issues 文本追加进 prompt,让 LLM 按意见
    修订(删除无出处句/纠正错配数字),而非从零重写。

    Args:
        topic: 研究主题(本维度).
        clusters: 事件分组骨架.
        raw_notes: 来源原文全文(compress_research 过滤后的).
        review_issues: 审查意见格式化文本;空串时不带修订指令(纯写作).
        config: 运行时配置.

    Returns:
        写好的 markdown 章节文本.
    """
    configurable = Configuration.from_runnable_config(config)
    model_config = get_model_config(
        configurable, configurable.research_model, configurable.research_model_max_tokens
    )
    writer = configurable_model.with_retry(**RETRY_KWARGS).with_config(model_config)
    # 审查意见非空时,把它格式化成"审查意见"块放进 prompt,指示 LLM 按意见修订
    # (只改被指出的句子,不重写整章);空串时是首次写作,无修订指令。
    feedback_block = (
        f"\n<审查意见>\n{review_issues}\n</审查意见>\n"
        f"\n以下是审查员对本章的修正意见。请逐条处理：引用错配的对照来源原文改正；"
        f"无出处的删除该句或标注无法核实；口径冲突的以更权威来源为准并注明。"
        f"只修改审查指出的内容，保留未出问题的部分，脚注引用保持 [^N] 格式。"
        if review_issues
        else ""
    )
    prompt_content = write_section_prompt.format(
        topic=topic,
        clusters=_clusters_to_text(clusters),
        raw_notes=raw_notes,
        review_issues=feedback_block,
    )
    response = await writer.ainvoke([HumanMessage(content=prompt_content)])
    return str(response.content)


async def write_section(
    state: ResearcherState, config: RunnableConfig
) -> dict:
    """基于分组结果写本维度的报告章节(markdown 含脚注引用)."""
    clusters = state.get("clusters", [])
    topic = state.get("research_topic", "")

    if not clusters:
        _log("write_section", "无 clusters,跳过", topic=topic)
        return {"section_text": "公开信息有限，未能获取足够数据撰写本章节。"}

    # raw_notes 是 compress_research 产出的全文(tool+ai 消息拼接),含原文细节
    # (金额/日期/人名原话);clusters 只给骨架,write 凭 raw_notes 充实章节
    raw_notes = "\n\n".join(state.get("raw_notes", []))
    section = await write_section_with_feedback(
        topic=topic,
        clusters=clusters,
        raw_notes=raw_notes,
        review_issues="",
        config=config,
    )
    _log("write_section", f"章节完成 {len(section)} 字符", topic=topic)
    return {"section_text": section}


# researcher 子图编译
researcher_builder = StateGraph(
    ResearcherState,
    output_schema=ResearcherOutputState,
    context_schema=Configuration,
)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_node("write_section", write_section)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", "write_section")
researcher_builder.add_edge("write_section", END)
researcher_subgraph = researcher_builder.compile()
