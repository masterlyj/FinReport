"""Company Report Kit 图节点实现.

流程：
  clarify_with_user → write_brief(interrupt) → supervisor ⇄ supervisor_tools
  → final_report_generation → END

clarify/write_brief 用 with_structured_output(strict=True) 关思考模式获取结构化输出.
final_report 开思考模式提升报告逻辑性.
supervisor 用 bind_tools 决策派发，supervisor_tools 用 ainvoke 并行调用 researcher_subgraph.
"""

import asyncio
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from company_report_kit.configuration import Configuration
from company_report_kit.graph.researcher import researcher_subgraph
from company_report_kit.graph.state import (
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearchQuestion,
)
from company_report_kit.prompts import (
    clarify_with_user_instructions,
    final_report_generation_prompt,
    lead_researcher_prompt,
    transform_messages_into_research_topic_prompt,
)
from company_report_kit.utils import (
    configurable_model,
    get_model_config,
    get_notes_from_tool_calls,
    get_today_str,
    think_tool,
    RETRY_KWARGS,
)


def _log(node: str, msg: str) -> None:
    """打印节点日志，便于在 Studio / 控制台追踪流程."""
    print(f"[{node}] {msg}")


async def clarify_with_user(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["write_brief", "__end__"]]:
    """判断是否需要向用户追问澄清.

    用 ClarifyWithUser 结构化输出约束 LLM. allow_clarification=False 时直接放行.
    思考模式关闭以保证 strict 工具调用兼容.
    """
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return Command(goto="write_brief")

    model_config = get_model_config(
        configurable, configurable.research_model, configurable.research_model_max_tokens
    )
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser, strict=True)
        .with_retry(**RETRY_KWARGS)
        .with_config(model_config)
    )
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(state["messages"]),
        date=get_today_str(),
    )
    response: ClarifyWithUser = await clarification_model.ainvoke(
        [HumanMessage(content=prompt_content)]
    )
    _log("clarify_with_user", "LLM 判断完成")
    if response.need_clarification:
        _log("clarify_with_user", "需追问")
        return Command(
            goto="__end__",
            update={"messages": [AIMessage(content=response.question)]},
        )
    _log("clarify_with_user", "无需追问，放行")
    return Command(
        goto="write_brief",
        update={"messages": [AIMessage(content=response.verification)]},
    )


async def write_brief(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["research_supervisor"]]:
    """生成研究简报并暂停等待人工确认.

    用 ResearchQuestion 结构化输出约束 LLM 生成 research_brief.
    调 interrupt() 暂停等待人工确认，resume 时图从本节点开头重新执行.
    """
    configurable = Configuration.from_runnable_config(config)
    model_config = get_model_config(
        configurable, configurable.research_model, configurable.research_model_max_tokens
    )
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion, strict=True)
        .with_retry(**RETRY_KWARGS)
        .with_config(model_config)
    )
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str(),
    )
    response: ResearchQuestion = await research_model.ainvoke(
        [HumanMessage(content=prompt_content)]
    )
    _log("write_brief", "研究简报生成完成")
    research_brief = response.research_brief

    # interrupt() 暂停图执行，把简报回传用户.
    # resume 时必须用 Command(resume=...) 包裹，图从本节点开头重新执行.
    user_decision = interrupt({"research_brief": research_brief})
    _log("write_brief", f"用户决策={user_decision}")
    return Command(
        goto="research_supervisor",
        update={
            "research_brief": research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=lead_researcher_prompt.format(
                        date=get_today_str(),
                        max_researcher_iterations=Configuration.from_runnable_config(config).max_researcher_iterations,
                        max_concurrent_research_units=Configuration.from_runnable_config(config).max_concurrent_research_units,
                    )),
                    HumanMessage(content=research_brief),
                ],
            },
        },
    )


async def supervisor(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["supervisor_tools"]]:
    """supervisor 节点：基于 research_brief 用 LLM 决策研究策略.

    bind ConductResearch / ResearchComplete / think_tool 三个工具，
    LLM 按简报拆分研究主题或标记完成，结果交由 supervisor_tools 执行.
    思考模式关闭以兼容工具调用.
    """
    configurable = Configuration.from_runnable_config(config)
    model_config = get_model_config(
        configurable, configurable.research_model, configurable.research_model_max_tokens
    )
    lead_tools = [ConductResearch, ResearchComplete, think_tool]
    research_model = (
        configurable_model
        .bind_tools(lead_tools)
        .with_retry(**RETRY_KWARGS)
        .with_config(model_config)
    )
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_researcher_iterations=configurable.max_researcher_iterations,
        max_concurrent_research_units=configurable.max_concurrent_research_units,
    )
    supervisor_messages = state.get("supervisor_messages", [])
    _iter = sum(1 for m in supervisor_messages if isinstance(m, AIMessage))
    response = await research_model.ainvoke(supervisor_messages)
    _log("supervisor", f"iterations={_iter} 决策完成")
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )


async def supervisor_tools(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["supervisor", "__end__"]]:
    """supervisor_tools: 执行工具后再检查 exceeded 退出.

    先执行当轮 ConductResearch（拿到 researcher 结果 ToolMessage），
    再检查 exceeded——这样 notes 汇聚时能提取到本轮结果.
    """
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = sum(1 for m in supervisor_messages if isinstance(m, AIMessage))
    most_recent = supervisor_messages[-1]

    no_tool_calls = not most_recent.tool_calls
    research_complete = any(tc["name"] == "ResearchComplete" for tc in most_recent.tool_calls)
    exceeded = research_iterations > configurable.max_researcher_iterations

    if research_complete or no_tool_calls:
        _log("supervisor_tools", f"退出(complete={research_complete} no_calls={no_tool_calls})")
        return Command(goto="__end__", update={"notes": get_notes_from_tool_calls(supervisor_messages), "research_brief": state.get("research_brief", "")})

    all_tool_messages = []
    for tc in most_recent.tool_calls:
        if tc["name"] == "think_tool":
            all_tool_messages.append(ToolMessage(content=f"反思: {tc['args']['reflection']}", name="think_tool", tool_call_id=tc["id"]))

    conduct_calls = [tc for tc in most_recent.tool_calls if tc["name"] == "ConductResearch"]
    if conduct_calls:
        allowed = conduct_calls[:configurable.max_concurrent_research_units]
        overflow = conduct_calls[configurable.max_concurrent_research_units:]
        _log("supervisor_tools", f"派发 {len(allowed)} 个 researcher(overflow {len(overflow)})")
        research_tasks = [researcher_subgraph.ainvoke({"researcher_messages": [HumanMessage(content=tc["args"]["research_topic"])], "research_topic": tc["args"]["research_topic"]}, config) for tc in allowed]
        tool_results = await asyncio.gather(*research_tasks, return_exceptions=True)
        for observation, tc in zip(tool_results, allowed):
            if isinstance(observation, Exception):
                all_tool_messages.append(ToolMessage(content=f"研究失败: {observation}", name=tc["name"], tool_call_id=tc["id"]))
                continue
            all_tool_messages.append(ToolMessage(content=observation.get("section_text", "研究章节生成失败"), name=tc["name"], tool_call_id=tc["id"]))
        for tc in overflow:
            all_tool_messages.append(ToolMessage(content=f"超出并行上限", name="ConductResearch", tool_call_id=tc["id"]))
        all_raw = []
        for obs in tool_results:
            if not isinstance(obs, Exception):
                all_raw.extend(obs.get("raw_notes", []))
        raw_notes_concat = "\n".join(all_raw)
        if exceeded:
            _log("supervisor_tools", "exceeded，执行完工具后退出")
            return Command(goto="__end__", update={"supervisor_messages": all_tool_messages, "notes": [str(m.content) for m in all_tool_messages], "raw_notes": [raw_notes_concat] if raw_notes_concat else [], "research_brief": state.get("research_brief", "")})
        return Command(goto="supervisor", update={"supervisor_messages": all_tool_messages, "raw_notes": [raw_notes_concat] if raw_notes_concat else []})

    if exceeded:
        _log("supervisor_tools", "exceeded，退出")
        return Command(goto="__end__", update={"supervisor_messages": all_tool_messages, "notes": get_notes_from_tool_calls(supervisor_messages + all_tool_messages), "research_brief": state.get("research_brief", "")})
    return Command(goto="supervisor", update={"supervisor_messages": all_tool_messages})


async def final_report_generation(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    """生成最终报告.

    基于 research_brief + notes + messages 调 LLM 生成报告.
    开思考模式提升金融报告的逻辑性与深度.
    token 超限时逐步截断 findings 重试，最多 3 次.
    """
    configurable = Configuration.from_runnable_config(config)
    model_config = get_model_config(
        configurable,
        configurable.final_report_model,
        configurable.final_report_model_max_tokens,
        thinking=True,
    )
    writer_model = configurable_model.with_config(model_config)

    notes = state.get("notes", [])
    findings = "\n".join(notes)
    cleared_state = {"notes": {"type": "override", "value": []}}

    max_retries = 3
    current_retry = 0
    findings_char_limit = None

    while current_retry <= max_retries:
        try:
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str(),
            )
            _log("final_report_generation", f"调用 LLM 生成报告（第 {current_retry + 1} 次）")
            final_report_msg = await writer_model.ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            return Command(
                goto="__end__",
                update={
                    "final_report": final_report_msg.content,
                    "messages": [final_report_msg],
                    **cleared_state,
                },
            )
        except Exception as e:
            current_retry += 1
            err_msg = str(e).lower()
            if "context length" in err_msg or "maximum" in err_msg or "token" in err_msg:
                if current_retry == 1:
                    findings_char_limit = 200000
                else:
                    findings_char_limit = int(findings_char_limit * 0.9)
                findings = findings[:findings_char_limit]
                _log("final_report_generation", f"token 超限，截断到 {findings_char_limit} 字符重试")
            else:
                _log("final_report_generation", f"API 失败({type(e).__name__})，第 {current_retry} 次重试")
                await asyncio.sleep(2 ** current_retry)
            if current_retry <= max_retries:
                continue
            error_report = f"生成报告失败：{e}"
            return Command(
                goto="__end__",
                update={
                    "final_report": error_report,
                    "messages": [AIMessage(content=error_report)],
                    **cleared_state,
                },
            )

    error_report = "生成报告失败：重试次数耗尽"
    return Command(
        goto="__end__",
        update={
            "final_report": error_report,
            "messages": [AIMessage(content=error_report)],
            **cleared_state,
        },
    )

