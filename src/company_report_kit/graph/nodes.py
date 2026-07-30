"""Company Report Kit 图节点实现.

流程：
  clarify_with_user → write_brief(interrupt) → supervisor ⇄ supervisor_tools
  → final_report_generation → END

clarify/write_brief 用 with_structured_output(strict=True) 关思考模式获取结构化输出.
final_report 开思考模式提升报告逻辑性.
supervisor/supervisor_tools 占位，阶段 3 接入 Send 派发时改造.
"""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from company_report_kit.configuration import Configuration
from company_report_kit.graph.state import AgentState, ClarifyWithUser, ResearchQuestion
from company_report_kit.prompts import (
    clarify_with_user_instructions,
    final_report_generation_prompt,
    transform_messages_into_research_topic_prompt,
)
from company_report_kit.utils import configurable_model, get_model_config, get_today_str


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
        .with_retry(stop_after_attempt=3)
        .with_config(model_config)
    )
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(state["messages"]),
        date=get_today_str(),
    )
    _log("clarify_with_user", "调用 LLM 判断是否需追问")
    response: ClarifyWithUser = await clarification_model.ainvoke(
        [HumanMessage(content=prompt_content)]
    )
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
        .with_retry(stop_after_attempt=3)
        .with_config(model_config)
    )
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str(),
    )
    _log("write_brief", "调用 LLM 生成研究简报")
    response: ResearchQuestion = await research_model.ainvoke(
        [HumanMessage(content=prompt_content)]
    )
    research_brief = response.research_brief

    # interrupt() 暂停图执行，把简报回传用户.
    # resume 时必须用 Command(resume=...) 包裹，图从本节点开头重新执行.
    user_decision = interrupt({"research_brief": research_brief})
    _log("write_brief", f"用户决策={user_decision}")
    return Command(
        goto="research_supervisor",
        update={"research_brief": research_brief},
    )


async def supervisor(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["supervisor_tools"]]:
    """supervisor 节点：基于 research_brief 决策研究策略（阶段 3 接入 LLM）."""
    brief = state.get("research_brief", "(无 brief)")
    _log("supervisor", f"brief={brief[:80]}（占位，阶段 3 接入 LLM）")
    return Command(goto="supervisor_tools")


async def supervisor_tools(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["supervisor", "__end__"]]:
    """supervisor_tools 节点：执行 supervisor 决策的工具调用（阶段 3 接入）."""
    _log("supervisor_tools", "占位，跳 END 结束子图")
    return Command(goto="__end__")


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
            err_msg = str(e).lower()
            if "context length" in err_msg or "maximum" in err_msg or "token" in err_msg:
                current_retry += 1
                if current_retry == 1:
                    findings_char_limit = 200000
                else:
                    findings_char_limit = int(findings_char_limit * 0.9)
                findings = findings[:findings_char_limit]
                _log("final_report_generation", f"token 超限，截断到 {findings_char_limit} 字符重试")
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

