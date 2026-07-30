"""Company Report Kit 图节点实现.

定义主图与 supervisor 子图的节点函数。每个节点接收当前 state 与运行时配置，
返回 Command 跳转下一节点并更新状态。

流程：
  clarify_with_user → write_brief(interrupt) → supervisor ⇄ supervisor_tools
  → final_report_generation → END

write_brief 节点产出研究简报后调 interrupt() 暂停等待人工确认，承担 PRD
第 5 节"研究计划生成与人工确认"环节。其余节点目前返回写死的占位内容，
后续接入 LLM、Variable Memory、researcher 子图时替换对应节点内部逻辑即可，
节点签名与跳转关系保持稳定。
"""

from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from company_report_kit.configuration import Configuration
from company_report_kit.graph.state import AgentState


def _log(node: str, msg: str) -> None:
    """打印节点日志，便于在 Studio / 控制台追踪流程."""
    print(f"[{node}] {msg}")


async def clarify_with_user(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["write_brief", "__end__"]]:
    """判断是否需要向用户追问澄清.

    读取 Configuration.allow_clarification 决定是否启用澄清环节。
    启用但无需追问时，直接进入 brief 生成；需追问时跳 END 并返回问题给用户.

    Args:
        state: 当前图状态，含用户输入的 messages.
        config: 运行时配置，用于加载 Configuration.

    Returns:
        Command 跳转 write_brief（当前默认放行）或 END（需追问时返回问题）.
    """
    configurable = Configuration.from_runnable_config(config)
    _log("clarify_with_user", f"allow_clarification={configurable.allow_clarification}")
    if not configurable.allow_clarification:
        return Command(goto="write_brief")
    return Command(goto="write_brief")


async def write_brief(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["research_supervisor"]]:
    """生成研究简报并暂停等待人工确认.

    从用户 messages 提取研究目标，构造 research_brief，调 interrupt() 暂停图执行
    把简报回传用户。用户用 Command(resume=...) 确认后，图从本节点开头重新执行，
    interrupt() 第二次调用返回 resume 值，节点继续跳转 research_supervisor.

    Args:
        state: 当前图状态，含用户输入的 messages.
        config: 运行时配置.

    Returns:
        Command 跳转 research_supervisor，update 写入 research_brief.
    """
    messages = state.get("messages", [])
    user_input = messages[-1].content if messages else "(空输入)"
    _log("write_brief", f"user_input={user_input[:80]}, 生成简报后等待人工确认")
    research_brief = (
        f"研究简报（占位）：对 {user_input} 进行公司深度研究.\n"
        "研究范围：公司概况、商业模式、财务分析、估值、竞争格局、风险."
    )
    # interrupt() 暂停图执行，把简报回传用户.
    # resume 时必须用 Command(resume=...) 包裹，图从本节点开头重新执行，
    # interrupt() 第二次调用返回 resume 值作为 user_decision，跳过 raise.
    user_decision = interrupt({"research_brief": research_brief})
    _log("write_brief", f"用户决策={user_decision}")
    return Command(
        goto="research_supervisor",
        update={"research_brief": research_brief},
    )


async def supervisor(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["supervisor_tools"]]:
    """supervisor 节点：基于 research_brief 决策研究策略.

    读取 research_brief 作为决策上下文，通过工具调用（ConductResearch 拆分研究
    主题、think_tool 反思、ResearchComplete 标记结束）驱动研究流程，结果交由
    supervisor_tools 执行.

    Args:
        state: 当前图状态，含 research_brief.
        config: 运行时配置.

    Returns:
        Command 跳转 supervisor_tools 执行工具调用.
    """
    brief = state.get("research_brief", "(无 brief)")
    _log("supervisor", f"brief={brief[:80]}")
    return Command(goto="supervisor_tools")


async def supervisor_tools(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["supervisor", "__end__"]]:
    """supervisor_tools 节点：执行 supervisor 决策的工具调用.

    处理三类工具调用：
      1. think_tool - supervisor 反思，循环回 supervisor
      2. ConductResearch - 派发 researcher 子图，循环回 supervisor
      3. ResearchComplete - 标记研究结束，跳 END 结束子图

    当 research_iterations 超过 max_researcher_iterations 时强制结束子图，
    由主图边 research_supervisor→final_report_generation 接管进入报告生成.

    Args:
        state: 当前 supervisor 子图状态，含 research_iterations.
        config: 运行时配置，用于加载 Configuration.

    Returns:
        Command 跳转 supervisor（继续循环）或 __end__（结束子图）.
    """
    configurable = Configuration.from_runnable_config(config)
    iterations = state.get("research_iterations", 0)
    _log("supervisor_tools", f"iterations={iterations} max={configurable.max_researcher_iterations}")
    # 跳 END 结束子图，主图边 research_supervisor→final_report_generation 接管.
    return Command(goto="__end__")


async def final_report_generation(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    """最终报告生成节点.

    汇总 research_brief 与 notes 生成最终报告，写入 final_report 并追加
    AIMessage 到 messages 供用户查看.

    Args:
        state: 当前图状态，含 research_brief、notes、raw_notes.
        config: 运行时配置.

    Returns:
        Command 跳 END，update 写入 final_report 与 messages.
    """
    _log("final_report_generation", "生成占位报告")
    final_report = "公司深度研究报告（占位）：待接入两阶段写作（outline → sections）."
    return Command(
        goto="__end__",
        update={
            "final_report": final_report,
            "messages": [AIMessage(content=final_report)],
        },
    )

