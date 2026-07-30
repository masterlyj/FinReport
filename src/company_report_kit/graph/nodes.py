"""Company Report Kit 的图节点实现（阶段 1 骨架）.

每个节点先留占位逻辑：print 状态 + 返回写死的 update，不调 LLM、不接 Memory、
不派发子图. 这样能在不依赖 venv 装包的前提下先跑通图的拓扑和 HIL interrupt 链路，
阶段 2-5 再逐步把占位换成真实业务.

节点顺序对应 PRD 第 5 节核心流程（合并后）：
  clarify → write_brief(interrupt 人工确认) → supervisor ⇄ supervisor_tools
  → final_report

相比原 PRD，合并了"研究计划生成与人工确认"为 write_brief 单一环节，
对齐 open_deep_research 的单次 HIL 模式.
图拓扑保留 supervisor ⇄ supervisor_tools 循环，阶段 3 接入 Send 派发时
只需替换 supervisor_tools 占位逻辑，不动其他节点.
"""

from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from company_report_kit.configuration import Configuration
from company_report_kit.graph.state import AgentState


def _log(node: str, msg: str) -> None:
    """统一打印节点日志，便于在 LangGraph Studio / 控制台追踪流程.

    阶段 1 不接日志框架，先用 print 保证骨架可观测；阶段 6 接入正式 logger.
    """
    print(f"[{node}] {msg}")


async def clarify_with_user(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["write_brief", "__end__"]]:
    """判断是否需要向用户追问澄清.

    阶段 1 先跳过真实澄清逻辑：若 allow_clarification=False 则直接进入 brief 生成，
    否则也直接放行（占位），等阶段 2 接入 LLM 后再用 ClarifyWithUser 结构化输出判断.

    Returns:
        Command 跳转到 write_brief. 阶段 2 接 LLM 后，需追问时改为跳 END 并返回问题.
    """
    configurable = Configuration.from_runnable_config(config)
    _log("clarify_with_user", f"allow_clarification={configurable.allow_clarification}")
    if not configurable.allow_clarification:
        return Command(goto="write_brief")
    # 占位：阶段 2 接 LLM 后，此处用 ClarifyWithUser 结构化输出判断是否追问.
    return Command(goto="write_brief")


async def write_brief(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["research_supervisor"]]:
    """生成研究简报并暂停等待人工确认.

    这是 PRD 第 5 节"研究计划生成与人工确认"环节的载体（合并后由 brief 承担）.
    节点产出 research_brief 后调 interrupt() 暂停图执行，把简报回传用户；
    用户 resume 时传入确认信息，节点继续跳转 supervisor.

    阶段 1 简报内容是写死的占位；阶段 2 接 init_chat_model 后，
    用 ResearchQuestion 结构化输出约束 LLM 生成.
    """
    messages = state.get("messages", [])
    user_input = messages[-1].content if messages else "(空输入)"
    _log("write_brief", f"user_input={user_input[:80]}, 生成简报后等待人工确认")
    placeholder_brief = (
        f"研究简报（占位）：对 {user_input} 进行公司深度研究.\n"
        "研究范围：公司概况、商业模式、财务分析、估值、竞争格局、风险."
    )
    # interrupt 暂停图执行，返回值作为 resume 时的输入.
    # resume 时必须用 Command(resume=...) 包裹，图从本节点开头重新执行，
    # interrupt 第二次调用返回 resume 值，跳过 raise.
    user_decision = interrupt({"research_brief": placeholder_brief})
    _log("write_brief", f"用户决策={user_decision}")
    # 阶段 1 默认信任用户确认，直接放行；阶段 2 加分支处理拒绝回退.
    return Command(
        goto="research_supervisor",
        update={"research_brief": placeholder_brief},
    )


async def supervisor(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["supervisor_tools"]]:
    """supervisor 占位：阶段 3 接入 Send 并行派发 researcher 子图.

    阶段 1 不真正派发研究，只跳转 supervisor_tools 保证图拓扑完整.
    阶段 3 在此用 ConductResearch 工具拆分研究主题，并通过 Send 并行派发.
    """
    brief = state.get("research_brief", "(无 brief)")
    _log("supervisor", f"brief={brief[:80]}")
    # 占位：阶段 3 在此用 ConductResearch 工具拆分主题 + Send 并行派发.
    return Command(goto="supervisor_tools")


async def supervisor_tools(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["supervisor", "__end__"]]:
    """supervisor_tools 占位：处理 supervisor 的工具调用（阶段 3 接入真实工具）.

    原项目此节点处理三类工具调用：
      1. think_tool - supervisor 反思，循环回 supervisor
      2. ConductResearch - 派发 researcher 子图，循环回 supervisor
      3. ResearchComplete - 标记研究结束，跳 END 结束子图，
         由主图边 research_supervisor→final_report_generation 接管进入报告生成

    阶段 1 不接工具调用，直接结束研究阶段进入报告生成，
    保留此节点是为了让图拓扑对齐原项目的 supervisor ⇄ supervisor_tools 循环，
    阶段 3 接入真实工具时只需替换本节点占位逻辑，不动其他节点.
    """
    configurable = Configuration.from_runnable_config(config)
    iterations = state.get("research_iterations", 0)
    _log("supervisor_tools", f"iterations={iterations} max={configurable.max_researcher_iterations}")
    # 占位：阶段 1 默认研究已完成，直接进入报告生成.
    # 阶段 3 在此处理 ConductResearch/think_tool/ResearchComplete 三类工具调用.
    # 跳 END 结束子图，主图边 research_supervisor→final_report_generation 接管.
    return Command(goto="__end__")


async def final_report_generation(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    """最终报告生成占位：阶段 5 接入两阶段写作（outline → sections）.

    阶段 1 返回写死的占位报告，让图能完整跑到 END.
    """
    _log("final_report_generation", "生成占位报告")
    placeholder_report = "公司深度研究报告（占位）：阶段 5 接入两阶段写作后替换."
    return Command(
        goto="__end__",
        update={
            "final_report": placeholder_report,
            "messages": [AIMessage(content=placeholder_report)],
        },
    )
