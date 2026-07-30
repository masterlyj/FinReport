"""Company Report Kit 主图定义与编译.

三层图结构（阶段1先实现前两层，researcher子图阶段3接入）：
  主图: clarify → write_brief → research_supervisor(挂supervisor子图) → final_report
  supervisor子图: supervisor ⇄ supervisor_tools（Command goto 路由）
  researcher子图: researcher ⇄ researcher_tools → compress_research（阶段3）

主图的 research_supervisor 节点挂的是 supervisor_subgraph 编译后的实例，
非普通函数节点，这样主图状态与子图状态通过同名字段自动映射流转.

checkpointer 用 MemorySaver，阶段0决策：先跑通后换 SQLite 持久化.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from company_report_kit.configuration import Configuration
from company_report_kit.graph.nodes import (
    clarify_with_user,
    final_report_generation,
    supervisor,
    supervisor_tools,
    write_brief,
)
from company_report_kit.graph.state import AgentInputState, AgentState, SupervisorState

###################
# supervisor 子图
###################
# 阶段1只有 supervisor + supervisor_tools 两节点，无 researcher 派发.
# 阶段3接入时，在 supervisor_tools 内部用 Send 派发 researcher_subgraph，
# 无需改动本子图的 add_node/add_edge 结构.
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
# supervisor ⇄ supervisor_tools 之间无显式 edge：
# 跳转由节点返回的 Command goto 决定（supervisor→supervisor_tools，
# supervisor_tools→supervisor/final_report_generation/__end__）.
supervisor_subgraph = supervisor_builder.compile()

###################
# 主图
###################
# research_supervisor 节点挂 supervisor_subgraph 实例，非函数.
# 主图 AgentState 与 SupervisorState 通过同名字段映射：
#   research_brief / notes / raw_notes 双向流通
#   supervisor_messages 独立于主图 messages，子图内部流转.
main_builder = StateGraph(
    AgentState,
    input=AgentInputState,
    config_schema=Configuration,
)
main_builder.add_node("clarify_with_user", clarify_with_user)
main_builder.add_node("write_brief", write_brief)
main_builder.add_node("research_supervisor", supervisor_subgraph)
main_builder.add_node("final_report_generation", final_report_generation)

# 显式边：clarify→write_brief 与 supervisor→final_report 是固定流.
# write_brief→supervisor 由 Command goto 路由（interrupt 后放行），
# 故不写显式 edge，避免与 Command 跳转冲突.
main_builder.add_edge(START, "clarify_with_user")
main_builder.add_edge("research_supervisor", "final_report_generation")
main_builder.add_edge("final_report_generation", END)

# checkpointer：MemorySaver 内存级，进程退出即丢失.
# 阶段2换 SqliteSaver 后支持跨进程续跑.
graph = main_builder.compile(checkpointer=MemorySaver())
