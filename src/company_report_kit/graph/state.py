"""Company Report Kit 的图状态定义.

按"图状态只承载编排数据，证据走 Variable Memory"的分层决策，
这里只定义跨节点传递的控制流数据结构：当前 brief、topic 列表、notes.
证据/分析结果/检索 embedding 不进图状态，阶段 2 通过 config 注入 Memory.

对齐三层状态划分：
  AgentState / SupervisorState / ResearcherState / ResearcherOutputState
并保留其结构化输出模型（ConductResearch / ResearchComplete / ClarifyWithUser /
ResearchQuestion），这些既是 LLM 输出 schema，也是工具调用参数定义，
阶段 3 接 Send 派发和阶段 2 接 LLM 时直接复用.
"""

from typing import Annotated, Literal, Optional, TypeVar

T = TypeVar("T")

def last_value(current: T, new: T) -> T:
    """覆盖式标量 reducer：直接取新值，丢弃旧值（计数器场景）."""
    return new

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict


def override_reducer(current_value, new_value):
    """覆盖式 reducer：允许节点整体覆盖某字段而非追加.

    LangGraph 默认对 list 字段做 append，但 brief / final_report 这类
    字段期望"后写覆盖前写"，所以用本 reducer 配合
    {"type": "override", "value": ...} 语法实现整体替换.

    Args:
        current_value: 当前 state 中的值.
        new_value: 节点 update 的新值. 若为 {"type": "override", ...}
            则整体替换；否则走默认 add 语义.

    Returns:
        替换后的值或追加后的值.
    """
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return current_value + new_value


###################
# 结构化输出模型
###################
class SourceGrouping(BaseModel):
    """把若干来源聚成事件簇：同一事件/转载聚一簇，孤立来源单独成簇."""

    event_summary: str = Field(description="该簇报道的事件简述（一句话）.")
    key_facts: str = Field(description="从来源原文精炼总结的关键事实.")
    primary_url: str = Field(description="代表 URL，优先一手/权威/信息最全来源.")
    supporting_urls: list[str] = Field(default_factory=list, description="佐证 URL 列表，报道同一事件的其他来源.")


class SourceGroupingBatch(BaseModel):
    """分组节点一次输出全部事件簇，避免两两比较的平方成本."""

    model_config = ConfigDict(extra="forbid")

    clusters: list[SourceGrouping] = Field(description="全部事件簇，每簇一个代表 URL + 佐证 URL 列表.")


class ReviewIssue(BaseModel):
    """审查发现的单条问题（引用错配/无出处/口径冲突）.

    每条问题显式携带所属章节编号与处理动作，供审查→修正闭环按章节归组：
      - action=fix 的章节反馈给对应 researcher 重新搜索+修正
      - action=adjudicate 的跨章节问题由主 agent 标注裁决，不交给单个 researcher
    """

    model_config = ConfigDict(extra="forbid")

    section: int = Field(
        description="问题所属章节编号（1=投融资, 2=竞品, 3=团队, 4=业务, 5=财务）. 跨章节问题取 0.",
    )
    issue_type: Literal["引用错配", "无出处", "口径冲突"] = Field(description="问题类型.")
    report_text: str = Field(description="报告原文片段（含脚注标记）.")
    url: str = Field(description="被指摘的脚注 URL；无出处时可多个 URL 用逗号分隔，无 URL 填空串.")
    evidence: str = Field(description="问题对应 URL 的原文实际内容摘录（保留关键措辞与数字，供修正时对照核实）.")
    action: Literal["fix", "adjudicate"] = Field(
        description="处理动作: fix=反馈给对应 researcher 修正; adjudicate=跨章节口径冲突，主 agent 标注裁决.",
    )


class ReviewResult(BaseModel):
    """审查的结构化输出容器.

    with_structured_output(list[ReviewIssue]) 在 DeepSeek 下返回 {"iterable": [...]}
    而非列表本身,故包一层容器让 langchain 走 pydantic schema 解析路径.
    """

    model_config = ConfigDict(extra="forbid")

    issues: list[ReviewIssue] = Field(default_factory=list, description="审查发现的全部问题.")


class ConductResearch(BaseModel):
    """supervisor 调用此工具派发研究任务给 researcher 子图.

    阶段 3 接 Send 并行派发时，此模型作为工具调用的参数 schema，
    也作为 supervisor LLM 的结构化输出约束.
    """

    research_topic: str = Field(
        description="研究主题. 应为单一主题，描述需详尽（至少一段话），便于 researcher 聚焦.",
    )


class ResearchComplete(BaseModel):
    """supervisor 调用此工具标记研究阶段结束.

    触发条件：supervisor 认为已收集足够证据，或达到 max_researcher_iterations.
    """


class ClarifyWithUser(BaseModel):
    """clarify 节点的 LLM 结构化输出，判断是否需要向用户追问."""

    model_config = ConfigDict(extra="forbid")

    need_clarification: bool = Field(
        description="是否需要向用户追问澄清.",
    )
    question: str = Field(
        description="向用户追问的问题，用于澄清报告范围.",
    )
    verification: str = Field(
        description="无需追问时返回给用户的确认信息，表示研究即将开始.",
    )


class ResearchQuestion(BaseModel):
    """write_brief 节点的 LLM 结构化输出，生成研究简报."""

    model_config = ConfigDict(extra="forbid")

    research_brief: str = Field(
        description="研究简报，将用于指导后续研究.",
    )


###################
# 状态定义
###################
class AgentInputState(MessagesState):
    """图的输入状态，只暴露 messages 字段.

    收窄输入 schema，避免用户线程意外注入 research_brief / notes 等内部字段，
    强制走 clarify → write_brief 节点产出.
    """


class AgentState(MessagesState):
    """主图状态，贯穿 clarify → write_brief → supervisor → report.

    Attributes:
        messages: 与用户的对话历史（含澄清问答）.
        supervisor_messages: supervisor 子图内部消息流，独立于主 messages，
            避免研究员的工具调用污染用户可见对话.
        research_brief: write_brief 节点产出、经 interrupt 人工确认的研究简报，
            同时承担研究范围与方向，驱动后续 supervisor.
        notes: 研究阶段产出的压缩笔记，汇总各 researcher 的 compressed_research.
        raw_notes: 原始工具输出（未经压缩），保留供最终报告引用追溯.
        sections: researcher 产出的章节文本(按派发顺序),供 assemble_sections 拼接.
        final_report: 最终报告文本.
    """

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str] = None
    notes: Annotated[list[str], override_reducer] = []
    raw_notes: Annotated[list[str], override_reducer] = []
    sections: Annotated[list[str], override_reducer] = []
    final_report: Optional[str] = None


class SupervisorState(TypedDict):
    """supervisor 子图状态，独立于主图.

    单独划分是因为 supervisor 的反思循环不需要污染用户 messages，
    其工具调用（ConductResearch / ResearchComplete）只在子图内流转.

    Attributes:
        supervisor_messages: supervisor 与 researcher 交互的消息流.
        research_brief: 继承自主图，作为 supervisor 决策上下文.
        notes: 各 researcher 回传的压缩笔记，用 override_reducer 汇聚.
        research_iterations: 当前反思轮数，用于触发 max_researcher_iterations 退出.
        raw_notes: 原始工具输出（未经压缩），保留供最终报告引用追溯.
        sections: researcher 产出的章节文本(按派发顺序),供主图拼接.
    """

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: Annotated[int, last_value] = 0
    raw_notes: Annotated[list[str], override_reducer] = []
    sections: Annotated[list[str], override_reducer] = []


class ResearcherState(TypedDict):
    """单个 researcher 子图状态.

    每个 researcher 由 supervisor 通过 Send 派发，独立运行.
    内部字段 researcher_messages / tool_call_iterations 不回传父图，
    通过 ResearcherOutputState 显式控制暴露给 supervisor 的字段.

    Attributes:
        researcher_messages: researcher 与工具交互的消息流，子图内部使用.
        tool_call_iterations: 当前工具调用轮数，触发 max_react_tool_calls 退出.
        research_topic: supervisor 指派的研究主题.
        compressed_research: 压缩后的研究摘要，回传 supervisor.
        raw_notes: 原始工具输出，保留供证据库写入.
    """

    researcher_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    tool_call_iterations: Annotated[int, last_value] = 0
    research_topic: str
    clusters: list[SourceGrouping] = []
    section_text: str = ""
    raw_notes: Annotated[list[str], override_reducer] = []


class ResearcherOutputState(BaseModel):
    """researcher 子图的输出 schema，显式控制回传父图的字段.

    只暴露章节文本和原始笔记，屏蔽 researcher_messages / tool_call_iterations /
    clusters 等内部状态，避免子图内部消息流污染 supervisor.
    """

    section_text: str = Field(
        description="本维度报告章节的 markdown 文本（含脚注引用）.",
    )
    raw_notes: Annotated[list[str], override_reducer] = Field(
        default_factory=list,
        description="原始工具输出（未经压缩），保留供证据库写入.",
    )
