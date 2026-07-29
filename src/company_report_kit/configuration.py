"""Company Report Kit 的可配置项定义.

对齐 open_deep_research 的 Configuration schema，只暴露阶段 1 骨架要用的字段.
后续阶段接入研究子图、MCP、搜索 API 时，再在此类追加字段，避免一次性引入未使用配置.
"""

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """管理 Company Report Kit 全流程的可调参数.

    所有字段带 x_oap_ui_config metadata，可在 LangGraph Studio 的
    "Manage Assistants" 面板可视化配置，无需改代码.

    Attributes:
        allow_clarification: 是否在研究前向用户追问澄清. 关闭则直接进入 brief 生成.
        max_concurrent_research_units: supervisor 并行派发的 researcher 子图上限.
            默认 3 对齐 FinSight max_concurrent，避免触发 provider rate limit.
        max_researcher_iterations: supervisor 反思轮数上限. 超过即结束研究阶段，
            防止无限循环消耗 token.
        research_model: 研究 LLM（clarify / write_brief / supervisor / researcher）.
            格式 provider:model，由 init_chat_model 解析.
        research_model_max_tokens: 研究模型单次调用最大输出 token.
        final_report_model: 最终报告生成 LLM. 可与研究模型不同，如换更大上下文模型.
        final_report_model_max_tokens: 报告模型单次调用最大输出 token.
    """

    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "是否在开始研究前向用户追问澄清（研究范围、报告深度等）",
            }
        },
    )

    max_concurrent_research_units: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "supervisor 并行派发的 researcher 子图上限. 默认 3 对齐 FinSight max_concurrent，避免 rate limit.",
            }
        },
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "supervisor 反思轮数上限. 超过即结束研究阶段进入报告生成.",
            }
        },
    )

    # 阶段 1 仅声明模型字段，真实 LLM 调用在阶段 2 接入 init_chat_model 后生效.
    # 先占位是为了让 Studio 能尽早展示配置项，且让节点签名稳定.
    research_model: str = Field(
        default="deepseek:deepseek-v4-flash",
        metadata={
            "x_oap_ui_config": {
                "type": "string",
                "default": "deepseek:deepseek-v4-flash",
                "description": "研究用 LLM（clarify / write_brief / supervisor / researcher 主体）. 格式 provider:model，由 init_chat_model 解析.",
            }
        },
    )
    research_model_max_tokens: int = Field(
        default=16384,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 16384,
                "description": "研究模型单次调用最大输出 token.",
            }
        },
    )
    final_report_model: str = Field(
        default="deepseek:deepseek-v4-flash",
        metadata={
            "x_oap_ui_config": {
                "type": "string",
                "default": "deepseek:deepseek-v4-flash",
                "description": "最终报告生成 LLM. 可与研究模型不同，如换更大上下文模型.",
            }
        },
    )
    final_report_model_max_tokens: int = Field(
        default=16384,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 16384,
                "description": "报告模型单次调用最大输出 token.",
            }
        },
    )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None) -> "Configuration":
        """从 LangGraph RunnableConfig 提取 Configuration.

        采用可选式加载：无 config 或 configurable 为空时返回全默认值，
        保证节点在纯本地测试（无 Studio 配置注入）时也能运行.

        Args:
            config: LangGraph 运行时配置，可含 configurable 字段.

        Returns:
            填充后的 Configuration 实例. 任意字段解析失败时回退到默认值，
            避免单个配置错误导致整个图不可启动.
        """
        try:
            configurable = (config or {}).get("configurable", {})
            return cls(**{k: v for k, v in configurable.items() if k in cls.model_fields})
        except Exception:
            return cls()

