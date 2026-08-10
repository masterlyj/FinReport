"""Company Report Kit 的可配置项定义.

统一走 OpenAI 兼容格式：所有 endpoint 用 ChatOpenAI 调用，
通过 api_base + api_key + model 三件套指向具体服务，
不依赖 provider 专属类（如 ChatDeepSeek），天然兼容官方/中转/自建代理.
"""

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """管理 Company Report Kit 全流程的可调参数.

    所有字段带 x_oap_ui_config metadata，可在 LangGraph Studio 的
    "Manage Assistants" 面板可视化配置，无需改代码.
    """

    # --- endpoint 配置（所有节点共用）---
    api_base: str = Field(
        default="",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "string",
                "default": "",
                "description": "OpenAI 兼容 endpoint 的 base URL. 空则用 SDK 默认地址，支持官方/中转/自建代理.",
            }
        },
    )
    api_key: str = Field(
        default="",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "string",
                "default": "",
                "description": "对应 endpoint 的 API key.",
            }
        },
    )

    # --- 流程控制 ---
    allow_clarification: bool = Field(
        default=True,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "是否在开始研究前向用户追问澄清（研究范围、报告深度等）",
            }
        },
    )
    max_concurrent_research_units: int = Field(
        default=3,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "supervisor 并行派发的 researcher 子图上限. 默认 3，避免 rate limit.",
            }
        },
    )
    max_researcher_iterations: int = Field(
        default=6,
        json_schema_extra={
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

    max_react_tool_calls: int = Field(
        default=10,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "单个 researcher 工具调用轮数上限. 超过即结束研究进入压缩.",
            }
        },
    )

    # --- 模型配置 ---
    research_model: str = Field(
        default="deepseek-v4-flash",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "string",
                "default": "deepseek-v4-flash",
                "description": "研究用 LLM 模型名（clarify/write_brief/supervisor/researcher）. 纯模型名，不含 provider 前缀.",
            }
        },
    )
    research_model_max_tokens: int = Field(
        default=65536,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 65536,
                "description": "研究模型单次调用最大输出 token.",
            }
        },
    )
    final_report_model: str = Field(
        default="deepseek-v4-flash",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "string",
                "default": "deepseek-v4-flash",
                "description": "最终报告生成 LLM 模型名. 可与研究模型不同.",
            }
        },
    )
    final_report_model_max_tokens: int = Field(
        default=65536,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 65536,
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
        except Exception:  # noqa: BLE001 - 配置异常时回退默认,不阻断流水线
            return cls()
