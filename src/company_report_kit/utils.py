"""LLM 调用层基础设施.

统一用 ChatDeepSeek（deepseek: provider），让 langchain-deepseek 的 strict 兼容生效.
deepseek-v4-flash 默认开启思考模式，但思考模式不支持 strict 的强制 tool_choice，
故结构化输出节点（clarify/write_brief）关闭思考，报告生成节点保留思考以提升逻辑性.

思考开关通过 extra_body={"thinking": {"type": "disabled"|"enabled"}} 传给服务端，
configurable_fields 加入 extra_body 让 with_config 运行时切换.

think_tool 是供 supervisor/researcher 反思的空工具，LLM 调用后原样返回 reflection，
用于在研究流程中创造显式的"暂停思考"动作.

RETRY_KWARGS 控制 with_retry 行为：
  - stop_after_attempt=8：最多重试 8 次
  - wait_exponential_jitter=True + exponential_jitter_params：
    指数退避（initial=2s, max=60s, exp_base=2）+ 随机抖动（0-3s），
    避免并发重试雪崩，应对 DeepSeek API 间歇性 500/503.
"""

from datetime import datetime
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.runnables.retry import ExponentialJitterParams
from langchain_core.tools import tool

from company_report_kit.configuration import Configuration

# 所有节点共用的可配置模型实例.
# configurable_fields 含 extra_body，节点运行时通过 with_config 切换思考开关.
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "extra_body"),
)


@tool
def think_tool(reflection: str) -> str:
    """反思工具：记录对研究进展的反思，规划下一步.

    supervisor 和 researcher 在搜索后或派发前调用，创造显式思考动作.
    调用后原样返回 reflection，不执行实际逻辑.
    """
    return reflection


# LLM 调用重试 kwargs.
# 指数退避 2s→4s→8s→16s→32s→60s + 0-3s 随机抖动，最多 8 次.
RETRY_KWARGS = {
    "stop_after_attempt": 8,
    "wait_exponential_jitter": True,
    "exponential_jitter_params": ExponentialJitterParams(initial=2, max=60, exp_base=2, jitter=3),
}


def get_today_str() -> str:
    """返回当前日期的可读字符串，供 prompt 填充."""
    now = datetime.now()
    return f"{now.year}年{now.month}月{now.day}日"


def get_model_config(
    configurable: Configuration,
    model: str,
    max_tokens: int,
    thinking: bool = False,
) -> dict[str, Any]:
    """构造传给 configurable_model.with_config 的运行时配置.

    Args:
        configurable: Configuration 实例，提供 api_key.
        model: 纯模型名（如 deepseek-v4-flash），加 deepseek: 前缀路由到 ChatDeepSeek.
        max_tokens: 单次调用最大输出 token.
        thinking: 是否开启思考模式. 结构化输出节点需 False（strict 要求），
            报告生成节点可 True（提升逻辑性）.

    Returns:
        含 model/max_tokens/api_key/extra_body/tags 的配置 dict.
    """
    return {
        "model": f"deepseek:{model}",
        "max_tokens": max_tokens,
        "api_key": configurable.api_key or None,
        "extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}},
        "tags": ["langsmith:nostream"],
    }


def get_notes_from_tool_calls(messages: list) -> list[str]:
    """从消息列表提取所有 ToolMessage 的 content 作为 notes.

    supervisor_tools 退出研究阶段时调用，把 ConductResearch 的结果汇聚成 notes，
    供 final_report_generation 引用.
    """
    from langchain_core.messages import filter_messages
    return [str(m.content) for m in filter_messages(messages, include_types="tool")]

