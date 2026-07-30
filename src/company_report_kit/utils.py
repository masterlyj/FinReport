"""LLM 调用层基础设施.

统一用 ChatDeepSeek（deepseek: provider），让 langchain-deepseek 的 strict 兼容生效.
deepseek-v4-flash 默认开启思考模式，但思考模式不支持 strict 的强制 tool_choice，
故结构化输出节点（clarify/write_brief）关闭思考，报告生成节点保留思考以提升逻辑性.

思考开关通过 extra_body={"thinking": {"type": "disabled"|"enabled"}} 传给服务端，
configurable_fields 加入 extra_body 让 with_config 运行时切换.
"""

from datetime import datetime
from typing import Any

from langchain.chat_models import init_chat_model

from company_report_kit.configuration import Configuration

# 所有节点共用的可配置模型实例.
# configurable_fields 含 extra_body，节点运行时通过 with_config 切换思考开关.
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "extra_body"),
)


def get_today_str() -> str:
    """返回当前日期的可读字符串，供中文研报 prompt 填充.

    Returns:
        形如 '2024年1月15日' 的日期字符串.
    """
    now = datetime.now()
    # 使用 %m 和 %d，并加上“年月日”汉字
    return now.strftime("%Y年%-m月%d日")

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

