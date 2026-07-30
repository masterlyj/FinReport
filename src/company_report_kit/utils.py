"""LLM 调用层基础设施.

所有 endpoint 统一走 OpenAI 兼容格式，通过 ChatOpenAI 调用：
  api_base + api_key + model 三件套指向具体服务（官方/中转/自建代理）

configurable_model 是所有节点共用的可配置模型实例，
节点在运行时通过 .with_config({"model": ..., "api_key": ..., "api_base": ...})
指定具体配置，而非编译期写死.
"""

from datetime import datetime

from langchain.chat_models import init_chat_model

# 所有节点共用的可配置模型实例.
# configurable_fields 允许运行时通过 with_config 覆盖 model/max_tokens/api_key/api_base，
# 这样一个实例即可服务 clarify/write_brief/supervisor/final_report 等多个节点，
# 各节点按 Configuration 指定不同模型与 endpoint.
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "api_base"),
)


def get_today_str() -> str:
    """返回当前日期的可读字符串，供 prompt 填充.

    Returns:
        形如 'Mon Jan 15, 2024' 的日期字符串.
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

