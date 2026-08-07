"""utils.py 的纯逻辑测试:get_today_str / get_model_config / get_notes_from_tool_calls / think_tool。

不触真实 LLM:get_model_config 只验返回 dict 的拼装;think_tool 是空工具原样返回。
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from company_report_kit.configuration import Configuration
from company_report_kit.utils import (
    RETRY_KWARGS,
    get_model_config,
    get_notes_from_tool_calls,
    get_today_str,
    think_tool,
)


def test_get_today_str_format() -> None:
    """get_today_str 返回「年月月日日」可读格式(不验具体日期,可重复)。"""
    out = get_today_str()
    assert re.fullmatch(r"\d{4}年\d{1,2}月\d{1,2}日", out)


def test_get_model_config_thinking_off() -> None:
    """thinking=False(结构化输出节点)时 thinking 显式 disabled。"""
    cfg = Configuration()
    mc = get_model_config(cfg, "deepseek-v4-flash", 65536, thinking=False)
    assert mc["model"] == "deepseek:deepseek-v4-flash"
    assert mc["max_tokens"] == 65536
    assert mc["extra_body"]["thinking"]["type"] == "disabled"
    assert mc["api_key"] is None  # Configuration 默认空串 → 透传 None
    assert mc["tags"] == ["langsmith:nostream"]


def test_get_model_config_thinking_on() -> None:
    """thinking=True(报告节点)时 thinking enabled。"""
    cfg = Configuration()
    mc = get_model_config(cfg, "rpt-model", 1024, thinking=True)
    assert mc["extra_body"]["thinking"]["type"] == "enabled"
    assert mc["model"] == "deepseek:rpt-model"


def test_get_model_config_api_key_passthrough() -> None:
    """configurable.api_key 非空时透传给配置。"""
    cfg = Configuration(api_key="sk-xxx")
    mc = get_model_config(cfg, "m", 100)
    assert mc["api_key"] == "sk-xxx"


def test_get_model_config_empty_api_key_becomes_none() -> None:
    """api_key 为空串时透传 None(锁住 or None 逻辑,防空串被当 key 传)。"""
    cfg = Configuration(api_key="")
    mc = get_model_config(cfg, "m", 100)
    assert mc["api_key"] is None


def test_get_notes_from_tool_calls_extracts_tool_only() -> None:
    """仅取 ToolMessage.content,HumanMessage/AIMessage 不进 notes。"""
    msgs = [
        HumanMessage(content="用户提问"),
        AIMessage(content="模型回答"),
        ToolMessage(content="工具结果A", name="t", tool_call_id="1"),
        ToolMessage(content="工具结果B", name="t", tool_call_id="2"),
    ]
    assert get_notes_from_tool_calls(msgs) == ["工具结果A", "工具结果B"]


def test_get_notes_from_tool_calls_empty() -> None:
    """无 ToolMessage 时返回空列表。"""
    assert get_notes_from_tool_calls([AIMessage(content="x")]) == []


def test_think_tool_returns_reflection() -> None:
    """think_tool 原样返回 reflection,不执行逻辑。"""
    assert think_tool.invoke({"reflection": "需要补查产能数据"}) == "需要补查产能数据"


def test_retry_kwargs_shape() -> None:
    """RETRY_KWARGS 开启指数退避抖动 + 足够次数上限,应对 API 限流。"""
    # 只守「重试开启 + 退避」语义,不锁死具体值(当前 stop_after_attempt=8)。
    assert RETRY_KWARGS["stop_after_attempt"] >= 5
    assert RETRY_KWARGS["wait_exponential_jitter"] is True
