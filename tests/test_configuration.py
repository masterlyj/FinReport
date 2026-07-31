"""Configuration 的纯逻辑测试:默认值与 from_runnable_config 四路径。

from_runnable_config 采用可选式加载:None/空 configurable→默认;未知键过滤;
构造异常回退默认,保证无 Studio 配置注入时图也能启动。
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from company_report_kit.configuration import Configuration


def test_defaults() -> None:
    """关键字段默认值抽查,锁定契约。"""
    c = Configuration()
    assert c.api_base == ""
    assert c.api_key == ""
    assert c.allow_clarification is True
    assert c.max_concurrent_research_units == 3
    assert c.max_researcher_iterations == 6
    assert c.max_react_tool_calls == 10
    assert c.research_model == "deepseek-v4-flash"
    assert c.research_model_max_tokens == 65536


def test_from_runnable_config_none() -> None:
    """config=None 时返回全默认实例。"""
    assert Configuration.from_runnable_config(None) == Configuration()


def test_from_runnable_config_empty_configurable() -> None:
    """configurable 为空 dict 时同样回退默认。"""
    assert Configuration.from_runnable_config({"configurable": {}}) == Configuration()


def test_from_runnable_config_partial_merge() -> None:
    """部分字段覆盖,其余保持默认。"""
    config: RunnableConfig = {
        "configurable": {
            "api_key": "sk-1",
            "allow_clarification": False,
            "max_react_tool_calls": 3,
        }
    }
    c = Configuration.from_runnable_config(config)
    assert c.api_key == "sk-1"
    assert c.allow_clarification is False
    assert c.max_react_tool_calls == 3
    # 未传字段保持默认
    assert c.max_concurrent_research_units == 3
    assert c.research_model == "deepseek-v4-flash"


def test_from_runnable_config_filters_unknown_keys() -> None:
    """未知键被过滤,不影响实例化(防 Studio 注入额外字段报错)。"""
    config: RunnableConfig = {
        "configurable": {
            "api_key": "sk-2",
            "unknown_field": "x",
            "another_unknown": 123,
        }
    }
    c = Configuration.from_runnable_config(config)
    assert c.api_key == "sk-2"


def test_from_runnable_config_invalid_falls_back() -> None:
    """字段类型非法导致构造异常时,回退全默认而非抛错。"""
    config: RunnableConfig = {
        "configurable": {
            "allow_clarification": "not-a-bool",  # str 给 bool 字段
            "max_react_tool_calls": "abc",  # str 给 int 字段
        }
    }
    # 整体构造失败 → except 捕获 → 返回 cls()
    c = Configuration.from_runnable_config(config)
    assert c == Configuration()
