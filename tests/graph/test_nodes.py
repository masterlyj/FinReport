"""graph 节点测试(mock LLM,测 Command goto 与状态更新逻辑)。

不真调 LLM:覆盖 allow_clarification=False 的放行路径(不触模型),
其余需 LLM 的路径后续补(mock configurable_model)。
"""

from __future__ import annotations

import pytest

from company_report_kit.graph.nodes import clarify_with_user


@pytest.mark.anyio
async def test_clarify_skip_when_disabled() -> None:
    """allow_clarification=False 时直接跳 write_brief,不调 LLM。"""
    # 关闭澄清:节点应短路放行,无需 LLM 即可决策
    config = {"configurable": {"allow_clarification": False}}
    cmd = await clarify_with_user(state={"messages": []}, config=config)  # type: ignore[arg-type]
    assert cmd.goto == "write_brief"
