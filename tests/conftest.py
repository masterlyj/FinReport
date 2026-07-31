"""pytest 公共 fixture:key 检测与 mock 工具,供 network/live 测试复用。"""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def skip_without_deepseek_key() -> None:
    """无 DEEPSEEK_API_KEY 时跳过 live 测试。"""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")
