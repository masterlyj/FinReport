"""Company Report Kit entrypoint.

这是占位入口，后续接入正式的公司研究报告生成流程：
范围澄清 → 研究计划 → 资料采集 → 证据结构化 → 公司建模 → 分析 → 报告 → 质量检查 → 导出。

日志走 logging + rich；占位提示用共享 rich Console 输出。
"""

from __future__ import annotations

from company_report_kit.logging_utils import console, setup_logging


def main() -> None:
    setup_logging()
    console.print("Company Report Kit - 公司深度研究报告生成系统", style="bold")
    console.print("正式流程待接入，参见 PRD.md 与 docs/PROJECT_FLOW.md")


if __name__ == "__main__":
    main()
