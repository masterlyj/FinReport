"""命令行入口层:各报告类型的 CLI 薄壳.

子模块按需导入(不在此预导入),避免 `python -m company_report_kit.cli.<子模块>`
时被预导入重复加载触发 RuntimeWarning。

用法:
    python -m company_report_kit.cli.run "腾讯控股"          # 通用深度报告
    python -m company_report_kit.cli.snapshot "月之暗面"     # 非上市研究快照
"""
