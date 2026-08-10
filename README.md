# Company Report Kit

公司深度研究报告生成系统。

## 一句话定位

用户输入一个公司，系统完成研究范围澄清、研究计划生成、资料采集、证据结构化、公司建模、分析写作、质量校验和报告导出，生成可追溯、可复核、可审校的公司深度研究报告初稿。

## 当前文档

- [PRD.md](PRD.md)：产品定位、边界、流程、模块、MVP 和验收标准。
- [docs/PROJECT_FLOW.md](docs/PROJECT_FLOW.md)：流程梳理与借鉴思路。
- [docs/MIGRATION_PLAN.md](docs/MIGRATION_PLAN.md)：FinSight → LangGraph 迁移方案与阶段进度。

## 核心分层

```text
Company Report Kit
  ├─ 研究流程层：澄清、任务说明、研究计划、任务编排、资料采集
  ├─ 公司领域层：基本面、业务拆解、财务分析、估值、风险
  ├─ 证据治理层：Evidence、来源评级、指标口径、结论映射
  ├─ 质量控制层：引用检查、口径冲突、缺口识别、报告审校
  └─ 交付层：公司报告模板、编辑审阅、导出
```

## 最小边界

- 做：公司深度研究报告生成（上市:A 股 / 港股 / 美股;未上市:如大模型厂商 Kimi）。
- 加强：证据链、来源评级、指标口径、公司业务结构和质量检查。
- 不做：通用聊天问答、任意主题百科报告、自动投资评级、无人审校最终研报。

## 快速开始

```bash
# 安装依赖
uv sync

# 配置环境变量
cp .env.example .env

# 通用深度报告（上市/开放大纲，clarify + write_brief 人工确认 + supervisor 动态拆分）
python -m company_report_kit.cli.run "腾讯控股"

# 非上市公司研究快照（固定 5 维度大纲：投融资/竞品/团队/业务/财务）
python -m company_report_kit.cli.snapshot "月之暗面"
```

## 目录结构

```text
src/company_report_kit/
  ├─ graph/         通用深度报告引擎（clarify→write_brief→supervisor→researcher→final_report）
  ├─ outlines/      标准大纲模板层（ResearchDimension/OutlineTemplate + 非上市 5 维度实例）
  ├─ workflows/     固化流水线（assembly 组装 / review 审查修正闭环 / snapshot 非上市流水线）
  ├─ cli/           命令行入口（run 通用 / snapshot 非上市快照）
  ├─ search_tools/  搜索工具（DeepSeek/Tavily/DuckDuckGo + 正文提取）
  ├─ logging_utils.py  日志与终端输出（logging + rich 统一配置）
  ├─ configuration.py / utils.py / prompts.py
```

## 日志与终端输出

进度/诊断日志走标准 `logging`（RichHandler 渲染，带级别与时间戳），面向用户的
报告正文/审查结果/文件路径走共享 rich Console。CLI 入口自动调用 `setup_logging()`，
无需手动配置。

- 默认日志级别 INFO，可用环境变量 `DEEPSEEK_LOG_LEVEL` 调整（如 `DEBUG` 排查）。
- 第三方库（urllib3/langgraph/ddgs…）默认静音到 WARNING，避免刷屏。
- 库代码内用 `from company_report_kit.logging_utils import get_logger` 取带命名空间的 logger。

