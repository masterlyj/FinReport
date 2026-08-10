# PRD：公司级别研究报告生成系统

## 1. 产品定位

Company Report Kit 是一个面向公司（含上市公司 A 股 / 港股 / 美股,及未上市公司如大模型厂商 Kimi）深度研究的报告生成系统。

它的目标不是简单生成一篇文章，而是把公司研究拆成一条可执行、可追溯、可复核的流程：明确研究范围、制定研究计划、采集资料、整理证据、完成财务与业务分析、生成报告并进行质量检查。

核心分层：

```text
Company Report Kit
  ├─ 研究流程层：澄清、任务说明、研究计划、任务编排、资料采集
  ├─ 公司领域层：基本面、业务拆解、财务分析、估值、风险
  ├─ 证据治理层：Evidence、来源评级、指标口径、结论映射
  ├─ 质量控制层：引用检查、口径冲突、缺口识别、报告审校
  └─ 交付层：公司报告模板、编辑审阅、导出
```

## 1.1 两套 workflow 与代码目录结构

系统内置两条报告流水线，共用同一套 researcher 子图、搜索工具与审查闭环：

| | 通用深度报告引擎 | 非上市研究快照（固化特例） |
|---|---|---|
| 入口 | `cli/run.py` | `cli/snapshot.py` |
| 大纲来源 | supervisor LLM 动态拆分（开放） | `outlines/unlisted.py` 固定 5 维度 |
| 流程 | clarify → write_brief(HIL) → supervisor → researcher → final_report | 模板 → 并行 researcher 各写章节 → 组装 → 审查 → 修正闭环 |
| 适用 | 上市公司深度报告 | 非上市公司（Kimi 等） |

目录结构：

```text
src/company_report_kit/
  ├─ graph/         通用深度报告引擎（并存保留）
  ├─ outlines/      标准大纲模板层：约束 researcher 调研范围的关键
  │   ├─ base.py    ResearchDimension / OutlineTemplate 抽象
  │   └─ unlisted.py 非上市 5 维度实例（投融资/竞品/团队/业务/财务）
  ├─ workflows/     固化流水线
  │   ├─ assembly.py   章节组装（编号层级 + 脚注重编号）
  │   ├─ review.py     审查→修正闭环（run_review / fix_section / 分组裁决）
  │   └─ snapshot.py   非上市流水线（模板→并行→组装→审查→修正）
  ├─ cli/           入口薄壳（run / snapshot）
  ├─ search_tools/  多源搜索 + 正文提取
  ├─ configuration.py / utils.py / prompts.py
```

**大纲模板约束机制**：researcher 的 `research_topic` 只能来自
`UNLISTED_TEMPLATE.topics_for(company)`——即模板维度 prompt 填充公司名。
子 Researcher 看不到模板之外的调研范围，实现"只能围绕大纲主题调研"；
再加 researcher system prompt 的"不扩展到大纲外内容"硬约束。

新增报告类型（行业报告、上市快报）只需新增一个 `outlines/*.py` 模板实例，
`workflows/snapshot.py` 流水线不改即复用。

## 2. 背景与问题

公司研究报告通常需要人工完成资料搜集、财报阅读、行业对标、数据核验、图表整理和文字撰写。主要痛点包括：

1. 资料分散：公告、财报、电话会、行业数据、新闻、公司官网、卖方研报分布在不同来源。
2. 口径不一：营收、毛利率、产能、渗透率、用户数等指标常有不同统计口径。
3. 过程难沉淀：每换一家公司，研究员都要重新搭建资料清单和分析框架。
4. 证据难追溯：报告结论和原始来源之间缺少结构化映射。
5. 质量不稳定：有的报告结构完整但证据弱，有的资料充分但逻辑不清。
6. 交付周期长：从选题到初稿需要多轮人工整理、判断和审校。

本产品要解决的是"公司研究生产流程标准化"，而不是单纯的文本生成。

## 3. 目标用户与场景

目标用户：

- 行业研究员：快速形成有证据链的公司报告初稿。
- 投资分析师：判断商业模式、竞争格局、估值和风险。
- 卖方/买方分析师：快速覆盖新公司、更新现有标的。
- 企业管理者：了解竞对、上下游议价权和客户结构。
- 产品/解决方案团队：理解客户业务、采购逻辑和战略方向。

典型任务：

- 腾讯控股公司深度研究。
- 贵州茅台公司深度研究。
- 特斯拉（TSLA）公司深度研究。
- 比亚迪公司深度研究。
- 英伟达（NVDA）公司深度研究。
- 月之暗面（Kimi）公司深度研究（未上市,验证非上市覆盖）。

## 4. 产品边界

### 做什么

- 公司深度研究报告生成（上市:A 股 / 港股 / 美股;未上市:如大模型厂商）。
- 研究范围澄清和研究任务说明。
- 研究计划生成与人工确认。
- 公开资料检索、用户上传资料解析、行情/财报数据接入。
- Evidence 证据库、来源评级、指标口径和引用管理。
- 公司基本面、业务拆解、财务分析、估值和风险分析。
- 报告初稿生成、质量检查、人工修订和导出。

### 不做什么

- 不做通用聊天问答。
- 不做任意主题百科报告。
- 不绕权抓取付费数据库。
- 不自动给投资评级、买卖建议或组合推荐。
- 不承诺生成无需审校的最终研报。
- MVP 不做多人权限、SaaS 计费、移动端、复杂 BI 仪表盘。

## 5. 核心流程

```text
用户输入公司（名称或股票代码）
  ↓
范围澄清（市场、时间、报告深度、关注点）
  ↓
生成研究任务说明（ResearchBrief）
  ↓
生成公司研究计划（ResearchPlan）
  ↓
用户确认计划
  ↓
专题研究单元采集资料
  ↓
Evidence 证据结构化
  ↓
公司建模与分析（业务、财务、估值、风险）
  ↓
报告初稿生成
  ↓
质量检查与人工修订
  ↓
导出报告
```

关键控制点：

- 研究计划未经确认，不进入正式采集。
- 资料必须先进入 Evidence，不直接拼进正文。
- 核心结论必须能追溯到 Evidence。
- 财务数字必须有来源、时间和口径。
- 来源冲突或资料不足必须显式标记。

## 6. 核心模块

| 模块 | 职责 | 输出 |
|---|---|---|
| 输入与澄清 | 标准化用户需求，补齐公司、市场、时间、报告类型 | ResearchTask |
| 研究任务说明 | 将用户需求转为可执行研究指令 | ResearchBrief |
| 研究计划 | 拆分公司研究专题和报告结构 | ResearchPlan |
| 研究编排 | 控制专题研究、顺序、并发和预算 | CollectTask / AnalysisTask |
| 专题研究 | 检索公告、行业、竞争、业务、财务、估值等资料 | RawFinding |
| Evidence Builder | 压缩、去重、评级和结构化证据 | Evidence |
| 公司建模 | 建立业务结构、竞争格局、产业链位置 | CompanyGraph |
| 分析引擎 | 形成业务、财务、估值、趋势和风险分析 | AnalysisResult |
| 报告生成 | 生成公司报告初稿 | ReportDraft |
| 质量检查 | 检查引用、事实、口径、逻辑和缺口 | QualityReport |
| 导出 | 输出 Markdown/Word/PDF | FinalReport |

## 7. 五层职责说明

### 研究流程层

负责把一个用户输入的公司（名称或股票代码）变成可执行任务，包括范围澄清、研究任务说明、研究计划、任务拆分、资料采集和过程调度。

不负责公司专业判断，也不直接负责最终审校。

### 公司领域层

负责定义公司报告应该研究什么，包括基本面、业务拆解、财务分析、估值、竞争格局和风险。

不负责底层资料检索和证据存储。

### 证据治理层

负责管理证据，包括来源、评级、发布时间、数据时间范围、指标、数值、单位、口径、原文链接和支撑结论。

不负责创造新观点。

### 质量控制层

负责检查报告质量，包括引用缺失、来源冲突、口径不一致、无证据结论、逻辑跳跃、重复段落和风险提示不足。

不负责替代研究员做最终判断。

### 交付层

负责面向用户展示、编辑、审阅和导出报告。

不负责研究流程本身。

## 8. 核心数据对象

### ResearchTask

- company_name
- stock_code
- market（A 股 / 港股 / 美股）
- time_range
- report_type（深度 / 点评 / 覆盖）
- depth
- focus_areas
- required_sources
- excluded_sources
- output_format
- status

### ResearchBrief

- research_goal
- research_scope
- key_questions
- business_focus
- output_requirements
- constraints

### Evidence

- evidence_id
- claim
- source_name
- source_type
- publish_date
- data_period
- metric
- value
- unit
- reliability
- url_or_file
- excerpt
- note

### CompanyNode

- segment：业务分部 / 产品线 / 子公司
- name
- description
- revenue_share
- representative_products
- key_resources
- bottlenecks
- bargaining_power

### AnalysisResult

- topic
- conclusion
- assumptions
- evidence_ids
- confidence
- risks

### Report

- outline
- sections
- charts
- references
- quality_report
- export_files

## 9. 默认报告结构

1. 核心结论与投资亮点。
2. 公司概况与发展历程。
3. 商业模式与业务拆解。
4. 行业地位与竞争格局。
5. 财务分析：盈利、成长、现金流与质量。
6. 估值分析：相对估值与绝对估值。
7. 成长驱动与未来展望。
8. 风险提示。
9. 附录：证据表与方法说明。

## 10. MVP 范围

### MVP 必须有

- 公司（名称或股票代码）输入。
- 范围澄清。
- 研究任务说明。
- 研究计划生成和人工确认。
- 有限专题研究。
- Evidence 证据库。
- 公司业务结构图和 CompanyGraph。
- 基础分析结果（业务、财务、估值、风险）。
- 报告初稿。
- 引用/口径/缺口质量检查。
- Markdown 和 Word 导出。

### MVP 可以简化

- 图表先做图表建议和表格，不强制复杂可视化。
- 数据源先支持公开网页和用户上传资料。
- UI 先做简单 Web/CLI 原型。
- PDF 导出可以后置。

### MVP 不做

- 通用研究平台。
- 多人协作权限。
- 付费数据库自动接入。
- 自动投资评级。
- 复杂 BI 仪表盘。
- 移动端。

## 11. 验收标准

- 能完成一个试点公司从输入到导出的完整流程。
- 报告覆盖业务、财务、估值、竞争和风险。
- 核心结论均有关联 Evidence。
- 关键数字大部分具备来源、时间和口径。
- 系统能标记资料不足、来源冲突和无证据结论。
- 输出报告可作为研究员初稿，而不是最终免审校报告。

## 12. 首批试点

优先选择资料公开、业务结构清晰、市场关注度高的公司：

1. 腾讯控股（00700.HK）。
2. 贵州茅台（600519.SH）。
3. 特斯拉（TSLA）。
4. 比亚迪（002594.SZ / 01211.HK）。
5. 英伟达（NVDA）。
6. 月之暗面（Kimi,未上市,验证非上市覆盖）。

## 13. 待定问题

- 首版 UI 是 Web、CLI 还是先 API？
- Evidence 存储用文件、SQLite 还是轻量数据库？
- Word/PDF 导出使用什么排版方案？
- 研究任务是否允许用户上传指定资料作为强制证据？
- 质量检查是否作为导出前强制门槛？
- 行情/财报数据是接 AkShare/efinance/yfinance 还是自建缓存？

