"""非上市公司标准大纲模板.

现有 run_snapshot.py 的 5 维度 OUTLINE 结构化迁入，作为非上市公司
研究的固化大纲。每个维度自包含，researcher 看不到其他维度的工作，
被 topics_for() 圈定在模板主题内。

维度顺序即报告章节顺序：
  1. 投融资  2. 竞品  3. 团队  4. 业务  5. 财务
"""

from __future__ import annotations

from company_report_kit.outlines.base import OutlineTemplate, ResearchDimension

_DIMENSIONS = (
    ResearchDimension(
        label="投融资",
        prompt=(
            "研究{company}的完整融资历史。梳理每一轮融资的:宣布/完成时间、融资金额(含币种)、"
            "投前/投后估值、轮次类型(天使/A/B/C/...)、领投方与跟投方(新老股东区分)。"
            "按时间线排列,标注估值变化轨迹与关键驱动因素(产品里程碑/技术突破/市场事件)。"
            "如有融资空窗期或终止的融资计划,也要记录。投资方信息尽量穿透到具体基金/机构。"
        ),
    ),
    ResearchDimension(
        label="竞品",
        prompt=(
            "研究{company}的竞品格局。首先准确识别直接竞品(同赛道、同产品形态)和间接竞品"
            "(替代方案),然后逐个对比:竞品的业务规模/融资阶段/市场地位、技术路线差异"
            "(架构/参数/性能基准/开源策略)、差异化优势与劣势。不要泛泛列举行业玩家,"
            "要聚焦与{company}直接争夺同一市场或同一技术路线的对手。如有公开的性能基准"
            "对比(榜单/评测),优先引用。"
        ),
    ),
    ResearchDimension(
        label="团队",
        prompt=(
            "研究{company}的组织架构与发展历程。包括:创始人背景(教育经历、过往创业/职业经历)、"
            "核心高管团队(CTO/CFO/COO 等关键岗位)、董事会构成(投资方董事席位)、"
            "公司发展历程(成立→关键产品节点→重要战略转折→现状)。"
            "如有重大人事变动(离职/仲裁/股权纠纷),也要记录。"
        ),
    ),
    ResearchDimension(
        label="业务",
        prompt=(
            "研究{company}的业务模式与商业化进展。包括:主力产品/服务及其营收模式、"
            "重大订单/政企合作/战略合作(含金额与时间)、上下游产业链位置(核心供应商/客户结构)、"
            "关键经营指标(用户规模/DAU/MAU/ARR/付费转化率等可得数据)。"
            "优先引用有数字的公开信息,不要笼统描述市场前景广阔。"
        ),
    ),
    ResearchDimension(
        label="财务",
        prompt=(
            "研究{company}可得的财务数据。非上市公司财务披露有限,重点搜集:各阶段公开的"
            "营收/利润/现金流数字(标注时间和口径)、融资节奏与累计融资额、资金储备(账面现金)、"
            "关键单位经济模型指标(如有)。每个数字必须标注来源和时间,区分实际披露 vs 市场传闻 vs 推测。"
        ),
    ),
)

UNLISTED_TEMPLATE = OutlineTemplate(
    name="unlisted-company",
    company_type="unlisted",
    dimensions=_DIMENSIONS,
)
