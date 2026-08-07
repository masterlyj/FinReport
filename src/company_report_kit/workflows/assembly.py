"""报告章节组装：拼接各 researcher 章节、加编号层级、重编脚注引用编号.

通用组件，供各固化 workflow（非上市快照/未来其他模板）复用。
组装逻辑与具体模板解耦——章节标题只认 markdown 层级，不做模板假设。
"""

from __future__ import annotations

import re

# 中文序数前缀(如"一、""十一、"),供剥离小节标题自编号.
_CN_ORDINAL_PREFIX = re.compile(r"^第?[一二三四五六七八九十百]+、")

# 脚注定义行:行首 [^N]: 形式。
# 用于剪除 write_section 偶尔生成的"定义了却未在正文引用"的孤儿脚注——
# 它们占编号、被 run_section_review 白抓一次原文,且让最终报告出现悬空脚注。
_FOOTNOTE_DEF_LINE = re.compile(r"^\[\^(\d+)\]:")


def _is_orphan_definition(line: str, referenced: set[int]) -> bool:
    """判断该行是否为'正文未引用的脚注定义'(orphan)。

    Args:
        line: 章节文本的一行.
        referenced: 本章节正文实际引用过的脚注号集合.

    Returns:
        True 表示该行是孤儿脚注定义(应剪除);False 表示保留.
    """
    mo = _FOOTNOTE_DEF_LINE.match(line)
    return bool(mo) and int(mo.group(1)) not in referenced


def assemble_sections(company: str, sections: list[str]) -> str:
    """拼接各 researcher 的章节,加报告级标题+编号层级,重编脚注引用编号.

    每个 researcher 产出 ### 开头的章节片段(不带编号),组装时统一加编号:
      # {company}研究报告          — 报告级标题
      ## N. 章节标题               — 第 N 个维度章节(剥离中文序数前缀)
      ### N.M 小节标题             — 章节内第 M 个小节(继承章节编号)
      #### N.M.K 更深小节          — 继承编号
    脚注引用 [^N] 按章节顺序重编为全局唯一;正文未引用的脚注定义(孤儿)
    在重编前剪除,正文引用了但无定义的 [^N](悬空引用)同时剥离——两者都不占
    编号、不被后续 review 抓取原文。

    Args:
        company: 公司名称,用于报告级标题.
        sections: 各 researcher 产出的 section_text(### 开头的章节片段,含脚注).

    Returns:
        拼接后的完整报告,含报告级标题+编号层级,脚注编号全局唯一且无悬空定义.
    """
    assembled: list[str] = ["# " + company + "研究报告", ""]
    footnote_offset = 0
    for idx, section in enumerate(sections, start=1):
        if not section:
            continue
        lines = section.split("\n")
        # 章节标题行:首个 ### 开头的行
        title_line = next((li for li, line in enumerate(lines) if line.startswith("### ")), None)
        if title_line is not None:
            title = _CN_ORDINAL_PREFIX.sub("", lines[title_line][4:])
            lines[title_line] = f"## {idx}. {title}"
            # 章节内小节 ### / #### 继承编号
            sub = 0
            for li in range(title_line + 1, len(lines)):
                m = re.match(r"^(#{3,4}) ", lines[li])
                if m:
                    sub += 1
                    level = m.group(1)
                    rest = _CN_ORDINAL_PREFIX.sub("", lines[li][len(level) + 1:])
                    lines[li] = f"{level} {idx}.{sub} {rest}"
        section = "\n".join(lines)
        # 正文引用号 [^N](后不跟 ":")与脚注定义号 [^N]:(行首)。
        body_refs = {int(m) for m in re.findall(r"\[\^(\d+)\](?!:)", section)}
        def_nums = {
            int(m) for m in re.findall(r"^\[\^(\d+)\]:", section, re.MULTILINE)
        }
        # 剪除孤儿定义(有定义、正文未引用):no-op 时不动。
        if def_nums - body_refs:
            section = "\n".join(
                line for line in section.split("\n")
                if not _is_orphan_definition(line, body_refs)
            )
        # 剥离悬空引用(正文引用了、但无定义):删掉无定义的 body [^N] 标记,
        # 避免读者点到不存在的脚注;保留有定义的引用。
        dangling = body_refs - def_nums
        if dangling:
            def _strip_dangling(m: re.Match, _d: set[int] = dangling) -> str:
                return "" if int(m.group(1)) in _d else m.group(0)

            section = re.sub(r"\[\^(\d+)\](?!:)", _strip_dangling, section)
        # 重编号:只编"既有定义又被引用"的脚注,孤儿与悬空都不占编号。
        ref_nums = sorted(body_refs & def_nums)
        if ref_nums:
            num_map = {old: footnote_offset + i + 1 for i, old in enumerate(ref_nums)}
            # 重编正文和脚注定义中的 [^N]。
            # 闭包显式绑定 num_map（本地变量）,避免 B023 引用外层循环变量。
            def _renum(m: re.Match, _num_map: dict[int, int] = num_map) -> str:
                return f"[^{_num_map[int(m.group(1))]}]"
            section = re.sub(r"\[\^(\d+)\]", _renum, section)
            footnote_offset = max(num_map.values())
        assembled.append(section)
    return "\n\n".join(assembled)
