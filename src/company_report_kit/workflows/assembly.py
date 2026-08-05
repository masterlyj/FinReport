"""报告章节组装：拼接各 researcher 章节、加编号层级、重编脚注引用编号.

通用组件，供各固化 workflow（非上市快照/未来其他模板）复用。
组装逻辑与具体模板解耦——章节标题只认 markdown 层级，不做模板假设。
"""

from __future__ import annotations

import re

# 中文序数前缀(如"一、""十一、"),供剥离小节标题自编号.
_CN_ORDINAL_PREFIX = re.compile(r"^第?[一二三四五六七八九十百]+、")


def assemble_sections(company: str, sections: list[str]) -> str:
    """拼接各 researcher 的章节,加报告级标题+编号层级,重编脚注引用编号.

    每个 researcher 产出 ### 开头的章节片段(不带编号),组装时统一加编号:
      # {company}研究报告          — 报告级标题
      ## N. 章节标题               — 第 N 个维度章节(剥离中文序数前缀)
      ### N.M 小节标题             — 章节内第 M 个小节(继承章节编号)
      #### N.M.K 更深小节          — 继承编号
    脚注引用 [^N] 按章节顺序重编为全局唯一.

    Args:
        company: 公司名称,用于报告级标题.
        sections: 各 researcher 产出的 section_text(### 开头的章节片段,含脚注).

    Returns:
        拼接后的完整报告,含报告级标题+编号层级,脚注编号全局唯一.
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
        # 收集本章节的脚注编号,建立旧→新映射.
        old_nums = sorted({int(m) for m in re.findall(r"\[\^(\d+)\]", section)})
        if old_nums:
            num_map = {old: footnote_offset + i + 1 for i, old in enumerate(old_nums)}
            # 重编正文和脚注定义中的 [^N]。
            # 闭包显式绑定 num_map（本地变量）,避免 B023 引用外层循环变量。
            def _renum(m: re.Match, _num_map: dict[int, int] = num_map) -> str:
                return f"[^{_num_map[int(m.group(1))]}]"
            section = re.sub(r"\[\^(\d+)\]", _renum, section)
            footnote_offset = max(num_map.values())
        assembled.append(section)
    return "\n\n".join(assembled)
