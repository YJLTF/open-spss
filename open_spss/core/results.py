"""分析结果结构:表格、图表(ECharts option)、智能解读,以及数值格式化工具。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def fnum(x, nd: int = 4) -> str:
    """数值格式化为字符串;空值/无穷返回空串。"""
    if x is None:
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v) or math.isinf(v):
        return ""
    if v != 0 and abs(v) < 1e-4:
        return f"{v:.2e}"
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


def stars(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def fp(p) -> str:
    """p 值格式化:0.000 → <0.001***"""
    if p is None:
        return ""
    try:
        v = float(p)
    except (TypeError, ValueError):
        return str(p)
    if math.isnan(v):
        return ""
    if v < 0.001:
        return "<0.001" + stars(v)
    s = f"{v:.3f}"
    if s == "1.000":
        s = ">0.999"
    return s + stars(v)


def judge_p(p, alpha: float = 0.05) -> str:
    return "是" if (p is not None and not math.isnan(p) and p < alpha) else "否"


import re as _re

# 半角 → 全角标点(中文语境,用码点写法避免源文件形近字符混淆);
# 千分位逗号 1,000 与 `代码`/链接片段原样保留
_FULL_COMMA = "\uff0c"      # ，
_FULL_SEMI = "\uff1b"       # ；
_FULL_COLON = "\uff1a"      # ：
_FULL_EXCL = "\uff01"       # ！
_FULL_QUEST = "\uff1f"      # ？
_FULL_LPAREN = "\uff08"     # （
_FULL_RPAREN = "\uff09"     # ）
_ELLIPSIS = "\u2026\u2026"  # ……
_PUNCT_MAP = [("...", _ELLIPSIS), (",", _FULL_COMMA), (";", _FULL_SEMI), (":", _FULL_COLON),
              ("!", _FULL_EXCL), ("?", _FULL_QUEST), ("(", _FULL_LPAREN), (")", _FULL_RPAREN)]


def zh_punct(text: str) -> str:
    parts = _re.split(r"(`[^`]*`|\[[^\]]*\]\([^)]*\))", text)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)
            continue
        # 仅保留两侧均为数字的逗号(千分位);t=2.1,p<0.05 这类数字后跟内容的必须转全角
        part = _re.sub(r"(?<=\d),(?=\d)", "\x00", part)
        for half, full in _PUNCT_MAP:
            part = part.replace(half, full)
        part = part.replace("\x00", ",")
        out.append(part)
    return "".join(out)


@dataclass
class Table:
    title: str
    headers: list[str]
    rows: list[list]
    note: str = ""
    raw: list[list] | None = None   # 导出 Excel 时用的原始数值(可选)

    def to_dict(self) -> dict:
        # 序列化时统一再做一次标点归一,兜住绕过 add_* 的直写字段
        return {"title": zh_punct(self.title), "headers": self.headers,
                "rows": self.rows, "note": zh_punct(self.note) if self.note else self.note}


@dataclass
class Result:
    tables: list[Table] = field(default_factory=list)
    charts: list[dict] = field(default_factory=list)      # {"title","option"}
    interpretation: list[str] = field(default_factory=list)  # markdown 段落
    notes: list[str] = field(default_factory=list)

    def add_table(self, title: str, headers: list[str], rows: list[list],
                  note: str = "", raw: list[list] | None = None) -> Table:
        t = Table(title=title, headers=headers, rows=rows, note=zh_punct(note) if note else note, raw=raw)
        self.tables.append(t)
        return t

    def add_chart(self, title: str, option: dict) -> None:
        self.charts.append({"title": zh_punct(title), "option": option})

    def add_interp(self, *paras: str) -> None:
        self.interpretation.extend(zh_punct(p) for p in paras if p)

    def add_note(self, *notes: str) -> None:
        self.notes.extend(zh_punct(n) for n in notes if n)

    def to_dict(self) -> dict:
        return {
            "tables": [t.to_dict() for t in self.tables],
            "charts": [{"title": zh_punct(c["title"]), "option": c["option"]} for c in self.charts],
            "interpretation": [zh_punct(p) for p in self.interpretation],
            "notes": [zh_punct(n) for n in self.notes],
        }


class ModelError(ValueError):
    """业务级错误(变量类型不符、数据不足等),前端以提示框展示。"""
