"""数据处理工作台:缺失值、异常值、标准化、哑变量、重编码、计算变量、筛选排序。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core.datasets import Dataset, as_numeric
from .core.results import ModelError


def _num(ds: Dataset, cols: list[str]) -> pd.DataFrame:
    return pd.concat([as_numeric(ds, c).rename(c) for c in cols], axis=1)


# ---------------- 缺失值 ----------------
def fill_missing(ds: Dataset, cols: list[str], method: str = "mean", value: float = 0) -> str:
    for c in cols:
        s = ds.df[c]
        if method in ("mean", "median"):
            s_num = pd.to_numeric(s, errors="coerce")
            fill = s_num.mean() if method == "mean" else s_num.median()
            ds.df[c] = s.fillna(fill)
        elif method == "mode":
            mode = s.mode(dropna=True)
            ds.df[c] = s.fillna(mode.iloc[0] if len(mode) else "未知")
        elif method == "ffill":
            ds.df[c] = s.ffill().bfill()
        elif method == "constant":
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                ds.df[c] = s.fillna(value)
            else:
                ds.df[c] = s.fillna(str(value))
        else:
            raise ModelError("未知填充方式")
    return f"缺失值填充({dict(mean='均值', median='中位数', mode='众数', ffill='前值', constant='常数').get(method, method)}): {', '.join(cols)}"


def drop_missing(ds: Dataset, cols: list[str], how: str = "any") -> str:
    before = len(ds.df)
    subset = cols if cols else None
    ds.df = ds.df.dropna(subset=subset, how=how).reset_index(drop=True)
    return f"删除缺失行({how}): {before} → {len(ds.df)} 行"


# ---------------- 异常值 ----------------
def winsorize(ds: Dataset, cols: list[str], method: str = "3sigma", suffix: str = "_win") -> str:
    data = _num(ds, cols)
    for c in cols:
        s = data[c].dropna()
        if method == "3sigma":
            mu, sd = s.mean(), s.std(ddof=1)
            lo, hi = mu - 3 * sd, mu + 3 * sd
        else:  # IQR
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        ds.df[c + suffix] = ds.df[c].where(~ds.df[c].notna(), np.nan).astype(float) \
            if not pd.api.types.is_numeric_dtype(ds.df[c]) else ds.df[c].astype(float)
        ds.df[c + suffix] = pd.to_numeric(ds.df[c], errors="coerce").clip(lo, hi)
    return f"异常值缩尾({('3σ' if method == '3sigma' else 'IQR')}): 生成列 {', '.join(c + suffix for c in cols)}"


# ---------------- 标准化 ----------------
def standardize(ds: Dataset, cols: list[str], method: str = "zscore") -> str:
    data = _num(ds, cols)
    tag = {"zscore": "_z", "minmax": "_mm", "center": "_c", "robust": "_rb"}[method]
    for c in cols:
        s = data[c]
        if method == "zscore":
            sd = s.std(ddof=1)
            out = (s - s.mean()) / (sd if sd else 1)
        elif method == "minmax":
            rng = s.max() - s.min()
            out = (s - s.min()) / (rng if rng else 1)
        elif method == "center":
            out = s - s.mean()
        else:
            iqr = s.quantile(0.75) - s.quantile(0.25)
            out = (s - s.median()) / (iqr if iqr else 1)
        ds.df[c + tag] = out
    name = {"zscore": "Z标准化", "minmax": "归一化[0,1]", "center": "中心化", "robust": "稳健标准化"}[method]
    return f"{name}: 生成列 {', '.join(c + tag for c in cols)}"


# ---------------- 哑变量 ----------------
def dummy_encode(ds: Dataset, col: str, drop_first: bool = True, prefix: str = "") -> str:
    s = ds.df[col]
    p = prefix or col
    dummies = pd.get_dummies(s.astype("category"), prefix=p, drop_first=drop_first, dtype=int)
    if dummies.shape[1] == 0:
        raise ModelError("该列取值过少,无法生成哑变量")
    for d in dummies.columns:
        ds.df[d] = dummies[d]
    return f"哑变量编码「{col}」: 新增 {dummies.shape[1]} 列"


# ---------------- 重编码 ----------------
def recode_bins(ds: Dataset, col: str, bins: int, labels: list[str] | None,
                new_name: str) -> str:
    s = pd.to_numeric(ds.df[col], errors="coerce")
    if labels:
        lab = labels[:bins]
        lab += [f"组{i+1}" for i in range(bins - len(lab))]
    else:
        lab = [f"组{i+1}" for i in range(bins)]
    ds.df[new_name] = pd.qcut(s, q=bins, labels=lab, duplicates="drop")
    ds.df[new_name] = ds.df[new_name].cat.add_categories(["缺失"]).fillna("缺失") \
        if hasattr(ds.df[new_name], "cat") else ds.df[new_name]
    return f"重编码「{col}」→「{new_name}」(按分位数分 {bins} 组)"


def recode_map(ds: Dataset, col: str, mapping: dict[str, str], new_name: str) -> str:
    ds.df[new_name] = ds.df[col].astype(str).map(mapping).fillna(ds.df[col])
    return f"重编码「{col}」→「{new_name}」(映射 {len(mapping)} 项)"


# ---------------- 计算变量 ----------------
def compute_var(ds: Dataset, expr: str, new_name: str) -> str:
    import re
    if not re.fullmatch(r"[A-Za-z0-9_\u4e00-\u9fa5+\-*/().,%^\s]+", expr or ""):
        raise ModelError("公式包含不允许的字符")
    expr = expr.replace("^", "**")
    try:
        res = ds.df.eval(expr)
    except Exception as e:  # noqa: BLE001
        raise ModelError(f"公式执行失败: {e}(请使用列名与 + - * / ( ) 数字)") from e
    ds.df[new_name] = res
    return f"计算变量「{new_name}」= {expr}"


# ---------------- 筛选 / 排序 ----------------
def filter_rows(ds: Dataset, expr: str) -> str:
    import re
    if not re.fullmatch(r"[A-Za-z0-9_\u4e00-\u9fa5+\-*/().,%<>=!&|~\s'\"]+", expr or ""):
        raise ModelError("筛选表达式包含不允许的字符")
    try:
        before = len(ds.df)
        ds.df = ds.df.query(expr).reset_index(drop=True)
        return f"筛选「{expr}」: {before} → {len(ds.df)} 行"
    except Exception as e:  # noqa: BLE001
        raise ModelError(f"筛选失败: {e}(示例: 年龄 > 30 and 性别 == '男')") from e


def sort_rows(ds: Dataset, cols: list[str], ascending: list[bool]) -> str:
    ds.df = ds.df.sort_values(by=cols, ascending=ascending).reset_index(drop=True)
    return f"排序: {', '.join(cols)}"


def cast_type(ds: Dataset, col: str, target: str) -> str:
    if target == "quant":
        ds.df[col] = pd.to_numeric(ds.df[col], errors="coerce")
    elif target == "cat":
        ds.df[col] = ds.df[col].astype(str)
    elif target == "ordinal":
        ds.df[col] = ds.df[col].astype(str)
    ds.set_dtype(col, target)
    return f"「{col}」类型 → {target}"


def apply_value_labels(ds: Dataset, cols: list[str], suffix: str = "") -> str:
    """把值标签转为文本:按列元数据里的值标签映射替换取值(sav 上传自动携带)。"""
    out = []
    for col in cols:
        meta = next((c for c in ds.columns if c["name"] == col), None)
        vals = (meta or {}).get("vals") or {}
        if not vals:
            continue
        target = col + suffix
        ds.df[target] = ds.df[col].map(lambda v, m=vals: m.get(str(v), v))
        if target == col:
            meta["dtype"] = "cat"          # 替换后取值变为标签文本
        else:
            ds.columns.append({"name": target, "dtype": "cat", "label": meta.get("label", ""),
                               "vals": {}})
        out.append(col)
    if not out:
        raise ValueError("所选列没有值标签;sav 文件的值标签会在上传时自动读取")
    return f"值标签已转为文本: {', '.join(out)}"


OPERATIONS = {
    "value_labels": apply_value_labels,
    "fill_missing": fill_missing,
    "drop_missing": drop_missing,
    "winsorize": winsorize,
    "standardize": standardize,
    "dummy": dummy_encode,
    "recode_bins": recode_bins,
    "recode_map": recode_map,
    "compute": compute_var,
    "filter": filter_rows,
    "sort": sort_rows,
    "cast": cast_type,
}
