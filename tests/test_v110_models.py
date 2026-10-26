"""v1.1.0 新增模型的数值断言:对照 scipy / statsmodels 的手工小样本结果。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from open_spss.core.datasets import Dataset
from open_spss.core.registry import REGISTRY
from open_spss.models import CATEGORIES  # noqa: F401

RNG = np.random.default_rng(42)


def _ds(df: pd.DataFrame) -> Dataset:
    return Dataset("t", df)


# ---------------------------------------------------------------- 偏相关
def test_partial_corr_matches_manual():
    n = 200
    z = RNG.normal(size=n)
    x = 2 * z + RNG.normal(size=n)
    y = 1.5 * z + 0.9 * x + RNG.normal(size=n)   # z 之外保留 x 对 y 的直接效应
    ds = _ds(pd.DataFrame({"x": x, "y": y, "z": z}))
    res = REGISTRY.get("partial_corr").runner(ds, {"vars": ["x", "y"], "ctrls": ["z"]}, {"alpha": 0.05})
    # 手工:对 z 回归取残差再求相关
    Z = np.column_stack([np.ones(n), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r_manual = float(np.corrcoef(rx, ry)[0, 1])
    row = res.tables[0].rows[0][2]         # 「x」行、「y」列(非对角线)
    assert abs(float(row.rstrip("*")) - r_manual) < 5.1e-4   # 表格展示保留 3 位小数
    # 星号与手工计算的显著性一致
    stars = row.lstrip("-0123456789.")
    p_manual = 2 * sps.t.sf(abs(r_manual * np.sqrt((n - 3) / (1 - r_manual**2))), n - 3)
    expect = "***" if p_manual < 0.01 else "**" if p_manual < 0.05 else "*" if p_manual < 0.1 else ""
    assert stars == expect
    assert abs(r_manual) > 0.3             # 数据有真实关联,不应退化


def test_partial_corr_rejects_overlap():
    ds = _ds(pd.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4], "z": [1, 1, 2]}))
    with pytest.raises(ValueError, match="重复"):
        REGISTRY.get("partial_corr").runner(ds, {"vars": ["x", "z"], "ctrls": ["z"]}, {"alpha": 0.05})


# ---------------------------------------------------------------- McDonald's ω
def test_omega_high_for_single_factor():
    n, k = 300, 6
    f = RNG.normal(size=n)
    X = np.column_stack([2.0 * f + RNG.normal(scale=0.8, size=n) for _ in range(k)])
    ds = _ds(pd.DataFrame(X, columns=[f"q{i}" for i in range(k)]))
    res = REGISTRY.get("omega").runner(ds, {"vars": list(ds.df.columns)}, {})
    omega = float(res.tables[1].rows[0][0])
    assert omega > 0.85                    # 强单因子结构,ω 应很高
    alpha = float(res.tables[1].rows[0][1])
    assert 0.5 < alpha <= 1.0


def test_omega_low_for_noise():
    X = RNG.normal(size=(200, 5))
    ds = _ds(pd.DataFrame(X, columns=[f"q{i}" for i in range(5)]))
    res = REGISTRY.get("omega").runner(ds, {"vars": list(ds.df.columns)}, {})
    omega = float(res.tables[1].rows[0][0])
    assert omega < 0.5


# ---------------------------------------------------------------- 中介效应
def test_mediation_detects_indirect_effect():
    n = 300
    x = RNG.normal(size=n)
    m = 0.6 * x + RNG.normal(scale=1.0, size=n)
    y = 0.3 * x + 0.8 * m + RNG.normal(scale=1.0, size=n)   # 间接效应=0.48,直接=0.3
    ds = _ds(pd.DataFrame({"x": x, "m": m, "y": y}))
    res = REGISTRY.get("mediation").runner(ds, {"y": "y", "x": "x", "ms": ["m"], "covs": ""},
                                           {"n_rep": 500, "alpha": 0.05})
    rows = res.tables[0].rows
    acme = next(r for r in rows if r[1].startswith("间接效应"))
    assert abs(float(acme[2]) - 0.48) < 0.06
    assert acme[7] == "显著"
    assert res.tables[1].rows[0][2] in ("部分中介", "完全中介")


def test_mediation_no_effect_when_m_independent():
    n = 300
    x = RNG.normal(size=n)
    m = RNG.normal(size=n)                       # 与 x 无关
    y = 0.5 * x + RNG.normal(size=n)
    ds = _ds(pd.DataFrame({"x": x, "m": m, "y": y}))
    res = REGISTRY.get("mediation").runner(ds, {"y": "y", "x": "x", "ms": ["m"], "covs": ""},
                                           {"n_rep": 300, "alpha": 0.05})
    assert res.tables[1].rows[0][2] == "未检出中介效应"


# ---------------------------------------------------------------- 调节效应
def test_moderation_detects_interaction():
    n = 400
    x = RNG.normal(size=n)
    mod = RNG.normal(size=n)
    y = 0.4 * x + 0.5 * mod + 0.9 * x * mod + RNG.normal(scale=1.0, size=n)
    ds = _ds(pd.DataFrame({"x": x, "mod": mod, "y": y}))
    res = REGISTRY.get("moderation").runner(ds, {"y": "y", "x": "x", "mod": "mod", "covs": []},
                                            {"alpha": 0.05})
    p3 = res.tables[0].rows[2][7]                # M3 步 p(ΔF)
    assert p3.startswith("<0.001") or float(p3.rstrip("*")) < 0.05
    # 简单斜率随调节水平递增
    slopes = [float(r[1]) for r in res.tables[2].rows]
    assert slopes[2] > slopes[1] > slopes[0]          # 交互为正,斜率随调节水平递增
    assert abs((slopes[2] - slopes[0]) - 2 * 0.9) < 0.15   # 1SD 差 ≈ 2×交互系数


def test_moderation_no_interaction():
    n = 400
    x = RNG.normal(size=n)
    mod = RNG.normal(size=n)
    y = 0.4 * x + 0.5 * mod + RNG.normal(scale=1.0, size=n)
    ds = _ds(pd.DataFrame({"x": x, "mod": mod, "y": y}))
    res = REGISTRY.get("moderation").runner(ds, {"y": "y", "x": "x", "mod": "mod", "covs": []},
                                            {"alpha": 0.05})
    p3_txt = res.tables[0].rows[2][7]
    p3 = 0.0 if p3_txt.startswith("<") else float(p3_txt.rstrip("*"))
    assert p3 >= 0.05


# ---------------------------------------------------------------- 重复测量 ANOVA
def test_anova_rm_equals_paired_t_squared():
    """2 水平重复测量:F = 配对 t 检验 t 的平方(经典恒等关系)。"""
    n = 25
    base = RNG.normal(size=n)
    y1, y2 = base + RNG.normal(scale=1.0, size=n), base + 0.9 + RNG.normal(scale=1.0, size=n)
    rows = [(f"S{i}", "前测", y1[i]) for i in range(n)] + [(f"S{i}", "后测", y2[i]) for i in range(n)]
    ds = _ds(pd.DataFrame(rows, columns=["sid", "t", "y"]))
    res = REGISTRY.get("anova_rm").runner(ds, {"y": "y", "subject": "sid", "time": "t"},
                                          {"alpha": 0.05})
    t = sps.ttest_rel(y2, y1)
    f_rep = float(res.tables[1].rows[0][4])
    assert abs(f_rep - t.statistic ** 2) < 5.1e-4   # 表格展示保留 3 位小数
    p_rep = float(res.tables[1].rows[0][5].rstrip("*").lstrip("<"))
    assert abs(p_rep - t.pvalue) < 5.1e-4 or t.pvalue < 0.001
    assert res.tables[2].rows[0][3] == "1"            # 2 水平球形度恒成立,GG ε=1


# ---------------------------------------------------------------- 混合模型
def test_mixed_lm_recovers_icc():
    n_sub, n_obs = 40, 4
    g = np.repeat(np.arange(n_sub), n_obs)
    x = RNG.normal(size=len(g))
    u = RNG.normal(scale=0.9, size=n_sub)
    y = 1.2 + 0.8 * x + u[g] + RNG.normal(scale=0.4, size=len(g))
    df = pd.DataFrame({"y": y, "x": x, "g": [f"G{i}" for i in g]})
    ds = _ds(df)
    res = REGISTRY.get("mixed_lm").runner(ds, {"y": "y", "group": "g", "xs": ["x"]},
                                          {"re_type": "ri", "reml": "reml", "alpha": 0.05})
    icc = float(res.tables[1].rows[-1]["估计值"] if False else res.tables[1].note.split("=")[-1])
    assert 0.5 < icc < 1.0                            # 真实 ICC≈0.9²/(0.9²+0.4²)≈0.83
    coef = float(res.tables[0].rows[1][1])            # x 的固定效应
    assert abs(coef - 0.8) < 0.1


# ---------------------------------------------------------------- GEE
def test_gee_gaussian_close_to_truth():
    n_sub, n_obs = 50, 4
    g = np.repeat(np.arange(n_sub), n_obs)
    x = RNG.normal(size=len(g))
    u = RNG.normal(scale=0.7, size=n_sub)
    y = 0.5 + 1.1 * x + u[g] + RNG.normal(scale=0.6, size=len(g))
    df = pd.DataFrame({"y": y, "x": x, "g": [f"G{i}" for i in g]})
    ds = _ds(df)
    res = REGISTRY.get("gee").runner(ds, {"y": "y", "group": "g", "xs": ["x"]},
                                     {"family": "gaussian", "corstr": "exchangeable", "alpha": 0.05})
    rows = res.tables[0].rows
    slope = next(r for r in rows if r[0] == "x")
    assert abs(float(slope[1]) - 1.1) < 0.12
    assert slope[5] == "显著"


# ---------------------------------------------------------------- 小提琴图
def test_violin_chart_structure():
    from open_spss.core.charts import violin_chart
    groups = {"A": RNG.normal(0, 1, 60).tolist(), "B": RNG.normal(2, 1, 60).tolist()}
    opt = violin_chart("t", groups, show_points=True)
    kinds = [s["type"] for s in opt["series"]]
    assert "line" in kinds and "scatter" in kinds     # 箱线也用折线多边形(类目轴会取整小数 x)
    assert opt["xAxis"]["type"] == "value" and opt["xAxis"]["_catLabels"] == ["A", "B"]
    violin = next(s for s in opt["series"] if s["type"] == "line" and s["name"] == "A")
    xs = [p[0] for p in violin["data"] if p[0] is not None]
    assert min(xs) < 0 < max(xs) < 2                  # 多边形围绕类别位置对称展开
    box = next(s for s in opt["series"] if s["name"] == "A·箱线")
    assert any(p[0] is None for p in box["data"])     # 箱线段以 null 断开
