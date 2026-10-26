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
