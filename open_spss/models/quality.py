"""质量管理:SPC 控制图(X̄-R、p 图)与过程能力分析。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..core import charts as ch
from ..core.datasets import Dataset, as_numeric, get_series
from ..core.registry import CAT, QUANT, ModelSpec, Option, Slot, register_model
from ..core.results import ModelError, Result, fnum

A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
D3 = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}


def _limit_line(series_name: str, value: float, n: int, color: str = "#ef4444") -> dict:
    return {"name": series_name, "type": "line", "data": [value] * n, "showSymbol": False,
            "smooth": False, "lineStyle": {"type": "dashed", "color": color, "width": 1.4},
            "tooltip": {"show": False}}


# ================================================================ X̄-R 控制图
def _run_xbar_r(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    val = as_numeric(ds, sl["value"])
    sub = get_series(ds, sl["subgroup"])
    data = pd.concat([val.rename("v"), sub.rename("g")], axis=1).dropna()
    groups = data.groupby("g", observed=True)["v"]
    means = groups.mean()
    ranges = groups.max() - groups.min()
    sizes = groups.size()
    if len(means) < 3:
        raise ModelError("子组数至少为 3")
    n = int(sizes.mode().iloc[0])
    if n < 2 or n > 10:
        raise ModelError(f"每个子组的样本量需在 2~10(当前众数 {n})")
    xbar = float(means.mean())
    rbar = float(ranges.mean())
    a2, d3, d4 = A2[n], D3[n], D4[n]
    ucl_x, lcl_x = xbar + a2 * rbar, xbar - a2 * rbar
    ucl_r, lcl_r = d4 * rbar, d3 * rbar
    ooc_x = int(((means > ucl_x) | (means < lcl_x)).sum())
    ooc_r = int(((ranges > ucl_r) | (ranges < lcl_r)).sum())
    res.add_table("控制限(X̄ 图)", ["CL(X̿)", "UCL", "LCL", "子组数", "子组容量n", "出界点数"],
                  [[fnum(xbar, 4), fnum(ucl_x, 4), fnum(lcl_x, 4), len(means), n, ooc_x]])
    res.add_table("控制限(R 图)", ["CL(R̄)", "UCL", "LCL", "出界点数"],
                  [[fnum(rbar, 4), fnum(ucl_r, 4), fnum(lcl_r, 4), ooc_r]])
    labels = [str(g) for g in means.index]
    opt_x = ch.line_chart("X̄ 控制图(子组均值)", labels,
                 {"子组均值": [round(float(v), 4) for v in means.values]}, x_name="子组", smooth=False)
    opt_x["series"] += [_limit_line("UCL", ucl_x, len(labels)), _limit_line("CL", xbar, len(labels), "#22c55e"),
                        _limit_line("LCL", lcl_x, len(labels))]
    res.add_chart("X̄ 控制图", opt_x)
    opt_r = ch.line_chart("R 控制图(子组极差)", labels,
                 {"子组极差": [round(float(v), 4) for v in ranges.values]}, x_name="子组", smooth=False)
    opt_r["series"] += [_limit_line("UCL", ucl_r, len(labels)), _limit_line("CL", rbar, len(labels), "#22c55e"),
                        _limit_line("LCL", lcl_r, len(labels))]
    res.add_chart("R 控制图", opt_r)
    ok = ooc_x == 0 and ooc_r == 0
    res.add_interp(
        f"X̄ 图出界点 {ooc_x} 个、R 图出界点 {ooc_r} 个。"
        + ("所有子组均值与极差均在控制限内,过程**受控**。" if ok
           else "存在超出控制限的点,提示过程存在**异常因素**(特殊原因变异),建议排查对应子组的设备/原料/操作条件。"))
    return res


register_model(ModelSpec(
    id="xbar_r", name="X̄-R 控制图", category="质量管理",
    runner=_run_xbar_r,
    desc="计量型 SPC 控制图:子组均值图与极差图,监控过程均值与波动是否受控。",
    tags=["SPC", "控制图", "休哈特"], demo_note="quality",
    slots=[Slot("value", "质量特性值(定量)", types=(QUANT,)),
           Slot("subgroup", "子组编号(定类/编号)", types=(CAT, QUANT))],
    options=[],
))


# ================================================================ p 控制图
def _run_p_chart(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    defect = as_numeric(ds, sl["defects"])
    size = as_numeric(ds, sl["size"])
    data = pd.concat([defect.rename("d"), size.rename("n")], axis=1).dropna()
    if (data["d"] > data["n"]).any():
        raise ModelError("不合格品数不能大于样本量")
    if len(data) < 5:
        raise ModelError("至少需要 5 个样本批次")
    pbar = float(data["d"].sum() / data["n"].sum())
    lim = 3 * np.sqrt(pbar * (1 - pbar) / data["n"].replace(0, np.nan))
    ucl = pbar + lim
    lcl = (pbar - lim).clip(lower=0)
    ooc = int(((data["d"] / data["n"] > ucl) | (data["d"] / data["n"] < lcl)).sum())
    res.add_table("p 控制图参数", ["平均不合格率p̄", "平均UCL", "平均LCL", "批次数", "出界批次数"],
                  [[fnum(pbar * 100, 3) + "%", fnum(ucl.mean(), 4), fnum(lcl.mean(), 4), len(data), ooc]])
    labels = [str(i + 1) for i in range(len(data))]
    pvals = (data["d"] / data["n"]).values
    opt = ch.line_chart("p 控制图(不合格品率)", labels,
                        {"批次不合格率": [round(float(v), 4) for v in pvals]}, x_name="批次", smooth=False)
    opt["series"] += [_limit_line("UCL", float(ucl.mean()), len(labels)), _limit_line("CL", pbar, len(labels), "#22c55e"),
                      _limit_line("LCL", float(lcl.mean()), len(labels))]
    res.add_chart("p 控制图", opt)
    res.add_interp(
        f"过程平均不合格率 p̄={fnum(pbar * 100, 2)}%,出界批次 {ooc} 个。"
        + ("过程受控。" if ooc == 0 else "存在异常批次(超出上下控制限),建议排查对应批次的工艺条件。"))
    return res


register_model(ModelSpec(
    id="p_chart", name="p 控制图(不合格品率)", category="质量管理",
    runner=_run_p_chart,
    desc="计件型 SPC 控制图:监控各批次不合格品率是否超出 3σ 控制限。",
    tags=["SPC", "p图", "不合格率"], demo_note="quality",
    slots=[Slot("defects", "不合格品数(定量)", types=(QUANT,)),
           Slot("size", "样本量(定量)", types=(QUANT,))],
    options=[],
))


# ================================================================ 过程能力分析
def _run_process_capability(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    s = as_numeric(ds, sl["var"]).dropna()
    usl, lsl = op["usl"], op["lsl"]
    if usl is None or lsl is None or usl <= lsl:
        raise ModelError("请设置规格上限 USL 与规格下限 LSL(USL > LSL)")
    mu = float(s.mean())
    sigma_within = float(s.std(ddof=1))  # 简化:组内 σ 用样本标准差估计
    cp = (usl - lsl) / (6 * sigma_within) if sigma_within else np.nan
    cpu = (usl - mu) / (3 * sigma_within) if sigma_within else np.nan
    cpl = (mu - lsl) / (3 * sigma_within) if sigma_within else np.nan
    cpk = min(cpu, cpl)
    target = (usl + lsl) / 2
    k = abs(target - mu) / ((usl - lsl) / 2) if usl != lsl else 0
    ppm_out = float((1 - sps.norm.cdf((usl - mu) / sigma_within) + sps.norm.cdf((lsl - mu) / sigma_within)) * 1e6)
    rate_out = float(((s > usl) | (s < lsl)).mean() * 1e6)
    grade = ("A++(≥1.67)" if cpk >= 1.67 else "A+(≥1.33)" if cpk >= 1.33 else
             "A(≥1.0)" if cpk >= 1.0 else "B(≥0.83)" if cpk >= 0.83 else "C(≥0.67)" if cpk >= 0.67 else "D(<0.67)")
    res.add_table("过程能力指数", ["样本量", "均值μ", "标准差σ", "Cp", "Cpk", "CPU", "CPL", "偏移系数K", "能力等级"],
                  [[len(s), fnum(mu, 4), fnum(sigma_within, 4), fnum(cp, 3), fnum(cpk, 3),
                    fnum(cpu, 3), fnum(cpl, 3), fnum(k, 3), grade]],
                  note="Cp 反映潜在能力;Cpk=Cp×(1-K) 同时考虑过程能力与中心偏移")
    res.add_table("不合格率估计", ["正态法估计PPM", "实际样本超差PPM", "规格下限LSL", "规格上限USL"],
                  [[fnum(ppm_out, 1), fnum(rate_out, 1), fnum(lsl, 3), fnum(usl, 3)]])
    hist = ch.hist_chart("过程分布直方图(含规格限)", s.values, x_name=sl["var"])
    res.add_chart("过程分布直方图", hist)
    res.add_interp(
        f"过程能力指数 Cpk=**{fnum(cpk, 3)}**(等级 {grade})。"
        + ("过程能力充足,可维持稳定生产。" if cpk >= 1.33 else
           "过程能力尚可,但中心偏移或波动偏大,建议持续监控。" if cpk >= 1.0 else
           "过程能力不足(Cpk<1.0),估计不合格率较高,需降低波动或调整中心(对准规格中心)。"))
    return res


register_model(ModelSpec(
    id="process_capability", name="过程能力分析", category="质量管理",
    runner=_run_process_capability,
    desc="计算 Cp/Cpk/CPU/CPL 过程能力指数,估计不合格率 PPM,附规格限直方图。",
    tags=["Cpk", "过程能力", "SPC"], demo_note="quality",
    slots=[Slot("var", "质量特性值(定量)", types=(QUANT,))],
    options=[
        Option("usl", "规格上限 USL", "number", default=0.65, step=0.05),
        Option("lsl", "规格下限 LSL", "number", default=0.35, step=0.05),
    ],
))

# ================================================================ X-MR 控制图(单值-移动极差)
def _run_xmchart(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    x = as_numeric(ds, sl["var"]).dropna()
    if len(x) < 10:
        raise ModelError("X-MR 控制图至少需要 10 个单值")
    v = x.values.astype(float)
    mr = np.abs(np.diff(v))
    xbar = v.mean()
    mrbar = mr.mean()
    ucl_x, lcl_x = xbar + 2.66 * mrbar, xbar - 2.66 * mrbar
    ucl_mr = 3.267 * mrbar
    out_x = [i for i, val in enumerate(v) if val > ucl_x or val < lcl_x]
    out_mr = [i for i, m in enumerate(mr) if m > ucl_mr]
    res.add_table("控制图参数", ["统计量", "中心线 CL", "上控制限 UCL", "下控制限 LCL", "出界点数"],
                  [["单值 X", fnum(xbar, 4), fnum(ucl_x, 4), fnum(lcl_x, 4), len(out_x)],
                   ["移动极差 MR", fnum(mrbar, 4), fnum(ucl_mr, 4), "0(无)", len(out_mr)]],
                  note="系数:d2=1.128 → E2=2.66, D4=3.267(移动极差窗口=2)")
    labels = [str(i + 1) for i in range(len(v))]
    tail = 120
    res.add_chart("X 单值控制图", ch.line_chart("X 单值控制图", labels[-tail:],
                 {"单值": [round(float(t), 4) for t in v[-tail:]],
                  "UCL": [round(float(ucl_x), 4)] * min(tail, len(v)),
                  "CL": [round(float(xbar), 4)] * min(tail, len(v)),
                  "LCL": [round(float(lcl_x), 4)] * min(tail, len(v))},
                 x_name="样本序号", y_name=sl["var"], smooth=False))
    res.add_chart("MR 移动极差控制图", ch.line_chart("MR 控制图", [str(i + 2) for i in range(len(mr))][-tail:],
                 {"MR": [round(float(m), 4) for m in mr[-tail:]],
                  "UCL": [round(float(ucl_mr), 4)] * min(tail, len(mr)),
                  "CL": [round(float(mrbar), 4)] * min(tail, len(mr))},
                 x_name="样本序号", y_name="移动极差", smooth=False))
    sigma = mrbar / 1.128
    res.add_table("过程估计", ["σ 估计(MR/d2)", "过程均值", "自然过程范围(±3σ)"],
                  [[fnum(sigma, 4), fnum(xbar, 4), f"{fnum(xbar - 3 * sigma, 3)} ~ {fnum(xbar + 3 * sigma, 3)}"]])
    if out_x:
        res.add_note(f"X 图出界点(第 {', '.join(str(i + 1) for i in out_x[:8])} 个{'…' if len(out_x) > 8 else ''})存在特殊原因变异,建议排查")
    else:
        res.add_note("X 图无出界点,过程均值稳定")
    res.add_interp(f"过程均值 {fnum(xbar, 3)},σ≈{fnum(sigma, 3)};X 图出界 {len(out_x)} 点、MR 图出界 {len(out_mr)} 点。"
                   + ("存在超出控制限的观测,过程**不受控**,需寻找并消除特殊原因后再评估过程能力。" if out_x or out_mr
                      else "两图均无出界点,过程统计受控,可进一步做过程能力分析。"))
    return res


register_model(ModelSpec(
    id="xmchart", name="X-MR 控制图(单值-移动极差)", category="质量管理",
    runner=_run_xmchart,
    desc="单值与移动极差控制图,适合小批量/破坏性检验/连续过程监控。",
    tags=["SPC", "控制图", "X-MR"], demo_note="quality",
    slots=[Slot("var", "质量特性(定量)", types=(QUANT,))],
    options=[],
))


# ================================================================ 验收抽样(OC 曲线)
def _run_oc_curve(ds: Dataset, sl: dict, op: dict) -> Result:  # noqa: ARG001
    from scipy import stats as sps
    res = Result()
    n = int(op["n"])
    c = int(op["c"])
    if n <= 0 or c < 0 or c >= n:
        raise ModelError("需满足 n≥1 且 0 ≤ c < n")
    p_grid = [round(p, 4) for p in np.arange(0.001, op["p_max"] + 0.0001, 0.001)]
    pa = [float(sps.binom.cdf(c, n, p)) for p in p_grid]
    p0, p1 = float(op["p0"]), float(op["p1"])
    alpha = float(sps.binom.sf(c, n, p0))      # 生产方风险 Pa(p0)=1-α
    beta = float(sps.binom.cdf(c, n, p1))      # 使用方风险
    aoq_max_p = 0.0
    aoq_max = 0.0
    N = int(op["N"] or 0)
    for p, pa_v in zip(p_grid, pa):
        aoq = p * pa_v * (1 - (N and n / N) or 1) if N else p * pa_v
        if aoq > aoq_max:
            aoq_max, aoq_max_p = aoq, p
    rows = [[f"{p:.3f}", f"{v:.4f}"] for p, v in zip(p_grid[::4], pa[::4])]
    res.add_table(f"OC 曲线数据(n={n}, c={c})", "不合格率 p".split(), [["p", "接收概率Pa"]][0][:0] or
                  [["p", "接收概率 Pa"], *rows][1:] if False else
                  [["p", "接收概率 Pa"]] + rows)
    res.add_table("方案风险核查", ["生产方风险点 p0(AQL)", "实际 α=1−Pa(p0)", "使用方风险点 p1(LTPD)", "实际 β=Pa(p1)", "鉴别比 p1/p0"],
                  [[p0, fnum(alpha, 4), p1, fnum(beta, 4), fnum(p1 / p0, 3) if p0 else ""]],
                  note="通常要求 α≈0.05、β≈0.10;不满足时需增大 n 或调整 c")
    res.add_chart("OC 曲线", ch.line_chart(f"抽样特性曲线(n={n}, c={c})",
                 [f"{p:.3f}" for p in p_grid], {"接收概率 Pa": [round(v, 4) for v in pa]},
                 x_name="批不合格率 p", y_name="接收概率 Pa", smooth=False, y_min=0, y_max=1))
    res.add_interp(f"抽样方案(n={n}, c={c}):当批不合格率 ≤ {p0:.1%} 时以 {1 - alpha:.1%} 概率接收(生产方风险 α={alpha:.3f});"
                   f"不合格率达 {p1:.1%} 时接收概率仅 {beta:.1%}(使用方风险 β={beta:.3f})。"
                   + ("方案鉴别能力良好(鉴别比<4)。" if p1 / p0 < 4 else "鉴别比偏大,可增大样本量 n 提升鉴别力。"))
    return res


register_model(ModelSpec(
    id="oc_curve", name="验收抽样(OC 曲线)", category="质量管理",
    runner=_run_oc_curve,
    desc="计数一次抽样方案的抽查特性曲线,核查生产方风险 α 与使用方风险 β。",
    tags=["抽样检验", "OC曲线", "AQL", "LTPD"], demo_note="quality",
    slots=[],
    options=[
        Option("n", "样本量 n", "number", default=50, min=1, max=2000),
        Option("c", "合格判定数 c", "number", default=2, min=0, max=500),
        Option("p0", "AQL 生产方风险质量(p0)", "number", default=0.01, min=0.0001, max=0.5, step=0.001),
        Option("p1", "LTPD 使用方风险质量(p1)", "number", default=0.08, min=0.0001, max=0.9, step=0.001),
        Option("N", "批量 N(选填,用于 AOQ)", "number", default=0, min=0, max=1000000),
        Option("p_max", "OC 曲线最大 p", "number", default=0.2, min=0.05, max=0.9, step=0.01),
    ],
))
