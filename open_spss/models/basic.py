"""通用统计:频数分析、描述性统计、分类汇总、正态性检验、方差齐性、P-P/Q-Q 图、相关分析。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..core import charts as ch
from ..core.datasets import Dataset, as_numeric, get_series
from ..core.registry import CAT, ORDINAL, QUANT, ModelSpec, Option, Slot, register_model
from ..core.results import ModelError, Result, fnum, fp, judge_p, stars


# ---------------------------------------------------------------- 频数分析
def _wcount(s: pd.Series, w: pd.Series | None) -> pd.Series:
    """按水平汇总权重;未加权时退化为计数。"""
    if w is None:
        return s.value_counts().astype(float)
    frame = pd.concat([s.rename("__v__"), w], axis=1).dropna()
    return frame.groupby("__v__", observed=True)["__w__"].sum().sort_values(ascending=False)


def _w_moments(x: np.ndarray, w: np.ndarray):
    """加权均值/无偏标准差/偏度/峰度(频数权重语义)。"""
    sw = w.sum()
    mu = float((x * w).sum() / sw)
    if sw <= 1:
        return mu, np.nan, np.nan, np.nan
    d = x - mu
    var = float((w * d * d).sum() / (sw - 1))
    sd = var ** 0.5
    m3 = float((w * d ** 3).sum() / sw)
    m4 = float((w * d ** 4).sum() / sw)
    skew = m3 / sd ** 3 if sd else np.nan
    kurt = m4 / sd ** 4 - 3 if sd else np.nan
    return mu, sd, skew, kurt


def _w_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """加权分位数:按累积权重的 CDF 线性插值。"""
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws)
    return float(np.interp(q * cw[-1], cw, xs))


def _run_frequency(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    cols = sl["vars"]
    w = ds.weights()
    for c in cols:
        s = get_series(ds, c).dropna().astype(str)
        if s.empty:
            raise ModelError(f"列「{c}」无有效数据")
        ws = w.reindex(s.index) if w is not None else None
        vc = _wcount(s, ws)
        total = float(vc.sum())
        cum = 0.0
        rows = []
        for lev, n in vc.items():
            pct = n / total * 100
            cum += pct
            rows.append([lev, fnum(n, 1), f"{pct:.2f}%", f"{cum:.2f}%"])
        rows.append(["合计", fnum(total, 1), "100%", ""])
        res.add_table(f"「{c}」频数分布", ["水平", "频数", "百分比", "累积百分比"], rows,
                      note=f"有效样本 N={fnum(total, 1)}")
        levels = [str(k) for k in vc.index][:10]
        res.add_chart(f"「{c}」频数柱状图",
                      ch.bar_chart(f"「{c}」频数分布", levels, {"频数": [round(float(vc[l]), 2) for l in levels]}, y_name="频数"))
        if op.get("chart_pie") and c == cols[0]:
            res.add_chart(f"「{c}」占比饼图", ch.pie_chart(f"「{c}」构成占比",
                          [(str(k), round(float(v), 2)) for k, v in vc.head(8).items()]))
    if w is not None:
        res.add_note(f"已按「{ds.weight_col}」加权:频数为权重合计,百分比基于加权频数。")
    return res


register_model(ModelSpec(
    id="frequency", name="频数分析", category="通用统计",
    desc="统计定类变量的各水平频数与占比，配合柱状图/饼图展示分布结构。",
    tags=["频数", "占比", "分类"], demo_note="survey",
    slots=[Slot("vars", "分析变量（定类）", types=(CAT, ORDINAL, QUANT), multiple=True)],
    options=[Option("chart_pie", "附加饼图", "switch", default=True)],
    runner=_run_frequency,
))


# ---------------------------------------------------------------- 描述性统计
def _run_descriptive(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    cols = sl["vars"]
    headers = ["变量名", "样本量", "最小值", "最大值", "总和", "均值", "标准差",
               "中位数", "变异系数", "偏度", "峰度"]
    rows = []
    w = ds.weights()
    for c in cols:
        s = as_numeric(ds, c).dropna()
        if len(s) < 2:
            raise ModelError(f"列「{c}」有效样本不足 2")
        if w is None:
            mu, sd = s.mean(), s.std(ddof=1)
            rows.append([
                c, int(len(s)), fnum(s.min(), 3), fnum(s.max(), 3), fnum(s.sum(), 3),
                fnum(mu), fnum(sd), fnum(s.median()), fnum(abs(sd / mu) if mu else np.nan, 3),
                fnum(s.skew()), fnum(s.kurt()),
            ])
        else:
            ws = w.reindex(s.index).fillna(0).values
            if ws.sum() <= 1 or (ws > 0).sum() < 2:
                raise ModelError(f"列「{c}」的加权有效样本不足 2")
            x = s.values
            mu, sd, skew, kurt = _w_moments(x, ws)
            med = _w_quantile(x, ws, 0.5)
            rows.append([
                c, fnum(ws.sum(), 1), fnum(s.min(), 3), fnum(s.max(), 3),
                fnum(float((x * ws).sum()), 3),
                fnum(mu), fnum(sd), fnum(med), fnum(abs(sd / mu) if mu else np.nan, 3),
                fnum(skew, 3), fnum(kurt, 3),
            ])
        if len(cols) <= 6:
            res.add_chart(f"「{c}」分布直方图", ch.hist_chart(f"「{c}」频数分布", s.values, x_name=c))
    if w is not None:
        headers[1] = "加权样本量"
    res.add_table("描述性统计结果", headers, rows,
                  note=("变异系数=标准差/均值;偏度、峰度以正态分布为参照(0)"
                        + (f";已按「{ds.weight_col}」加权,均值/标准差/分位数等为加权估计" if w is not None else "")))
    res.add_interp(
        f"本次共对 **{len(cols)}** 个定量变量进行描述性统计,有效样本量 "
        f"**{rows[0][1]}**。其中「{cols[0]}」的平均值为 **{rows[0][5]}**,标准差 **{rows[0][6]}**,"
        f"取值范围 [{rows[0][2]}, {rows[0][3]}]。",
        "偏度反映分布的不对称方向:偏度>0 呈右偏(长尾在右),<0 呈左偏;峰度>0 表示比正态更尖峰。"
        "结合直方图可判断数据是否近似正态、是否存在极端值。",
    )
    if len(cols) > 1:
        res.add_chart("各变量箱线图对比",
                      ch.box_chart("定量变量箱线图", {c: as_numeric(ds, c).dropna().tolist() for c in cols}))
    return res


register_model(ModelSpec(
    id="descriptive", name="描述性统计", category="通用统计",
    runner=_run_descriptive,
    desc="输出均值、标准差、偏度、峰度等集中与离散趋势指标，附直方图与箱线图。",
    tags=["均值", "标准差", "描述"], demo_note="survey",
    slots=[Slot("vars", "分析变量（定量）", types=(QUANT,), multiple=True)],
    options=[],
))


# ---------------------------------------------------------------- 分类汇总
def _run_aggregate(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    group, values = sl["group"], sl["vars"]
    funcs = op["funcs"] if isinstance(op["funcs"], list) else [op["funcs"]]
    fname = {"mean": "均值", "sum": "总和", "max": "最大", "min": "最小",
             "median": "中位数", "std": "标准差", "count": "计数"}
    g = ds.df.groupby(group, dropna=True, observed=True)
    headers = [group, "样本量"] + [f"{v}({fname[f]})" for v in values for f in funcs]
    rows = []
    for lev, sub in g:
        row: list = [str(lev), int(len(sub))]
        for v in values:
            sv = pd.to_numeric(sub[v], errors="coerce").dropna()
            for f in funcs:
                row.append(fnum(getattr(sv, "count" if f == "count" else f)()) if f == "count" else fnum(getattr(sv, f)()))
        rows.append(row)
    res.add_table(f"按「{group}」分类汇总", headers, rows)
    if values:
        v0 = values[0]
        means = {}
        for lev, sub in g:
            sv = pd.to_numeric(sub[v0], errors="coerce").dropna()
            if op.get("agg_chart") == "mean":
                means[str(lev)] = round(float(sv.mean()), 4)
            elif op.get("agg_chart") == "sum":
                means[str(lev)] = round(float(sv.sum()), 4)
            elif op.get("agg_chart") == "count":
                means[str(lev)] = int(len(sv))
        res.add_chart(f"各组{fname[op.get('agg_chart', 'mean')]}对比({v0})",
                      ch.bar_chart(f"各组{fname[op.get('agg_chart', 'mean')]}对比", list(means), {v0: list(means.values())}, x_name=group))
    res.add_interp(f"已按「{group}」将样本划分为 **{len(rows)}** 组并完成汇总,可用右侧图表比较各组的量级差异。")
    return res


register_model(ModelSpec(
    id="aggregate", name="分类汇总", category="通用统计",
    runner=_run_aggregate,
    desc="按定类变量分组，对定量变量计算均值/总和/中位数等汇总指标（类似 GROUP BY）。",
    tags=["分组", "汇总", "透视"], demo_note="survey",
    slots=[Slot("group", "分组变量（定类）", types=(CAT, ORDINAL)),
           Slot("vars", "汇总变量（定量，可多选）", types=(QUANT,), multiple=True)],
    options=[
        Option("funcs", "汇总指标", "select", default="mean",
               choices=[{"value": v, "label": l} for v, l in
                        [("mean", "均值"), ("sum", "总和"), ("max", "最大"), ("min", "最小"),
                         ("median", "中位数"), ("std", "标准差"), ("count", "计数")]]),
        Option("agg_chart", "图表指标", "select", default="mean",
               choices=[{"value": "mean", "label": "均值"}, {"value": "sum", "label": "总和"},
                        {"value": "count", "label": "计数"}]),
    ],
))


# ---------------------------------------------------------------- 正态性检验
def _run_normality(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    rows = []
    from statsmodels.stats.diagnostic import lilliefors
    for c in sl["vars"]:
        s = as_numeric(ds, c).dropna()
        n = len(s)
        if n < 4:
            raise ModelError(f"列「{c}」有效样本不足 4")
        sw_w, sw_p = (np.nan, np.nan)
        if n <= 5000:
            sw_w, sw_p = sps.shapiro(s)
        try:
            ks_stat, ks_p = lilliefors(s, dist="norm")
        except Exception:  # noqa: BLE001
            ks_stat, ks_p = np.nan, np.nan
        jb_stat, jb_p = sps.jarque_bera(s)
        alpha = op["alpha"]
        rows.append([
            c, n,
            fnum(sw_w), fp(sw_p), judge_p(sw_p, alpha),
            fnum(ks_stat, 4), fp(ks_p), judge_p(ks_p, alpha),
            fnum(jb_stat, 4), fp(jb_p), judge_p(jb_p, alpha),
        ])
        if c == sl["vars"][0]:
            res.add_chart(f"「{c}」直方图与正态曲线", ch.hist_chart(f"「{c}」频数分布", s.values, x_name=c))
            res.add_chart(f"「{c}」Q-Q 图", ch.qq_chart(f"「{c}」Q-Q 图(正态)", s.values))
    res.add_table("正态性检验结果", ["变量名", "样本量", "S-W统计量", "S-W p", "S-W正态?",
                                    "K-S统计量", "K-S p", "K-S正态?", "JB统计量", "JB p", "JB正态?"],
                  rows, note="Shapiro-Wilk 适用于 n≤5000;K-S(Lilliefors 校正)、Jarque-Bera 适用于大样本")
    alpha_pct = op["alpha"]
    sig = [r for r in rows if r[4] == "否"]
    res.add_interp(
        f"以 α={alpha_pct} 为显著性水平:" +
        (f"变量 **{', '.join(r[0] for r in rows if r[4] == '是')}** 满足正态性假设;"
         if any(r[4] == "是" for r in rows) else "") +
        (f"变量 **{', '.join(r[0] for r in sig)}** 不满足正态性,建议改用非参数检验方法。" if sig else "全部变量均满足正态性假设,可使用参数检验方法。"),
    )
    return res


register_model(ModelSpec(
    id="normality", name="正态性检验", category="通用统计",
    runner=_run_normality,
    desc="Shapiro-Wilk / K-S（Lilliefors） / Jarque-Bera 三种正态性检验，附直方图与 Q-Q 图。",
    tags=["正态性", "Shapiro-Wilk", "K-S"], demo_note="survey",
    slots=[Slot("vars", "分析变量（定量）", types=(QUANT,), multiple=True)],
    options=[Option("alpha", "显著性水平 α", "select", default=0.05,
                    choices=[{"value": 0.1, "label": "0.1"}, {"value": 0.05, "label": "0.05"}, {"value": 0.01, "label": "0.01"}])],
))


# ---------------------------------------------------------------- 方差齐性检验
def _run_var_homogeneity(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    y, g = as_numeric(ds, sl["y"]).dropna(), get_series(ds, sl["group"])
    data = pd.concat([y.rename("y"), g.rename("g")], axis=1).dropna()
    groups = [grp["y"].values for _, grp in data.groupby("g", observed=True) if len(grp) >= 2]
    if len(groups) < 2:
        raise ModelError("分组后每组至少需要 2 个有效样本")
    groups = [g.astype(float) for g in groups]
    lv_w, lv_p = sps.levene(*groups, center=op["center"])
    if all(g.var() > 0 for g in groups):
        bt_w, bt_p = sps.bartlett(*groups)
    else:
        bt_w, bt_p = np.nan, np.nan
        res.add_note("存在方差为 0 的组,Bartlett 检验不适用(已跳过),请以 Levene 结果为准")
    rows = [[f"Levene({op['center']})", fnum(lv_w, 4), int(sum(len(x) for x in groups) - len(groups)), fp(lv_p), judge_p(lv_p, op["alpha"])],
            ["Bartlett", fnum(bt_w, 4), len(groups) - 1, fp(bt_p), "—" if np.isnan(bt_p) else judge_p(bt_p, op["alpha"])]]
    res.add_table("方差齐性检验", ["方法", "统计量", "自由度", "p 值", "方差齐?"], rows,
                  note="p > α 表示各组方差满足齐性(齐性)")
    res.add_interp(
        f"Levene 检验 p={fp(lv_p)} {'>' if lv_p >= op['alpha'] else '≤'} α={op['alpha']},"
        + ("各组方差齐,可使用方差分析、独立样本 T 检验(合并方差)等参数方法。"
           if lv_p >= op["alpha"] else "各组方差不齐,建议使用 Welch 校正或非参数检验。"),
    )
    stat_tbl = [[str(k), int(len(v["y"])), fnum(v["y"].mean()), fnum(v["y"].std(ddof=1))]
                for k, v in data.groupby("g", observed=True)]
    res.add_table("分组描述统计", ["组别", "N", "均值", "标准差"], stat_tbl)
    res.add_chart("分组箱线图", ch.box_chart("各组的分布箱线图",
                 {str(k): v["y"].tolist() for k, v in data.groupby("g", observed=True)}, y_name=sl["y"]))
    return res


register_model(ModelSpec(
    id="var_homogeneity", name="方差齐性检验", category="通用统计",
    runner=_run_var_homogeneity,
    desc="Levene / Bartlett 检验各组方差是否相等，是方差分析/T 检验的前置条件。",
    tags=["方差齐性", "Levene"], demo_note="survey",
    slots=[Slot("y", "因变量（定量）", types=(QUANT,)),
           Slot("group", "分组变量（定类）", types=(CAT, ORDINAL))],
    options=[
        Option("center", "Levene 中心", "select", default="median",
               choices=[{"value": "median", "label": "中位数（Brown-Forsythe）"}, {"value": "mean", "label": "均值"}]),
        Option("alpha", "显著性水平 α", "select", default=0.05,
               choices=[{"value": 0.1, "label": "0.1"}, {"value": 0.05, "label": "0.05"}, {"value": 0.01, "label": "0.01"}]),
    ],
))


# ---------------------------------------------------------------- P-P 图 / Q-Q 图
def _run_pp_qq(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    s = as_numeric(ds, sl["var"]).dropna()
    if len(s) < 3:
        raise ModelError("有效样本不足 3")
    res.add_table("分布概要", ["样本量N", "均值", "标准差", "偏度", "峰度"],
                  [[len(s), fnum(s.mean(), 4), fnum(s.std(ddof=1), 4), fnum(s.skew(), 3), fnum(s.kurt(), 3)]])
    kind = op["kind"]
    if kind == "qq":
        res.add_chart("Q-Q 图(正态分布)", ch.qq_chart("Q-Q 图", s.values))
        res.add_interp("Q-Q 图中样本点越贴近红色参考线,数据越接近正态分布;两端点偏离通常表示尾部厚度异常。")
    else:
        from ..core.charts import scatter_chart
        n = len(s)
        ordered = np.sort(s.values)
        p_emp = (np.arange(1, n + 1) - 0.5) / n
        theo_cdf = sps.norm.cdf(ordered, loc=s.mean(), scale=s.std(ddof=1))
        pts = [[round(float(a), 4), round(float(b), 4)] for a, b in zip(theo_cdf, p_emp)]
        res.add_chart("P-P 图(正态分布)", scatter_chart("P-P 图", {"经验累积概率": pts},
                     x_name="理论累积概率", y_name="经验累积概率",
                     trend=[[0, 0], [1, 1]], trend_name="45°线"))
        res.add_interp("P-P 图中样本点越贴近 45° 参考线,数据的累积分布越接近理论正态分布。")
    return res


register_model(ModelSpec(
    id="pp_qq", name="P-P图 / Q-Q图", category="通用统计",
    runner=_run_pp_qq,
    desc="以图形方式直观检验数据与正态分布的接近程度。",
    tags=["P-P图", "Q-Q图", "正态"], demo_note="survey",
    slots=[Slot("var", "分析变量（定量）", types=(QUANT,))],
    options=[Option("kind", "图形类型", "select", default="qq",
                    choices=[{"value": "qq", "label": "Q-Q 图"}, {"value": "pp", "label": "P-P 图"}])],
))


# ---------------------------------------------------------------- 相关分析
def _run_correlation(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    cols = sl["vars"]
    if len(cols) < 2:
        raise ModelError("相关分析至少需要 2 个变量")
    data = pd.concat([as_numeric(ds, c).rename(c) for c in cols], axis=1)
    method = op["method"]
    fn = {"pearson": sps.pearsonr, "spearman": sps.spearmanr, "kendall": sps.kendalltau}[method]
    mname = {"pearson": "Pearson 积差相关", "spearman": "Spearman 等级相关", "kendall": "Kendall tau 相关"}[method]
    nvar = len(cols)
    R = np.full((nvar, nvar), np.nan)
    P = np.full((nvar, nvar), np.nan)
    for i in range(nvar):
        for j in range(i, nvar):
            if i == j:
                R[i, j] = 1.0
                P[i, j] = 0.0
            else:
                xy = data[[cols[i], cols[j]]].dropna()
                r, p = fn(xy.iloc[:, 0], xy.iloc[:, 1])
                R[i, j] = R[j, i] = r
                P[i, j] = P[j, i] = p
    rows = []
    for i, c in enumerate(cols):
        rows.append([c] + [f"{fnum(R[i, j], 3)}{stars(P[i, j])}" for j in range(nvar)])
    res.add_table(f"{mname}系数矩阵", ["变量"] + cols, rows,
                  note="*** p<0.01,** p<0.05,* p<0.1(双尾);对角线为 1")
    prow = []
    for i, c in enumerate(cols):
        prow.append([c] + [fp(P[i, j]) if i != j else "—" for j in range(nvar)])
    res.add_table("p 值矩阵(双尾)", ["变量"] + cols, prow)
    res.add_chart("相关系数热力图", ch.heatmap_chart("相关系数热力图", cols, cols, R.tolist(),
                 fmt="{c}", v_min=-1, v_max=1))
    res.add_interp(
        f"采用 **{mname}**,共分析 {nvar} 个变量的两两相关关系。" +
        (f"其中「{cols[0]}」与「{cols[1]}」的相关系数为 **{fnum(R[0,1],3)}**(p={fp(P[0,1])}),"
         f"呈{'显著' if P[0,1]<op['alpha'] else '不显著'}的"
         f"{'正' if R[0,1]>0 else '负'}相关,相关程度为"
         f"{'强' if abs(R[0,1])>=0.6 else '中等' if abs(R[0,1])>=0.4 else '弱'}。" if nvar >= 2 else ""),
    )
    xy = data[[cols[0], cols[1]]].dropna()
    if len(xy) <= 500:
        pts = [[float(a), float(b)] for a, b in xy.values]
        res.add_chart(f"「{cols[0]}」与「{cols[1]}」散点图",
                      ch.scatter_chart("散点图", {"样本": pts}, x_name=cols[0], y_name=cols[1]))
    return res


register_model(ModelSpec(
    id="correlation", name="相关分析", category="通用统计",
    runner=_run_correlation,
    desc="Pearson / Spearman / Kendall 相关矩阵与显著性检验，附热力图与散点图。",
    tags=["相关系数", "热力图"], demo_note="survey",
    slots=[Slot("vars", "分析变量（定量，≥2）", types=(QUANT,), multiple=True)],
    options=[
        Option("method", "相关系数类型", "select", default="pearson",
               choices=[{"value": "pearson", "label": "Pearson（双变量正态）"},
                        {"value": "spearman", "label": "Spearman（非正态/等级）"},
                        {"value": "kendall", "label": "Kendall tau（小样本/有序）"}]),
        Option("alpha", "显著性水平 α", "select", default=0.05,
               choices=[{"value": 0.1, "label": "0.1"}, {"value": 0.05, "label": "0.05"}, {"value": 0.01, "label": "0.01"}]),
    ],
))


# ================================================================ 偏相关分析
def _run_partial_corr(ds: Dataset, sl: dict, op: dict) -> Result:
    """偏相关:控制变量回归取残差后的 Pearson 相关,t 检验推断。"""
    res = Result()
    cols, ctrls = list(sl["vars"]), list(sl["ctrls"])
    overlap = [c for c in cols if c in ctrls]
    if overlap:
        raise ModelError(f"分析变量与控制变量重复:「{overlap[0]}」")
    data = pd.concat([as_numeric(ds, c).rename(c) for c in cols + ctrls], axis=1).dropna()
    n = len(data)
    dfree = n - len(ctrls) - 2
    if dfree < 1:
        raise ModelError(f"有效样本不足:控制 {len(ctrls)} 个变量后自由度 ≤0(当前有效 {n} 行)")
    Y = data[cols].values.astype(float)
    if ctrls:
        Z = np.column_stack([np.ones(n), data[ctrls].values.astype(float)])
        beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
        Y = Y - Z @ beta          # 残差:剔除控制变量的线性影响
    R = np.corrcoef(Y.T)
    P = np.full_like(R, np.nan)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j:
                r = float(np.clip(R[i, j], -0.999999, 0.999999))
                t = r * np.sqrt(dfree / (1 - r * r))
                P[i, j] = float(2 * sps.t.sf(abs(t), dfree))
    rows = [[c] + [f"{fnum(R[i, j], 3)}{stars(P[i, j])}" if i != j else "1" for j in range(len(cols))]
            for i, c in enumerate(cols)]
    ctrl_desc = "、".join(ctrls) if ctrls else "无"
    res.add_table("偏相关系数矩阵", ["变量"] + cols, rows,
                  note=f"控制变量:{ctrl_desc};*** p<0.01,** p<0.05,* p<0.1(双尾),自由度 df={dfree}")
    res.add_chart("偏相关系数热力图", ch.heatmap_chart("偏相关系数热力图", cols, cols, R.tolist(),
                  fmt="{c}", v_min=-1, v_max=1))
    r01, p01 = R[0, 1], P[0, 1]
    res.add_interp(
        f"在控制 {ctrl_desc} 的条件下,「{cols[0]}」与「{cols[1]}」的偏相关系数为 **{fnum(r01, 3)}**"
        f"(p={fp(p01)}),偏相关关系{'显著' if p01 < op['alpha'] else '不显著'},"
        f"{'正' if r01 > 0 else '负'}相关且程度"
        f"{'较强' if abs(r01) >= 0.5 else '中等' if abs(r01) >= 0.3 else '较弱'}。"
        f"偏相关剔除了控制变量的混杂影响,比简单相关更能反映两变量的直接关联。")
    return res


register_model(ModelSpec(
    id="partial_corr", name="偏相关分析", category="通用统计",
    runner=_run_partial_corr,
    desc="在控制一个或多个变量的条件下分析两变量的净相关，输出偏相关矩阵与显著性检验。",
    tags=["偏相关", "控制变量", "净相关"], demo_note="survey",
    slots=[
        Slot("vars", "分析变量（定量，≥2）", types=(QUANT,), multiple=True),
        Slot("ctrls", "控制变量（定量，≥1）", types=(QUANT,), multiple=True, max_count=10),
    ],
    options=[Option("alpha", "显著性水平 α", "select", default=0.05,
                    choices=[{"value": 0.1, "label": "0.1"}, {"value": 0.05, "label": "0.05"},
                             {"value": 0.01, "label": "0.01"}])],
))


# ================================================================ RFM 模型
def _run_rfm(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    cust, rcol, fcol, mcol = sl["cust"], sl["r"], sl["f"], sl["m"]
    data = pd.DataFrame({
        "cust": get_series(ds, cust).astype(str),
        "R": as_numeric(ds, rcol),
        "F": as_numeric(ds, fcol),
        "M": as_numeric(ds, mcol),
    }).dropna()
    if len(data) < 10:
        raise ModelError(f"有效数据不足 10 行(当前 {len(data)})")
    agg = data.groupby("cust", observed=True).agg(R=("R", "min"), F=("F", "sum"), M=("M", "sum"))
    if len(agg) < 5:
        raise ModelError("客户数不足 5,无法分层")
    split = {"median": agg.median, "mean": agg.mean}[op["split"]]()
    # R:数值越小(越近期)得分越高;F/M:越大得分越高
    r_score = (agg["R"] <= split["R"]).astype(int) + 1
    f_score = (agg["F"] >= split["F"]).astype(int) + 1
    m_score = (agg["M"] >= split["M"]).astype(int) + 1
    seg_map = {
        (2, 2, 2): "重要价值客户", (2, 2, 1): "一般价值客户",
        (2, 1, 2): "重要发展客户", (2, 1, 1): "一般发展客户",
        (1, 2, 2): "重要保持客户", (1, 2, 1): "一般保持客户",
        (1, 1, 2): "重要挽留客户", (1, 1, 1): "一般挽留客户",
    }
    seg = [seg_map[(r, f, m)] for r, f, m in zip(r_score, f_score, m_score)]
    agg = agg.assign(R得分=r_score, F得分=f_score, M得分=m_score, 客户分层=seg)
    split_name = {"median": "中位数", "mean": "均值"}[op["split"]]
    res.add_table("分层切分点", ["指标", "切分点(" + split_name + ")", "高分定义"],
                  [["R(最近购买间隔)", fnum(split["R"], 2), "≤切分点(近期有购买)"],
                   ["F(购买频次)", fnum(split["F"], 2), "≥切分点(购买频繁)"],
                   ["M(消费金额)", fnum(split["M"], 2), "≥切分点(消费高)"]])
    n_show = min(len(agg), 100)
    res.add_table("客户 RFM 明细(前 " + str(n_show) + " 位)",
                  ["客户", "R", "F", "M", "R得分", "F得分", "M得分", "客户分层"],
                  [[c, fnum(row["R"], 1), fnum(row["F"], 0), fnum(row["M"], 1), row["R得分"], row["F得分"], row["M得分"], row["客户分层"]]
                   for c, row in agg.head(n_show).iterrows()],
                  note="共 " + str(len(agg)) + " 位客户" + ("(仅展示前 100)" if len(agg) > 100 else ""))
    stat = agg.groupby("客户分层", observed=True).agg(
        人数=("M", "size"), 平均R=("R", "mean"), 平均F=("F", "mean"), 平均M=("M", "mean"), 金额合计=("M", "sum"))
    stat = stat.sort_values("金额合计", ascending=False)
    total_n = int(stat["人数"].sum())
    total_amt = float(stat["金额合计"].sum()) or 1
    res.add_table("客户分层统计(按金额贡献降序)",
                  ["客户分层", "人数", "人数占比", "平均R", "平均F", "平均M", "金额合计", "金额占比"],
                  [[i, int(row["人数"]), f"{row['人数'] / total_n * 100:.1f}%", fnum(row["平均R"], 1),
                    fnum(row["平均F"], 1), fnum(row["平均M"], 1), fnum(row["金额合计"], 1),
                    f"{row['金额合计'] / total_amt * 100:.1f}%"] for i, row in stat.iterrows()])
    res.add_chart("客户分层人数占比", ch.pie_chart("RFM 客户分层", [(i, int(row["人数"])) for i, row in stat.iterrows()]))
    res.add_chart("各层金额贡献", ch.bar_chart("各客户分层消费金额合计", list(stat.index),
                 {"金额合计": [round(float(v), 2) for v in stat["金额合计"]]}, y_name="金额合计"))
    top_seg = stat.index[0]
    res.add_interp(f"共 {len(agg)} 位客户,按 {split_name} 切分为 8 层。**{top_seg}** 人数 {int(stat.iloc[0]['人数'])} 人"
                   f"({stat.iloc[0]['人数'] / total_n * 100:.1f}%),金额贡献 {stat.iloc[0]['金额合计'] / total_amt * 100:.1f}%。"
                   "建议:重要价值客户保持服务;重要发展/保持客户重点激活;重要挽留客户优先召回。")
    return res


register_model(ModelSpec(
    id="rfm", name="RFM 客户价值模型", category="通用统计",
    runner=_run_rfm,
    desc="按最近购买（R）、频次（F）、金额（M）将客户切分为 8 层，输出分层画像与金额贡献。",
    tags=["RFM", "客户分层", "会员营销"], demo_note="ecommerce",
    slots=[
        Slot("cust", "客户标识（每行一笔交易）", types=("any",)),
        Slot("r", "R-最近购买间隔（定量）", types=(QUANT,)),
        Slot("f", "F-购买频次（定量）", types=(QUANT,)),
        Slot("m", "M-消费金额（定量）", types=(QUANT,)),
    ],
    options=[Option("split", "得分切分方式", "select", default="median",
                    choices=[{"value": "median", "label": "中位数"}, {"value": "mean", "label": "均值"}])],
))
