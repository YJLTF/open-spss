"""统计绘图:直方图、箱线图、散点图、折线图、条形/饼图等通用可视化工具。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..core import charts as ch
from ..core.datasets import Dataset, as_numeric, get_series
from ..core.registry import CAT, ORDINAL, QUANT, ModelSpec, Option, Slot, register_model
from ..core.results import ModelError, Result, fnum, fp


def _run_hist(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    for c in sl["vars"][:4]:
        s = as_numeric(ds, c).dropna()
        if len(s) < 5:
            raise ModelError(f"列「{c}」有效样本不足 5")
        res.add_chart(f"「{c}」直方图", ch.hist_chart(f"「{c}」频数分布(含正态曲线)", s.values,
                     x_name=c, bins=int(op["bins"]) if op["bins"] else None))
        mean, std = s.mean(), s.std(ddof=1)
        sw_w, sw_p = sps.shapiro(s) if len(s) <= 5000 else (np.nan, np.nan)
        res.add_table(f"「{c}」分布概要", ["N", "均值", "标准差", "偏度", "峰度", "S-W p"],
                      [[len(s), fnum(mean, 3), fnum(std, 3), fnum(s.skew(), 3), fnum(s.kurt(), 3), fp(sw_p)]])
    return res


register_model(ModelSpec(
    id="plot_hist", name="直方图", category="统计绘图",
    runner=_run_hist,
    desc="频数分布直方图与正态拟合曲线，快速查看数据分布形态。",
    tags=["直方图", "分布"], demo_note="survey",
    slots=[Slot("vars", "分析变量（定量，最多4个）", types=(QUANT,), multiple=True, max_count=4)],
    options=[Option("bins", "分箱数（0=自动）", "number", default=0, min=0, max=60)],
))


def _run_box(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    groups = {}
    ycol = sl.get("y") or (sl["vars"][0] if sl["vars"] else "")
    if sl.get("group") and ycol:
        g = get_series(ds, sl["group"]).astype(str)
        y = as_numeric(ds, ycol)
        data = pd.concat([y.rename("y"), g.rename("g")], axis=1).dropna()
        for lev, sub in data.groupby("g", observed=True):
            if len(sub) >= 3:
                groups[str(lev)] = sub["y"].tolist()
        if len(groups) < 2:
            raise ModelError("分组后有效组数不足 2")
        res.add_chart("分组箱线图", ch.box_chart(f"「{ycol}」按「{sl['group']}」分组箱线图", groups, y_name=ycol))
        stat_rows = []
        for lev, sub in data.groupby("g", observed=True):
            sv = sub["y"]
            stat_rows.append([str(lev), len(sv), fnum(sv.mean(), 3), fnum(sv.median(), 3),
                              fnum(sv.std(ddof=1), 3), fnum(sv.min(), 3), fnum(sv.max(), 3)])
        res.add_table("分组统计", ["组别", "N", "均值", "中位数", "标准差", "最小", "最大"], stat_rows)
    else:
        for c in sl["vars"]:
            groups[c] = as_numeric(ds, c).dropna().tolist()
        if not groups:
            raise ModelError("请选择分析变量或分组变量")
        res.add_chart("箱线图", ch.box_chart("多变量箱线图", groups, y_name=""))
    return res


register_model(ModelSpec(
    id="plot_box", name="箱线图", category="统计绘图",
    runner=_run_box,
    desc="展示中位数、四分位距与离群点，支持多变量对比或按组比较。",
    tags=["箱线图", "离群点"], demo_note="survey",
    slots=[Slot("vars", "分析变量（不分组时，多选）", types=(QUANT,), multiple=True, max_count=8, required=False),
           Slot("y", "因变量（分组时）", types=(QUANT,), required=False),
           Slot("group", "分组变量（选填）", types=(CAT, ORDINAL), required=False)],
    options=[],
))


def _run_scatter(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    x, y = as_numeric(ds, sl["x"]), as_numeric(ds, sl["y"])
    data = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    r, p = sps.pearsonr(data["x"], data["y"])
    slope, intercept = np.polyfit(data["x"], data["y"], 1)
    xs = np.linspace(data["x"].min(), data["x"].max(), 40)
    series = {}
    if sl.get("group"):
        g = get_series(ds, sl["group"]).astype(str)
        gg = g.loc[data.index]
        for lev in pd.unique(gg):
            mask = gg == lev
            series[str(lev)] = [[float(a), float(b)] for a, b in data.loc[mask.values].values]
    else:
        series["样本"] = [[float(a), float(b)] for a, b in data.values]
    trend = [[float(a), float(slope * a + intercept)] for a in xs]
    res.add_chart("散点图", ch.scatter_chart(f"散点图(r={fnum(r, 3)}, p={fp(p)})",
                 series, x_name=sl["x"], y_name=sl["y"], trend=trend, trend_name="线性拟合"))
    res.add_table("相关与拟合", ["Pearson r", "p", "线性方程", "样本量"],
                  [[fnum(r, 4), fp(p), f"y = {fnum(slope, 4)}x + {fnum(intercept, 4)}", len(data)]])
    res.add_interp(f"两变量 Pearson 相关系数 r={fnum(r, 3)}(p={fp(p)}),"
                   f"呈{'显著' if p < 0.05 else '不显著'}的{'正' if r > 0 else '负'}相关;虚线为最小二乘拟合线。")
    return res


register_model(ModelSpec(
    id="plot_scatter", name="散点图", category="统计绘图",
    runner=_run_scatter,
    desc="两变量散点图与线性拟合线，支持按组着色，附相关系数。",
    tags=["散点图", "拟合"], demo_note="survey",
    slots=[Slot("x", "X 变量（定量）", types=(QUANT,)),
           Slot("y", "Y 变量（定量）", types=(QUANT,)),
           Slot("group", "着色分组（选填）", types=(CAT, ORDINAL), required=False)],
    options=[],
))


def _run_line(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    ys = sl["ys"]
    if sl.get("x"):
        x = get_series(ds, sl["x"]).astype(str)
        series = {}
        for c in ys:
            s = as_numeric(ds, c)
            data = pd.concat([x.rename("x"), s.rename("y")], axis=1).dropna()
            agg = data.groupby("x", observed=True)["y"].mean()
            series[c] = [round(float(v), 4) for v in agg.values]
        cats = list(agg.index)
    else:
        cats = [f"第{i+1}行" for i in range(len(ds.df))]
        series = {c: [round(float(v), 4) if pd.notna(v) else None
                      for v in as_numeric(ds, c).values] for c in ys}
    if not series:
        raise ModelError("请至少选择一个 Y 变量")
    res.add_chart("折线图", ch.line_chart("折线图", cats, series, x_name=sl["x"] or "序号",
                 y_name="/".join(ys), smooth=op.get("smooth", True)))
    res.add_table("数据概要", ["变量", "N", "均值", "最大值", "最小值"],
                  [[c, int(as_numeric(ds, c).notna().sum()), fnum(as_numeric(ds, c).mean(), 3),
                    fnum(as_numeric(ds, c).max(), 3), fnum(as_numeric(ds, c).min(), 3)] for c in ys])
    return res


register_model(ModelSpec(
    id="plot_line", name="折线图", category="统计绘图",
    runner=_run_line,
    desc="多序列折线图，可按类别变量聚合取均值，适合趋势与对比展示。",
    tags=["折线图", "趋势"], demo_note="timeseries",
    slots=[Slot("ys", "Y 变量（可多个）", types=(QUANT,), multiple=True, max_count=6),
           Slot("x", "X 轴类别（选填）", types=(CAT, ORDINAL, QUANT), required=False)],
    options=[],
))


def _run_bar_pie(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    cat = sl["cat"]
    if sl.get("value"):
        val = as_numeric(ds, sl["value"])
        g = get_series(ds, cat).astype(str)
        data = pd.concat([g.rename("g"), val.rename("v")], axis=1).dropna()
        agg = data.groupby("g", observed=True)["v"].agg(["mean", "sum", "count"])
        metric = op["metric"]
        pairs = [(str(i), float(agg.loc[i, metric])) for i in agg.index]
        res.add_chart("条形图", ch.bar_chart(f"各{cat}的{metric}", [p[0] for p in pairs],
                     {sl["value"]: [round(p[1], 4) for p in pairs]}, y_name=sl["value"]))
    else:
        vc = get_series(ds, cat).astype(str).value_counts()
        pairs = [(str(k), int(v)) for k, v in list(vc.items())[:12]]
        res.add_chart("条形图", ch.bar_chart(f"「{cat}」频数", [p[0] for p in pairs],
                     {"频数": [p[1] for p in pairs]}, y_name="频数"))
    if op["kind"] == "pie":
        res.add_chart("饼图", ch.pie_chart("构成占比", pairs))
    res.add_table("数据明细", ["类别", "数值"],
                  [[p[0], fnum(p[1], 3)] for p in pairs])
    return res


register_model(ModelSpec(
    id="plot_bar", name="条形图 / 饼图", category="统计绘图",
    runner=_run_bar_pie,
    desc="按类别统计频数或均值，输出条形图与饼图。",
    tags=["条形图", "饼图"], demo_note="survey",
    slots=[Slot("cat", "类别变量（定类）", types=(CAT, ORDINAL)),
           Slot("value", "数值变量（选填，默认频数）", types=(QUANT,), required=False)],
    options=[
        Option("metric", "聚合方式（有数值变量时）", "select", default="mean",
               choices=[{"value": "mean", "label": "均值"}, {"value": "sum", "label": "总和"}, {"value": "count", "label": "计数"}]),
        Option("kind", "图表类型", "select", default="bar",
               choices=[{"value": "bar", "label": "条形图"}, {"value": "pie", "label": "条形图+饼图"}]),
    ],
))


# ================================================================ 饼图
def _run_pie(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    s = get_series(ds, sl["var"]).dropna().astype(str)
    if s.nunique() < 2:
        raise ModelError("分类变量至少需要 2 个类别")
    if s.nunique() > 12:
        raise ModelError(f"类别数过多({s.nunique()} 个),饼图建议 ≤12 类,请先合并类别")
    counts = s.value_counts()
    pairs = [(str(k), int(v)) for k, v in counts.items()]
    res.add_table("频数分布", ["类别", "频数", "占比"],
                  [[k, v, f"{v / len(s) * 100:.2f}%"] for k, v in pairs],
                  note=f"N={int(len(s))}")
    res.add_chart("饼图", ch.pie_chart(f"「{sl['var']}」构成", pairs, donut=op["donut"]))
    res.add_interp(f"「{sl['var']}」共 {len(s)} 个有效观测,占比最高的类别为「{pairs[0][0]}」({pairs[0][1] / len(s) * 100:.1f}%),"
                   f"最低为「{pairs[-1][0]}」({pairs[-1][1] / len(s) * 100:.1f}%)。")
    return res


register_model(ModelSpec(
    id="plot_pie", name="饼图 / 环形图", category="统计绘图",
    runner=_run_pie,
    desc="展示分类变量的构成比例，支持环形样式。",
    tags=["饼图", "占比", "构成"], demo_note="survey",
    slots=[Slot("var", "分类变量", types=(CAT, ORDINAL))],
    options=[Option("donut", "环形样式", "switch", default=True)],
))


# ================================================================ 雷达图
def _run_radar(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    cols = sl["vars"][:8]
    if len(cols) < 3:
        raise ModelError("雷达图至少需要 3 个维度")
    means = {}
    for c in cols:
        s = as_numeric(ds, c).dropna()
        if len(s) < 3:
            raise ModelError(f"列「{c}」有效样本不足")
        means[c] = float(s.mean())
    rows = [[c, fnum(v, 3)] for c, v in means.items()]
    res.add_table("各维度均值", ["维度", "均值"], rows)
    vals = np.asarray(list(means.values()), dtype=float)
    if op["normalize"] == "minmax":
        rng = vals.max() - vals.min()
        norm = (vals - vals.min()) / rng * 100 if rng > 0 else np.full(len(vals), 50.0)
        tip = "按各维度最小-最大归一化到 0~100"
    elif op["normalize"] == "z":
        norm = (vals - vals.mean()) / vals.std(ddof=1)
        norm = 50 + norm * 15  # z 分数映射到 50±15 的可读区间
        tip = "按 z 分数标准化映射(50 为平均水平)"
    else:
        mx = np.abs(vals).max() or 1
        norm = vals / mx * 100
        tip = "按绝对值占比归一化"
    res.add_chart("雷达图", ch.radar_chart(
        f"「{'/'.join(cols[:3])}{'…' if len(cols) > 3 else ''}」维度对比",
        [(c, 100.0) for c in cols], {"均值(归一化)": [round(float(v), 1) for v in norm]}))
    res.add_table("归一化结果", ["维度", "原始均值", "归一化(0~100)"],
                  [[c, fnum(m, 3), fnum(n, 1)] for c, m, n in zip(cols, means.values(), norm)],
                  note=tip)
    top = int(np.argmax(norm))
    res.add_interp(f"按「{tip}」后,表现最强的维度是「{cols[top]}」({norm[top]:.1f}),最弱的是「{cols[int(np.argmin(norm))]}」"
                   f"({norm.min():.1f})。雷达图面积越大表示整体水平越均衡且越高。")
    return res


register_model(ModelSpec(
    id="plot_radar", name="雷达图", category="统计绘图",
    runner=_run_radar,
    desc="多维度指标对比雷达图，自动归一化到同一量纲（3~8 个维度）。",
    tags=["雷达图", "多维", "对比"], demo_note="survey",
    slots=[Slot("vars", "维度变量（定量/定级，3~8个）", types=(QUANT, ORDINAL), multiple=True, max_count=8)],
    options=[Option("normalize", "归一化方式", "select", default="minmax",
                    choices=[{"value": "minmax", "label": "最小-最大（0~100）"},
                             {"value": "z", "label": "z 分数映射"},
                             {"value": "ratio", "label": "绝对值占比"}])],
))


# ================================================================ 帕累托图
def _run_pareto(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    s = get_series(ds, sl["var"]).dropna().astype(str)
    if s.nunique() < 2:
        raise ModelError("分类变量至少需要 2 个类别")
    counts = s.value_counts()
    if len(counts) > 15:
        raise ModelError(f"类别数过多({len(counts)}),建议先合并次要类别(帕累托图通常展示前几大主因)")
    pairs = [(str(k), int(v)) for k, v in counts.items()]
    total = sum(v for _, v in pairs)
    cum, acc = [], 0
    for _, v in pairs:
        acc += v
        cum.append(acc / total * 100)
    n80 = next((i + 1 for i, c in enumerate(cum) if c >= 80), len(pairs))
    res.add_table("频数与累计占比", ["类别", "频数", "占比", "累计占比"],
                  [[k, v, f"{v / total * 100:.2f}%", f"{c:.2f}%"] for (k, v), c in zip(pairs, cum)],
                  note=f"N={int(total)}")
    res.add_chart("帕累托图", ch.pareto_chart(f"「{sl['var']}」帕累托分析", pairs))
    res.add_interp(
        f"前 {n80} 个类别累计贡献 **{cum[n80 - 1]:.1f}%** 的频数"
        + (f",符合“关键少数”原则({n80}/{len(pairs)} 的类别承担了主要部分)。" if n80 <= len(pairs) * 0.6 else ",分布较分散,没有明显的主导类别。")
        + f"最主要类别为「{pairs[0][0]}」({pairs[0][1] / total * 100:.1f}%)。")
    return res


register_model(ModelSpec(
    id="plot_pareto", name="帕累托图", category="统计绘图",
    runner=_run_pareto,
    desc="按频数降序排列并叠加累计百分比曲线，识别“关键的少数”类别（质量管理常用）。",
    tags=["帕累托", "二八原则", "主次因素"], demo_note="survey",
    slots=[Slot("var", "分类变量", types=(CAT, ORDINAL))],
    options=[],
))


# ================================================================ 漏斗图
def _run_funnel(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    s = get_series(ds, sl["var"]).dropna().astype(str)
    if s.nunique() < 2:
        raise ModelError("分类变量至少需要 2 个类别")
    counts = s.value_counts()
    pairs = [(str(k), int(v)) for k, v in counts.items()]
    conv = [pairs[0][1]]
    for i in range(1, len(pairs)):
        conv.append(pairs[i][1] / pairs[0][1] * 100 if pairs[0][1] else np.nan)
    res.add_table("转化概览", ["阶段", "数量", "相对首层转化率"],
                  [[k, v, fnum(c, 2) + "%"] for (k, v), c in zip(pairs, conv)])
    res.add_chart("漏斗图", ch.funnel_chart(f"「{sl['var']}」漏斗", pairs))
    res.add_interp(f"以数量最多的「{pairs[0][0]}」({pairs[0][1]})为漏斗首层,末层「{pairs[-1][0]}」仅为其 "
                   f"{conv[-1] if conv and not np.isnan(conv[-1]) else 0:.1f}%。漏斗图适用于展示流程各环节的递减与转化情况。")
    return res


register_model(ModelSpec(
    id="plot_funnel", name="漏斗图", category="统计绘图",
    runner=_run_funnel,
    desc="按数量递减展示流程各环节，常用于转化/流失分析。",
    tags=["漏斗图", "转化率", "流程"], demo_note="ecommerce",
    slots=[Slot("var", "阶段/分类变量", types=(CAT, ORDINAL))],
    options=[],
))


# ================================================================ 小提琴图
def _run_plot_violin(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    ycol, xcol = sl["y"], sl["x"]
    y = as_numeric(ds, ycol)
    if xcol:
        frame = pd.DataFrame({"x": get_series(ds, xcol).astype(str), "y": y}).dropna()
        groups = {k: g["y"].tolist() for k, g in frame.groupby("x", observed=True)}
        if len(groups) > 20:
            raise ModelError("分组数超过 20,小提琴图过于拥挤,请先合并水平或换变量")
        res.add_table("分组描述统计", ["分组", "n", "均值", "标准差", "中位数"],
                      [[k, int(len(v)), fnum(float(np.mean(v)), 3), fnum(float(np.std(v, ddof=1)), 3),
                        fnum(float(np.median(v)), 3)] for k, v in groups.items()])
    else:
        groups = {"全部样本": y.dropna().tolist()}
    res.add_chart("小提琴图", ch.violin_chart(f"「{ycol}」分布{'(按' + xcol + '分组)' if xcol else ''}",
                 groups, y_name=ycol, show_points=bool(op.get("points"))))
    res.add_interp(
        f"小提琴图展示「{ycol}」的核密度分布形态"
        + (f"(按「{xcol}」分组):外廓宽度对应密度,内部箱线为中位数与四分位距,散点为原始观测。" if xcol
           else ":外廓宽度对应密度,内部箱线为中位数与四分位距,散点为原始观测。")
        + "多峰或偏态形态提示该变量不适合直接使用均值类参数方法。")
    return res


register_model(ModelSpec(
    id="plot_violin", name="小提琴图", category="统计绘图",
    runner=_run_plot_violin,
    desc="核密度分布 + 箱线 + 抖动散点三合一，直观比较各组分布形态、偏态与多峰。",
    tags=["小提琴图", "分布", "KDE"], demo_note="panel",
    slots=[Slot("y", "数值变量（定量）", types=(QUANT,)),
           Slot("x", "分组变量（定类/定级，可选）", types=(CAT, ORDINAL), required=False)],
    options=[Option("points", "显示原始散点", "switch", default=True)],
))
