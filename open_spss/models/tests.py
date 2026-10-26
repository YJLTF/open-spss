"""差异检验:参数检验(T 检验、方差分析、协方差分析)、卡方族、非参数检验。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..core import charts as ch
from ..core.datasets import Dataset, as_numeric, get_series
from ..core.registry import CAT, ORDINAL, QUANT, ModelSpec, Option, Slot, register_model
from ..core.results import ModelError, Result, fnum, fp, judge_p

ALPHA_OPT = Option("alpha", "显著性水平 α", "select", default=0.05,
                   choices=[{"value": 0.1, "label": "0.1"}, {"value": 0.05, "label": "0.05"}, {"value": 0.01, "label": "0.01"}])


# ================================================================ T 检验
def _run_ttest_1sample(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    mu = float(op["popmean"])
    rows = []
    for c in sl["vars"]:
        s = as_numeric(ds, c).dropna()
        if len(s) < 2:
            raise ModelError(f"列「{c}」有效样本不足 2")
        t, p = sps.ttest_1samp(s, mu)
        se = s.std(ddof=1) / np.sqrt(len(s))
        dfree = len(s) - 1
        d = (s.mean() - mu) / (s.std(ddof=1) if s.std(ddof=1) else 1)
        crit = sps.t.ppf(1 - op["alpha"] / 2, dfree)
        ci = (s.mean() - mu - crit * se, s.mean() - mu + crit * se)
        rows.append([c, len(s), fnum(s.mean()), fnum(s.std(ddof=1)), fnum(mu),
                     fnum(t), dfree, fp(p), fnum(d, 3), f"{fnum(ci[0])} ~ {fnum(ci[1])}", judge_p(p, op["alpha"])])
        res.add_chart(f"「{c}」分布与检验值", ch.hist_chart(f"「{c}」分布(虚线=检验值 μ0={mu})", s.values, x_name=c))
    res.add_table("单样本 T 检验", ["变量", "N", "均值", "标准差", "检验值μ0", "t", "df", "p", "Cohen's d", "差值95%CI", "显著?"],
                  rows, note="H0: 总体均值 = μ0")
    r0 = rows[0]
    res.add_interp(
        f"以「{r0[0]}」为例:样本均值 {r0[2]} vs 检验值 {r0[4]},t({r0[6]})={r0[5]},p={r0[7]}"
        f"{'<α,拒绝原假设,该变量均值与检验值存在显著差异' if r0[10] == '是' else '≥α,不能拒绝原假设,尚无证据表明均值与检验值有差异'}。"
        f"效应量 Cohen's d={r0[8]}({'小' if abs(float(r0[8]))<0.5 else '中' if abs(float(r0[8]))<0.8 else '大'})。")
    return res


register_model(ModelSpec(
    id="ttest_1sample", name="单样本 T 检验", category="差异检验",
    runner=_run_ttest_1sample,
    desc="检验样本均值与指定总体均值 μ0 是否存在显著差异。",
    tags=["T检验", "均值"], demo_note="survey",
    slots=[Slot("vars", "分析变量(定量)", types=(QUANT,), multiple=True)],
    options=[Option("popmean", "检验值 μ0", "number", default=0.0, step=0.1), ALPHA_OPT],
))


def _run_ttest_indep(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    gcol = sl["group"]
    data = pd.concat([as_numeric(ds, sl["y"]).rename("y"), get_series(ds, gcol).astype(str).rename("g")], axis=1).dropna()
    levels = data["g"].unique().tolist()
    if len(levels) != 2:
        raise ModelError(f"分组变量需恰好 2 个水平(当前 {len(levels)}:{', '.join(map(str, levels[:6]))})")
    a = data.loc[data.g == levels[0], "y"]
    b = data.loc[data.g == levels[1], "y"]
    if min(len(a), len(b)) < 2:
        raise ModelError("每组至少需要 2 个有效样本")
    res.add_table("分组描述统计", ["组别", "N", "均值", "标准差", "标准误"],
                  [[str(levels[0]), len(a), fnum(a.mean()), fnum(a.std(ddof=1)), fnum(a.std(ddof=1) / np.sqrt(len(a)))],
                   [str(levels[1]), len(b), fnum(b.mean()), fnum(b.std(ddof=1)), fnum(b.std(ddof=1) / np.sqrt(len(b)))]])
    lev_f, lev_p = sps.levene(a, b, center="mean")
    equal = bool(lev_p >= op["alpha"])
    t, p = sps.ttest_ind(a, b, equal_var=equal)
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    if equal:
        dfree = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / dfree
        se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
    else:
        se = np.sqrt(v1 / n1 + v2 / n2)
        dfree = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    diff = a.mean() - b.mean()
    crit = sps.t.ppf(1 - op["alpha"] / 2, dfree)
    s_pool = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    cohen = diff / (s_pool if s_pool else 1)
    res.add_table("Levene 方差齐性检验", ["F", "p", "结论"],
                  [[fnum(lev_f, 4), fp(lev_p), "方差齐(合并方差)" if equal else "方差不齐(Welch 校正)"]])
    res.add_table("独立样本 T 检验", ["t", "df", "p", "均值差", "差值95%CI", "Cohen's d", "显著?"],
                  [[fnum(t), fnum(dfree, 2), fp(p), fnum(diff),
                    f"{fnum(diff - crit * se)} ~ {fnum(diff + crit * se)}", fnum(cohen, 3),
                    judge_p(p, op["alpha"])]],
                  note="H0: 两组总体均值相等")
    res.add_chart("分组箱线图", ch.box_chart(f"「{sl['y']}」按「{gcol}」分组箱线图",
                 {str(l): data.loc[data.g == l, "y"].tolist() for l in levels}, y_name=sl["y"]))
    res.add_interp(
        f"组「{levels[0]}」均值 {fnum(a.mean())},组「{levels[1]}」均值 {fnum(b.mean())}。"
        f"Levene 检验 p={fp(lev_p)}({'方差齐' if equal else '方差不齐'}),采用{'合并方差' if equal else 'Welch 校正'} t 检验:"
        f"t={fnum(t)},p={fp(p)}{',' if p < op['alpha'] else ''}"
        f"{'两组均值差异显著' if p < op['alpha'] else ',两组均值差异不显著'}"
        f"(Cohen's d={fnum(cohen, 3)})。")
    return res


register_model(ModelSpec(
    id="ttest_indep", name="独立样本 T 检验", category="差异检验",
    runner=_run_ttest_indep,
    desc="比较两组独立样本的均值差异,自动进行 Levene 方差齐性判断与 Welch 校正。",
    tags=["T检验", "两组比较"], demo_note="survey",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,)),
           Slot("group", "分组变量(二分类)", types=(CAT, ORDINAL))],
    options=[ALPHA_OPT],
))


def _run_ttest_paired(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    flat = [c for c in sl["pairs"] if c]
    pairs = [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]
    if not pairs:
        raise ModelError("请至少添加一对配对变量(按顺序两两配对)")
    rows = []
    for col_a, col_b in pairs:
        x = as_numeric(ds, col_a)
        y = as_numeric(ds, col_b)
        d = (x - y).dropna()
        if len(d) < 2:
            raise ModelError(f"配对「{col_a} - {col_b}」有效样本不足")
        t, p = sps.ttest_rel(x.dropna(), y.dropna()) if len(x.dropna()) == len(y.dropna()) else sps.ttest_1samp(d, 0)
        se = d.std(ddof=1) / np.sqrt(len(d))
        dfree = len(d) - 1
        crit = sps.t.ppf(1 - op["alpha"] / 2, dfree)
        cohen = d.mean() / (d.std(ddof=1) if d.std(ddof=1) else 1)
        rows.append([f"{col_a} - {col_b}", len(d), fnum(x.mean() if len(x.dropna()) == len(y.dropna()) else (x - y).mean()),
                     fnum(y.mean() if len(x.dropna()) == len(y.dropna()) else np.nan),
                     fnum(d.mean()), fnum(d.std(ddof=1)), fnum(t), dfree, fp(p),
                     fnum(cohen, 3), f"{fnum(d.mean() - crit * se)} ~ {fnum(d.mean() + crit * se)}",
                     judge_p(p, op["alpha"])])
    res.add_table("配对样本 T 检验", ["配对", "N", "前项均值", "后项均值", "差值均值", "差值标准差",
                                    "t", "df", "p", "Cohen's d", "差值95%CI", "显著?"], rows,
                  note="H0: 两配对总体均值之差 = 0")
    r0 = rows[0]
    res.add_interp(
        f"配对「{r0[0]}」:差值均值 {r0[4]},t({r0[7]})={r0[6]},p={r0[8]},"
        f"{'前后差异显著' if r0[11] == '是' else '前后差异不显著'}(Cohen's d={r0[9]})。")
    return res


register_model(ModelSpec(
    id="ttest_paired", name="配对样本 T 检验", category="差异检验",
    runner=_run_ttest_paired,
    desc="检验同一对象的两次测量(前测-后测)均值差异,支持多对配对。",
    tags=["T检验", "配对", "前测后测"], demo_note="survey",
    slots=[Slot("pairs", "配对变量(添加若干对)", types=(QUANT,), multiple=True, max_count=6,
                tip="列表中的变量按顺序两两配对:第1个与第2个、第3个与第4个…")],
    options=[ALPHA_OPT],
))


# ================================================================ 方差分析
def _posthoc(groups: dict[str, np.ndarray], method: str, alpha: float) -> list[list]:
    """事后多重比较:LSD / Tukey HSD / Scheffe"""
    names = list(groups.keys())
    k = len(names)
    N = sum(len(v) for v in groups.values())
    rows = []
    if method == "tukey":
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        vals, labs = [], []
        for nm, v in groups.items():
            vals.extend(v)
            labs.extend([nm] * len(v))
        tk = pairwise_tukeyhsd(np.asarray(vals, dtype=float), np.asarray(labs), alpha=alpha)
        for row in tk.summary().data[1:]:
            g1, g2, mdiff, padj, low, high, rej = row
            rows.append([str(g1), str(g2), fnum(mdiff), f"{fnum(low)} ~ {fnum(high)}",
                         fp(float(padj)), "是" if str(rej) == "True" else "否"])
        return rows
    allv = np.concatenate(list(groups.values()))
    mse = allv.var(ddof=1)  # 组内均方近似(各组方差加权)
    ms_list = [v.var(ddof=1) * (len(v) - 1) for v in groups.values()]
    dfw = N - k
    mse = sum(ms_list) / dfw
    fcrit = sps.f.ppf(1 - alpha, k - 1, dfw)
    tcrit = sps.t.ppf(1 - alpha / 2, dfw)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = groups[names[i]], groups[names[j]]
            ni, nj = len(a), len(b)
            diff = a.mean() - b.mean()
            se = np.sqrt(mse * (1 / ni + 1 / nj))
            tval = diff / se if se else np.nan
            p = 2 * sps.t.sf(abs(tval), dfw) if se else np.nan
            if method == "scheffe":
                psche = (k - 1) * sps.f.sf((tval ** 2) / (k - 1), k - 1, dfw)
                rows.append([names[i], names[j], fnum(diff), fnum(tval), fp(psche), judge_p(psche, alpha)])
            else:  # LSD
                rows.append([names[i], names[j], fnum(diff), fnum(tval), fp(p), judge_p(p, alpha)])
    return rows


def _run_anova_1way(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    ycol, gcol = sl["y"], sl["group"]
    data = pd.concat([as_numeric(ds, ycol).rename("y"), get_series(ds, gcol).astype(str).rename("g")], axis=1).dropna()
    groups: dict[str, np.ndarray] = {}
    for lev, sub in data.groupby("g", observed=True):
        if len(sub) >= 2:
            groups[str(lev)] = sub["y"].values
    if len(groups) < 2:
        raise ModelError("分组后每组至少需要 2 个有效样本")
    stat_rows = [[k, len(v), fnum(v.mean()), fnum(v.std(ddof=1)), fnum(v.std(ddof=1) / np.sqrt(len(v)))]
                 for k, v in groups.items()]
    res.add_table("分组描述统计", ["组别", "N", "均值", "标准差", "标准误"], stat_rows)
    F, p = sps.f_oneway(*groups.values())
    grand = data["y"].mean()
    ssb = sum(len(v) * (v.mean() - grand) ** 2 for v in groups.values())
    ssw = sum(((v - v.mean()) ** 2).sum() for v in groups.values())
    k, N = len(groups), len(data)
    eta2 = ssb / (ssb + ssw) if (ssb + ssw) else np.nan
    lev_f, lev_p = sps.levene(*groups.values(), center="mean")
    res.add_table("Levene 方差齐性检验", ["F", "p", "结论"],
                  [[fnum(lev_f, 4), fp(lev_p), "方差齐" if lev_p >= op["alpha"] else "方差不齐(结论请谨慎/改用Welch)"]])
    res.add_table("单因素方差分析(ANOVA)", ["差异源", "平方和SS", "自由度df", "均方MS", "F", "p"],
                  [["组间(因子)", fnum(ssb), k - 1, fnum(ssb / (k - 1)), fnum(F), fp(p)],
                   ["组内(误差)", fnum(ssw), N - k, fnum(ssw / (N - k)), "", ""],
                   ["总计", fnum(ssb + ssw), N - 1, "", "", ""]],
                  note=f"效应量 η²={fnum(eta2, 3)}")
    if op["posthoc"] != "none" and len(groups) > 2:
        ph = _posthoc(groups, op["posthoc"], op["alpha"])
        mname = {"lsd": "LSD", "tukey": "Tukey HSD", "scheffe": "Scheffe"}[op["posthoc"]]
        res.add_table(f"事后多重比较({mname})", ["组1", "组2", "均值差", "t" if op["posthoc"] != "tukey" else "p-adj区间", "p", "显著?"], ph,
                      note=f"H0: 两两均值相等;α={op['alpha']}")
    res.add_chart("分组箱线图", ch.box_chart(f"「{ycol}」按「{gcol}」分组箱线图", groups, y_name=ycol))
    gsig = judge_p(p, op["alpha"]) == "是"
    txt = (f"F({k - 1},{N - k})={fnum(F)},p={fp(p)}{'<α' if gsig else '≥α'},"
           + ("不同组的均值存在显著差异" if gsig else "尚不能认为各组均值存在差异")
           + f",效应量 η²={fnum(eta2, 3)}。")
    if gsig and op["posthoc"] != "none" and len(groups) > 2:
        txt += "事后比较显示差异主要来自上表中标为「显著」的组对。"
    res.add_interp(txt)
    return res


register_model(ModelSpec(
    id="anova_1way", name="单因素方差分析", category="差异检验",
    runner=_run_anova_1way,
    desc="比较三组及以上独立样本均值差异,支持 LSD / Tukey / Scheffe 事后多重比较。",
    tags=["ANOVA", "F检验", "事后检验"], demo_note="survey",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,)),
           Slot("group", "分组变量(定类)", types=(CAT, ORDINAL))],
    options=[
        Option("posthoc", "事后多重比较", "select", default="tukey",
               choices=[{"value": "none", "label": "不进行"}, {"value": "lsd", "label": "LSD(最小显著差)"},
                        {"value": "tukey", "label": "Tukey HSD"}, {"value": "scheffe", "label": "Scheffe"}]),
        ALPHA_OPT,
    ],
))


def _run_anova_multi(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    ycol = sl["y"]
    factors = sl["factors"]
    if len(factors) < 2:
        raise ModelError("多因素方差分析至少需要 2 个因子变量")
    data = pd.concat([as_numeric(ds, ycol).rename("y")] +
                     [get_series(ds, f).astype(str).rename(f) for f in factors], axis=1).dropna()
    safe = [f"f{i}" for i in range(len(factors))]
    formula = "y ~ C(" + ") * C(".join(safe) + ")"
    fit = ols(formula, data=data.assign(**{safe[i]: data[factors[i]] for i in range(len(factors))})).fit()
    table = sm.stats.anova_lm(fit, typ=2)
    rows = []
    rename = {f"C({s})": factors[i] for i, s in enumerate(safe)}
    rename["C(" + ") * C(".join(safe) + ")"] = f"{' × '.join(factors)}(交互)"
    for idx, row in table.iterrows():
        label = rename.get(str(idx), str(idx))
        if label == "Residual":
            label = "残差"
        rows.append([label, fnum(row["sum_sq"]), int(row["df"]), fnum(row["sum_sq"] / row["df"]),
                     fnum(row["F"]) if not np.isnan(row["F"]) else "", fp(row["PR(>F)"])])
    res.add_table(f"多因素方差分析(II 型)", ["差异源", "平方和SS", "自由度df", "均方MS", "F", "p"], rows,
                  note=f"R²={fnum(fit.rsquared,3)},调整R²={fnum(fit.rsquared_adj,3)}")
    res.add_interp(
        f"模型整体 R²={fnum(fit.rsquared, 3)}。" +
        (f"因子「{factors[0]}」(p={fp(table.iloc[0]['PR(>F)'])})与「{factors[1]}」(p={fp(table.iloc[1]['PR(>F)'])})"
         if len(table) >= 2 else "") +
        ("对因变量的主效应显著;请关注交互项是否显著——交互显著时,一个因子的作用随另一因子水平而变化,应先解读交互效应。"
         if "交互" in " ".join(r[0] for r in rows) and rows and any(r[5] and "0." in r[5] for r in rows) else "各因子效应解读见上表。"))
    # 交互图
    v0 = sl["vars"][0] if sl.get("vars") else None
    if v0:
        f2 = factors[1]
        piv = data.groupby([factors[0], f2], observed=True)["y"].mean().reset_index()
        series: dict[str, list] = {}
        cats = sorted(piv[factors[0]].astype(str).unique())
        for lev2, sub in piv.groupby(f2, observed=True):
            m = dict(zip(sub[factors[0]].astype(str), sub["y"]))
            series[str(lev2)] = [round(float(m[c]), 4) if c in m else None for c in cats]
        res.add_chart("交互效应图", ch.line_chart("交互效应图(均值)", cats, series,
                     x_name=factors[0], y_name=f"{ycol}均值", smooth=False))
    return res


register_model(ModelSpec(
    id="anova_multi", name="多因素方差分析", category="差异检验",
    runner=_run_anova_multi,
    desc="分析两个及以上分类因子对定量因变量的主效应与交互效应(II 型平方和)。",
    tags=["ANOVA", "交互效应"], demo_note="survey",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,)),
           Slot("factors", "因子变量(定类,2~4个)", types=(CAT, ORDINAL), multiple=True, max_count=4),
           Slot("vars", "交互图第二因子(选填)", types=(CAT, ORDINAL), required=False)],
    options=[ALPHA_OPT],
))


def _run_ancova(ds: Dataset, sl: dict, op: dict) -> Result:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    res = Result()
    ycol, gcol = sl["y"], sl["group"]
    covs = sl["covs"]
    data = pd.concat([as_numeric(ds, ycol).rename("y"),
                      get_series(ds, gcol).astype(str).rename("grp")] +
                     [as_numeric(ds, c).rename(f"x{i}") for i, c in enumerate(covs)], axis=1).dropna()
    formula = "y ~ C(grp)" + "".join(f" + x{i}" for i in range(len(covs)))
    fit = ols(formula, data=data).fit()
    table = sm.stats.anova_lm(fit, typ=2)
    rows = []
    for idx, row in table.iterrows():
        label = {"C(grp)": f"分组({gcol})", "Residual": "残差"}.get(str(idx), str(idx))
        rows.append([label, fnum(row["sum_sq"]), int(row["df"]), fnum(row["sum_sq"] / row["df"]),
                     fnum(row["F"]) if not np.isnan(row["F"]) else "", fp(row["PR(>F)"])])
    res.add_table("协方差分析(ANCOVA)", ["差异源", "平方和SS", "自由度df", "均方MS", "F", "p"], rows,
                  note=f"R²={fnum(fit.rsquared,3)};协变量:{', '.join(covs)}")
    # 调整后均值(最小二乘均值):协变量取总均值
    xm = {f"x{i}": data[f"x{i}"].mean() for i in range(len(covs))}
    adj_rows = []
    for lev in data["grp"].unique():
        probe = pd.DataFrame([{**xm, "grp": lev, "x0": xm["x0"]}])
        for i in range(1, len(covs)):
            probe[f"x{i}"] = xm[f"x{i}"]
        pred = fit.get_prediction(probe).summary_frame(alpha=op["alpha"])
        adj_rows.append([str(lev), fnum(pred["mean"].iloc[0]), f"{fnum(pred['mean_ci_lower'].iloc[0])} ~ {fnum(pred['mean_ci_upper'].iloc[0])}"])
    res.add_table("调整后均值(协变量固定于总均值)", ["组别", "调整均值", f"{int((1-op['alpha'])*100)}%CI"], adj_rows)
    gp = table.loc[table.index.astype(str) == "C(grp)"]
    p_grp = float(gp["PR(>F)"].iloc[0]) if len(gp) else np.nan
    res.add_interp(
        f"控制协变量 {', '.join(covs)} 后,分组因素「{gcol}」对「{ycol}」的效应 p={fp(p_grp)},"
        f"{'剔除协变量影响后组间差异显著,调整后均值见上表' if p_grp < op['alpha'] else '尚不能认为剔除协变量影响后组间存在差异'}。")
    res.add_chart("原始均值与调整均值对比",
                  ch.bar_chart("原始均值 vs 调整均值",
                               [str(l) for l in data["grp"].unique()],
                               {"原始均值": [round(float(data.loc[data.grp == l, "y"].mean()), 4) for l in data["grp"].unique()],
                                "调整均值": [float(r[1]) for r in adj_rows]}))
    return res


register_model(ModelSpec(
    id="ancova", name="协方差分析", category="差异检验",
    runner=_run_ancova,
    desc="在控制连续协变量(如基线)的影响后,检验分组变量对因变量的效应,输出调整后均值。",
    tags=["ANCOVA", "协变量", "调整均值"], demo_note="clinical",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,)),
           Slot("group", "分组变量(定类)", types=(CAT, ORDINAL)),
           Slot("covs", "协变量(定量,1~4个)", types=(QUANT,), multiple=True, max_count=4)],
    options=[ALPHA_OPT],
))


# ================================================================ 卡方族
def _run_chisq(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    a, b = sl["x"], sl["y"]
    if a == b:
        raise ModelError("两个变量不能相同")
    sub = ds.df[[a, b]].dropna()
    w = None
    wname = None
    if sl.get("freq"):
        wname = sl["freq"]
    elif ds.weight_col:                       # 加权个案:数据集权重充当频数
        wname = ds.weight_col
    if wname:
        w = pd.to_numeric(ds.df[wname], errors="coerce")
        sub = ds.df[[a, b, wname]].dropna()
        w = sub[wname]
    if w is not None and len(w) > 0:
        ct = pd.pivot_table(sub, index=a, columns=b, values=wname, aggfunc="sum").fillna(0)
    else:
        ct = pd.crosstab(sub[a].astype(str), sub[b].astype(str))
    ct = ct.loc[(ct.sum(axis=1) > 0), ct.columns[(ct.sum(axis=0) > 0)]]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise ModelError("交叉表维度不足 2×2,请检查变量取值")
    chi2, p, dof, exp = sps.chi2_contingency(ct.values, correction=op["correction"] and ct.shape == (2, 2))
    n = float(ct.values.sum())
    cramer = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    rows = []
    for i, lev in enumerate(ct.index):
        row_total = ct.iloc[i].sum()
        rows.append([str(lev)] + [f"{int(ct.iloc[i, j])} ({ct.iloc[i, j] / row_total * 100:.1f}%)" for j in range(ct.shape[1])]
                    + [int(row_total)])
    rows.append(["列合计"] + [int(ct.iloc[:, j].sum()) for j in range(ct.shape[1])] + [int(n)])
    wt_tag = f"(按「{wname}」加权)" if wname else "(行百分比)"
    res.add_table(f"交叉表:「{a}」×「{b}」{wt_tag}", [f"{a}\\{b}"] + [str(c) for c in ct.columns] + ["行合计"], rows)
    low_exp = float((exp < 5).mean())
    note = (";" + (f"E<5 的单元格占 {low_exp * 100:.0f}%," + ("满足卡方检验要求(<20%)" if low_exp < 0.2 else "超过 20%,结论仅供参考(建议合并水平或 Fisher)")) if ct.shape == (2, 2) else "")
    res.add_table("卡方检验结果", ["卡方值χ²", "自由度df", "p 值", "Cramér's V", "显著?"],
                  [[fnum(chi2, 4), dof, fp(p), fnum(cramer, 3), judge_p(p, op["alpha"])]],
                  note="H0: 两变量相互独立" + note)
    if ct.shape == (2, 2) and op["fisher"]:
        odds, pfisher = sps.fisher_exact(ct.values)
        res.add_table("Fisher 精确检验(2×2)", ["精确 p 值", "优势比OR"], [[fp(pfisher), fnum(odds, 4)]])
    cats_x = [str(c) for c in ct.index]
    series = {str(c): [float(ct.loc[r, c]) for r in ct.index] for c in ct.columns}
    res.add_chart("交叉频数堆积图", ch.bar_chart(f"「{b}」在各「{a}」水平的频数", cats_x, series, x_name=a, y_name="频数", stack=True))
    sig = p < op["alpha"]
    res.add_interp(
        f"卡方检验显示 χ²({dof})={fnum(chi2, 3)},p={fp(p)}{',' if sig else ''}"
        + (f"「{a}」与「{b}」不独立,存在显著关联(关联强度 Cramér's V={fnum(cramer, 3)},"
           f"{'强' if cramer >= 0.5 else '中等' if cramer >= 0.3 else '弱'})。"
           if sig else f"尚无证据表明「{a}」与「{b}」存在关联。"))
    return res


register_model(ModelSpec(
    id="chisq", name="卡方检验(交叉分析)", category="差异检验",
    runner=_run_chisq,
    desc="检验两个定类变量是否独立,输出交叉表、卡方统计量、Cramér's V 与 2×2 Fisher 精确检验。",
    tags=["卡方", "交叉表", "独立性"], demo_note="survey",
    slots=[Slot("x", "自变量(定类)", types=(CAT, ORDINAL)),
           Slot("y", "因变量(定类)", types=(CAT, ORDINAL)),
           Slot("freq", "频数列(已汇总数据,选填)", types=(QUANT,), required=False)],
    options=[
        Option("correction", "2×2 连续性校正(Yates)", "switch", default=True),
        Option("fisher", "附 2×2 Fisher 精确检验", "switch", default=True),
        ALPHA_OPT,
    ],
))


def _run_mcnemar(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    a, b = sl["x"], sl["y"]
    sub = ds.df[[a, b]].dropna().astype(str)
    levels = sorted(set(sub[a]) | set(sub[b]))
    if len(levels) != 2:
        raise ModelError(f"配对卡方要求两变量均为二分类(当前取值:{', '.join(levels[:6])})")
    n11 = int(((sub[a] == levels[0]) & (sub[b] == levels[0])).sum())
    n12 = int(((sub[a] == levels[0]) & (sub[b] == levels[1])).sum())
    n21 = int(((sub[a] == levels[1]) & (sub[b] == levels[0])).sum())
    n22 = int(((sub[a] == levels[1]) & (sub[b] == levels[1])).sum())
    res.add_table("配对四格表", ["", f"前:{levels[0]}", f"前:{levels[1]}", "合计"],
                  [[f"后:{levels[0]}", n11, n21, n11 + n21],
                   [f"后:{levels[1]}", n12, n22, n12 + n22],
                   ["合计", n11 + n12, n21 + n22, n11 + n12 + n21 + n22]])
    bc, cb = n12, n21
    n_discord = bc + cb
    if n_discord == 0:
        raise ModelError("不一致对数为 0,两分类结果完全一致,无需检验")
    if op["exact"] or n_discord < 25:
        p = min(1.0, 2 * sps.binom.cdf(min(bc, cb), n_discord, 0.5))
        stat = fnum((bc - cb) ** 2 / n_discord, 4)
        method = "精确二项检验"
    else:
        stat = fnum((abs(bc - cb) - 1) ** 2 / n_discord, 4) if op["correction"] else fnum((bc - cb) ** 2 / n_discord, 4)
        p = sps.chi2.sf((abs(bc - cb) - (1 if op["correction"] else 0)) ** 2 / n_discord, 1)
        method = "连续性校正" if op["correction"] else "未校正"
    kappa = None
    po = (n11 + n22) / (n11 + n12 + n21 + n22)
    pe = ((n11 + n12) * (n11 + n21) + (n12 + n22) * (n21 + n22)) / (n11 + n12 + n21 + n22) ** 2
    kappa = (po - pe) / (1 - pe) if pe != 1 else np.nan
    res.add_table(f"McNemar 配对卡方检验({method})", ["不一致对(b,c)", "统计量", "p 值", "Kappa一致性系数", "显著?"],
                  [[f"b={bc}, c={cb}", stat, fp(p), fnum(kappa, 3), judge_p(p, op["alpha"])]],
                  note="H0: 两配对分类结果分布一致")
    res.add_interp(
        f"两分类不一致的对数为 {n_discord}(b={bc}, c={cb}),p={fp(p)},"
        f"{'前后分类结果的变化具有统计学意义' if p < op['alpha'] else '尚不能认为前后分类结果有变化'};"
        f"Kappa={fnum(kappa, 3)},一致性{'较差' if kappa < 0.4 else '中等' if kappa < 0.6 else '较强' if kappa < 0.8 else '极强'}。")
    return res


register_model(ModelSpec(
    id="mcnemar", name="配对卡方检验(McNemar)", category="差异检验",
    runner=_run_mcnemar,
    desc="检验同一批对象前后两次二分类结果(如转阴/阳性)的变化是否显著。",
    tags=["McNemar", "配对", "四格表"], demo_note="clinical",
    slots=[Slot("x", "前测(二分类)", types=(CAT, ORDINAL)),
           Slot("y", "后测(二分类)", types=(CAT, ORDINAL))],
    options=[Option("correction", "连续性校正", "switch", default=True),
             Option("exact", "使用精确二项检验", "switch", default=False),
             ALPHA_OPT],
))


# ================================================================ 非参数检验
def _run_wilcoxon_1(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    rows = []
    for c in sl["vars"]:
        s = as_numeric(ds, c).dropna() - float(op["med"])
        d = s[s != 0]
        if len(d) < 6:
            raise ModelError(f"列「{c}」有效非零差值不足 6 个")
        w, p = sps.wilcoxon(d, alternative=op["alternative"], correction=op["correction"])
        n = len(d)
        mu_w = n * (n + 1) / 4
        sd_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (w - mu_w) / sd_w if sd_w else np.nan
        rows.append([c, len(s), int(len(s[s > op["med"]])), int(len(s[s < op["med"]])),
                     fnum(w), fnum(z, 3), fp(p), judge_p(p, op["alpha"])])
    res.add_table("单样本 Wilcoxon 符号秩检验", ["变量", "N", "大于μ0数", "小于μ0数", "W统计量", "Z", "p", "显著?"],
                  rows, note=f"H0: 中位数 = {op['med']}")
    r0 = rows[0]
    res.add_interp(f"以中位数 μ0={op['med']} 检验:W={r0[4]},p={r0[6]},"
                   f"{'中位数与检验值差异显著' if r0[7] == '是' else '尚不能认为中位数与检验值有差异'}。")
    return res


register_model(ModelSpec(
    id="wilcoxon_1", name="单样本 Wilcoxon 检验", category="差异检验",
    runner=_run_wilcoxon_1,
    desc="非参数方法,检验样本中位数与给定值是否有差异,适用于非正态数据。",
    tags=["Wilcoxon", "符号秩", "非参数"], demo_note="survey",
    slots=[Slot("vars", "分析变量(定量)", types=(QUANT,), multiple=True)],
    options=[
        Option("med", "检验值(中位数)", "number", default=0.0, step=0.1),
        Option("alternative", "备择假设", "select", default="two-sided",
               choices=[{"value": "two-sided", "label": "双侧 ≠"}, {"value": "greater", "label": "单侧 >"}, {"value": "less", "label": "单侧 <"}]),
        Option("correction", "连续性校正", "switch", default=True),
        ALPHA_OPT,
    ],
))


def _run_mannwhitney(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    ycol, gcol = sl["y"], sl["group"]
    data = pd.concat([as_numeric(ds, ycol).rename("y"), get_series(ds, gcol).astype(str).rename("g")], axis=1).dropna()
    levels = data["g"].unique().tolist()
    if len(levels) != 2:
        raise ModelError(f"需要恰好 2 个组(当前 {len(levels)} 个)")
    a = data.loc[data.g == levels[0], "y"].values
    b = data.loc[data.g == levels[1], "y"].values
    if min(len(a), len(b)) < 2:
        raise ModelError("每组至少需要 2 个有效样本")
    u, p = sps.mannwhitneyu(a, b, alternative=op["alternative"])
    # 秩与效应量
    allv = np.concatenate([a, b])
    ranks = sps.rankdata(allv)
    ra, rb = ranks[:len(a)], ranks[len(a):]
    n1, n2 = len(a), len(b)
    mu_u = n1 * n2 / 2
    # 带结校正的标准差
    _, counts = np.unique(allv, return_counts=True)
    tie = (counts ** 3 - counts).sum()
    sd_u = np.sqrt(n1 * n2 / 12 * ((n1 + n2 + 1) - tie / (n1 + n2) / (n1 + n2 - 1)))
    z = (u - mu_u) / sd_u if sd_u else np.nan
    rbiserial = 1 - 2 * u / (n1 * n2)
    res.add_table("分组秩统计", ["组别", "N", "平均秩", "秩和"],
                  [[str(levels[0]), n1, fnum(ra.mean(), 2), fnum(ra.sum(), 1)],
                   [str(levels[1]), n2, fnum(rb.mean(), 2), fnum(rb.sum(), 1)]])
    res.add_table("Mann-Whitney U 检验", ["U", "Z", "p", "秩双列相关 r", "显著?"],
                  [[fnum(u), fnum(z, 3), fp(p), fnum(rbiserial, 3), judge_p(p, op["alpha"])]],
                  note="H0: 两组分布位置相同(非正态两独立样本检验)")
    res.add_chart("分组箱线图", ch.box_chart(f"「{ycol}」按「{gcol}」分组箱线图",
                 {str(l): data.loc[data.g == l, "y"].tolist() for l in levels}, y_name=ycol))
    res.add_interp(
        f"组「{levels[0]}」平均秩 {fnum(ra.mean(), 1)},组「{levels[1]}」平均秩 {fnum(rb.mean(), 1)};"
        f"U={fnum(u)},p={fp(p)},{'两组水平差异显著' if p < op['alpha'] else '尚不能认为两组水平有差异'}(效应量 r={fnum(rbiserial, 3)})。")
    return res


register_model(ModelSpec(
    id="mannwhitney", name="Mann-Whitney U 检验", category="差异检验",
    runner=_run_mannwhitney,
    desc="独立样本 T 检验的非参数替代,基于秩比较两组分布位置差异。",
    tags=["Mann-Whitney", "秩和检验", "非参数"], demo_note="survey",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,)),
           Slot("group", "分组变量(二分类)", types=(CAT, ORDINAL))],
    options=[Option("alternative", "备择假设", "select", default="two-sided",
                    choices=[{"value": "two-sided", "label": "双侧 ≠"}, {"value": "greater", "label": "单侧 >"}, {"value": "less", "label": "单侧 <"}]),
             ALPHA_OPT],
))


def _run_kruskal(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    ycol, gcol = sl["y"], sl["group"]
    data = pd.concat([as_numeric(ds, ycol).rename("y"), get_series(ds, gcol).astype(str).rename("g")], axis=1).dropna()
    groups: dict[str, np.ndarray] = {}
    for lev, sub in data.groupby("g", observed=True):
        if len(sub) >= 2:
            groups[str(lev)] = sub["y"].values
    if len(groups) < 3:
        raise ModelError("Kruskal-Wallis 检验至少需要 3 组(两组请用 Mann-Whitney U)")
    h, p = sps.kruskal(*groups.values())
    N = sum(len(v) for v in groups.values())
    k = len(groups)
    eta2h = (h - k + 1) / (N - k) if N > k else np.nan
    res.add_table("Kruskal-Wallis 检验", ["组数k", "总样本N", "H统计量", "df", "p", "效应量 ε²(H校正)", "显著?"],
                  [[k, N, fnum(h, 4), k - 1, fp(p), fnum(eta2h, 3), judge_p(p, op["alpha"])]],
                  note="H0: 各组分布位置相同")
    stat_rows = []
    for kk, v in groups.items():
        ranks = sps.rankdata(np.concatenate(list(groups.values())))
        idx = np.concatenate([[i] * len(v) for i, v in enumerate(groups.values())])
        mean_ranks = {nm: ranks[idx == i].mean() for i, nm in enumerate(groups.keys())}
    for kk, v in groups.items():
        stat_rows.append([kk, len(v), fnum(v.mean()), fnum(np.median(v)), fnum(mean_ranks[kk], 2)])
    res.add_table("分组统计", ["组别", "N", "均值", "中位数", "平均秩"], stat_rows)
    if op["posthoc"]:
        ph_rows = []
        keys = list(groups.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                u, pj = sps.mannwhitneyu(groups[keys[i]], groups[keys[j]], alternative="two-sided")
                ph_rows.append([keys[i], keys[j], fp(pj), judge_p(pj, op["alpha"] / (len(keys) * (len(keys) - 1) / 2))])
        res.add_table("事后两两比较(Mann-Whitney + Bonferroni 校正)", ["组1", "组2", "p", "校正后显著?"], ph_rows)
    res.add_chart("分组箱线图", ch.box_chart(f"「{ycol}」按「{gcol}」分组箱线图", groups, y_name=ycol))
    res.add_interp(
        f"H={fnum(h, 3)},p={fp(p)},{'至少有两组的分布位置存在显著差异,具体见两两比较' if p < op['alpha'] else '尚不能认为各组分布位置有差异'}。")
    return res


register_model(ModelSpec(
    id="kruskal", name="Kruskal-Wallis 检验", category="差异检验",
    runner=_run_kruskal,
    desc="多组独立样本的单因素方差分析非参数替代,附 Bonferroni 校正的两两比较。",
    tags=["Kruskal-Wallis", "非参数", "多组"], demo_note="survey",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,)),
           Slot("group", "分组变量(定类)", types=(CAT, ORDINAL))],
    options=[Option("posthoc", "两两比较", "switch", default=True), ALPHA_OPT],
))


def _run_friedman(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    cols = sl["vars"]
    if len(cols) < 3:
        raise ModelError("Friedman 检验至少需要 3 个配对测量(变量)")
    data = pd.concat([as_numeric(ds, c).rename(c) for c in cols], axis=1).dropna()
    if len(data) < 4:
        raise ModelError("完整配对样本不足 4 行")
    stat, p = sps.friedmanchisquare(*[data[c].values for c in cols])
    k = len(cols)
    n = len(data)
    ranks = data.rank(axis=1)
    mean_ranks = ranks.mean()
    wkk = (12 * n / (k * (k + 1)) * ((mean_ranks - (k + 1) / 2) ** 2).sum()) / (n * (k ** 2 - 1)) if n * (k**2 - 1) else np.nan
    rows = [[c, int(data[c].count()), fnum(data[c].mean()), fnum(data[c].median()), fnum(mean_ranks[c], 3)] for c in cols]
    res.add_table("描述统计与平均秩", ["测量", "N", "均值", "中位数", "平均秩"], rows)
    res.add_table("Friedman 检验", ["样本量N", "测量数k", "χ²统计量", "df", "p", "Kendall's W 一致性", "显著?"],
                  [[n, k, fnum(stat, 4), k - 1, fp(p), fnum(wkk, 3), judge_p(p, op["alpha"])]],
                  note="H0: 各配对测量的分布位置相同")
    if op["posthoc"]:
        ph_rows = []
        m = k * (k - 1) / 2
        for i in range(k):
            for j in range(i + 1, k):
                wv, pj = sps.wilcoxon(data[cols[i]], data[cols[j]])
                ph_rows.append([cols[i], cols[j], fnum((mean_ranks[cols[i]] - mean_ranks[cols[j]]), 3),
                                fp(pj), judge_p(pj, op["alpha"] / m)])
        res.add_table("事后两两比较(Wilcoxon + Bonferroni 校正)", ["测量1", "测量2", "平均秩差", "p", "校正后显著?"], ph_rows)
    res.add_chart("各测量均值对比", ch.bar_chart("各测量均值对比", cols,
                 {"均值": [round(float(data[c].mean()), 4) for c in cols]}, y_name="均值"))
    res.add_interp(
        f"χ²({k - 1})={fnum(stat, 3)},p={fp(p)},"
        f"{'各次测量之间存在显著差异,两两比较见上表' if p < op['alpha'] else '尚不能认为各次测量有差异'};"
        f"Kendall's W={fnum(wkk, 3)} 表示评分者/重复测量的一致性程度。")
    return res


register_model(ModelSpec(
    id="friedman", name="Friedman 检验", category="差异检验",
    runner=_run_friedman,
    desc="多配对样本(重复测量)的非参数检验,如同一批对象在多个时点/方案的取值比较。",
    tags=["Friedman", "重复测量", "非参数"], demo_note="survey",
    slots=[Slot("vars", "配对测量(定量,≥3列)", types=(QUANT,), multiple=True, max_count=10)],
    options=[Option("posthoc", "两两比较", "switch", default=True), ALPHA_OPT],
))


def _run_sign_test(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    a = as_numeric(ds, sl["x"])
    b = as_numeric(ds, sl["y"])
    d = (a - b).dropna()
    d_nz = d[d != 0]
    pos, neg = int((d_nz > 0).sum()), int((d_nz < 0).sum())
    n = pos + neg
    if n == 0:
        raise ModelError("差值全为 0,无法检验")
    p = sps.binomtest(pos, n, 0.5, alternative=op["alternative"]).pvalue
    res.add_table("符号检验", ["正差值数", "负差值数", "零差值数", "有效N", "p 值", "显著?"],
                  [[pos, neg, int((d == 0).sum()), n, fp(p), judge_p(p, op["alpha"])]],
                  note="H0: P(正差) = P(负差) = 0.5")
    res.add_interp(f"正差 {pos} 个、负差 {neg} 个,p={fp(p)},"
                   f"{'两侧差异具有统计学意义' if p < op['alpha'] else '尚不能认为两侧有差异'}。")
    return res


register_model(ModelSpec(
    id="sign_test", name="符号检验", category="差异检验",
    runner=_run_sign_test,
    desc="仅利用差值正负号的简单配对非参数检验,对分布无任何假设。",
    tags=["符号检验", "非参数"], demo_note="survey",
    slots=[Slot("x", "变量1(定量)", types=(QUANT,)), Slot("y", "变量2(定量)", types=(QUANT,))],
    options=[Option("alternative", "备择假设", "select", default="two-sided",
                    choices=[{"value": "two-sided", "label": "双侧 ≠"}, {"value": "greater", "label": "单侧 >"}, {"value": "less", "label": "单侧 <"}]),
             ALPHA_OPT],
))


# ================================================================ 事后多重比较(独立)
def _run_posthoc(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    ycol, gcol = sl["y"], sl["group"]
    data = pd.concat([as_numeric(ds, ycol).rename("y"), get_series(ds, gcol).astype(str).rename("g")], axis=1).dropna()
    groups: dict[str, np.ndarray] = {}
    for lev, sub in data.groupby("g", observed=True):
        if len(sub) >= 2:
            groups[str(lev)] = sub["y"].values
    if len(groups) < 3:
        raise ModelError("事后多重比较至少需要 3 个组(两组请用 T 检验)")
    method = op["method"]
    names = list(groups.keys())
    rows: list[list] = []
    pair_stats = []
    if method in ("lsd", "bonferroni", "holm", "scheffe"):
        allv = np.concatenate(list(groups.values()))
        N, k = len(allv), len(groups)
        ms_list = [v.var(ddof=1) * (len(v) - 1) for v in groups.values()]
        dfw = N - k
        mse = float(sum(ms_list) / dfw)
        for i in range(k):
            for j in range(i + 1, k):
                a, b = groups[names[i]], groups[names[j]]
                ni, nj = len(a), len(b)
                diff = a.mean() - b.mean()
                se = np.sqrt(mse * (1 / ni + 1 / nj))
                t = diff / se if se > 0 else np.inf
                pv = 2 * sps.t.sf(abs(t), dfw)
                pair_stats.append([names[i], names[j], float(diff), float(se), float(t), float(pv), dfw])
        pvals = [r[5] for r in pair_stats]
        if method == "lsd":
            padj = pvals
            note = "LSD 法:不进行多重校正,灵敏但 I 类错误率较高"
        elif method == "scheffe":
            kk = len(groups)
            padj = [float(sps.f.sf(t_ ** 2 / (kk - 1), kk - 1, dfw)) for t_ in [r[4] for r in pair_stats]]
            note = "Scheffe 法:基于 F 分布的两两对比,适用于任意复杂对比,最保守"
        else:
            from statsmodels.stats.multitest import multipletests
            mt = multipletests(pvals, alpha=op["alpha"], method={"bonferroni": "bonferroni", "holm": "holm"}[method])
            padj = list(mt[1])
            note = {"bonferroni": "Bonferroni 校正:按 α/比较次数,保守",
                    "holm": "Holm 逐步法:控制 FWER,较 Bonferroni 更有效"}[method]
        for r, p_adj in zip(pair_stats, padj):
            lo = r[2] - sps.t.ppf(1 - op["alpha"] / 2, r[6]) * r[3]
            hi = r[2] + sps.t.ppf(1 - op["alpha"] / 2, r[6]) * r[3]
            rows.append([r[0], r[1], fnum(r[2]), fnum(r[4]), fp(r[5]), fp(p_adj),
                         f"{fnum(lo, 3)} ~ {fnum(hi, 3)}", "是" if p_adj < op["alpha"] else "否"])
    else:  # tukey
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        vals, labs = [], []
        for nm, v in groups.items():
            vals.extend(v)
            labs.extend([nm] * len(v))
        tk = pairwise_tukeyhsd(np.asarray(vals, dtype=float), np.asarray(labs), alpha=op["alpha"])
        note = "Tukey HSD:基于学生化极差分布,各组样本量相等时功效最佳"
        for row in tk.summary().data[1:]:
            g1, g2, mdiff, padj, low, high, rej = row
            t1, t2 = groups[str(g1)], groups[str(g2)]
            se = np.sqrt(t1.var(ddof=1) / len(t1) + t2.var(ddof=1) / len(t2))
            tt = float(mdiff) / se if se > 0 else np.inf
            rows.append([str(g1), str(g2), fnum(mdiff), fnum(tt), "", fp(float(padj)),
                         f"{fnum(float(low), 3)} ~ {fnum(float(high), 3)}", "是" if str(rej) == "True" else "否"])
    mname = {"lsd": "LSD", "bonferroni": "Bonferroni", "holm": "Holm", "scheffe": "Scheffe", "tukey": "Tukey HSD"}[method]
    res.add_table("分组均值概览", ["组别", "N", "均值", "标准差"],
                  [[k2, len(v), fnum(v.mean()), fnum(v.std(ddof=1))] for k2, v in groups.items()])
    res.add_table(f"事后多重比较({mname})",
                  ["组1", "组2", "均值差(1-2)", "t", "原始p", "校正p", f"{int((1 - op['alpha']) * 100)}%CI", "显著?"],
                  rows, note=note + f";α={op['alpha']}")
    sig_pairs = [f"{r[0]}-{r[1]}" for r in rows if r[7] == "是"]
    if rows:
        big = max(rows, key=lambda r: abs(float(r[2])))
    res.add_interp(
        f"共进行 {len(rows)} 组两两比较({mname}校正),其中 **{len(sig_pairs)} 组对** 均值差异显著"
        + (f":{';'.join(sig_pairs[:6])}{'…' if len(sig_pairs) > 6 else ''}。" if sig_pairs else "。")
        + (f"均值差异最大的组对为「{big[0]}」与「{big[1]}」(差 {fnum(abs(float(big[2])))})。" if rows else ""))
    res.add_chart("各组均值条形图", ch.bar_chart(f"「{ycol}」各组均值", names,
                 {"均值": [round(float(groups[k2].mean()), 4) for k2 in names]}, y_name="均值"))
    return res


register_model(ModelSpec(
    id="posthoc", name="事后多重比较", category="差异检验",
    runner=_run_posthoc,
    desc="方差分析显著后的两两比较:LSD / Bonferroni / Holm / Tukey HSD / Scheffe,输出校正 p 值与置信区间。",
    tags=["事后检验", "多重比较", "两两比较"], demo_note="survey",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,)),
           Slot("group", "分组变量(定类,≥3组)", types=(CAT, ORDINAL))],
    options=[
        Option("method", "比较方法", "select", default="tukey",
               choices=[{"value": "tukey", "label": "Tukey HSD"}, {"value": "lsd", "label": "LSD"},
                        {"value": "bonferroni", "label": "Bonferroni"}, {"value": "holm", "label": "Holm"},
                        {"value": "scheffe", "label": "Scheffe"}]),
        ALPHA_OPT,
    ],
))


# ================================================================ 卡方拟合优度
def _run_chisq_gof(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    s = get_series(ds, sl["var"]).dropna().astype(str)
    if s.nunique() < 2:
        raise ModelError("分类变量至少需要 2 个类别")
    counts = s.value_counts().sort_index()
    k = len(counts)
    raw = str(op.get("ratios") or "").replace("，", ",").strip()
    if raw:
        parts = [v for v in raw.split(",") if v.strip()]
        if len(parts) != k:
            raise ModelError(f"期望比例需为 {k} 个数(与类别数一致),用逗号分隔")
        ratio = np.asarray([float(v) for v in parts], dtype=float)
        if ratio.min() < 0 or ratio.sum() <= 0:
            raise ModelError("期望比例须为非负且不全为 0")
        ratio = ratio / ratio.sum()
        src = "自定义比例 " + ",".join(f"{v:.3f}" for v in ratio)
    else:
        ratio = np.full(k, 1 / k)
        src = "等概率(均匀分布)"
    expected = counts.sum() * ratio
    mask = expected > 0
    if (expected < 5).any():
        res.add_note("存在期望频数 <5 的类别,卡方近似可能不准,建议合并类别")
    chi2, p = sps.chisquare(counts.values[mask], f_exp=expected[mask])
    res.add_table("拟合优度检验", ["类别", "观察频数", "期望频数", "期望比例", "残差(观-期)"],
                  [[str(idx), int(cnt), fnum(exp, 2), f"{ratio[i] * 100:.2f}%", fnum(cnt - exp, 2)]
                   for i, (idx, cnt, exp) in enumerate(zip(counts.index, counts.values, expected))],
                  note=f"期望分布:{src}")
    res.add_table("检验结果", ["卡方 χ²", "自由度 df", "p", "结论(α=0.05)"],
                  [[fnum(chi2, 4), int(mask.sum() - 1), fp(p), judge_p(p)]],
                  note=f"N={int(counts.sum())}")
    res.add_chart("观察 vs 期望频数", ch.bar_chart("卡方拟合优度", [str(i) for i in counts.index],
                 {"观察频数": [int(v) for v in counts.values], "期望频数": [round(float(v), 2) for v in expected]},
                 y_name="频数"))
    res.add_interp(
        f"χ²={fnum(chi2, 3)},df={int(mask.sum() - 1)},p={fp(p)}。"
        + ("观察分布与期望分布存在显著差异,数据不服从该期望分布。" if p < 0.05
           else "尚不能认为观察分布与期望分布存在差异。")
        + (f"各类别观察频数与期望的偏离以「{(counts - expected).abs().idxmax()}」最大。" if k > 2 else ""))
    return res


register_model(ModelSpec(
    id="chisq_gof", name="卡方拟合优度检验", category="差异检验",
    runner=_run_chisq_gof,
    desc="检验分类变量的观察分布是否符合等概率或自定义比例的期望分布。",
    tags=["卡方", "拟合优度", "分布"], demo_note="survey",
    slots=[Slot("var", "分类变量", types=(CAT, ORDINAL))],
    options=[
        Option("ratios", "期望比例(空=等概率)", "text", default="", tip="与类别数一致,如 0.5,0.3,0.2"),
    ],
))


# ================================================================ 重复测量方差分析
def _run_anova_rm(ds: Dataset, sl: dict, op: dict) -> Result:
    """单因素重复测量方差分析:单变量 F 检验 + Mauchly 球形度检验与 Greenhouse-Geisser 校正。"""
    res = Result()
    yname, sname, tname = sl["y"], sl["subject"], sl["time"]
    data = pd.DataFrame({
        "y": as_numeric(ds, yname),
        "s": get_series(ds, sname).astype(str),
        "t": get_series(ds, tname).astype(str),
    }).dropna()
    # 每个被试在同一水平最多一条记录:重复取均值并提示
    dup = int(data.duplicated(["s", "t"]).sum())
    wide = data.pivot_table(index="s", columns="t", values="y", aggfunc="mean")
    order = [t for t in dict.fromkeys(data["t"]) if t in wide.columns]   # 保持数据中的水平先后
    wide = wide[order].dropna()                      # 缺失任一水平的被试整行剔除
    n, k = wide.shape
    if k < 2:
        raise ModelError("组内因素至少需要 2 个水平")
    if n < 5:
        raise ModelError(f"完整被试不足 5 人(当前 {n}),重复测量分析不可靠")
    if dup:
        res.add_note(f"检测到 {dup} 条同一被试同一水平的重复记录,已按均值聚合。")

    Y = wide.values
    grand = Y.mean()
    lv_means = Y.mean(axis=0)
    sub_means = Y.mean(axis=1)
    ss_time = n * float(((lv_means - grand) ** 2).sum())
    ss_subj = k * float(((sub_means - grand) ** 2).sum())
    ss_total = float(((Y - grand) ** 2).sum())
    ss_err = ss_total - ss_time - ss_subj
    df_time, df_subj, df_err = k - 1, n - 1, (k - 1) * (n - 1)
    ms_time, ms_err = ss_time / df_time, ss_err / df_err
    f_val = ms_time / ms_err if ms_err > 0 else np.nan
    p_raw = float(sps.f.sf(f_val, df_time, df_err))
    p_eta2 = ss_time / (ss_time + ss_err) if (ss_time + ss_err) else np.nan

    drows = [[t, int(len(wide[t])), fnum(lv_means[j], 3), fnum(float(wide[t].std(ddof=1)), 3)]
             for j, t in enumerate(wide.columns)]
    res.add_table("描述统计(完整被试)", ["水平", "n", "均值", "标准差"], drows)
    res.add_table("重复测量方差分析(未校正)",
                  ["变异源", "SS", "df", "MS", "F", "p", "偏η²"],
                  [["组内(时间/条件)", fnum(ss_time, 3), df_time, fnum(ms_time, 3), fnum(f_val, 3), fp(p_raw), fnum(p_eta2, 3)],
                   ["组内误差", fnum(ss_err, 3), df_err, fnum(ms_err, 3), "", "", ""],
                   ["组间(被试间)", fnum(ss_subj, 3), df_subj, fnum(ss_subj / df_subj, 3), "", "", ""]],
                  note="剔除任一水平缺失的被试(完整案例分析)")

    # 球形度:正交对比协方差矩阵的特征值
    p_lv = k - 1
    basis = np.column_stack([np.eye(k)[:, j] - np.eye(k)[:, -1] for j in range(p_lv)])
    Q, _ = np.linalg.qr(basis)                        # 正交化对比矩阵(k × (k-1))
    Z = (Y - Y.mean(axis=0)) @ Q
    S = np.atleast_2d(np.cov(Z, rowvar=False, ddof=1))   # k=2 时 cov 返回标量
    eig = np.linalg.eigvalsh(S)
    eig = np.clip(eig.real, 1e-12, None)
    tr, tr2 = float(eig.sum()), float((eig ** 2).sum())
    eps_gg = tr * tr / (p_lv * tr2)
    eps_lb = 1.0 / p_lv
    W = float(np.prod(eig) / (tr / p_lv) ** p_lv)
    chi2_m = -(n - 1 - (2 * p_lv ** 2 + p_lv + 2) / (6 * p_lv)) * np.log(W)
    df_m = p_lv * (p_lv + 1) / 2 - 1
    p_mauchly = float(sps.chi2.sf(chi2_m, df_m)) if k > 2 else np.nan
    if k == 2:
        eps_gg = 1.0
        note = "2 个水平时球形度恒成立,无需校正"
    else:
        note = "Greenhouse-Geisser 校正;ε<0.75 时建议以校正结果为准"
    p_gg = float(sps.f.sf(f_val, eps_gg * df_time, eps_gg * df_err))
    res.add_table("球形度检验与校正结果",
                  ["Mauchly W", "χ²", "p", "GG ε", "下界 ε", "校正 df(组内,误差)", "校正后 p"],
                  [[fnum(W, 4), fnum(chi2_m, 3), fp(p_mauchly), fnum(eps_gg, 3), fnum(eps_lb, 3),
                    f"{fnum(eps_gg * df_time, 2)},{fnum(eps_gg * df_err, 2)}", fp(p_gg)]],
                  note=note)
    res.add_chart("各水平均值轮廓图", ch.line_chart("均值轮廓图", [str(c) for c in wide.columns],
                 {"均值": [round(float(v), 4) for v in lv_means]}, x_name=tname, y_name=f"「{yname}」均值"))
    sig = (p_gg if k > 2 and eps_gg < 0.75 else p_raw) < op["alpha"]
    res.add_interp(
        f"{n} 名被试在 {k} 个水平( {('、'.join(map(str, wide.columns)))} )上的重复测量方差分析:"
        f"F({fnum(df_time, 0)},{fnum(df_err, 0)})=**{fnum(f_val, 3)}**,未校正 p={fp(p_raw)},偏η²={fnum(p_eta2, 3)}。"
        + (f"Mauchly 球形度检验 p={fp(p_mauchly)},"
           f"{'满足球形度假设,以未校正结果为准' if (np.isnan(p_mauchly) or p_mauchly >= 0.05) else '违反球形度假设,建议采用 GG 校正结果'}"
           f"(GG 校正后 p={fp(p_gg)})。" if k > 2 else "")
        + f"结论:各水平均值差异{'显著' if sig else '不显著'},"
        f"{'时间/条件因素对指标存在显著影响' if sig else '尚无证据表明各水平间存在差异'}。")
    return res


register_model(ModelSpec(
    id="anova_rm", name="重复测量方差分析", category="差异检验",
    runner=_run_anova_rm,
    desc="同一批被试在多个时间点/条件下的均值比较:单变量 F 检验、Mauchly 球形度检验与 GG 校正。",
    tags=["重复测量", "球形度", "组内设计"], demo_note="panel",
    slots=[Slot("y", "观测指标(定量)", types=(QUANT,)),
           Slot("subject", "被试编号(同一被试编号相同)", types=(CAT, ORDINAL, QUANT)),
           Slot("time", "组内因素(时间/条件)", types=(CAT, ORDINAL))],
    options=[ALPHA_OPT],
))
