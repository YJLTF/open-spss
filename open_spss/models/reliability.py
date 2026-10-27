"""问卷研究:信度分析、效度分析、项目分析、多选题分析、统计绘图工具。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..core import charts as ch
from ..core.datasets import Dataset, as_numeric, get_series
from ..core.registry import CAT, ORDINAL, QUANT, ModelSpec, Option, Slot, register_model
from ..core.results import ModelError, Result, fnum, fp, judge_p
from .dimreduc import _matrix, _varimax

ALPHA_OPT = Option("alpha", "显著性水平 α", "select", default=0.05,
                   choices=[{"value": 0.1, "label": "0.1"}, {"value": 0.05, "label": "0.05"}, {"value": 0.01, "label": "0.01"}])


# ================================================================ 信度分析
def _cronbach(X: np.ndarray) -> float:
    k = X.shape[1]
    if k < 2:
        return np.nan
    var_items = X.var(axis=0, ddof=1).sum()
    var_total = X.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - var_items / var_total)) if var_total else np.nan


def _run_reliability(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    cols = sl["vars"]
    X = _matrix(ds, cols, standardize=False).values
    n, k = X.shape
    alpha = _cronbach(X)
    # CITC 与删除后 α(保留数值列,避免经 fnum 字符串往返;常数题/缺失使 CITC 为 NaN)
    rows = []
    citcs: list[float] = []
    for j, c in enumerate(cols):
        rest = np.delete(X, j, axis=1).sum(axis=1)
        citc = float(np.corrcoef(X[:, j], rest)[0, 1])
        citcs.append(citc)
        a_del = _cronbach(np.delete(X, j, axis=1))
        rows.append([c, fnum(X[:, j].mean(), 3), fnum(X[:, j].std(ddof=1), 3),
                     fnum(citc, 3) or "—", fnum(a_del, 3) or "—",
                     "可保留" if citc == citc and citc >= 0.4 else "建议检查"])
    res.add_table("信度分析(克隆巴赫 α)", ["题目", "均值", "标准差", "CITC(校正后题总相关)", "删除该题后的α", "判断"],
                  rows,
                  note=f"总量表 Cronbach's α={fnum(alpha, 3)};CITC≥0.4 且删除后 α 不明显升高,说明该题目合理")
    # 折半信度
    half1, half2 = X[:, :k // 2].sum(axis=1), X[:, k // 2:].sum(axis=1)
    r = float(np.corrcoef(half1, half2)[0, 1])
    sb = 2 * r / (1 + r) if (1 + r) else np.nan
    res.add_table("折半信度(Spearman-Brown 校正)", ["折半数", "折半相关系数r", "Spearman-Brown 系数"],
                  [[2, fnum(r, 3), fnum(sb, 3)]])
    grade = ("优秀" if alpha >= 0.9 else "良好" if alpha >= 0.8 else
             "可接受" if alpha >= 0.7 else "偏低" if alpha >= 0.6 else "不可接受")
    res.add_chart("CITC 条形图", ch.bar_chart("各题目 CITC", [r[0] for r in rows],
                 {"CITC": [round(v, 3) if v == v else None for v in citcs]}, y_name="CITC"))
    bad = [c for c, v in zip(cols, citcs) if not v >= 0.4]
    res.add_interp(
        f"量表共 {k} 个题目、{n} 个有效样本。总量表 Cronbach's α=**{fnum(alpha, 3)}**,信度**{grade}**"
        f"(α≥0.8 良好、≥0.7 可接受);折半信度 Spearman-Brown={fnum(sb, 3)}。"
        + (f"建议重点关注 CITC<0.4 的题目:{', '.join(bad)}。" if bad else "各题目 CITC 均达标,无需删题。"))
    return res


register_model(ModelSpec(
    id="reliability", name="信度分析", category="问卷研究",
    runner=_run_reliability,
    desc="克隆巴赫 α 系数、CITC、删除后 α 与折半信度，评估量表内部一致性。",
    tags=["信度", "Cronbachα", "CITC"], demo_note="survey",
    slots=[Slot("vars", "量表题目（定量/定级，≥2）", types=(QUANT, ORDINAL), multiple=True, max_count=50)],
    options=[],
))


# ================================================================ 效度分析
def _run_validity(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    cols = sl["vars"]
    data = _matrix(ds, cols, standardize=True)
    p = data.shape[1]
    X = data.values
    corr = np.corrcoef(X.T)
    # KMO
    inv = np.linalg.pinv(corr)
    d = np.sqrt(np.diag(inv))
    partial = np.zeros_like(corr)
    for i in range(p):
        for j in range(p):
            partial[i, j] = -inv[i, j] / (d[i] * d[j]) if i != j else 0
    r2 = sum(corr[i, j] ** 2 for i in range(p) for j in range(p) if i != j)
    p2 = sum(partial[i, j] ** 2 for i in range(p) for j in range(p) if i != j)
    kmo = r2 / (r2 + p2) if (r2 + p2) else np.nan
    det = np.linalg.det(corr)
    chi2 = -(len(data) - 1 - (2 * p + 5) / 6) * np.log(det) if det > 0 else np.nan
    df = p * (p - 1) / 2
    pval = sps.chi2.sf(chi2, df) if not np.isnan(chi2) else np.nan
    res.add_table("KMO 与 Bartlett 检验", ["KMO", "Bartlett χ²", "df", "p"],
                  [[fnum(kmo, 3), fnum(chi2, 3), int(df), fp(pval)]],
                  note="KMO≥0.8 极适合、≥0.7 适合、≥0.6 尚可、<0.5 不适合;Bartlett p<0.05 说明适合因子分析")
    # 因子载荷
    eig, vec = np.linalg.eigh(corr)
    order = np.argsort(eig)[::-1]
    eig, vec = np.clip(eig[order], 1e-9, None), vec[:, order]
    nfac = max(1, int((eig > 1).sum()))
    load = vec[:, :nfac] * np.sqrt(eig[:nfac])
    load = _varimax(load)
    cum = np.cumsum(eig / p)
    res.add_table("方差解释", [f"因子{i+1}" for i in range(nfac)],
                  [["特征值"] + [fnum(v, 3) for v in eig[:nfac]],
                   ["方差解释率"] + [fnum(eig[i] / p * 100, 2) + "%" for i in range(nfac)],
                   ["累积解释率"] + [fnum(cum[i] * 100, 2) + "%" for i in range(nfac)]])
    lrows = [[c] + [fnum(load[i, j], 3) for j in range(nfac)] + [fnum((load[i] ** 2).sum(), 3)]
             for i, c in enumerate(cols)]
    res.add_table("旋转后载荷矩阵", ["变量"] + [f"因子{i+1}" for i in range(nfac)] + ["共同度"], lrows)
    res.add_chart("载荷热力图", ch.heatmap_chart("因子载荷热力图", [f"因子{i+1}" for i in range(nfac)],
                 cols, load.tolist(), fmt="{c:.2f}", v_min=-1, v_max=1))
    res.add_interp(
        f"KMO={fnum(kmo, 3)}({'适合' if kmo >= 0.7 else '勉强适合' if kmo >= 0.6 else '不适合'})。"
        f"按特征值>1 提取 {nfac} 个因子,累积解释率 {fnum(cum[nfac-1] * 100, 1)}%(≥60% 结构效度可接受);"
        "请结合载荷归属为各因子命名,以确认量表的结构效度。")
    return res


register_model(ModelSpec(
    id="validity", name="效度分析", category="问卷研究",
    runner=_run_validity,
    desc="KMO、Bartlett 球形检验与探索性因子载荷结构，评估量表结构效度。",
    tags=["效度", "KMO", "结构效度"], demo_note="survey",
    slots=[Slot("vars", "量表题目（定量/定级，≥3）", types=(QUANT, ORDINAL), multiple=True, max_count=50)],
    options=[],
))


# ================================================================ 项目分析
def _run_item_analysis(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    cols = sl["vars"]
    X = _matrix(ds, cols, standardize=False)
    total = X.sum(axis=1)
    rows = []
    for c in cols:
        s = X[c]
        q_lo, q_hi = s.quantile(0.27), s.quantile(0.73)
        lo = total[s <= q_lo].index
        hi = total[s >= q_hi].index
        # 临界比值法:高低分组在该题上的 t 检验
        a = s[hi].values
        b = s[lo].values
        t, p = sps.ttest_ind(a, b, equal_var=False) if min(len(a), len(b)) > 1 else (np.nan, np.nan)
        d = (a.mean() - b.mean()) / s.std(ddof=1) if s.std(ddof=1) else np.nan
        citc = float(pd.concat([s, total.rename("t")], axis=1).corr().iloc[0, 1])
        rows.append([c, len(a) + len(b), fnum(a.mean(), 3), fnum(b.mean(), 3),
                     fnum(t, 3), fp(p), fnum(abs(d), 3), fnum(citc, 3),
                     "区分度良好" if (not np.isnan(p) and p < op["alpha"] and abs(d) >= 0.4) else "需修订/删除"])
    res.add_table("项目分析(临界比值法 + 题总相关)", ["题目", "高低组N", "高分组均值", "低分组均值",
                                    "CR(t)", "p", "区分度|d|", "题总相关r", "结论"], rows,
                  note="高低分组按总分 27% 分位;CR 显著且 |d|≥0.4、r≥0.3 说明题目区分度良好")
    res.add_chart("区分度对比", ch.bar_chart("各题目区分度 |d|", [r[0] for r in rows],
                 {"区分度": [round(float(r[6]), 3) for r in rows]}, y_name="|d|"))
    bad = [r[0] for r in rows if r[8].startswith("需")]
    res.add_interp(
        f"共分析 {len(cols)} 个题目。"
        + (f"区分度良好的题目 {len(cols) - len(bad)} 个;" if len(bad) < len(cols) else "")
        + (f"建议修订或删除:{'、'.join(bad)}(高低组差异不显著或区分度<0.4)。" if bad else "全部题目区分度良好。"))
    return res


register_model(ModelSpec(
    id="item_analysis", name="项目分析(区分度)", category="问卷研究",
    runner=_run_item_analysis,
    desc="临界比值法（CR）与题总相关评估每个题目的区分度，筛选劣质题目。",
    tags=["项目分析", "区分度", "临界比值"], demo_note="survey",
    slots=[Slot("vars", "量表题目（定量/定级）", types=(QUANT, ORDINAL), multiple=True, max_count=50)],
    options=[ALPHA_OPT],
))


# ================================================================ 多选题分析
def _run_multi_choice(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    cols = sl["vars"]
    X = pd.concat([get_series(ds, c) for c in cols], axis=1).dropna()
    n = len(X)
    # 1=选中
    selected = X.apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0) > 0)
    counts = selected.sum().astype(int)
    total_sel = int(counts.sum())
    rows = []
    for c in cols:
        rows.append([c, int(counts[c]), f"{counts[c] / total_sel * 100:.2f}%" if total_sel else "—",
                     f"{counts[c] / n * 100:.2f}%", n])
    rows.sort(key=lambda r: -r[1])
    res.add_table("多选题响应分析", ["选项", "响应频次", "响应率", "普及率", "样本量N"], rows,
                  note="响应率=该选项选择数÷总选择数;普及率=该选项选择数÷样本量(可多选故合计>100%)")
    res.add_chart("选项响应频次", ch.bar_chart("多选题响应频次", [r[0] for r in rows],
                 {"频次": [int(r[1]) for r in rows]}, y_name="频次"))
    res.add_chart("选项普及率", ch.pie_chart("普及率构成", [(r[0], int(r[1])) for r in rows[:8]]))
    res.add_interp(
        f"共 {n} 人作答,总选择 {total_sel} 次,人均选择 {fnum(total_sel / n, 2)} 项。"
        f"最受欢迎的选项为「{rows[0][0]}」(普及率 {rows[0][3]})。")
    return res


register_model(ModelSpec(
    id="multi_choice", name="多选题分析", category="问卷研究",
    runner=_run_multi_choice,
    desc="对 0/1 编码的多选题输出响应率与普及率，发现最受欢迎的选项。",
    tags=["多选题", "响应率", "普及率"], demo_note="survey",
    slots=[Slot("vars", "选项列（0/1 编码，每列一个选项）", types=(QUANT, CAT), multiple=True, max_count=30)],
    options=[],
))

# ================================================================ McDonald's ω
def _run_omega(ds: Dataset, sl: dict, op: dict) -> Result:
    """McDonald's ω:单因子模型下 (Σλ)²/((Σλ)²+Σθ),θ=1-λ²(标准化题目)。"""
    res = Result()
    cols = sl["vars"]
    data = _matrix(ds, cols, standardize=True).dropna()
    X = data.values
    n, k = X.shape
    corr = np.corrcoef(X.T)
    eig, vec = np.linalg.eigh(corr)
    order = np.argsort(eig)[::-1]
    eig, vec = eig[order], vec[:, order]
    load = np.clip(vec[:, 0] * np.sqrt(max(eig[0], 1e-9)), -0.999, 0.999)
    if (load < 0).sum() > k / 2:            # 载荷方向统一(反向题常见)
        load = -load
    theta = 1 - load ** 2                    # 标准化误差方差
    sum_l = load.sum()
    omega = float(sum_l ** 2 / (sum_l ** 2 + theta.sum())) if (sum_l ** 2 + theta.sum()) else np.nan
    alpha = _cronbach(X)
    grade = ("优秀" if omega >= 0.9 else "良好" if omega >= 0.8 else
             "可接受" if omega >= 0.7 else "偏低" if omega >= 0.6 else "不可接受")
    rows = [[c, fnum(load[j], 3), fnum(theta[j], 3), fnum(load[j] ** 2, 3)] for j, c in enumerate(cols)]
    res.add_table("单因子载荷与误差方差", ["题目", "标准化载荷 λ", "误差方差 θ=1-λ²", "共同度 λ²"], rows,
                  note="负载荷提示反向计分题目,建议先反向处理后再分析")
    res.add_table("信度系数对比", ["McDonald's ω", "Cronbach's α", "信度评价(ω)"],
                  [[fnum(omega, 3), fnum(alpha, 3), grade]],
                  note="ω≥0.9 优秀、≥0.8 良好、≥0.7 可接受;ω 基于因子模型,不要求各题目载荷相等,通常更接近量表真实信度")
    res.add_chart("题目标准化载荷", ch.bar_chart("单因子标准化载荷", cols,
                 {"λ": [round(float(v), 3) for v in load]}, y_name="载荷 λ"))
    low = [c for c, v in zip(cols, load) if v < 0.4]
    res.add_interp(
        f"量表 {k} 个题目、{n} 个有效样本。McDonald's ω=**{fnum(omega, 3)}**,信度**{grade}**;"
        f"同期 Cronbach's α={fnum(alpha, 3)}。"
        + (f"载荷低于 0.4 的题目:{', '.join(low)},对总分贡献有限,可考虑修订或删除。" if low
           else "各题目标准化载荷均 ≥0.4,单因子结构清晰,无需删题。"))
    return res


register_model(ModelSpec(
    id="omega", name="McDonald's ω 信度", category="问卷研究",
    runner=_run_omega,
    desc="基于单因子模型的组合信度 McDonald's ω，与 Cronbach's α 对比，不要求 tau 等价。",
    tags=["信度", "ω", "组合信度"], demo_note="survey",
    slots=[Slot("vars", "量表题目（定量/定级，≥3）", types=(QUANT, ORDINAL), multiple=True, max_count=50)],
    options=[],
))

# ================================================================ IPA 分析
_QUAD_LABEL = {
    "A": "优势保持区(高重要×高满意)",
    "B": "重点改进区(高重要×低满意)",
    "C": "次要改进区(低重要×低满意)",
    "D": "过度供给区(低重要×高满意)",
}


def _run_ipa(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    imps, perfs = list(sl["imp"]), list(sl["perf"])
    if len(imps) != len(perfs):
        raise ModelError(f"重要性与满意度变量需一一对应(当前 {len(imps)} vs {len(perfs)})")
    if not imps:
        raise ModelError("请选择变量")
    rows = []
    for c1, c2 in zip(imps, perfs):
        a, b = as_numeric(ds, c1), as_numeric(ds, c2)
        d = pd.concat([a.rename("i"), b.rename("p")], axis=1).dropna()
        if len(d) < 3:
            raise ModelError(f"变量对「{c1}/{c2}」有效样本不足")
        rows.append([c1, float(d["i"].mean()), float(d["p"].mean())])
    imps_m = np.array([r[1] for r in rows])
    perfs_m = np.array([r[2] for r in rows])
    split = str(op["split"])
    if split == "custom":
        try:
            ci, cp = [float(v) for v in str(op["cutvals"]).replace("，", ",").split(",")[:2]]
        except Exception as e:  # noqa: BLE001
            raise ModelError("自定义分割点需两个数字,如 3,3") from e
    elif split == "median":
        ci, cp = float(np.median(imps_m)), float(np.median(perfs_m))
    else:
        ci, cp = float(imps_m.mean()), float(perfs_m.mean())
    split_name = {"mean": "均值", "median": "中位数", "custom": "自定义"}[split]
    quads = []
    for name, im, pm in rows:
        hi_i, hi_p = im >= ci, pm >= cp
        q = ("A" if hi_i and hi_p else "B" if hi_i else "D" if hi_p else "C")
        quads.append(q)
    res.add_table("IPA 象限结果", ["项目", "重要性均值", "满意度均值", "象限", "管理策略"],
                  [[r[0], fnum(r[1], 3), fnum(r[2], 3), quads[i], _QUAD_LABEL[quads[i]].split("(")[0]]
                   for i, r in enumerate(rows)],
                  note=f"分割点:重要性={fnum(ci, 3)},满意度={fnum(cp, 3)}({split_name})")
    xs = [[round(float(imps_m[i]), 4), round(float(perfs_m[i]), 4)] for i in range(len(rows))]
    xlo, xhi = float(imps_m.min()) - 0.3, float(imps_m.max()) + 0.3
    ylo, yhi = float(perfs_m.min()) - 0.3, float(perfs_m.max()) + 0.3
    series = {
        "A 优势保持": [xs[i] for i in range(len(rows)) if quads[i] == "A"],
        "B 重点改进": [xs[i] for i in range(len(rows)) if quads[i] == "B"],
        "C 次要改进": [xs[i] for i in range(len(rows)) if quads[i] == "C"],
        "D 过度供给": [xs[i] for i in range(len(rows)) if quads[i] == "D"],
    }
    series = {k: v for k, v in series.items() if v}
    opt = ch.scatter_chart("IPA 象限图(横轴重要性,纵轴满意度)", series, x_name="重要性", y_name="满意度", symbol=12)
    opt["series"].append({"name": "分割线", "type": "line", "data": [[ci, ylo], [ci, yhi]],
                          "showSymbol": False, "lineStyle": {"type": "dashed", "color": "#94a3b8"}, "tooltip": {"show": False}})
    opt["series"].append({"name": "分割线2", "type": "line", "data": [[xlo, cp], [xhi, cp]],
                          "showSymbol": False, "lineStyle": {"type": "dashed", "color": "#94a3b8"}, "tooltip": {"show": False}})
    res.add_chart("IPA 四象限图", opt)
    names_q = {q: [rows[i][0] for i in range(len(rows)) if quads[i] == q] for q in "ABCD"}
    parts = []
    if names_q["A"]:
        parts.append("优势保持:" + "、".join(names_q["A"]))
    if names_q["B"]:
        parts.append("**重点改进:" + "、".join(names_q["B"]) + "**")
    if names_q["D"]:
        parts.append("过度供给(可优化资源配置):" + "、".join(names_q["D"]))
    if names_q["C"]:
        parts.append("次要改进:" + "、".join(names_q["C"]))
    res.add_interp("IPA 分析按重要性-满意度将项目划入四个象限。" + ";".join(parts) + "。"
                   "优先改进 B 区(重要但不满意)项目,A 区维持并总结经验。")
    return res


register_model(ModelSpec(
    id="ipa", name="IPA 分析(重要性-满意度)", category="问卷研究",
    runner=_run_ipa,
    desc="重要性-表现四象限分析，定位需优先改进的服务/产品项目。",
    tags=["IPA", "四象限", "满意度"], demo_note="survey",
    slots=[
        Slot("imp", "重要性变量（定量/定级，与满意度对应）", types=(QUANT, ORDINAL), multiple=True, max_count=12),
        Slot("perf", "满意度变量（定量/定级，与重要性对应）", types=(QUANT, ORDINAL), multiple=True, max_count=12),
    ],
    options=[
        Option("split", "分割点", "select", default="mean",
               choices=[{"value": "mean", "label": "总均值"}, {"value": "median", "label": "中位数"},
                        {"value": "custom", "label": "自定义（量表中点）"}]),
        Option("cutvals", "自定义分割点（重要性，满意度）", "text", default="3,3", show_if={"split": "custom"}),
    ],
))
