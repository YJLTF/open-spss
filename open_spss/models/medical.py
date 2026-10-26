"""医学统计:ROC 曲线、Kappa 一致性、Cox 比例风险回归、Kaplan-Meier 生存曲线。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import charts as ch
from ..core.datasets import Dataset, as_numeric, get_series
from ..core.registry import CAT, ORDINAL, QUANT, ModelSpec, Option, Slot, register_model
from ..core.results import ModelError, Result, fnum, fp


def _concordance_index(risk: "np.ndarray", time: "np.ndarray", event: "np.ndarray") -> float:
    """Harrell 一致性指数:风险高者应更早发生事件"""
    conc = pairs = 0
    n = len(time)
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] == time[j]:
                continue
            if time[i] < time[j]:
                a, b = (i, j) if event[i] else (None, None)
            else:
                a, b = (j, i) if event[j] else (None, None)
            if a is None:
                continue
            pairs += 1
            if risk[a] > risk[b]:
                conc += 1
            elif risk[a] == risk[b]:
                conc += 0.5
    return conc / pairs if pairs else np.nan


# ================================================================ ROC 曲线
def _run_roc(ds: Dataset, sl: dict, op: dict) -> Result:
    from sklearn.metrics import auc as sk_auc
    from sklearn.metrics import roc_auc_score, roc_curve
    res = Result()
    yraw = get_series(ds, sl["y"]).astype(str)
    levels = sorted(yraw.dropna().unique(), key=str)
    if len(levels) != 2:
        raise ModelError(f"状态变量需为二分类(当前 {len(levels)} 个水平)")
    pos = op["pos_level"] if str(op["pos_level"]).strip() else levels[1]
    if pos not in levels:
        pos = levels[1]
    y = (yraw == pos).astype(int)
    rows = []
    interp_parts = []
    for i, col in enumerate(sl["scores"]):
        score = as_numeric(ds, col)
        data = pd.concat([score.rename("s"), y.rename("y")], axis=1).dropna()
        if data["y"].nunique() < 2:
            raise ModelError(f"「{col}」对应的金标准只有一个类别")
        fpr, tpr, thr = roc_curve(data["y"], data["s"])
        auc_val = sk_auc(fpr, tpr)
        n1 = int(data["y"].sum())
        n2 = int(len(data) - n1)
        # Hanley-McNeil 标准误
        q1 = auc_val / (2 - auc_val)
        q2 = 2 * auc_val ** 2 / (1 + auc_val)
        se = np.sqrt((auc_val * (1 - auc_val) + (n1 - 1) * (q1 - auc_val ** 2) + (n2 - 1) * (q2 - auc_val ** 2)) / (n1 * n2)) if n1 and n2 else np.nan
        youden = tpr - fpr
        bi = int(np.argmax(youden))
        best_thr = thr[bi] if bi < len(thr) else thr[-1]
        grade = ("优秀" if auc_val >= 0.9 else "良好" if auc_val >= 0.8 else
                 "中等" if auc_val >= 0.7 else "较低" if auc_val >= 0.6 else "接近随机")
        rows.append([col, fnum(auc_val, 4), f"+{fnum(1.96 * se, 4)} / -{fnum(1.96 * se, 4)}",
                     f"{fnum(max(auc_val - 1.96 * se, 0), 4)} ~ {fnum(min(auc_val + 1.96 * se, 1), 4)}",
                     fnum(best_thr, 4), fnum(tpr[bi], 4), fnum(1 - fpr[bi], 4), fnum(youden[bi], 4), grade])
        interp_parts.append(f"「{col}」AUC={fnum(auc_val, 3)}({grade}),最佳截断值 {fnum(best_thr, 3)}(灵敏度 {fnum(tpr[bi], 3)},特异度 {fnum(1-fpr[bi], 3)})。")
        series_name = col if len(sl["scores"]) > 1 else "ROC 曲线"
        pts = [[float(a), float(b)] for a, b in zip(fpr, tpr)]
        if i == 0:
            roc_series = {series_name: pts}
        else:
            roc_series[series_name] = pts
    res.add_table("ROC 曲线结果", ["检验变量", "AUC", "标准误×1.96", "AUC 95%CI", "最佳截断值",
                                  "灵敏度", "特异度", "约登指数", "诊断价值"], rows,
                  note="AUC≥0.9 优秀、≥0.8 良好、≥0.7 中等、≥0.6 较低、=0.5 无诊断价值")
    roc_series["随机(45°)"] = [[0, 0], [1, 1]]
    res.add_chart("ROC 曲线", ch.scatter_chart("ROC 曲线(受试者工作特征)", roc_series,
                 x_name="假阳性率(1-特异度)", y_name="真阳性率(灵敏度)", symbol=4))
    res.add_interp(f"以「{pos}」为阳性:" + " ".join(interp_parts))
    return res


register_model(ModelSpec(
    id="roc", name="ROC 曲线", category="医学统计",
    runner=_run_roc,
    desc="诊断试验评价:AUC 及其置信区间、约登指数最佳截断值、灵敏度/特异度。",
    tags=["ROC", "AUC", "诊断"], demo_note="clinical",
    slots=[Slot("y", "金标准(二分类)", types=(CAT, ORDINAL)),
           Slot("scores", "检验变量(定量,可多个)", types=(QUANT,), multiple=True, max_count=5)],
    options=[Option("pos_level", "阳性水平(空=较大值)", "text", default="")],
))


# ================================================================ Kappa 一致性
def _run_kappa(ds: Dataset, sl: dict, op: dict) -> Result:
    from sklearn.metrics import cohen_kappa_score
    res = Result()
    a, b = sl["x"], sl["y"]
    sub = ds.df[[a, b]].dropna().astype(str)
    weights = op["weights"] if op["weights"] != "none" else None
    levels = sorted(set(sub[a]) | set(sub[b]), key=str)
    if len(levels) < 2:
        raise ModelError("至少需要 2 个类别")
    kappa = cohen_kappa_score(sub[a], sub[b], weights=weights)
    # 渐近标准误与 z 检验(基于混淆矩阵 delta 法的简化版:使用 bootstrap)
    ct = pd.crosstab(sub[a], sub[b])
    n = len(sub)
    po = np.trace(ct.values) / n
    pe = float((ct.sum(axis=1) / n * ct.sum(axis=0) / n).sum())
    rng = np.random.default_rng(42)
    boots = []
    arr = sub.values
    for _ in range(300):
        idx = rng.integers(0, n, n)
        smp = arr[idx]
        try:
            boots.append(cohen_kappa_score(smp[:, 0], smp[:, 1], weights=weights))
        except Exception:  # noqa: BLE001
            continue
    se = float(np.std(boots)) if len(boots) > 10 else np.nan
    z = kappa / se if se and se > 0 else np.nan
    p = 2 * (1 - __import__("scipy.stats", fromlist=["norm"]).norm.cdf(abs(z))) if not np.isnan(z) else np.nan
    strength = ("几乎完全一致" if kappa >= 0.81 else "高度一致" if kappa >= 0.61 else
                "中度一致" if kappa >= 0.41 else "一般一致" if kappa >= 0.21 else "轻微一致" if kappa > 0 else "极差")
    res.add_table("交叉表(评估者A × 评估者B)", ["A\\B"] + [str(c) for c in ct.columns] + ["合计"],
                  [[str(idx)] + [int(v) for v in ct.loc[idx]] + [int(ct.loc[idx].sum())] for idx in ct.index]
                  + ["合计"] + [int(ct.sum(axis=0)[c]) for c in ct.columns] + [n])
    res.add_table("Kappa 一致性检验", ["Kappa", "标准误", "Z", "p", "观察一致率po", "期望一致率pe", "一致性强度"],
                  [[fnum(kappa, 4), fnum(se, 4), fnum(z, 3), fp(p), fnum(po, 4), fnum(pe, 4), strength]],
                  note=("加权方式:" + {"none": "未加权", "linear": "线性加权", "quadratic": "二次加权"}[op["weights"]]) if weights else "未加权 Kappa")
    res.add_interp(
        f"Kappa={fnum(kappa, 3)}(p={fp(p)}),两变量一致性强度为 **{strength}**;"
        f"观察一致率 {fnum(po * 100, 1)}%,剔除机遇一致后的真实一致率见 Kappa 值。")
    return res


register_model(ModelSpec(
    id="kappa", name="Kappa 一致性检验", category="医学统计",
    runner=_run_kappa,
    desc="评估两位评估者/两次测量分类结果的一致性,支持线性/二次加权 Kappa。",
    tags=["Kappa", "一致性"], demo_note="clinical",
    slots=[Slot("x", "评估者A/测量1(定类)", types=(CAT, ORDINAL)),
           Slot("y", "评估者B/测量2(定类)", types=(CAT, ORDINAL))],
    options=[Option("weights", "加权方式", "select", default="none",
                    choices=[{"value": "none", "label": "未加权"}, {"value": "linear", "label": "线性加权"},
                             {"value": "quadratic", "label": "二次加权"}])],
))


# ================================================================ Cox 比例风险回归
def _run_cox(ds: Dataset, sl: dict, op: dict) -> Result:
    import statsmodels.api as sm
    res = Result()
    time_col, event_col = sl["time"], sl["event"]
    xs = list(sl["xs"])
    ev_raw = get_series(ds, event_col)
    ev_levels = sorted(pd.unique(ev_raw.dropna()).tolist(), key=str)
    if len(ev_levels) > 2:
        raise ModelError(f"结局变量需为二分类(当前 {len(ev_levels)} 个水平)")
    pos_ev = op["event_level"] if str(op["event_level"]).strip() else (ev_levels[-1] if len(ev_levels) > 1 else ev_levels[0])
    event = (ev_raw == pos_ev).astype(float)
    cols = [time_col] + xs
    data = pd.concat([as_numeric(ds, time_col).rename("__t__")] +
                     [as_numeric(ds, c).rename(f"x{i}") for i, c in enumerate(xs)] +
                     [event.rename("__e__")], axis=1).dropna()
    if int(data["__e__"].sum()) < 5:
        raise ModelError("事件(死亡/复发)发生数不足 5,无法拟合 Cox 模型")
    X = data[[f"x{i}" for i in range(len(xs))]]
    cox = sm.PHReg(data["__t__"], X, status=data["__e__"]).fit()
    risk = np.asarray(X @ np.asarray(cox.params))
    cidx = _concordance_index(risk, data["__t__"].values, data["__e__"].values)
    rows = []
    for i, c in enumerate(xs):
        b = float(cox.params[i])
        se = float(cox.bse[i])
        z = float(b / se) if se else np.nan
        p = float(2 * __import__("scipy.stats", fromlist=["norm"]).norm.sf(abs(z))) if not np.isnan(z) else np.nan
        ci = cox.conf_int(alpha=op["alpha"])[i]
        rows.append([c, fnum(b, 4), fnum(se, 4), fnum(z, 3), fp(p), fnum(np.exp(b), 4),
                     f"{fnum(np.exp(ci[0]), 4)} ~ {fnum(np.exp(ci[1]), 4)}",
                     "显著" if (not np.isnan(p) and p < op["alpha"]) else "不显著"])
    res.add_table("Cox 比例风险回归", ["变量", "系数B", "标准误", "Z", "p", "HR(风险比)", "HR 95%CI", "结论"], rows,
                  note=f"事件水平:「{pos_ev}」;HR>1 表示风险增加,<1 表示保护因素;"
                       f"一致性指数 C-index={fnum(cidx, 3)}")
    res.add_interp(
        f"Cox 回归拟合完成,C-index={fnum(cidx, 3)}(≥0.7 区分度良好)。"
        + (f"显著危险/保护因素:{'、'.join(r[0] + f'(HR={r[5]})' for r in rows if r[7] == '显著')}。" if any(r[7] == "显著" for r in rows) else "暂无显著变量。"))
    return res


register_model(ModelSpec(
    id="cox", name="Cox 比例风险回归", category="医学统计",
    runner=_run_cox,
    desc="生存分析多因素方法,输出各协变量的 HR 风险比与 C-index,不要求生存时间分布。",
    tags=["Cox", "生存分析", "HR"], demo_note="clinical",
    slots=[Slot("time", "生存时间(定量)", types=(QUANT,)),
           Slot("event", "结局事件(二分类)", types=(CAT, ORDINAL)),
           Slot("xs", "协变量(定量)", types=(QUANT,), multiple=True)],
    options=[Option("event_level", "事件发生水平(空=较大值)", "text", default=""),
             Option("alpha", "显著性水平 α", "select", default=0.05,
                    choices=[{"value": 0.1, "label": "0.1"}, {"value": 0.05, "label": "0.05"}, {"value": 0.01, "label": "0.01"}])],
))


# ================================================================ Kaplan-Meier 生存曲线
def _run_km(ds: Dataset, sl: dict, op: dict) -> Result:
    from statsmodels.duration.survfunc import SurvfuncRight, survdiff
    res = Result()
    time_col, event_col = sl["time"], sl["event"]
    ev_raw = get_series(ds, event_col)
    ev_levels = sorted(pd.unique(ev_raw.dropna()).tolist(), key=str)
    pos_ev = op["event_level"] if str(op["event_level"]).strip() else (ev_levels[-1] if len(ev_levels) > 1 else ev_levels[0])
    event = (ev_raw == pos_ev).astype(int)
    gcol = sl.get("group")
    if gcol:
        data = pd.concat([as_numeric(ds, time_col).rename("__t__"), event.rename("__e__"),
                          get_series(ds, gcol).astype(str).rename("__g__")], axis=1).dropna()
        groups = [g for g, sub in data.groupby("__g__", observed=True) if len(sub) >= 3 and sub["__e__"].sum() >= 1]
        if len(groups) < 1:
            raise ModelError("分组后每组需要 ≥3 例且至少 1 个事件")
        stat, p = survdiff(data["__t__"].values, data["__e__"].values, data["__g__"].values)
        rows = []
        series = {}
        for g in groups:
            sub = data[data["__g__"] == g]
            sf = SurvfuncRight(sub["__t__"].values, sub["__e__"].values)
            med = float(np.median(sub["__t__"][sub["__e__"] == 1])) if (sub["__e__"] == 1).any() else np.nan
            rows.append([str(g), len(sub), int(sub["__e__"].sum()), fnum(med, 2),
                         fnum(float(sf.surv_prob[-1]), 4) if len(sf.surv_prob) else "—"])
            xs = np.repeat(sf.surv_times, 2)[1:]
            ys = np.repeat(sf.surv_prob, 2)[:-1]
            series[str(g)] = [[round(float(a), 3), round(float(b), 4)] for a, b in zip(xs, ys)]
        res.add_table("各组生存概况", ["组别", "例数", "事件数", "中位生存时间(事件者)", "末期生存率"], rows)
        res.add_chart("Kaplan-Meier 生存曲线", ch.line_chart("Kaplan-Meier 生存曲线",
                     [str(v) for v in range(2000)], series, x_name="时间", y_name="生存概率",
                     smooth=False))
        res.add_interp(
            f"Log-rank 检验统计量={fnum(stat, 3)},p={fp(float(p))},"
            f"{'各组生存曲线差异显著' if float(p) < op['alpha'] else '各组生存曲线差异不显著'}。")
    else:
        data = pd.concat([as_numeric(ds, time_col).rename("__t__"), event.rename("__e__")], axis=1).dropna()
        sf = SurvfuncRight(data["__t__"].values, data["__e__"].values)
        med = float(np.median(data["__t__"][data["__e__"] == 1])) if (data["__e__"] == 1).any() else np.nan
        res.add_table("总体生存概况", ["例数", "事件数", "中位生存时间(事件者)", "末期生存率"],
                      [[len(data), int(data["__e__"].sum()), fnum(med, 2),
                        fnum(float(sf.surv_prob[-1]), 4) if len(sf.surv_prob) else "—"]])
        xs = np.repeat(sf.surv_times, 2)[1:]
        ys = np.repeat(sf.surv_prob, 2)[:-1]
        res.add_chart("Kaplan-Meier 生存曲线", ch.line_chart("Kaplan-Meier 生存曲线(总体)",
                     [str(v) for v in range(len(xs))], {"生存概率": [[round(float(a), 3), round(float(b), 4)] for a, b in zip(xs, ys)]},
                     x_name="时间", y_name="生存概率", smooth=False))
        res.add_interp(f"总体中位生存时间(事件者)为 {fnum(med, 2)},末期生存率 {fnum(float(sf.surv_prob[-1]), 3) if len(sf.surv_prob) else '—'}。")
    res.add_note("分组时进行 Log-rank 检验;曲线阶梯下降处为事件发生,竖直虚线状下降后平台期表示删失后无新事件")
    return res


register_model(ModelSpec(
    id="kaplan_meier", name="Kaplan-Meier 生存曲线", category="医学统计",
    runner=_run_km,
    desc="非参数生存估计,支持分组绘制生存曲线并进行 Log-rank 检验。",
    tags=["Kaplan-Meier", "生存曲线", "Log-rank"], demo_note="clinical",
    slots=[Slot("time", "生存时间(定量)", types=(QUANT,)),
           Slot("event", "结局事件(二分类)", types=(CAT, ORDINAL)),
           Slot("group", "分组变量(选填)", types=(CAT, ORDINAL), required=False)],
    options=[Option("event_level", "事件发生水平(空=较大值)", "text", default=""),
             Option("alpha", "显著性水平 α", "select", default=0.05,
                    choices=[{"value": 0.1, "label": "0.1"}, {"value": 0.05, "label": "0.05"}, {"value": 0.01, "label": "0.01"}])],
))

# ================================================================ Log-rank 生存曲线比较
def _run_logrank(ds: Dataset, sl: dict, op: dict) -> Result:
    from statsmodels.duration.survfunc import SurvfuncRight, survdiff
    res = Result()
    data = pd.concat([as_numeric(ds, sl["time"]).rename("__t__"),
                      (get_series(ds, sl["event"]).astype(str)).rename("__e__"),
                      get_series(ds, sl["group"]).astype(str).rename("__g__")], axis=1).dropna()
    ev_levels = sorted(data["__e__"].unique().tolist(), key=str)
    if len(ev_levels) != 2:
        raise ModelError(f"事件变量需为二分类(当前 {len(ev_levels)} 个水平)")
    pos_ev = op["event_level"] if str(op["event_level"]).strip() else ev_levels[-1]
    data["__e__"] = (data["__e__"] == pos_ev).astype(int)
    groups = [g for g, sub in data.groupby("__g__", observed=True) if len(sub) >= 3]
    if len(groups) < 2:
        raise ModelError("分组后至少需要 2 组、每组 ≥3 例")
    data = data[data["__g__"].isin(groups)]
    stat, p = survdiff(data["__t__"].values, data["__e__"].values, data["__g__"].values)
    rows, series = [], {}
    for g in groups:
        sub = data[data["__g__"] == g]
        sf = SurvfuncRight(sub["__t__"].values, sub["__e__"].values)
        med = float(np.median(sub["__t__"][sub["__e__"] == 1])) if (sub["__e__"] == 1).any() else np.nan
        rows.append([str(g), len(sub), int(sub["__e__"].sum()), fnum(med, 2),
                     fnum(float(sf.surv_prob[-1]), 4) if len(sf.surv_prob) else "—"])
        xs = np.repeat(sf.surv_times, 2)[1:]
        ys = np.repeat(sf.surv_prob, 2)[:-1]
        series[str(g)] = [[round(float(a), 3), round(float(b), 4)] for a, b in zip(xs, ys)]
    res.add_table("各组生存概况", ["组别", "例数", "事件数", "中位生存时间(事件者)", "末期生存率"], rows)
    res.add_table("Log-rank 检验", ["统计量 χ²", "自由度 df", "p", "结论(α=0.05)"],
                  [[fnum(stat, 4), len(groups) - 1, fp(float(p)), "是" if float(p) < 0.05 else "否"]],
                  note=f"H0:各组生存曲线相同;事件定义=「{pos_ev}」")
    res.add_chart("各组生存曲线比较", ch.line_chart("Kaplan-Meier 生存曲线比较", [str(v) for v in range(2000)],
                 series, x_name="时间", y_name="生存概率", smooth=False))
    sig = float(p) < op["alpha"]
    meds = {str(g): v for g, *rest in [(r[0], r[3]) for r in rows] for v in [rest[0]]}
    res.add_interp(f"Log-rank χ²={fnum(stat, 3)},p={fp(float(p))}:"
                   + ("各组生存曲线差异**显著**," if sig else "尚不能认为各组生存曲线有差异,")
                   + f"其中位生存时间分别为 {';'.join(f'{k}:{v}' for k, v in meds.items())}。")
    return res


register_model(ModelSpec(
    id="logrank", name="Log-rank 生存曲线比较", category="医学统计",
    runner=_run_logrank,
    desc="比较两组/多组生存曲线是否来自同一生存分布(Kaplan-Meier + Log-rank 检验)。",
    tags=["Log-rank", "生存分析", "组间比较"], demo_note="clinical",
    slots=[Slot("time", "生存时间(定量)", types=(QUANT,)),
           Slot("event", "事件状态(二分类)", types=(CAT, ORDINAL)),
           Slot("group", "分组变量(定类,2~5组)", types=(CAT, ORDINAL))],
    options=[
        Option("event_level", "事件发生的水平(空=自动取后一水平)", "text", default=""),
        Option("alpha", "显著性水平 α", "select", default=0.05,
               choices=[{"value": 0.1, "label": "0.1"}, {"value": 0.05, "label": "0.05"}, {"value": 0.01, "label": "0.01"}]),
    ],
))


# ================================================================ 诊断试验评价
def _run_dxtest(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    gold_raw = get_series(ds, sl["gold"]).dropna().astype(str)
    test_col = sl["test"]
    thr = float(op["threshold"] or 0)
    if thr > 0:
        t = as_numeric(ds, test_col)
        test_raw = pd.Series(np.where(t >= thr, "阳性", "阴性"), index=t.index).dropna().astype(str)
        test_desc = f"数值 ≥ {thr} 判为阳性"
    else:
        test_raw = get_series(ds, test_col).dropna().astype(str)
        test_desc = ""
    gl = sorted(gold_raw.unique().tolist(), key=str)
    tl = sorted(test_raw.unique().tolist(), key=str)
    if len(gl) != 2 or len(tl) != 2:
        raise ModelError(f"金标准与诊断结果均需为二分类(金标准 {len(gl)} 个水平,诊断 {len(tl)} 个水平)")
    gp = op["gold_pos"] if str(op["gold_pos"]).strip() else gl[-1]
    tp = op["test_pos"] if str(op["test_pos"]).strip() else tl[-1]
    d = pd.concat([gold_raw.rename("g"), test_raw.rename("t")], axis=1).dropna()
    tp_n = int(((d["g"] == gp) & (d["t"] == tp)).sum())
    fn_n = int(((d["g"] == gp) & (d["t"] != tp)).sum())
    fp_n = int(((d["g"] != gp) & (d["t"] == tp)).sum())
    tn_n = int(((d["g"] != gp) & (d["t"] != tp)).sum())
    total = tp_n + fn_n + fp_n + tn_n
    if tp_n + fn_n == 0 or tn_n + fp_n == 0:
        raise ModelError("金标准两个水平均需有样本")
    sens = tp_n / (tp_n + fn_n)
    spec = tn_n / (tn_n + fp_n)
    youden = sens + spec - 1
    ppv = tp_n / (tp_n + fp_n) if tp_n + fp_n else np.nan
    npv = tn_n / (tn_n + fn_n) if tn_n + fn_n else np.nan
    acc = (tp_n + tn_n) / total
    prev = (tp_n + fn_n) / total
    lrp = sens / (1 - spec) if spec < 1 else np.inf
    lrn = (1 - sens) / spec if spec > 0 else np.inf
    res.add_table("诊断四格表", ["\\", f"诊断阳性({tp})", f"诊断阴性({tl[0] if tl[0] != tp else tl[0]})", "合计"],
                  [[f"金标准阳性({gp})", tp_n, fn_n, tp_n + fn_n],
                   [f"金标准阴性({gl[0] if gl[0] != gp else gl[0]})", fp_n, tn_n, fp_n + tn_n],
                   ["合计", tp_n + fp_n, fn_n + tn_n, total]],
                  note=test_desc or f"阳性定义:金标准=「{gp}」,诊断=「{tp}」")
    res.add_table("诊断试验评价指标", ["指标", "数值", "说明"],
                  [["灵敏度(敏感度)", f"{sens * 100:.2f}%", "真阳性率:有病者被正确诊断的比例"],
                   ["特异度", f"{spec * 100:.2f}%", "真阴性率:无病者被正确排除的比例"],
                   ["约登指数(Youden)", fnum(youden, 4), "灵敏度+特异度−1,越大诊断效能越高"],
                   ["阳性预测值 PPV", f"{ppv * 100:.2f}%", "诊断阳性中真有病的比例"],
                   ["阴性预测值 NPV", f"{npv * 100:.2f}%", "诊断阴性中真无病的比例"],
                   ["准确率", f"{acc * 100:.2f}%", "(真阳+真阴)/总数"],
                   ["患病率", f"{prev * 100:.2f}%", "样本中金标准阳性比例"],
                   ["阳性似然比 LR+", fnum(lrp, 4), "阳性结果出现在病人的倍数,>10 优秀"],
                   ["阴性似然比 LR−", fnum(lrn, 4), "阴性结果出现在病人的倍数,<0.1 优秀"]])
    pairs = [("灵敏度", round(sens, 4)), ("特异度", round(spec, 4)), ("约登指数", round(max(youden, 0), 4))]
    res.add_chart("诊断效能概览", ch.bar_chart("灵敏度 / 特异度 / 约登指数", [k for k, _ in pairs],
                 {"数值": [v for _, v in pairs]}, y_name="数值"))
    res.add_interp(f"灵敏度 {sens * 100:.1f}%、特异度 {spec * 100:.1f}%,约登指数 {fnum(youden, 3)};"
                   f"PPV {ppv * 100:.1f}%、NPV {npv * 100:.1f}%。"
                   + ("该诊断对「有病」的识别能力强。" if sens >= 0.8 else "灵敏度偏低,漏诊风险较高,可降低阳性判定阈值;")
                   + ("特异度良好,误诊率低。" if spec >= 0.8 else "特异度偏低,误诊风险较高。"))
    return res


register_model(ModelSpec(
    id="dxtest", name="诊断试验评价", category="医学统计",
    runner=_run_dxtest,
    desc="由金标准与诊断结果的四格表计算灵敏度、特异度、约登指数、PPV/NPV 与似然比。",
    tags=["诊断试验", "灵敏度", "特异度", "四格表"], demo_note="clinical",
    slots=[Slot("gold", "金标准(二分类)", types=(CAT, ORDINAL)),
           Slot("test", "诊断结果(二分类或定量)", types=(CAT, ORDINAL, QUANT))],
    options=[
        Option("threshold", "定量阈值(0=按分类)", "number", default=0, min=0),
        Option("gold_pos", "金标准阳性水平(空=自动)", "text", default=""),
        Option("test_pos", "诊断阳性水平(空=自动)", "text", default=""),
    ],
))
