"""时间序列与预测:ARIMA、灰色预测 GM(1,1)、指数平滑、移动平均。"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..core import charts as ch
from ..core.datasets import Dataset, as_numeric, get_series
from ..core.registry import CAT, ORDINAL, QUANT, ModelSpec, Option, Slot, register_model
from ..core.results import ModelError, Result, fnum, fp


def _series(ds: Dataset, sl: dict) -> tuple[pd.Series, list]:
    s = as_numeric(ds, sl["var"]).dropna()
    if len(s) < 10:
        raise ModelError("时间序列至少需要 10 个观测值")
    labels = None
    tcol = sl.get("time")
    if tcol:
        labels = [str(v) for v in ds.df[tcol].loc[s.index]]
    if labels is None:
        labels = [str(i + 1) for i in range(len(s))]
    return s, labels


# ================================================================ ARIMA
def _run_arima(ds: Dataset, sl: dict, op: dict) -> Result:
    from statsmodels.tsa.arima.model import ARIMA
    res = Result()
    s, labels = _series(ds, sl)
    steps = int(op["steps"])
    order_raw = str(op["order"]).replace("，", ",").strip()
    if order_raw and order_raw not in ("auto", "自动"):
        try:
            p, d, q = (int(v) for v in order_raw.split(","))
            order = (p, d, q)
        except ValueError:
            raise ModelError("阶数格式应为 p,d,q,如 1,0,0")
        aic = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(s.values, order=order).fit()
        bic = model.bic
        aic = model.aic
    else:
        best = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for d in (0, 1):
                for p in range(0, 3):
                    for q in range(0, 3):
                        if p == 0 and q == 0:
                            continue
                        try:
                            m = ARIMA(s.values, order=(p, d, q)).fit()
                            if best is None or m.aic < best[0].aic:
                                best = (m, (p, d, q))
                        except Exception:  # noqa: BLE001
                            continue
        if best is None:
            raise ModelError("ARIMA 网格搜索失败,请检查序列")
        model, order = best
        aic, bic = model.aic, model.bic
    resid = np.asarray(model.resid)
    lb = None
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(resid, lags=[min(10, max(2, len(resid) // 4))], return_df=True).iloc[0]
    except Exception:  # noqa: BLE001
        pass
    res.add_table("ARIMA 模型摘要", ["阶数(p,d,q)", "样本量", "AIC", "BIC", "Ljung-Box p(残差白噪声)"],
                  [[f"{order[0]},{order[1]},{order[2]}", len(s), fnum(aic, 2), fnum(bic, 2),
                    fp(float(lb["lb_pvalue"])) if lb is not None else "—"]],
                  note="残差 Ljung-Box 检验 p>0.05 说明残差近似白噪声,模型信息提取充分")
    rows = []
    for name, v in zip(model.param_names, np.asarray(model.params)):
        rows.append([name, fnum(v, 4)])
    if rows:
        res.add_table("模型参数", ["参数", "估计值"], rows)
    fc = model.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    frows = []
    ci = np.asarray(ci)
    for i in range(steps):
        frows.append([f"未来{i + 1}", fnum(mean[i], 3),
                      f"{fnum(ci[i, 0], 3)} ~ {fnum(ci[i, 1], 3)}"])
    res.add_table(f"未来 {steps} 期预测", ["期数", "预测值", "95%置信区间"], frows)
    hist = [round(float(v), 4) for v in s.values]
    tail = min(len(hist), 60)
    fc_labels = [f"+{i+1}" for i in range(steps)]
    series = {"历史值": hist[-tail:] + [None] * steps,
              "预测值": [None] * tail + [round(float(v), 4) for v in mean]}
    opt = ch.line_chart("ARIMA 预测", [str(l) for l in labels[-tail:]] + fc_labels, series,
                        x_name="时间", y_name=sl["var"], smooth=False)
    opt["series"].append({
        "name": "95%下限", "type": "line", "smooth": False, "showSymbol": False,
        "data": [None] * tail + [round(float(ci[i, 0]), 4) for i in range(steps)],
        "lineStyle": {"type": "dashed", "width": 1.4, "color": "#94a3b8"}, "tooltip": {"show": False},
    })
    opt["series"].append({
        "name": "95%上限", "type": "line", "smooth": False, "showSymbol": False,
        "data": [None] * tail + [round(float(ci[i, 1]), 4) for i in range(steps)],
        "lineStyle": {"type": "dashed", "width": 1.4, "color": "#94a3b8"}, "tooltip": {"show": False},
    })
    res.add_chart("ARIMA 历史与预测", opt)
    res.add_interp(
        f"自动/指定选择 ARIMA{order} 模型(AIC={fnum(aic, 2)}),"
        f"未来 {steps} 期预测值依次为:{', '.join(fnum(v, 2) for v in mean[:5])}{'…' if steps > 5 else ''}。"
        + ("残差白噪声检验通过,模型可用于预测。" if lb is None or float(lb["lb_pvalue"]) > 0.05 else "注意:残差白噪声检验未通过(p<0.05),建议尝试其他阶数。"))
    return res


register_model(ModelSpec(
    id="arima", name="ARIMA 时间序列预测", category="时间预测",
    runner=_run_arima,
    desc="差分整合移动平均自回归模型，AIC 网格自动定阶，输出预测值与 95% 置信带。",
    tags=["ARIMA", "时间序列", "预测"], demo_note="timeseries",
    slots=[Slot("var", "序列变量（定量，按行序）", types=(QUANT,)),
           Slot("time", "时间标签列（选填）", types=("any",), required=False)],
    options=[
        Option("order", "阶数 p，d，q（auto=自动）", "text", default="auto"),
        Option("steps", "预测期数", "number", default=6, min=1, max=36),
    ],
))


# ================================================================ 灰色预测 GM(1,1)
def _run_gm11(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    s, labels = _series(ds, sl)
    x0 = s.values.astype(float)
    if (x0 <= 0).any():
        raise ModelError("GM(1,1) 要求序列全部为正数")
    n = len(x0)
    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[1:] + x1[:-1])
    B = np.column_stack([-z1, np.ones(n - 1)])
    a, b = np.linalg.lstsq(B, x0[1:], rcond=None)[0]
    steps = int(op["steps"])
    f1 = (x0[0] - b / a) * np.exp(-a * np.arange(n + steps)) + b / a
    fitted_all = np.r_[x0[0], np.diff(f1)]
    fitted, forecast = fitted_all[:n], fitted_all[n:]
    # 后验差检验
    e = x0 - fitted[:n]
    C = e.std(ddof=0) / x0.std(ddof=0)
    eps = np.abs(e.mean())
    p_prob = float((np.abs(e - e.mean()) < 0.6745 * x0.std(ddof=0)).mean())
    grade = ("优" if C < 0.35 and p_prob >= 0.95 else
             "良" if C < 0.5 and p_prob >= 0.8 else
             "合格" if C < 0.65 and p_prob >= 0.7 else "不合格")
    frows = []
    for i in range(int(op["steps"])):
        frows.append([f"未来{i + 1}", fnum(forecast[i], 3)])
    res.add_table(f"GM(1,1) 未来 {op['steps']} 期预测", ["期数", "预测值"], frows)
    res.add_table("模型检验(后验差)", ["发展系数a", "灰色作用量b", "后验差比值C", "小误差概率p", "模型等级"],
                  [[fnum(a, 4), fnum(b, 4), fnum(C, 4), fnum(p_prob, 4), grade]],
                  note="C<0.35 且 p>0.95 为「优」;C>0.65 不建议使用该模型")
    allv = np.r_[fitted, forecast]
    tail = min(len(x0), 60)
    series = {"实际值": [round(float(v), 4) for v in x0[-tail:]] + [None] * int(op["steps"]),
              "拟合/预测": [round(float(v), 4) for v in allv[-tail:]]}
    res.add_chart("GM(1,1) 拟合与预测",
                  ch.line_chart("灰色预测 GM(1,1)", [str(l) for l in labels[-tail:]] + [f"+{i+1}" for i in range(int(op["steps"]))],
                                series, x_name="时间", y_name=sl["var"], smooth=False))
    res.add_interp(
        f"GM(1,1) 模型等级为 **{grade}**(C={fnum(C, 3)}, p={fnum(p_prob, 3)}),"
        f"未来 {op['steps']} 期预测值:{', '.join(fnum(v, 2) for v in forecast[:5])}{'…' if op['steps'] > 5 else ''}。"
        "适用于小样本、指数趋势明显的单序列预测。")
    return res


register_model(ModelSpec(
    id="gm11", name="灰色预测 GM(1,1)", category="时间预测",
    runner=_run_gm11,
    desc="小样本单序列预测，后验差检验给出模型等级，适合指数增长型数据。",
    tags=["灰色预测", "GM11", "小样本"], demo_note="timeseries",
    slots=[Slot("var", "序列变量（定量，>0）", types=(QUANT,)),
           Slot("time", "时间标签列（选填）", types=("any",), required=False)],
    options=[Option("steps", "预测期数", "number", default=6, min=1, max=36)],
))


# ================================================================ 指数平滑
def _run_exp_smooth(ds: Dataset, sl: dict, op: dict) -> Result:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    res = Result()
    s, labels = _series(ds, sl)
    steps = int(op["steps"])
    kind = op["kind"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "simple":
            model = SimpleExpSmoothing(s.values, initialization_method="estimated").fit()
        elif kind == "holt":
            model = ExponentialSmoothing(s.values, trend="add", initialization_method="estimated").fit()
        else:  # winters 季节
            m = int(op["seasonal_periods"])
            if m < 2 or len(s) < 2 * m:
                raise ModelError(f"Holt-Winters 需要序列长度 ≥ 2×周期(当前 {len(s)},周期 {m})")
            model = ExponentialSmoothing(s.values, trend="add", seasonal="add",
                                         seasonal_periods=m, initialization_method="estimated").fit()
    fc = model.forecast(steps)
    name = {"simple": "一次指数平滑", "holt": "Holt 二次指数平滑(含趋势)", "winters": "Holt-Winters 三次指数平滑(含季节)"}[kind]
    res.add_table("模型摘要", ["方法", "样本量", "SSE", "AIC"],
                  [[name, len(s), fnum(model.sse, 3), fnum(model.aic, 2)]])
    frows = [[f"未来{i + 1}", fnum(fc[i], 3)] for i in range(steps)]
    res.add_table(f"未来 {steps} 期预测", ["期数", "预测值"], frows)
    tail = min(len(s), 60)
    series = {"实际值": [round(float(v), 4) for v in s.values[-tail:]] + [None] * steps,
              "预测值": [None] * tail + [round(float(v), 4) for v in np.asarray(fc)]}
    res.add_chart("指数平滑预测", ch.line_chart(name, [str(l) for l in labels[-tail:]] + [f"+{i+1}" for i in range(steps)],
                 series, x_name="时间", y_name=sl["var"], smooth=False))
    res.add_interp(f"{name} 完成,未来 {steps} 期预测值:{', '.join(fnum(v, 2) for v in np.asarray(fc)[:5])}{'…' if steps > 5 else ''};SSE={fnum(model.sse, 2)}。")
    return res


register_model(ModelSpec(
    id="exp_smooth", name="指数平滑预测", category="时间预测",
    runner=_run_exp_smooth,
    desc="一次/Holt/Holt-Winters 三种指数平滑，支持趋势与季节成分。",
    tags=["指数平滑", "季节", "预测"], demo_note="timeseries",
    slots=[Slot("var", "序列变量（定量）", types=(QUANT,)),
           Slot("time", "时间标签列（选填）", types=("any",), required=False)],
    options=[
        Option("kind", "平滑方法", "select", default="holt",
               choices=[{"value": "simple", "label": "一次（无趋势）"}, {"value": "holt", "label": "二次 Holt（线性趋势）"},
                        {"value": "winters", "label": "三次 Holt-Winters（趋势+季节）"}]),
        Option("seasonal_periods", "季节周期数（Holt-Winters）", "number", default=12, min=2, max=52),
        Option("steps", "预测期数", "number", default=6, min=1, max=36),
    ],
))


# ================================================================ 移动平均
def _run_moving_avg(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    s, labels = _series(ds, sl)
    k = int(op["window"])
    if k < 2 or k >= len(s):
        raise ModelError(f"窗口期需在 2 与 {len(s) - 1} 之间")
    ma = s.rolling(k).mean()
    next_val = float(s.tail(k).mean())
    frows = [[f"未来{i + 1}", fnum(next_val, 3)] for i in range(int(op["steps"]))]
    res.add_table(f"移动平均预测(窗口={k})", ["期数", "预测值"], frows,
                  note="简单移动平均:下一期 = 最近 k 期均值;未来多期预测值为常数")
    tail = min(len(s), 80)
    series = {"实际值": [round(float(v), 4) for v in s.values[-tail:]],
              f"{k}期移动平均": [None if pd.isna(v) else round(float(v), 4) for v in ma.values[-tail:]]
              + [round(next_val, 4)] * int(op["steps"])}
    res.add_chart("移动平均", ch.line_chart(f"{k} 期移动平均", [str(l) for l in labels[-tail:]] + [f"+{i+1}" for i in range(int(op["steps"]))],
                 series, x_name="时间", y_name=sl["var"], smooth=False))
    res.add_interp(f"以最近 {k} 期均值作为预测,下一期预测值 **{fnum(next_val, 3)}**;移动平均能平滑随机波动、凸显趋势。")
    return res


register_model(ModelSpec(
    id="moving_avg", name="移动平均预测", category="时间预测",
    runner=_run_moving_avg,
    desc="简单移动平均平滑序列并预测下一期，快速消除随机波动。",
    tags=["移动平均", "平滑"], demo_note="timeseries",
    slots=[Slot("var", "序列变量（定量）", types=(QUANT,)),
           Slot("time", "时间标签列（选填）", types=("any",), required=False)],
    options=[
        Option("window", "窗口期 k", "number", default=3, min=2, max=60),
        Option("steps", "预测期数", "number", default=3, min=1, max=12),
    ],
))


# ================================================================ 马尔科夫预测
def _run_markov(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    s = get_series(ds, sl["var"]).dropna()
    if len(s) < 12:
        raise ModelError("马尔科夫预测至少需要 12 个连续观测")
    vals = s.tolist()
    uniq = sorted(set(str(v) for v in vals), key=str)
    nbins = int(op["bins"] or 0)
    if nbins == 0 and len(uniq) > 10:
        nbins = 5
        res_note = f"序列取值 {len(uniq)} 个,已自动离散为 {nbins} 个区间"
    else:
        res_note = ""
    if nbins > 0:
        snum = pd.to_numeric(s, errors="coerce").dropna()
        if len(snum) < 12:
            raise ModelError("数值化后观测不足 12 个")
        try:
            cuts = pd.qcut(snum, nbins, duplicates="drop")
            state = cuts.astype(str).tolist()
            uniq = sorted(set(state), key=str)
        except Exception:  # noqa: BLE001
            raise ModelError("无法将序列离散化,请检查数据")
        raw = snum.tolist()
    else:
        state = [str(v) for v in vals]
    n = len(state)
    states = uniq
    k = len(states)
    if k < 2:
        raise ModelError("状态数不足 2,无法建立转移矩阵")
    idx = {s2: i for i, s2 in enumerate(states)}
    C = np.zeros((k, k))
    for a, b in zip(state[:-1], state[1:]):
        C[idx[a], idx[b]] += 1
    P = C / C.sum(axis=1, keepdims=True).clip(min=1)
    res.add_table("状态转移频数矩阵", ["当前\下一"] + states,
                  [[states[i]] + [int(v) for v in C[i]] for i in range(k)],
                  note=f"有效转移 {n - 1} 次" + (";" + res_note if res_note else ""))
    res.add_table("状态转移概率矩阵", ["当前\下一"] + states,
                  [[states[i]] + [f"{v:.2%}" for v in P[i]] for i in range(k)],
                  note="行和为 1,表示当前状态下期的去向概率")
    # 预测:初始分布 = 最后一个观测所在状态
    cur = state[-1]
    steps = min(int(op["steps"]), 12)
    dist = np.zeros(k)
    dist[idx[cur]] = 1.0
    all_d = [dist @ np.linalg.matrix_power(P, t) for t in range(1, steps + 1)]
    tbl_rows = []
    for t, d in enumerate(all_d, 1):
        best = states[int(np.argmax(d))]
        tbl_rows.append([f"+{t}", cur if t == 1 else "", *[f"{v:.2%}" for v in d], best])
    res.add_table(f"未来 {steps} 期状态概率分布", ["期数", "当前状态"] + states + ["最可能状态"],
                  tbl_rows, note="基于当前状态「" + str(cur) + "」的马尔科夫链演化")
    res.add_chart("状态转移概率热力图",
                  ch.heatmap_chart("一步转移概率", states, states, P.tolist(), x_name="下一状态",
                                   fmt="{c}", v_min=0, v_max=1))
    res.add_chart("未来状态概率演化",
                  ch.line_chart("各状态概率随期数变化", [f"+{t}" for t in range(1, steps + 1)],
                                {s2: [round(float(all_d[t][i]), 4) for t in range(steps)] for i, s2 in enumerate(states)},
                                x_name="未来期数", y_name="概率", y_min=0, y_max=1))
    d1 = all_d[0]
    res.add_interp(f"当前状态为「{cur}」。下一期最可能处于「{states[int(np.argmax(d1))]}」(概率 {d1.max():.1%})。"
                   f"马尔科夫链假设状态只依赖上一期(无后效性),长期会趋向平稳分布。")
    return res


register_model(ModelSpec(
    id="markov", name="马尔科夫预测", category="时间预测",
    runner=_run_markov,
    desc="由状态转移概率矩阵预测未来各状态分布，适合市场份额、等级流转、客户状态迁移等。",
    tags=["马尔科夫", "转移矩阵", "状态预测"], demo_note="timeseries",
    slots=[Slot("var", "状态序列（定量或分类）", types=(QUANT, CAT, ORDINAL))],
    options=[
        Option("steps", "预测期数", "number", default=3, min=1, max=12),
        Option("bins", "离散化区间数（0=按类别）", "number", default=0, min=0, max=10),
    ],
))


# ================================================================ 灰色 Verhulst
def _run_verhulst(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    s, labels = _series(ds, sl)
    x0 = s.values.astype(float)
    if (x0 <= 0).any():
        raise ModelError("灰色 Verhulst 要求序列全部为正数(饱和增长型数据)")
    n = len(x0)
    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[1:] + x1[:-1])
    B = np.column_stack([z1, -z1 ** 2])
    a, b = np.linalg.lstsq(B, x0[1:], rcond=None)[0]
    if b == 0 or a <= 0:
        raise ModelError("数据不满足 Verhulst 饱和增长形态(a 需为正、b 非零)")
    cap = a / b  # 饱和量(承载量)
    steps = int(op["steps"])
    tt = np.arange(n + steps)
    c = cap / x0[0] - 1
    f1 = cap / (1 + c * np.exp(-a * tt))
    fitted = f1[:n]
    fitted_all = np.r_[x0[0], np.diff(f1[:n + steps])]
    fitted_v, forecast = fitted_all[:n], fitted_all[n:]
    e = x0 - fitted_v
    denom = x0.std(ddof=0)
    C = e.std(ddof=0) / denom if denom else np.inf
    p_prob = float((np.abs(e - e.mean()) < 0.6745 * denom).mean()) if denom else np.nan
    grade = ("优" if C < 0.35 and p_prob >= 0.95 else
             "良" if C < 0.5 and p_prob >= 0.8 else
             "合格" if C < 0.65 and p_prob >= 0.7 else "不合格")
    res.add_table(f"Verhulst 未来 {steps} 期预测", ["期数", "预测值"],
                  [[f"未来{i + 1}", fnum(forecast[i], 3)] for i in range(steps)])
    res.add_table("模型检验(后验差)", ["发展系数a", "灰色作用量b", "饱和量a/b", "后验差比值C", "小误差概率p", "模型等级"],
                  [[fnum(a, 4), fnum(b, 4), fnum(cap, 3), fnum(C, 4), fnum(p_prob, 4), grade]],
                  note="Verhulst 适合 S 型饱和增长序列,饱和量 ≈ " + fnum(cap, 2))
    tail = min(len(x0), 60)
    series = {"实际值": [round(float(v), 4) for v in x0[-tail:]] + [None] * steps,
              "拟合/预测": [round(float(v), 4) for v in fitted_all[-tail:]]}
    res.add_chart("灰色 Verhulst 拟合与预测",
                  ch.line_chart("灰色 Verhulst 预测", [str(l) for l in labels[-tail:]] + [f"+{i + 1}" for i in range(steps)],
                                series, x_name="时间", y_name=sl["var"], smooth=False))
    res.add_interp(f"Verhulst 模型等级 **{grade}**(C={fnum(C, 3)}, p={fnum(p_prob, 3)}),估计饱和量为 **{fnum(cap, 2)}**;"
                   f"未来 {steps} 期预测:{', '.join(fnum(v, 2) for v in forecast[:5])}{'…' if steps > 5 else ''}。"
                   "模型呈 S 型曲线,适合接近饱和上限的序列(如产品渗透率、种群/市场容量)。")
    return res


register_model(ModelSpec(
    id="verhulst", name="灰色 Verhulst 预测", category="时间预测",
    runner=_run_verhulst,
    desc="S 型饱和增长序列预测（GM（1,1） 的饱和推广），自动估计增长上限，附后验差检验。",
    tags=["Verhulst", "灰色预测", "饱和", "S曲线"], demo_note="timeseries",
    slots=[Slot("var", "序列变量（定量，>0）", types=(QUANT,)),
           Slot("time", "时间标签列（选填）", types=("any",), required=False)],
    options=[Option("steps", "预测期数", "number", default=6, min=1, max=36)],
))
