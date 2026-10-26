"""回归建模:多元线性回归(含逐步)、分层回归、Logistic 族、泊松、岭/Lasso、曲线估计。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import charts as ch
from ..core.datasets import Dataset, as_numeric, get_series
from ..core.registry import CAT, ORDINAL, QUANT, ModelSpec, Option, Slot, register_model
from ..core.results import ModelError, Result, fnum, fp, judge_p

ALPHA_OPT = Option("alpha", "显著性水平 α", "select", default=0.05,
                   choices=[{"value": 0.1, "label": "0.1"}, {"value": 0.05, "label": "0.05"}, {"value": 0.01, "label": "0.01"}])


def _design(ds: Dataset, xs: list[str], df: pd.DataFrame | None = None) -> pd.DataFrame:
    """自变量设计矩阵:定量标准化不处理;定类自动哑变量(以首水平为参照)。"""
    out = pd.DataFrame(index=range(len(df) if df is not None else len(ds.df)))
    parts = []
    for x in xs:
        s = (ds.df[x] if df is None else df[x])
        if x in [c["name"] for c in ds.columns if c["dtype"] in ("cat", "ordinal")]:
            d = pd.get_dummies(s.astype("category"), prefix=x, drop_first=True, dtype=float)
            parts.append(d)
        else:
            parts.append(pd.to_numeric(s, errors="coerce").rename(x))
    out = pd.concat(parts, axis=1)
    return out


def _clean_xy(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.concat([X, y.rename("__y__")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < X.shape[1] + 5:
        raise ModelError(f"有效样本不足(当前 {len(data)} 行,变量数 {X.shape[1]}),请减少变量或检查缺失")
    return data.drop(columns="__y__"), data["__y__"]


def _durbin_watson(resid: np.ndarray) -> float:
    return float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2))


def _vif(X: pd.DataFrame) -> list[tuple[str, float]]:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xc = sm_add_const(X)
    return [(X.columns[i - 1], variance_inflation_factor(Xc.values, i)) for i in range(1, Xc.shape[1])]


def sm_add_const(X: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.api as sm
    return sm.add_constant(X, has_constant="add")


# ================================================================ 多元线性回归
def _run_linear(ds: Dataset, sl: dict, op: dict) -> Result:
    import statsmodels.api as sm
    res = Result()
    yname = sl["y"]
    xs = list(sl["xs"])
    X_raw = _design(ds, xs)
    y = as_numeric(ds, yname)
    X, y = _clean_xy(X_raw, y)
    if op["method"] in ("stepwise", "both"):
        X, dropped = _stepwise(X, y, op["alpha"])
    else:
        X = X.loc[:, X.std(ddof=1) > 0]
    if X.shape[1] == 0:
        raise ModelError("没有可用的自变量(方差为 0)")
    model = sm.OLS(y, sm_add_const(X)).fit()
    # 系数表
    rows = []
    for name in model.params.index:
        ci = model.conf_int(alpha=op["alpha"]).loc[name]
        rows.append([name, fnum(model.params[name], 4), fnum(model.bse[name], 4),
                     fnum(model.tvalues[name], 3), fp(model.pvalues[name]),
                     f"{fnum(ci[0], 4)} ~ {fnum(ci[1], 4)}",
                     "常数项" if name == "const" else ("显著" if model.pvalues[name] < op["alpha"] else "不显著")])
    res.add_table("回归系数与显著性检验", ["变量", "系数B", "标准误", "t", "p", f"{int((1-op['alpha'])*100)}%CI", "结论"], rows,
                  note="因变量:" + yname + (f";哑变量以首水平为参照" if any(("_" in n and n != "const") for n in model.params.index) else ""))
    res.add_table("模型摘要", ["R²", "调整R²", "F", "F检验p", "Durbin-Watson"],
                  [[fnum(model.rsquared, 4), fnum(model.rsquared_adj, 4),
                    fnum(model.fvalue, 3), fp(model.f_pvalue), fnum(_durbin_watson(model.resid), 3)]],
                  note=f"样本量 N={int(model.nobs)}")
    try:
        vif_rows = [[n, fnum(v, 3), "共线性严重" if v >= 10 else "可接受" if v < 5 else "需关注"] for n, v in _vif(X)]
        res.add_table("共线性诊断(VIF)", ["变量", "VIF", "判断"], vif_rows, note="VIF≥10 存在严重多重共线性")
    except Exception:  # noqa: BLE001
        pass
    resid = np.asarray(model.resid)
    fitted = np.asarray(model.fittedvalues)
    rng = np.linspace(fitted.min(), fitted.max(), 50)
    res.add_chart("残差散点图(残差 vs 拟合值)",
                  ch.scatter_chart("残差图", {"残差": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(fitted, resid)]},
                                   x_name="拟合值", y_name="残差", trend=[[float(rng[0]), 0], [float(rng[-1]), 0]], trend_name="零线"))
    from ..core.charts import qq_chart
    res.add_chart("残差正态 Q-Q 图", qq_chart("残差 Q-Q 图", resid))
    sig_names = [r[0] for r in rows if r[6] == "显著" and r[0] != "const"]
    res.add_interp(
        f"模型 F={fnum(model.fvalue, 3)}(p={fp(model.f_pvalue)}),整体{'显著' if model.f_pvalue < op['alpha'] else '不显著'};"
        f"R²={fnum(model.rsquared, 3)},说明自变量能解释因变量 **{fnum(model.rsquared * 100, 1)}%** 的变异。"
        + (f"显著的自变量:{'、'.join(sig_names)}。" if sig_names else "没有自变量达到显著水平。")
        + (f"Durbin-Watson={fnum(_durbin_watson(model.resid), 2)},残差独立性{'良好' if 1.5 < _durbin_watson(model.resid) < 2.5 else '欠佳'}。"))
    res.add_note("回归方程:" + " ".join([f"{yname} = {fnum(model.params['const'], 3)}"] +
                [f"{'+' if model.params[n] >= 0 else '-'} {fnum(abs(model.params[n]), 3)}×{n}" for n in model.params.index if n != "const"]))
    return res


def _stepwise(X: pd.DataFrame, y: pd.Series, alpha: float) -> tuple[pd.DataFrame, list[str]]:
    """双向逐步回归(F 检验进入/剔除)"""
    import statsmodels.api as sm
    remain = list(X.columns)
    selected: list[str] = []
    dropped: list[str] = []
    while True:
        changed = False
        best_p, best_c = 1.1, None
        for c in remain:
            model = sm.OLS(y, sm_add_const(X[selected + [c]])).fit()
            pv = model.pvalues[c]
            if pv < best_p:
                best_p, best_c = pv, c
        if best_c and best_p < alpha:
            selected.append(best_c)
            remain.remove(best_c)
            changed = True
            # 剔除已选变量中 p 超标者(常数项除外)
            model = sm.OLS(y, sm_add_const(X[selected])).fit()
            worst = max(((model.pvalues[c], c) for c in selected), default=(0, None))
            if worst[0] > alpha and worst[1] and len(selected) > 1:
                selected.remove(worst[1])
                remain.append(worst[1])
                dropped.append(worst[1])
        if not changed or not remain:
            break
    return X[selected], dropped


register_model(ModelSpec(
    id="linear", name="多元线性回归", category="回归建模",
    runner=_run_linear,
    desc="OLS 多元线性回归:系数检验、VIF 共线性诊断、DW 检验、残差图;支持向前/向后/双向逐步筛选。",
    tags=["线性回归", "逐步回归", "OLS"], demo_note="survey",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,)),
           Slot("xs", "自变量(可含定类,自动哑变量)", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[
        Option("method", "变量进入方式", "select", default="enter",
               choices=[{"value": "enter", "label": "全部进入(Enter)"}, {"value": "stepwise", "label": "双向逐步(Stepwise)"},
                        {"value": "both", "label": "向前选择"}]),
        ALPHA_OPT,
    ],
))


# ================================================================ 分层回归
def _run_hierarchical(ds: Dataset, sl: dict, op: dict) -> Result:
    import statsmodels.api as sm
    res = Result()
    yname = sl["y"]
    xs = list(sl["xs"])
    counts_raw = str(op["blocks"]).strip()
    try:
        counts = [int(v) for v in counts_raw.split(",") if v.strip()]
    except ValueError:
        raise ModelError("分层格式应为逗号分隔的数字,如 1,2,3")
    if sum(counts) != len(xs):
        raise ModelError(f"每层变量数之和({sum(counts)})必须等于自变量个数({len(xs)})")
    X_all = _design(ds, xs)
    y = as_numeric(ds, yname)
    X_all, y = _clean_xy(X_all, y)
    # 将哑变量归属其原始变量层:重算每层列名
    col_layer: dict[str, int] = {}
    cur = 0
    for i, x in enumerate(xs):
        layer_cols = [c for c in X_all.columns if c == x or c.startswith(x + "_")]
        for c in layer_cols:
            col_layer[c] = i
    # 排除被 _clean_xy 丢掉的列
    rows, f_rows = [], []
    prev_r2 = 0.0
    prev_model = None
    for li in range(len(counts)):
        keep = [c for c in X_all.columns if col_layer.get(c, 99) <= li]
        X = X_all[keep]
        model = sm.OLS(y, sm_add_const(X)).fit()
        dr2 = model.rsquared - prev_r2
        rows.append([f"第{li + 1}层", "、".join(xs[cur:cur + counts[li]]), fnum(model.rsquared, 4),
                     fnum(dr2, 4), fnum(model.rsquared_adj, 4), fnum(model.fvalue, 3), fp(model.f_pvalue)])
        if prev_model is not None:
            try:
                ft, pv = _nested_f(prev_model, model)
                f_rows.append([f"第{li}层 → 第{li + 1}层", fnum(ft, 3), fp(pv), judge_p(pv, op["alpha"])])
            except Exception:  # noqa: BLE001
                pass
        prev_r2 = model.rsquared
        prev_model = model
        cur += counts[li]
    res.add_table("分层回归模型汇总", ["模型层", "进入变量", "R²", "ΔR²", "调整R²", "F", "p"], rows)
    if f_rows:
        res.add_table("层间增量 F 检验(ΔR² 是否显著)", ["比较", "ΔF", "p", "显著?"], f_rows)
    last = prev_model
    rows2 = []
    for name in last.params.index:
        rows2.append([name, fnum(last.params[name], 4), fnum(last.bse[name], 4), fnum(last.tvalues[name], 3), fp(last.pvalues[name])])
    res.add_table("最终模型系数", ["变量", "B", "标准误", "t", "p"], rows2)
    res.add_interp(
        f"共构建 {len(counts)} 层模型。每加入一层变量,ΔR² 表示该层带来的解释力增量;"
        + (f"最后一层后模型 R²={fnum(last.rsquared, 3)},显著自变量:{'、'.join(n for n in last.params.index if n != 'const' and last.pvalues[n] < op['alpha']) or '无'}。" if True else ""))
    return res


def _nested_f(reduced, full):
    """嵌套模型 F 检验"""
    import numpy as np
    rss_r = float(np.sum(reduced.resid ** 2))
    rss_f = float(np.sum(full.resid ** 2))
    df1 = full.df_resid - reduced.df_resid
    df2 = int(full.df_resid)
    f = ((rss_r - rss_f) / df1) / (rss_f / df2)
    from scipy import stats as sps
    return f, sps.f.sf(f, df1, df2)


register_model(ModelSpec(
    id="hierarchical", name="分层(层次)回归", category="回归建模",
    runner=_run_hierarchical,
    desc="按层依次加入自变量,考察每层新增变量对 R² 的增量贡献(ΔR² 及其 F 检验)。",
    tags=["分层回归", "ΔR²", "控制变量"], demo_note="survey",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,)),
           Slot("xs", "自变量(按进入顺序)", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[
        Option("blocks", "每层变量数(逗号分隔)", "text", default="1,2",
               tip="例如自变量有 3 个,填 1,2 表示:第1层放入第1个变量,第2层放入后两个"),
        ALPHA_OPT,
    ],
))


# ================================================================ Logistic 族
def _logit_common_tables(res: Result, model, alpha: float, label: str) -> None:
    params = model.params
    conf = model.conf_int(alpha=alpha)
    rows = []
    for name in params.index:
        b = float(params[name])
        se = float(model.bse[name])
        z = float(model.tvalues[name])
        p = float(model.pvalues[name])
        ci = conf.loc[name]
        rows.append([name, fnum(b, 4), fnum(se, 4), fnum(z, 3), fp(p),
                     f"{fnum(np.exp(b), 4)}", f"{fnum(np.exp(ci[0]), 4)} ~ {fnum(np.exp(ci[1]), 4)}",
                     "显著" if p < alpha else "不显著"])
    res.add_table(f"{label}回归系数", ["变量", "回归系数B", "标准误", "Z", "p", "OR(优势比)", "OR 95%CI", "结论"], rows,
                  note="OR>1 表示该变量增加时事件发生概率上升")


def _run_logit_binary(ds: Dataset, sl: dict, op: dict) -> Result:
    import statsmodels.api as sm
    res = Result()
    yname = sl["y"]
    yraw = get_series(ds, yname)
    levels = yraw.dropna().unique().tolist()
    if len(levels) != 2:
        raise ModelError(f"因变量需为二分类(当前 {len(levels)} 个水平)")
    lv01 = sorted(levels, key=str)
    y = (yraw == lv01[1]).astype(float)
    X_raw = _design(ds, list(sl["xs"]))
    X, yy = _clean_xy(X_raw, y)
    model = sm.Logit(yy, sm_add_const(X)).fit(disp=0)
    _logit_common_tables(res, model, op["alpha"], "二元 Logistic")
    pred = (model.predict() >= 0.5).astype(int)
    acc = float((pred == yy.values).mean())
    llnull = sm.Logit(yy, np.ones((len(yy), 1))).fit(disp=0).llf
    mcfadden = 1 - model.llf / llnull
    res.add_table("模型摘要", ["样本量N", "-2对数似然", "McFadden R²", "预测准确率"],
                  [[int(model.nobs), fnum(-2 * model.llf, 3), fnum(mcfadden, 4), f"{acc * 100:.2f}%"]])
    cm = pd.crosstab(yy, pred, rownames=["实际"], colnames=["预测"])
    res.add_table("混淆矩阵(阈值 0.5)", ["实际\\预测", "预测=0", "预测=1"],
                  [[str(idx), int(row.get(0, 0)), int(row.get(1, 0))] for idx, row in cm.iterrows()])
    sig = [n for n in model.params.index if n != "const" and model.pvalues[n] < op["alpha"]]
    res.add_interp(
        f"以「{lv01[1]}」为事件发生(1),共 {int(yy.sum())} 例。模型 McFadden R²={fnum(mcfadden, 3)},预测准确率 {acc * 100:.1f}%。"
        + (f"显著影响变量:{'、'.join(sig)}。" if sig else "暂无变量达到显著水平。"))
    return res


register_model(ModelSpec(
    id="logit_binary", name="二元 Logistic 回归", category="回归建模",
    runner=_run_logit_binary,
    desc="因变量为二分类时的回归建模,输出 OR 优势比、显著性、混淆矩阵与预测准确率。",
    tags=["Logistic", "OR值", "分类"], demo_note="survey",
    slots=[Slot("y", "因变量(二分类)", types=(CAT, ORDINAL)),
           Slot("xs", "自变量(可含定类)", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[ALPHA_OPT],
))


def _run_logit_multi(ds: Dataset, sl: dict, op: dict) -> Result:
    import statsmodels.api as sm
    res = Result()
    yname = sl["y"]
    yraw = get_series(ds, yname).astype(str)
    levels = sorted(yraw.dropna().unique(), key=str)
    if len(levels) < 3:
        raise ModelError("多分类 Logistic 至少需要 3 个类别(2 类请用二元 Logistic)")
    y = pd.Categorical(yraw, categories=levels).codes
    X_raw = _design(ds, list(sl["xs"]))
    X, yy = _clean_xy(X_raw, pd.Series(y))
    model = sm.MNLogit(yy, sm_add_const(X)).fit(disp=0)
    base = levels[0]
    for i in range(1, len(levels)):
        col = i - 1
        rows = []
        for name in model.params.index:
            b = float(model.params.iloc[:, col].loc[name])
            p = float(model.pvalues.iloc[:, col].loc[name])
            rows.append([name, fnum(b, 4), fnum(model.bse.iloc[:, col].loc[name], 4), fp(p), fnum(np.exp(b), 4),
                         "显著" if p < op["alpha"] else "不显著"])
        res.add_table(f"Logit({levels[i]} / 参照:{base})", ["变量", "B", "标准误", "p", "OR", "结论"], rows)
    llnull = sm.MNLogit(yy, np.ones((len(yy), 1))).fit(disp=0).llf
    mcfadden = 1 - model.llf / llnull
    pred = np.asarray(model.predict()).argmax(axis=1)
    acc = float((pred == yy.values).mean())
    res.add_table("模型摘要", ["样本量N", "-2对数似然", "McFadden R²", "预测准确率", "类别数"],
                  [[int(model.nobs), fnum(-2 * model.llf, 3), fnum(mcfadden, 4), f"{acc * 100:.2f}%", len(levels)]])
    res.add_interp(
        f"多分类 Logistic 以「{base}」为参照组,共 {len(levels) - 1} 组对比方程。"
        f"McFadden R²={fnum(mcfadden, 3)},预测准确率 {acc * 100:.1f}%,各变量效应解读见上表。")
    return res


register_model(ModelSpec(
    id="logit_multi", name="多分类 Logistic 回归", category="回归建模",
    runner=_run_logit_multi,
    desc="因变量为无序多分类(≥3 类)时的 Logistic 回归,以首类别为参照输出各组 OR。",
    tags=["Logistic", "多分类"], demo_note="survey",
    slots=[Slot("y", "因变量(≥3类)", types=(CAT, ORDINAL)),
           Slot("xs", "自变量(可含定类)", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[ALPHA_OPT],
))


def _run_logit_ordinal(ds: Dataset, sl: dict, op: dict) -> Result:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    res = Result()
    yname = sl["y"]
    yraw = get_series(ds, yname).dropna().astype(str)
    try:
        key = pd.to_numeric(yraw)
        levels = [str(v) for v in sorted(key.unique(), key=float)]
    except (TypeError, ValueError):
        levels = sorted(yraw.unique())
    y_cat = pd.Categorical(yraw, categories=levels, ordered=True)
    X_raw = _design(ds, list(sl["xs"]))
    data = pd.concat([X_raw, pd.Series(y_cat.astype(str), name="__y__")], axis=1).dropna()
    X, yv = data.drop(columns="__y__"), data["__y__"]
    y_codes = pd.Series(yv.map({l: i for i, l in enumerate(levels)}), index=yv.index).astype(float)
    model = OrderedModel(y_codes, X, distr="logit").fit(disp=0, method="bfgs", maxiter=200)
    rows = []
    for name in model.params.index:
        if name in [c for c in levels[:-1]]:
            continue
        b = float(model.params[name])
        p = float(model.pvalues[name])
        rows.append([name, fnum(b, 4), fnum(model.bse[name], 4), fp(p), fnum(np.exp(b), 4),
                     "阈值" if name.startswith(tuple(levels[:-1])) or "/" in name else ("显著" if p < op["alpha"] else "不显著")])
    res.add_table("有序 Logistic 回归参数", ["变量/阈值", "系数B", "标准误", "p", "OR", "结论"], rows,
                  note=f"因变量水平(由低到高):{' < '.join(levels)};阈值对应相邻水平的累积分界")
    pred = np.asarray(model.model.predict(model.params, X)).argmax(axis=1)
    y_codes = yv.map({l: i for i, l in enumerate(levels)}).astype(float).values
    acc = float((pred == y_codes).mean())
    res.add_table("模型摘要", ["样本量N", "对数似然", "伪R²(比例优势近似)", "预测准确率"],
                  [[int(len(yv)), fnum(model.llf, 3), "—", f"{acc * 100:.2f}%"]])
    sig = [r[0] for r in rows if r[5] == "显著"]
    res.add_interp(
        f"有序 Logistic(比例优势模型)以「{levels[0]}」为最低水平。预测准确率 {acc * 100:.1f}%。"
        + (f"显著自变量:{'、'.join(sig)};OR>1 表示该变量取值越大,因变量落在更高等级的概率越大。" if sig else ""))
    return res


register_model(ModelSpec(
    id="logit_ordinal", name="有序 Logistic 回归", category="回归建模",
    runner=_run_logit_ordinal,
    desc="因变量为有序等级(如满意程度:低→高)的比例优势模型,输出累加 OR。",
    tags=["有序Logistic", "比例优势", "等级"], demo_note="survey",
    slots=[Slot("y", "因变量(有序等级)", types=(ORDINAL, CAT)),
           Slot("xs", "自变量(可含定类)", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[ALPHA_OPT],
))


# ================================================================ 泊松回归
def _run_poisson(ds: Dataset, sl: dict, op: dict) -> Result:
    import statsmodels.api as sm
    res = Result()
    y = as_numeric(ds, sl["y"])
    X_raw = _design(ds, list(sl["xs"]))
    X, yy = _clean_xy(X_raw, y)
    if (yy < 0).any():
        raise ModelError("泊松回归要求因变量为非负计数")
    model = sm.GLM(yy, sm_add_const(X), family=sm.families.Poisson()).fit()
    rows = []
    for name in model.params.index:
        p = float(model.pvalues[name])
        rows.append([name, fnum(model.params[name], 4), fnum(model.bse[name], 4), fnum(model.tvalues[name], 3), fp(p),
                     fnum(np.exp(model.params[name]), 4), "显著" if p < op["alpha"] else "不显著"])
    res.add_table("泊松回归系数", ["变量", "B", "标准误", "Z", "p", "exp(B)(发生率比IRR)", "结论"], rows,
                  note="exp(B)>1 表示该变量每增加 1 单位,事件发生次数期望变为原来的 exp(B) 倍")
    deviance_ratio = 1 - model.deviance / model.null_deviance if model.null_deviance else np.nan
    res.add_table("模型摘要", ["样本量N", "偏差Deviance", "Null偏差", "伪R²", "AIC"],
                  [[int(model.nobs), fnum(model.deviance, 3), fnum(model.null_deviance, 3),
                    fnum(deviance_ratio, 4), fnum(model.aic, 2)]])
    res.add_interp(
        f"泊松 GLM 拟合完成,伪 R²={fnum(deviance_ratio, 3)}。"
        + (f"显著变量:{'、'.join(r[0] for r in rows if r[6] == '显著' and r[0] != 'const')}。" if any(r[6] == "显著" and r[0] != "const" for r in rows) else ""))
    return res


register_model(ModelSpec(
    id="poisson", name="泊松回归", category="回归建模",
    runner=_run_poisson,
    desc="因变量为计数数据(次数/频次)时的广义线性模型,输出 IRR 发生率比。",
    tags=["泊松", "计数", "GLM"], demo_note="survey",
    slots=[Slot("y", "因变量(非负计数)", types=(QUANT,)),
           Slot("xs", "自变量(可含定类)", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[ALPHA_OPT],
))


# ================================================================ 岭回归 / Lasso
def _run_ridge(ds: Dataset, sl: dict, op: dict) -> Result:
    from sklearn.linear_model import Ridge, RidgeCV
    res = Result()
    y = as_numeric(ds, sl["y"])
    X_raw = _design(ds, list(sl["xs"]))
    X, yy = _clean_xy(X_raw, y)
    X = X.loc[:, X.std(ddof=1) > 0]
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    alphas = np.logspace(-3, 3, 60)
    rcv = RidgeCV(alphas=alphas).fit(Xs, yy)
    k = float(op["lambda"]) if op["lambda"] and op["lambda"] > 0 else float(rcv.alpha_)
    model = Ridge(alpha=k).fit(Xs, yy)
    r2 = model.score(Xs, yy)
    rows = []
    for i, name in enumerate(X.columns):
        b_raw = model.coef_[i] / sc.scale_[i]
        rows.append([name, fnum(b_raw, 4)])
    res.add_table(f"岭回归系数(λ={fnum(k, 4)})", ["变量", "原始尺度系数B"],
                  [["截距(常数项)", fnum(model.intercept_ - np.sum(model.coef_ / sc.scale_ * sc.mean_), 4)]] + rows)
    res.add_table("模型摘要", ["λ(惩罚系数)", "R²", "样本量N"],
                  [[fnum(k, 4), fnum(r2, 4), len(yy)]])
    # 岭迹图
    traces: dict[str, list] = {c: [] for c in X.columns}
    for a in np.logspace(-3, 3, 40):
        m = Ridge(alpha=float(a)).fit(Xs, yy)
        for i, c in enumerate(X.columns):
            traces[c].append(round(float(m.coef_[i]), 4))
    res.add_chart("岭迹图", ch.line_chart("岭迹图(标准化系数随 λ 变化)",
                 [f"{a:.1e}" for a in np.logspace(-3, 3, 40)], traces,
                 x_name="log(λ)", smooth=False, y_min=None))
    res.add_interp(
        f"岭回归通过 L2 惩罚压缩系数以缓解多重共线性。当前 λ={fnum(k, 4)}"
        f"{'(自动交叉验证选择)' if not op['lambda'] else '(手动指定)'},R²={fnum(r2, 3)};岭迹图中系数趋于稳定的 λ 可作为最终选择。")
    return res


register_model(ModelSpec(
    id="ridge", name="岭回归", category="回归建模",
    runner=_run_ridge,
    desc="L2 正则化线性回归,处理自变量多重共线性,附岭迹图;支持自动交叉验证选 λ。",
    tags=["岭回归", "共线性", "正则化"], demo_note="survey",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,)),
           Slot("xs", "自变量(定量)", types=(QUANT,), multiple=True)],
    options=[Option("lambda", "λ 惩罚系数(0=交叉验证自动)", "number", default=0, min=0, step=0.1)],
))


def _run_lasso(ds: Dataset, sl: dict, op: dict) -> Result:
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.preprocessing import StandardScaler
    res = Result()
    y = as_numeric(ds, sl["y"])
    X_raw = _design(ds, list(sl["xs"]))
    X, yy = _clean_xy(X_raw, y)
    X = X.loc[:, X.std(ddof=1) > 0]
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    if op["lambda"] and op["lambda"] > 0:
        k = float(op["lambda"])
        model = Lasso(alpha=k, max_iter=50000).fit(Xs, yy)
    else:
        cv = LassoCV(cv=5, max_iter=50000).fit(Xs, yy)
        k, model = float(cv.alpha_), cv
    r2 = model.score(Xs, yy)
    kept = [c for c, co in zip(X.columns, model.coef_) if abs(co) > 1e-8]
    rows = []
    for i, name in enumerate(X.columns):
        b_raw = model.coef_[i] / sc.scale_[i]
        rows.append([name, fnum(b_raw, 4), "保留" if abs(model.coef_[i]) > 1e-8 else "压缩为 0"])
    res.add_table(f"Lasso 系数(λ={fnum(k, 4)})", ["变量", "原始尺度系数B", "筛选结果"], rows)
    res.add_table("模型摘要", ["λ", "R²", "保留变量数", "样本量N"],
                  [[fnum(k, 4), fnum(r2, 4), len(kept), len(yy)]])
    # 正则化路径
    from sklearn.linear_model import lasso_path
    eps = 1e-4
    alphas, coefs, _ = lasso_path(Xs, yy, eps=eps, n_alphas=60, max_iter=50000)
    traces = {c: [round(float(coefs[i, j]), 4) for j in range(coefs.shape[1])] for i, c in enumerate(X.columns)}
    res.add_chart("Lasso 正则化路径", ch.line_chart("系数随 log(λ) 收缩路径",
                 [f"{a:.2e}" for a in alphas], traces, x_name="log(λ)", smooth=False))
    res.add_interp(
        f"Lasso(L1 正则)自动将不重要的变量系数压缩为 0,实现变量筛选。"
        f"当前 λ={fnum(k, 4)},共保留 **{len(kept)}** 个变量:{'、'.join(kept) or '无'};R²={fnum(r2, 3)}。")
    return res


register_model(ModelSpec(
    id="lasso", name="Lasso 回归", category="回归建模",
    runner=_run_lasso,
    desc="L1 正则化回归,自动压缩无关变量系数至 0 完成特征筛选,附正则化路径图。",
    tags=["Lasso", "特征筛选", "正则化"], demo_note="survey",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,)),
           Slot("xs", "自变量(定量)", types=(QUANT,), multiple=True)],
    options=[Option("lambda", "λ 惩罚系数(0=交叉验证自动)", "number", default=0, min=0, step=0.1)],
))


# ================================================================ 曲线估计
CURVE_FUNCS = {
    "线性": (lambda x: x, "y = b0 + b1·x"),
    "二次": (lambda x: np.column_stack([x, x ** 2]), "y = b0 + b1·x + b2·x²"),
    "三次": (lambda x: np.column_stack([x, x ** 2, x ** 3]), "y = b0 + b1·x + b2·x² + b3·x³"),
    "对数": (lambda x: np.log(x), "y = b0 + b1·ln(x)"),
    "S曲线": (lambda x: 1 / x, "y = e^(b0 + b1/x)"),
    "指数": (lambda x: x, "y = b0·e^(b1·x)"),
    "幂函数": (lambda x: np.log(x), "y = b0·x^b1"),
}


def _run_curve(ds: Dataset, sl: dict, op: dict) -> Result:
    import statsmodels.api as sm
    res = Result()
    x = as_numeric(ds, sl["x"])
    y = as_numeric(ds, sl["y"])
    data = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    xv, yv = data["x"].values.astype(float), data["y"].values.astype(float)
    fits = []
    for name in CURVE_FUNCS:
        try:
            if name == "线性":
                Xm = sm_add_const(pd.DataFrame({"x": xv}))
                model = sm.OLS(yv, Xm).fit()
                pred = model.fittedvalues
            elif name == "二次":
                Xm = sm_add_const(pd.DataFrame({"x": xv, "x2": xv ** 2}))
                model = sm.OLS(yv, Xm).fit()
                pred = model.fittedvalues
            elif name == "三次":
                Xm = sm_add_const(pd.DataFrame({"x": xv, "x2": xv ** 2, "x3": xv ** 3}))
                model = sm.OLS(yv, Xm).fit()
                pred = model.fittedvalues
            elif name == "对数":
                Xm = sm_add_const(pd.DataFrame({"lnx": np.log(xv)}))
                model = sm.OLS(yv, Xm).fit()
                pred = model.fittedvalues
            elif name == "S曲线":
                model = sm.OLS(np.log(yv), sm_add_const(pd.DataFrame({"invx": 1 / xv}))).fit()
                pred = np.exp(model.fittedvalues)
            elif name == "指数":
                model = sm.OLS(np.log(yv), sm_add_const(pd.DataFrame({"x": xv}))).fit()
                pred = np.exp(model.fittedvalues)
            elif name == "幂函数":
                model = sm.OLS(np.log(yv), sm_add_const(pd.DataFrame({"lnx": np.log(xv)}))).fit()
                pred = np.exp(model.fittedvalues)
            else:
                continue
            sst = ((yv - yv.mean()) ** 2).sum()
            r2 = 1 - ((yv - pred) ** 2).sum() / sst
            fits.append({"name": name, "model": model, "r2": float(r2),
                         "pred": np.asarray(pred), "params": [float(v) for v in model.params]})
        except Exception:  # noqa: BLE001
            continue
    if not fits:
        raise ModelError("所有曲线模型拟合失败,请检查数据(对数/幂函数要求 x>0,指数/S 曲线要求 y>0)")
    fits.sort(key=lambda f: -f["r2"])
    rows = []
    for f in fits:
        rows.append([f["name"], CURVE_FUNCS[f["name"]][1], fnum(f["r2"], 4), int(len(xv))])
    res.add_table("曲线模型汇总(按 R² 降序)", ["模型", "方程形式", "R²", "N"], rows,
                  note=f"最佳模型:{fits[0]['name']}")
    order = np.argsort(xv)
    series = {"样本点": None}
    scatter_series = {"实际值": [[float(a), float(b)] for a, b in zip(xv, yv)]}
    for f in fits[:4]:
        scatter_series[f["name"]] = [[float(xv[i]), float(f["pred"][i])] for i in order]
    res.add_chart("曲线拟合对比(前4)", ch.scatter_chart("曲线估计拟合对比", scatter_series,
                 x_name=sl["x"], y_name=sl["y"], symbol=5))
    res.add_interp(
        f"共拟合 {len(fits)} 种曲线。最佳模型为 **{fits[0]['name']}**(R²={fnum(fits[0]['r2'], 4)}),"
        f"方程:{CURVE_FUNCS[fits[0]['name']][1]};R² 越大拟合越好,请结合曲线形态与业务意义选择。")
    return res


register_model(ModelSpec(
    id="curve", name="曲线估计(曲线回归)", category="回归建模",
    runner=_run_curve,
    desc="一次拟合线性/二次/三次/对数/指数/幂函数等多种曲线,按 R² 排序寻找最优形态。",
    tags=["曲线回归", "非线性"], demo_note="timeseries",
    slots=[Slot("x", "自变量(定量,>0)", types=(QUANT,)),
           Slot("y", "因变量(定量)", types=(QUANT,))],
    options=[
        Option("models", "候选模型", "select", default="全部",
               choices=[{"value": "全部", "label": "全部 7 种"}],
               tip="默认一次性拟合全部模型并排序"),
    ],
))


# ================================================================ Probit 回归
def _run_probit(ds: Dataset, sl: dict, op: dict) -> Result:
    import statsmodels.api as sm
    res = Result()
    yraw = get_series(ds, sl["y"])
    levels = yraw.dropna().unique().tolist()
    if len(levels) != 2:
        raise ModelError(f"因变量需为二分类(当前 {len(levels)} 个水平)")
    lv01 = sorted(levels, key=str)
    y = (yraw == lv01[1]).astype(float)
    X, yy = _clean_xy(_design(ds, list(sl["xs"])), y)
    try:
        model = sm.Probit(yy, sm_add_const(X)).fit(disp=0)
    except Exception as e:  # noqa: BLE001
        raise ModelError(f"Probit 模型未收敛({e.__class__.__name__}),请检查因变量分布或减少变量") from e
    conf = model.conf_int(alpha=op["alpha"])
    rows = []
    for name in model.params.index:
        b, p = float(model.params[name]), float(model.pvalues[name])
        rows.append([name, fnum(b, 4), fnum(float(model.bse[name]), 4), fnum(float(model.tvalues[name]), 3),
                     fp(p), f"{fnum(conf.loc[name][0], 4)} ~ {fnum(conf.loc[name][1], 4)}",
                     "显著" if p < op["alpha"] else "不显著"])
    res.add_table("Probit 回归系数", ["变量", "系数B", "标准误", "Z", "p", f"{int((1 - op['alpha']) * 100)}%CI", "结论"], rows)
    try:
        mfx = model.get_margeff(at="overall")
        mrows = [[n, fnum(float(m), 5), fnum(float(se), 5), fp(float(pv)), "显著" if pv < op["alpha"] else "不显著"]
                 for n, m, se, pv in zip(model.params.index[1:], mfx.margeff, mfx.margeff_se, mfx.pvalues)]
        res.add_table("平均边际效应(∂P/∂X)", ["变量", "边际效应", "标准误", "p", "结论"], mrows,
                      note="边际效应表示该变量增加 1 单位时事件发生概率的平均变化")
    except Exception:  # noqa: BLE001
        pass
    pred = (model.predict() >= 0.5).astype(int)
    acc = float((pred == yy.values).mean())
    llnull = sm.Probit(yy, np.ones((len(yy), 1))).fit(disp=0).llf
    res.add_table("模型摘要", ["样本量N", "-2对数似然", "McFadden R²", "预测准确率"],
                  [[int(model.nobs), fnum(-2 * model.llf, 3), fnum(1 - model.llf / llnull, 4), f"{acc * 100:.2f}%"]])
    cm = pd.crosstab(yy, pred, rownames=["实际"], colnames=["预测"])
    res.add_table("混淆矩阵(阈值 0.5)", ["实际\预测", "预测=0", "预测=1"],
                  [[str(idx), int(row.get(0, 0)), int(row.get(1, 0))] for idx, row in cm.iterrows()])
    sig = [r[0] for r in rows if r[6] == "显著" and r[0] != "const"]
    res.add_interp(f"以「{lv01[1]}」为事件(1)。Probit 模型预测准确率 {acc * 100:.1f}%。"
                   + (f"显著变量:{'、'.join(sig)}。Probit 与 Logistic 结论通常一致,系数因分布假设不同而不可直接比较。"
                      if sig else "暂无变量达到显著水平。"))
    return res


register_model(ModelSpec(
    id="probit", name="Probit 回归", category="回归建模",
    runner=_run_probit,
    desc="因变量为二分类、误差服从正态分布的回归,输出系数、平均边际效应与预测准确率。",
    tags=["Probit", "二分类", "边际效应"], demo_note="survey",
    slots=[Slot("y", "因变量(二分类)", types=(CAT, ORDINAL)),
           Slot("xs", "自变量(可含定类)", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[ALPHA_OPT],
))


# ================================================================ 负二项回归
def _run_negbin(ds: Dataset, sl: dict, op: dict) -> Result:
    import statsmodels.api as sm
    res = Result()
    y = as_numeric(ds, sl["y"])
    X, yy = _clean_xy(_design(ds, list(sl["xs"])), y)
    if (yy < 0).any() or (yy % 1 != 0).any():
        raise ModelError("负二项回归要求因变量为非负整数(计数数据)")
    ybar = float(yy.mean())
    over = float(yy.var(ddof=1))
    res.add_table("因变量概览", ["N", "均值", "方差", "方差/均值(离散指数)"],
                  [[len(yy), fnum(ybar, 3), fnum(over, 3), fnum(over / ybar, 3) if ybar else ""]],
                  note="离散指数>1 提示过离散,适合负二项;≈1 可用泊松")
    import warnings
    use_nb = op["family"] == "negbin"
    model = None
    if use_nb:
        for meth in ("newton", "nm"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cand = sm.NegativeBinomial(yy, sm_add_const(X), loglike_method="nb2").fit(disp=0, maxiter=500, method=meth)
                if cand.mle_retvals.get("converged", True):
                    model = cand
                    break
            except Exception:  # noqa: BLE001
                continue
        if model is None:
            res.add_note("负二项模型未收敛,已自动退回泊松回归")
    if model is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.Poisson(yy, sm_add_const(X)).fit(disp=0, maxiter=300)
    conf = model.conf_int(alpha=op["alpha"])
    rows = []
    for name in model.params.index:
        b, p = float(model.params[name]), float(model.pvalues[name])
        rows.append([name, fnum(b, 4), fnum(float(model.bse[name]), 4), fnum(float(model.tvalues[name]), 3),
                     fp(p), fnum(np.exp(b), 4),
                     f"{fnum(np.exp(conf.loc[name][0]), 4)} ~ {fnum(np.exp(conf.loc[name][1]), 4)}",
                     "显著" if p < op["alpha"] else "不显著"])
    label = "负二项" if (use_nb and "alpha" not in model.params.index or use_nb and model.__class__.__name__ == "NegativeBinomialResults") else "泊松"
    res.add_table(f"{label}回归系数", ["变量", "系数B", "标准误", "Z", "p", "IRR(发生率比)", "IRR 95%CI", "结论"],
                  rows, note="IRR=exp(B):>1 表示该变量增加时事件次数增多")
    llnull = model.llnull if hasattr(model, "llnull") else None
    res.add_table("模型摘要", ["样本量N", "-2对数似然", "伪R²(McFadden)", "LLR p"],
                  [[int(model.nobs), fnum(-2 * model.llf, 3),
                    fnum(1 - model.llf / llnull, 4) if llnull else "", fp(model.llr_pvalue) if hasattr(model, "llr_pvalue") else ""]])
    sig = [r[0] for r in rows if r[7] == "显著" and r[0] not in ("const", "alpha")]
    res.add_interp(f"拟合{label}回归(N={len(yy)})。模型整体 LLR p={fp(model.llr_pvalue)},"
                   + (f"显著解释变量:{'、'.join(sig)}。" if sig else "暂无显著变量。")
                   + ("方差/均值明显大于 1,存在过离散,负二项优于泊松。" if over / ybar > 1.5 and ybar > 0 else ""))
    return res


register_model(ModelSpec(
    id="negbin", name="负二项 / 泊松回归(计数)", category="回归建模",
    runner=_run_negbin,
    desc="因变量为计数的回归建模,自动检验过离散,支持泊松与负二项(NB2)切换,输出发生率比 IRR。",
    tags=["泊松", "负二项", "计数数据", "IRR"], demo_note="ecommerce",
    slots=[Slot("y", "因变量(非负整数)", types=(QUANT,)),
           Slot("xs", "自变量(可含定类)", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[
        Option("family", "模型族", "select", default="negbin",
               choices=[{"value": "negbin", "label": "负二项(NB2)"}, {"value": "poisson", "label": "泊松"}]),
        ALPHA_OPT,
    ],
))


# ================================================================ 分位数回归
def _run_quantile(ds: Dataset, sl: dict, op: dict) -> Result:
    import statsmodels.api as sm
    res = Result()
    y = as_numeric(ds, sl["y"])
    X, yy = _clean_xy(_design(ds, list(sl["xs"])), y)
    X = X.loc[:, X.std(ddof=1) > 0]
    if X.shape[1] == 0:
        raise ModelError("没有可用的自变量(方差为 0)")
    q = float(op["q"])
    model = sm.QuantReg(yy, sm_add_const(X)).fit(q=q, max_iter=3000)
    conf = model.conf_int(alpha=op["alpha"])
    rows = []
    for name in model.params.index:
        p = float(model.pvalues[name])
        rows.append([name, fnum(float(model.params[name]), 4), fnum(float(model.bse[name]), 4),
                     fnum(float(model.tvalues[name]), 3), fp(p),
                     f"{fnum(conf.loc[name][0], 4)} ~ {fnum(conf.loc[name][1], 4)}",
                     "显著" if p < op["alpha"] else "不显著"])
    res.add_table(f"分位数回归系数(τ={q})", ["变量", "系数B", "标准误", "t", "p", f"{int((1 - op['alpha']) * 100)}%CI", "结论"], rows)

    def qloss(r):
        return float(np.sum(np.where(r >= 0, q * r, (q - 1) * r)))

    resid = np.asarray(yy) - np.asarray(model.fittedvalues)
    null_resid = np.asarray(yy) - np.quantile(yy, q)
    pseudo_r2 = 1 - qloss(resid) / qloss(null_resid)
    res.add_table("模型摘要", ["分位点τ", "伪R²(Koenker-Machado)", "样本量N"],
                  [[q, fnum(pseudo_r2, 4), len(yy)]],
                  note="中位数回归(τ=0.5)对离群值比均值回归更稳健")
    res.add_chart("实际 vs 拟合", ch.scatter_chart("分位数回归拟合", {"样本": [[round(float(f), 4), round(float(a), 4)] for f, a in zip(model.fittedvalues, yy)]},
                 x_name="拟合值", y_name="实际值"))
    sig = [r[0] for r in rows if r[6] == "显著" and r[0] != "const"]
    res.add_interp(f"τ={q} 分位数回归(伪R²={fnum(pseudo_r2, 3)}):研究自变量对因变量第 {q * 100:.0f}% 分位的影响。"
                   + (f"显著变量:{'、'.join(sig)}。可与 OLS 对比:若不同分位系数差异明显,提示存在异质性效应。" if sig else "暂无显著变量。"))
    return res


register_model(ModelSpec(
    id="quantile", name="分位数回归", category="回归建模",
    runner=_run_quantile,
    desc="研究自变量对因变量不同分位数(如中位数、P90)的影响,对离群值稳健,可揭示异质性效应。",
    tags=["分位数回归", "中位数回归", "稳健"], demo_note="survey",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,)),
           Slot("xs", "自变量(可含定类)", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[
        Option("q", "分位点 τ", "select", default=0.5,
               choices=[{"value": 0.1, "label": "0.1"}, {"value": 0.25, "label": "0.25"},
                        {"value": 0.5, "label": "0.5(中位数)"}, {"value": 0.75, "label": "0.75"},
                        {"value": 0.9, "label": "0.9"}]),
        ALPHA_OPT,
    ],
))


# ================================================================ 稳健回归
def _run_robust(ds: Dataset, sl: dict, op: dict) -> Result:
    import statsmodels.api as sm
    res = Result()
    y = as_numeric(ds, sl["y"])
    X, yy = _clean_xy(_design(ds, list(sl["xs"])), y)
    X = X.loc[:, X.std(ddof=1) > 0]
    norms = {"huber": sm.robust.norms.HuberT(), "tukey": sm.robust.norms.TukeyBiweight(), "hampel": sm.robust.norms.Hampel()}
    model = sm.RLM(yy, sm_add_const(X), M=norms[op["norm"]]).fit(maxiter=200)
    conf = model.conf_int(alpha=op["alpha"])
    ols = sm.OLS(yy, sm_add_const(X)).fit()
    rows = []
    for name in model.params.index:
        p = float(model.pvalues[name])
        rows.append([name, fnum(float(model.params[name]), 4), fnum(float(model.bse[name]), 4),
                     fnum(float(model.tvalues[name]), 3), fp(p), fnum(float(ols.params[name]), 4),
                     f"{fnum(conf.loc[name][0], 4)} ~ {fnum(conf.loc[name][1], 4)}",
                     "显著" if p < op["alpha"] else "不显著"])
    res.add_table("稳健回归系数(与OLS对比)", ["变量", "稳健B", "标准误", "Z", "p", "OLS B", f"{int((1 - op['alpha']) * 100)}%CI", "结论"],
                  rows, note={"huber": "Huber 加权:小残差按平方处理、大残差按线性处理",
                              "tukey": "Tukey Biweight:大残差权重趋零,抗离群最强",
                              "hampel": "Hampel:三段式降权"}[op["norm"]])
    n_out = int(np.sum(np.abs(np.asarray(ols.resid)) > 2.5 * float(np.std(ols.resid))))
    res.add_table("模型摘要", ["样本量N", "稳健尺度(scale)", "OLS疑似离群点(|残差|>2.5σ)"],
                  [[len(yy), fnum(float(model.scale), 4), n_out]])
    res.add_chart("实际 vs 拟合", ch.scatter_chart("稳健回归拟合",
                 {"样本": [[round(float(f), 4), round(float(a), 4)] for f, a in zip(model.fittedvalues, yy)]},
                 x_name="拟合值", y_name="实际值"))
    sig = [r[0] for r in rows if r[7] == "显著" and r[0] != "const"]
    res.add_interp("稳健回归(M-估计)对离群点自动降权;当稳健系数与 OLS 差异较大时,说明结果受离群值影响,应以稳健结果为准。"
                   + f"检测到 {n_out} 个疑似离群点。"
                   + (f"显著变量:{'、'.join(sig)}。" if sig else "暂无显著变量。"))
    return res


register_model(ModelSpec(
    id="robust", name="稳健回归(RLM)", category="回归建模",
    runner=_run_robust,
    desc="使用 Huber/Tukey/Hampel 稳健估计削弱离群值影响,输出与 OLS 系数对比。",
    tags=["稳健回归", "离群值", "RLM"], demo_note="survey",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,)),
           Slot("xs", "自变量(可含定类)", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[
        Option("norm", "权函数", "select", default="huber",
               choices=[{"value": "huber", "label": "Huber"}, {"value": "tukey", "label": "Tukey Biweight"},
                        {"value": "hampel", "label": "Hampel"}]),
        ALPHA_OPT,
    ],
))


# ================================================================ 曲面响应回归(RSM)
def _run_rsm(ds: Dataset, sl: dict, op: dict) -> Result:
    import statsmodels.api as sm
    res = Result()
    y = as_numeric(ds, sl["y"])
    xs = list(sl["xs"])[:4]
    if len(xs) < 2:
        raise ModelError("曲面响应回归至少需要 2 个自变量")
    data = pd.concat([as_numeric(ds, c).rename(c) for c in xs] + [y.rename("__y__")], axis=1).dropna()
    if len(data) < 3 * len(xs) + 5:
        raise ModelError(f"有效样本不足(当前 {len(data)} 行),曲面回归需要更多样本")
    yv = data["__y__"]
    parts = [data[xs]]
    for c in xs:
        parts.append((data[c] ** 2).rename(f"{c}²"))
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            parts.append((data[xs[i]] * data[xs[j]]).rename(f"{xs[i]}×{xs[j]}"))
    X = pd.concat(parts, axis=1)
    model = sm.OLS(yv, sm_add_const(X)).fit()
    rows = []
    for name in model.params.index:
        p = float(model.pvalues[name])
        rows.append([name, fnum(float(model.params[name]), 4), fnum(float(model.bse[name]), 4),
                     fnum(float(model.tvalues[name]), 3), fp(p), "显著" if p < op["alpha"] else "不显著"])
    res.add_table("二次响应面系数", ["项", "系数B", "标准误", "t", "p", "结论"], rows)
    res.add_table("模型摘要", ["R²", "调整R²", "F", "F检验p", "样本量N"],
                  [[fnum(model.rsquared, 4), fnum(model.rsquared_adj, 4), fnum(model.fvalue, 3), fp(model.f_pvalue), len(data)]])
    # 驻点:∂ŷ/∂x = 0 → quad·x = -lin
    k = len(xs)
    quad = np.zeros((k, k))
    lin = np.zeros(k)
    for i in range(k):
        lin[i] = float(model.params[xs[i]])
        quad[i, i] = 2 * float(model.params[f"{xs[i]}²"])
        for j in range(i + 1, k):
            v = float(model.params[f"{xs[i]}×{xs[j]}"])
            quad[i, j] += v
            quad[j, i] += v
    station_txt = "无法求解驻点(二次项矩阵奇异)"
    try:
        opt_pt = np.linalg.solve(quad, -lin)
        eigv = np.linalg.eigvalsh(quad)
        kind = ("驻点为极大值点" if eigv.max() < 0 else "驻点为极小值点" if eigv.min() > 0 else "驻点为鞍点")
        outside = [n for n, v in zip(xs, opt_pt) if v < data[n].min() or v > data[n].max()]
        station_txt = kind + "(" + ", ".join(f"{n}={fnum(v, 3)}" for n, v in zip(xs, opt_pt)) + ")"
        if outside:
            station_txt += f";注意:「{','.join(outside)}」方向驻点在数据范围之外,仅为外推参考"
        res.add_table("驻点与曲面形状", ["结论"], [[station_txt]],
                      note="二次项矩阵特征值全负→极大值;全正→极小值;有正有负→鞍点")
    except np.linalg.LinAlgError:
        pass
    if len(xs) == 2:
        g1 = np.linspace(data[xs[0]].min(), data[xs[0]].max(), 24)
        g2 = np.linspace(data[xs[1]].min(), data[xs[1]].max(), 24)
        A = pd.DataFrame([(a, b) for a in g1 for b in g2], columns=xs)
        Ap = [A]
        for c in xs:
            Ap.append((A[c] ** 2).rename(f"{c}²"))
        Ap.append((A[xs[0]] * A[xs[1]]).rename(f"{xs[0]}×{xs[1]}"))
        zz = np.asarray(model.predict(sm_add_const(pd.concat(Ap, axis=1)))).reshape(24, 24)
        res.add_chart(f"响应曲面等高线({xs[0]} × {xs[1]})",
                      ch.heatmap_chart("响应面预测值", [fnum(v, 2) for v in g2], [fnum(v, 2) for v in g1],
                                       zz.tolist(), x_name=xs[1], fmt="{c}",
                                       v_min=float(np.nanmin(zz)), v_max=float(np.nanmax(zz))))
    sig_terms = [r[0] for r in rows if r[5] == "显著" and r[0] != "const"]
    quad_sig = [t for t in sig_terms if "²" in t or "×" in t]
    res.add_interp(f"二次响应面模型 R²={fnum(model.rsquared, 3)},调整R²={fnum(model.rsquared_adj, 3)}。"
                   + (f"二次/交互项({', '.join(quad_sig)})显著,存在弯曲效应与变量间协同作用。" if quad_sig else "二次与交互项均不显著,关系接近平面,可考虑改用线性回归。")
                   + f"{station_txt}。")
    return res


register_model(ModelSpec(
    id="rsm", name="曲面响应回归(二次RSM)", category="回归建模",
    runner=_run_rsm,
    desc="拟合含平方项与交互项的二次曲面并求解驻点,常用于实验设计(DOE)参数优化。",
    tags=["响应面", "RSM", "实验设计", "优化"], demo_note="clinical",
    slots=[Slot("y", "响应变量(定量)", types=(QUANT,)),
           Slot("xs", "因子变量(定量,2~4个)", types=(QUANT,), multiple=True, max_count=4)],
    options=[ALPHA_OPT],
))
# ================================================================ 中介效应分析
def _run_mediation(ds: Dataset, sl: dict, op: dict) -> Result:
    """简单/多重中介:系数乘积法点估计 + case 重抽样百分位 Bootstrap 推断(PROCESS 风格)。"""
    res = Result()
    yname, xname, mnames = sl["y"], sl["x"], list(sl["ms"])
    covs = [sl["covs"]] if sl["covs"] else []
    allv = [yname, xname] + mnames + covs
    if len(set(allv)) != len(allv):
        raise ModelError("因变量、自变量、中介变量与协变量之间不能重复选择")
    data = pd.concat([pd.to_numeric(get_series(ds, c), errors="coerce") for c in allv], axis=1).dropna()
    n = len(data)
    if n < 20:
        raise ModelError(f"有效样本不足 20 行(当前 {n}),中介效应 Bootstrap 检验不可靠")
    Y = data[yname].values
    X = data[xname].values
    Mcols = [data[m].values for m in mnames]
    Ccols = [data[c].values for c in covs]
    one = np.ones(n)
    # 三套设计矩阵:a 路径(M~X+C)、b/c' 路径(Y~X+M+C)、c 路径(Y~X+C)
    Xa = np.column_stack([one, X] + Ccols)
    Xr = np.column_stack([one, X] + Ccols)
    n_rep = int(op["n_rep"])
    rng = np.random.default_rng(20240001)      # 固定种子,结果可复现

    def _ols(A, y):
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return beta

    def _path_betas(mi: int):
        """返回全样本点估计与 Bootstrap 分布:(a, b, cprime, c) 与各自的重抽样数组。"""
        Xo = np.column_stack([one, X, Mcols[mi]] + Ccols)
        ba = _ols(Xa, Mcols[mi])
        bo = _ols(Xo, Y)
        br = _ols(Xr, Y)
        pts = (ba[1], bo[2], bo[1], br[1])
        boots = np.empty((n_rep, 4))
        for r in range(n_rep):
            idx = rng.integers(0, n, n)
            ba_ = _ols(Xa[idx], Mcols[mi][idx])
            bo_ = _ols(Xo[idx], Y[idx])
            br_ = _ols(Xr[idx], Y[idx])
            boots[r] = (ba_[1], bo_[2], bo_[1], br_[1])
        return pts, boots

    rows, summary_rows = [], []
    alpha = op["alpha"]
    for mi, m in enumerate(mnames):
        (a, b, cp, c), boots = _path_betas(mi)   # Xa/Xo/Xr 的系数顺序:a, b, c', c
        acme_pt, ade_pt, tot_pt = a * b, cp, c
        acme_b, ade_b, tot_b = boots[:, 0] * boots[:, 1], boots[:, 2], boots[:, 3]
        for label, pt, arr in (("间接效应(ACME)", acme_pt, acme_b),
                               ("直接效应(ADE)", ade_pt, ade_b),
                               ("总效应", tot_pt, tot_b)):
            p = max(float(2 * min((arr <= 0).mean(), (arr >= 0).mean())), 1.0 / (2 * n_rep))
            rows.append([m, label, fnum(pt, 4), fnum(float(arr.std(ddof=1)), 4),
                         fnum(float(np.percentile(arr, 2.5)), 4), fnum(float(np.percentile(arr, 97.5)), 4),
                         fp(p), "显著" if p < alpha else "不显著"])
        p_ind = max(float(2 * min((acme_b <= 0).mean(), (acme_b >= 0).mean())), 1.0 / (2 * n_rep))
        p_dir = max(float(2 * min((ade_b <= 0).mean(), (ade_b >= 0).mean())), 1.0 / (2 * n_rep))
        if p_ind < alpha:
            kind = "部分中介" if p_dir < alpha else "完全中介"
        else:
            kind = "未检出中介效应"
        prop = acme_pt / tot_pt if tot_pt else np.nan
        summary_rows.append([m, fnum(prop, 4), kind])
    res.add_table("效应分解(Bootstrap 检验)",
                  ["中介变量", "效应类型", "效应值", "Boot SE", "95%CI 下限", "95%CI 上限", "p", "显著性"],
                  rows, note=f"case 重抽样百分位 Bootstrap({n_rep} 次,固定随机种子);ACME=a×b")
    res.add_table("中介效应判定", ["中介变量", "中介效应占比", "判定"], summary_rows,
                  note="判定规则:间接效应显著且直接效应显著→部分中介;间接效应显著、直接效应不显著→完全中介")
    first = summary_rows[0]
    sig_list = "、".join(r[0] for r in summary_rows if "中介" in r[2])
    res.add_interp(
        f"以「{xname}」为自变量、「{yname}」为因变量、「{('、'.join(mnames))}」为中介变量,"
        f"Bootstrap({n_rep} 次)检验显示:{sig_list or '所选中介变量均'}"
        + (f"存在**{first[2]}**(占比 {first[1]},即「{xname}」对「{yname}」的影响约 "
           f"{fnum(float(first[1]) * 100, 1)}% 经由「{mnames[0]}」传递)。" if "中介" in first[2]
           else "未发现显著的间接传导路径,总效应主要由直接路径贡献。"))
    return res


register_model(ModelSpec(
    id="mediation", name="中介效应分析", category="回归建模",
    runner=_run_mediation,
    desc="系数乘积法 + Bootstrap 检验的中介模型:间接/直接/总效应分解、中介占比与完全/部分中介判定。",
    tags=["中介", "PROCESS", "Bootstrap"], demo_note="survey",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,)),
           Slot("x", "自变量(定量)", types=(QUANT,)),
           Slot("ms", "中介变量(定量,可多个)", types=(QUANT,), multiple=True, max_count=5),
           Slot("covs", "协变量(定量,可选)", types=(QUANT,), multiple=False, required=False)],
    options=[Option("n_rep", "Bootstrap 次数", "select", default=1000,
                    choices=[{"value": 200, "label": "200(快)"},
                             {"value": 500, "label": "500"},
                             {"value": 1000, "label": "1000(推荐)"},
                             {"value": 2000, "label": "2000(慢)"}]),
             ALPHA_OPT],
))


# ================================================================ 调节效应分析
def _run_moderation(ds: Dataset, sl: dict, op: dict) -> Result:
    """层次回归检验调节:中心化交互项 + ΔR² 的 F 变化检验 + 简单斜率与交互图。"""
    import statsmodels.api as sm
    from scipy import stats as sps

    res = Result()
    yname, xname, mname = sl["y"], sl["x"], sl["mod"]
    covs = [c for c in sl["covs"]]
    if len({yname, xname, mname}) != 3 or yname in covs or xname in covs or mname in covs:
        raise ModelError("因变量、自变量、调节变量与协变量之间不能重复选择")
    frame = pd.concat([pd.to_numeric(get_series(ds, c), errors="coerce")
                       for c in [yname, xname] + covs]
                      + [get_series(ds, mname).rename(mname)], axis=1)
    xc = (frame[xname] - frame[xname].mean()).rename("X_c")
    parts, mod_desc = [xc], ""
    m_meta = next((c for c in ds.columns if c["name"] == mname), None)
    if m_meta and m_meta["dtype"] in ("cat", "ordinal"):
        dummies = pd.get_dummies(frame[mname].astype("category"), prefix="M", drop_first=True, dtype=float)
        if dummies.shape[1] == 0:
            raise ModelError("调节变量只有一个水平,无法检验调节效应")
        parts.append(dummies)
        mod_terms = list(dummies.columns)
        mod_desc = "定类/定级变量(哑变量化,以首个水平为参照)"
    else:
        ms = pd.to_numeric(frame[mname], errors="coerce")
        sd = ms.std(ddof=1)
        parts.append((ms - ms.mean()).rename("M_c"))
        mod_terms = ["M_c"]
        mod_desc = f"定量变量(中心化,SD={fnum(sd, 3)})"
    for i, c in enumerate(covs):
        parts.append(frame[c].rename(f"C{i + 1}"))
    D = pd.concat(parts, axis=1).dropna()
    if len(D) < 20:
        raise ModelError(f"有效样本不足 20 行(当前 {len(D)})")
    Y = pd.to_numeric(get_series(ds, yname), errors="coerce").reindex(D.index)
    Xc = D["X_c"]
    base_covs = [c for c in D.columns if c.startswith("C")]
    m_cols = [c for c in D.columns if c.startswith("M")]
    inter_cols = [f"X_x_{m}" for m in m_cols]
    inter = pd.DataFrame({c: Xc * D[m] for c, m in zip(inter_cols, m_cols)})
    ordered = [Xc] + [D[c] for c in m_cols] + [D[c] for c in base_covs] + [inter[c] for c in inter_cols]

    def _fit(cols: list):
        X = pd.concat(cols, axis=1)
        return sm.OLS(Y, sm.add_constant(X, has_constant="add")).fit()

    M1 = _fit([Xc] + [D[c] for c in base_covs])
    M2 = _fit([Xc] + [D[c] for c in m_cols] + [D[c] for c in base_covs])
    M3 = _fit(ordered)

    def _fchange(small, big):
        from scipy import stats as _sps
        df1, df2 = int(big.df_model - small.df_model), int(big.df_resid)
        if df1 <= 0 or df2 <= 0:
            return np.nan, np.nan
        f = (big.rsquared - small.rsquared) / df1 / ((1 - big.rsquared) / df2)
        return float(f), float(_sps.f.sf(f, df1, df2))

    f2, p2 = _fchange(M1, M2)
    f3, p3 = _fchange(M2, M3)
    res.add_table("层次回归模型比较",
                  ["模型", "自变量X", "调节变量M", "交互项X×M", "R²", "ΔR²", "F 变化", "p(ΔF)"],
                  [["M1", "✓", "", "", fnum(M1.rsquared, 4), "—", "—", "—"],
                   ["M2", "✓", "✓", "", fnum(M2.rsquared, 4), fnum(M2.rsquared - M1.rsquared, 4), fnum(f2, 3), fp(p2)],
                   ["M3", "✓", "✓", "✓", fnum(M3.rsquared, 4), fnum(M3.rsquared - M2.rsquared, 4), fnum(f3, 3), fp(p3)]],
                  note="调节效应看 M3 步:交互项加入后 ΔR² 显著(F 变化 p<0.05)即存在调节效应")

    def _pretty(nm: str) -> str:
        return (nm.replace("X_c", f"「{xname}」(中心化)")
                  .replace("M_c", f"「{mname}」(中心化)")
                  .replace("X_x_M_c", f"「{xname}」×「{mname}」"))
    irows = []
    for nm in M3.params.index:
        if nm == "const":
            continue
        irows.append([_pretty(nm), fnum(M3.params[nm], 4), fnum(M3.bse[nm], 4), fnum(M3.tvalues[nm], 3),
                      fp(M3.pvalues[nm]), "显著" if M3.pvalues[nm] < op["alpha"] else "不显著"])
    res.add_table("交互模型(M3)回归系数", ["项", "系数", "标准误", "t", "p", "显著性"], irows)

    # 简单斜率:自变量对因变量的效应在不同调节水平下的差异
    b, V = M3.params, M3.cov_params()
    srows = []
    if mod_terms == ["M_c"]:
        msd = float(D["M_c"].std(ddof=1))
        levels = [("M = 均值-1SD", -msd), ("M = 均值", 0.0), ("M = 均值+1SD", msd)]
        for label, mv in levels:
            slope = float(b["X_c"] + b["X_x_M_c"] * mv)
            var = float(V.loc["X_c", "X_c"] + mv ** 2 * V.loc["X_x_M_c", "X_x_M_c"]
                        + 2 * mv * V.loc["X_c", "X_x_M_c"])
            se = float(np.sqrt(max(var, 1e-300)))
            t = slope / se
            p = float(2 * sps.t.sf(abs(t), int(M3.df_resid)))
            srows.append([label, fnum(slope, 4), fnum(se, 4), fnum(t, 3), fp(p),
                          "显著" if p < op["alpha"] else "不显著"])
    else:
        srows.append(["M = 参照水平", fnum(b["X_c"], 4), fnum(M3.bse["X_c"], 4), fnum(M3.tvalues["X_c"], 3),
                      fp(M3.pvalues["X_c"]), "显著" if M3.pvalues["X_c"] < op["alpha"] else "不显著"])
        for m in mod_terms:
            lvl = m.replace("M_", "")
            slope = float(b["X_c"] + b[f"X_x_{m}"])
            var = float(V.loc["X_c", "X_c"] + V.loc[f"X_x_{m}", f"X_x_{m}"]
                        + 2 * V.loc["X_c", f"X_x_{m}"])
            se = float(np.sqrt(max(var, 1e-300)))
            t = slope / se
            p = float(2 * sps.t.sf(abs(t), int(M3.df_resid)))
            srows.append([f"M = {lvl}", fnum(slope, 4), fnum(se, 4), fnum(t, 3), fp(p),
                          "显著" if p < op["alpha"] else "不显著"])
    res.add_table("简单斜率检验(自变量→因变量的效应)", ["调节水平", "斜率", "标准误", "t", "p", "显著性"], srows,
                  note="不同调节水平下斜率差异明显,说明调节效应具有实质意义")

    # 交互图
    xgrid = np.linspace(float(Xc.min()), float(Xc.max()), 20)
    series: dict[str, list] = {}
    if mod_terms == ["M_c"]:
        msd = float(D["M_c"].std(ddof=1))
        for lab, mv in (("M低(-1SD)", -msd), ("M均值", 0.0), ("M高(+1SD)", msd)):
            series[lab] = [round(float(M3.params["const"] + M3.params["X_c"] * xv + M3.params["X_x_M_c"] * mv), 4)
                           for xv in xgrid]
    else:
        series["参照水平"] = [round(float(M3.params["const"] + M3.params["X_c"] * xv), 4) for xv in xgrid]
        for m in mod_terms:
            series[f"{m.replace('M_', '')}"] = [
                round(float(M3.params["const"] + M3.params["X_c"] * xv + M3.params[f"X_x_{m}"]), 4) for xv in xgrid]
    res.add_chart("调节效应交互图", ch.line_chart("简单斜率图(预测值)", [round(v, 2) for v in xgrid], series,
                 x_name=f"「{xname}」(中心化)", y_name=f"「{yname}」预测值"))
    sig_int = any(M3.pvalues[c] < op["alpha"] for c in inter_cols)
    direction = ""
    if sig_int and mod_terms == ["M_c"]:
        v = float(M3.params["X_x_M_c"])
        direction = ",调节方向为" + ("增强(正向调节)" if v > 0 else "削弱(负向调节)")
    if sig_int:
        core = (f"交互项显著,「{mname}」对「{xname}」→「{yname}」的关系存在显著调节效应{direction}。"
                f"简单斜率检验显示自变量效应随调节水平变化。")
    else:
        core = f"交互项不显著(p={fp(p3)}),未检出「{mname}」的调节效应,各调节水平下回归线近似平行。"
    res.add_interp(
        f"层次回归显示:加入交互项后 ΔR²={fnum(M3.rsquared - M2.rsquared, 4)}(F 变化 p={fp(p3)})。{core}"
        f"调节变量处理方式:{mod_desc}。")
    return res


register_model(ModelSpec(
    id="moderation", name="调节效应分析", category="回归建模",
    runner=_run_moderation,
    desc="层次回归检验调节效应:中心化交互项、ΔR² 的 F 变化检验、简单斜率与交互图。",
    tags=["调节", "交互效应", "简单斜率"], demo_note="survey",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,)),
           Slot("x", "自变量(定量)", types=(QUANT,)),
           Slot("mod", "调节变量(定量/定类/定级)", types=(QUANT, CAT, ORDINAL)),
           Slot("covs", "协变量(定量,可选)", types=(QUANT,), multiple=True, max_count=8)],
    options=[ALPHA_OPT],
))


# ================================================================ 线性混合模型
def _run_mixed_lm(ds: Dataset, sl: dict, op: dict) -> Result:
    """线性混合模型:固定效应 + 随机截距/随机斜率,输出 ICC 与方差成分。"""
    import statsmodels.api as sm

    res = Result()
    yname, xs, gname = sl["y"], list(sl["xs"]), sl["group"]
    if yname == gname or gname in xs or yname in xs:
        raise ModelError("因变量、固定效应与分组变量之间不能重复选择")
    if not xs:
        raise ModelError("请至少选择 1 个固定效应变量")
    X = _design(ds, xs)
    frame = pd.concat([as_numeric(ds, yname), X, get_series(ds, gname).astype(str).rename("G")], axis=1).dropna()
    Y = frame[yname].values
    Xd = frame[X.columns].values
    groups = frame["G"].values
    if len(frame) < 10:
        raise ModelError(f"有效样本不足 10 行(当前 {len(frame)})")
    n_groups = len(set(groups))
    if n_groups < 3:
        raise ModelError(f"分组水平数过少({n_groups} 组),随机效应无法估计")

    reml = op["reml"] == "reml"
    if op["re_type"] == "ris":
        # 随机斜率挂在第一个定量固定效应上
        slope_col = next((c for c in X.columns if c in xs), None)
        if slope_col is None:
            raise ModelError("随机斜率需要一个定量固定效应;当前固定效应均为哑变量,请改用随机截距")
        j = X.columns.get_loc(slope_col)
        Z = Xd[:, [j]]
        md = sm.MixedLM(Y, sm.add_constant(Xd, has_constant="add"), groups=groups, exog_re=Z)
    else:
        md = sm.MixedLM(Y, sm.add_constant(Xd, has_constant="add"), groups=groups)
    with np.errstate(all="ignore"):
        r = md.fit(reml=reml)

    if not r.converged:
        res.add_note("模型未完全收敛,固定效应估计仅供参考;可尝试减少随机效应或固定效应变量。")
    fe_names = ["const"] + list(X.columns)
    fe_vals = np.asarray(r.fe_params)
    fe_bse = np.asarray(r.bse)[:len(fe_vals)]
    ci = np.asarray(r.conf_int())[:len(fe_vals)]
    from scipy import stats as sps
    frows = []
    for i, nm in enumerate(fe_names):
        z = fe_vals[i] / fe_bse[i] if fe_bse[i] else np.nan
        p = float(2 * sps.norm.sf(abs(z)))
        frows.append(["常数项" if nm == "const" else nm, fnum(fe_vals[i], 4), fnum(fe_bse[i], 4),
                      fnum(z, 3), fp(p), fnum(ci[i][0], 4), fnum(ci[i][1], 4)])
    res.add_table("固定效应估计", ["项", "系数", "标准误", "z", "p", "95%CI 下限", "95%CI 上限"], frows,
                  note="定类自变量已哑变量化(以首水平为参照)" if any(c not in xs for c in X.columns) else "")

    scale = float(r.scale)
    var_re = np.diag(np.asarray(r.cov_re))
    rrows = [["残差方差", fnum(scale, 4)],
             ["随机截距方差", fnum(var_re[0], 4)],
             ["随机斜率方差", fnum(var_re[1], 4) if len(var_re) > 1 else "—"],
             ["组数", n_groups]]
    icc = var_re[0] / (var_re[0] + scale) if (var_re[0] + scale) else np.nan
    res.add_table("随机效应与方差成分", ["成分", "估计值"], rrows,
                  note="组内相关系数 ICC=随机截距方差/(随机截距方差+残差方差)=" + fnum(icc, 3))
    fitrows = [["样本量", len(frame)], ["分组数", n_groups], ["对数似然", fnum(r.llf, 3)]]
    if not reml:
        fitrows += [["AIC", fnum(r.aic, 3)], ["BIC", fnum(r.bic, 3)]]
    else:
        fitrows.append(["估计方法", "REML(AIC/BIC 仅 ML 下可比)"])
    res.add_table("模型拟合信息", ["指标", "值"], fitrows)
    res.add_chart("固定效应系数图", ch.bar_chart("固定效应系数", [r[0] for r in frows],
                 {"系数": [round(float(r[1]), 4) if r[1] else None for r in frows]}, y_name="系数"))
    sig_fe = [r[0] for r in frows if r[4] and r[4].startswith("<0.001") or (r[4] and not r[4].startswith("<") and float(r[4].rstrip("*")) < op["alpha"]) if r[4]]
    res.add_interp(
        f"线性混合模型以「{gname}」为分组单元({n_groups} 组):固定效应{'、'.join(sig_fe[:3]) or '均不显著'}"
        f"达到显著水平;随机截距方差={fnum(var_re[0], 3)},ICC={fnum(icc, 3)},"
        + (f"组间差异明显(ICC>0.059),混合模型比普通回归更合适。" if icc > 0.059 else
           "组间差异较小,普通回归与混合模型结论可能接近。"))
    return res


register_model(ModelSpec(
    id="mixed_lm", name="线性混合模型", category="回归建模",
    runner=_run_mixed_lm,
    desc="含随机截距/随机斜率的分层模型:固定效应、方差成分与 ICC,适用于重复测量与嵌套数据。",
    tags=["混合模型", "多层模型", "ICC"], demo_note="panel",
    slots=[Slot("y", "因变量(定量)", types=(QUANT,)),
           Slot("group", "分组变量(被试/聚类单元)", types=(CAT, ORDINAL, QUANT)),
           Slot("xs", "固定效应(定量/定类,可多个)", types=(QUANT, CAT, ORDINAL), multiple=True, max_count=10)],
    options=[Option("re_type", "随机效应结构", "select", default="ri",
                    choices=[{"value": "ri", "label": "随机截距"},
                             {"value": "ris", "label": "随机截距+随机斜率"}]),
             Option("reml", "估计方法", "select", default="reml",
                    choices=[{"value": "reml", "label": "REML(推荐)"},
                             {"value": "ml", "label": "极大似然 ML"}]),
             ALPHA_OPT],
))
