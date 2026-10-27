"""机器学习:决策树、随机森林、SVM、KNN、朴素贝叶斯、神经网络、梯度提升(自动识别分类/回归任务)。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import charts as ch
from ..core.datasets import Dataset, get_series
from ..core.registry import CAT, ORDINAL, QUANT, ModelSpec, Option, Slot, register_model
from ..core.results import ModelError, Result, fnum

TEST_RATIO_OPT = Option("test_ratio", "测试集比例", "number", default=0.3, min=0.1, max=0.5, step=0.05)
SEED_OPT = Option("seed", "随机种子", "number", default=42, min=0)
SAVE_OPT = Option("save_pred", "保存预测值到数据集", "switch", default=True)


def _prepare_ml(ds: Dataset, sl: dict):
    """返回 X(数值化 DataFrame), y, task, levels"""
    ycol = sl["y"]
    xs = list(sl["xs"])
    if ycol in xs:
        raise ModelError("因变量不能同时作为自变量")
    ymeta = next((c for c in ds.columns if c["name"] == ycol), None)
    task = "classification" if (ymeta and ymeta["dtype"] in ("cat", "ordinal")) else "regression"
    yraw = get_series(ds, ycol)
    levels = None
    if task == "classification":
        yraw = yraw.astype(str)
        levels = sorted(yraw.dropna().unique(), key=str)
        y = pd.Categorical(yraw, categories=levels).codes
    else:
        y = pd.to_numeric(yraw, errors="coerce")
    parts = []
    for x in xs:
        s = ds.df[x]
        meta = next((c for c in ds.columns if c["name"] == x), None)
        if meta and meta["dtype"] in ("cat", "ordinal"):
            parts.append(pd.get_dummies(s.astype("category"), prefix=x, drop_first=True, dtype=float))
        else:
            parts.append(pd.to_numeric(s, errors="coerce").rename(x))
    X = pd.concat(parts, axis=1)
    data = pd.concat([X, pd.Series(y, name="__y__")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 20:
        raise ModelError(f"机器学习模型需要 ≥20 行完整样本(当前 {len(data)})")
    if task == "classification" and len(levels) < 2:
        raise ModelError("分类因变量至少需要 2 个水平")
    return data.drop(columns="__y__"), data["__y__"], task, levels


def _split(X, y, op):
    from sklearn.model_selection import train_test_split
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 and y.dtype.kind in "iu" else None
    return train_test_split(X, y, test_size=float(op["test_ratio"]), random_state=int(op["seed"]),
                            stratify=stratify)


def _classification_report(res: Result, name: str, model, Xte, yte, levels) -> float:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    avg = "binary" if len(levels) == 2 else "macro"
    prec = precision_score(yte, pred, average=avg, zero_division=0)
    rec = recall_score(yte, pred, average=avg, zero_division=0)
    f1 = f1_score(yte, pred, average=avg, zero_division=0)
    res.add_table("分类性能(测试集)", ["模型", "准确率", "精确率", "召回率", "F1"],
                  [[name, f"{acc * 100:.2f}%", fnum(prec, 4), fnum(rec, 4), fnum(f1, 4)]],
                  note="多分类指标为宏平均")
    cm = confusion_matrix(yte, pred, labels=range(len(levels)))
    res.add_table("混淆矩阵(测试集)", ["实际\\预测"] + [str(l) for l in levels] + ["正确率"],
                  [[str(levels[i])] + [int(v) for v in cm[i]] + [f"{cm[i][i] / max(cm[i].sum(), 1) * 100:.1f}%"]
                   for i in range(len(levels))])
    res.add_chart("混淆矩阵热力图", ch.heatmap_chart("混淆矩阵", [str(l) for l in levels],
                 [str(l) for l in levels], cm.tolist(), fmt="{c:.0f}"))
    return acc


def _regression_report(res: Result, name: str, model, Xte, yte) -> float:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    pred = model.predict(Xte)
    r2 = r2_score(yte, pred)
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    mae = mean_absolute_error(yte, pred)
    with np.errstate(all="ignore"):
        mape = float(np.nanmean(np.abs((yte - pred) / np.where(np.abs(yte) < 1e-9, np.nan, yte))) * 100)
    res.add_table("回归性能(测试集)", ["模型", "R²", "RMSE", "MAE", "MAPE%"],
                  [[name, fnum(r2, 4), fnum(rmse, 4), fnum(mae, 4), fnum(mape, 2)]])
    idx = np.argsort(np.asarray(yte))
    res.add_chart("预测值 vs 实际值", ch.line_chart("实际值与预测值对比(按实际值排序)",
                 [str(i) for i in range(len(idx))],
                 {"实际值": [round(float(np.asarray(yte)[i]), 4) for i in idx],
                  "预测值": [round(float(np.asarray(pred)[i]), 4) for i in idx]},
                 x_name="样本", smooth=False))
    return r2


def _importance_chart(res: Result, model, columns: list[str], topn: int = 12) -> None:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return
    order = np.argsort(-imp)[:topn]
    res.add_chart(f"特征重要性 Top{topn}",
                  ch.bar_chart("特征重要性", [columns[i] for i in order][::-1],
                               {"重要性": [round(float(imp[i]), 4) for i in order][::-1]},
                               horizontal=True))


def _ml_runner(model_factory, model_name: str):
    """model_factory(op, task) -> 已配置的 sklearn 模型"""
    def run(ds: Dataset, sl: dict, op: dict) -> Result:
        res = Result()
        X, y, task, levels = _prepare_ml(ds, sl)
        Xtr, Xte, ytr, yte = _split(X, y, op)
        model = model_factory(op, task)
        model.fit(Xtr, ytr)
        if task == "classification":
            acc = _classification_report(res, model_name, model, Xte, yte, levels)
            res.add_interp(f"{model_name} 分类模型训练完成:测试集准确率 **{acc * 100:.1f}%**,各项指标与混淆矩阵见上表;"
                           f"特征重要性反映各变量对分类的贡献。")
        else:
            r2 = _regression_report(res, model_name, model, Xte, yte)
            res.add_interp(f"{model_name} 回归模型训练完成:测试集 R²=**{fnum(r2, 3)}**(越接近 1 拟合越好);"
                           f"特征重要性反映各变量对预测的贡献。")
        _importance_chart(res, model, list(X.columns))
        pred_all = model.predict(X)
        if op.get("save_pred"):
            name = f"预测_{sl['y']}"
            col = pd.Series(np.nan, index=ds.df.index, dtype="object" if task == "classification" else float)
            if task == "classification":
                col.loc[X.index] = [levels[int(v)] for v in np.asarray(pred_all).ravel()]
            else:
                col.loc[X.index] = np.asarray(pred_all).ravel()
            ds.write_col(name, col, "cat" if task == "classification" else "quant")
            res.add_note(f"预测结果已写回数据集(列:{name})")
        preview_rows = []
        for i in range(min(20, len(X))):
            actual = str(levels[int(y.iloc[i])]) if task == "classification" else fnum(y.iloc[i], 3)
            predicted = str(levels[int(pred_all[i])]) if task == "classification" else fnum(pred_all[i], 3)
            preview_rows.append([str(X.index[i]), actual, predicted, "✓" if actual == predicted else "✗"])
        res.add_table("预测结果预览(前20行)", ["样本", "实际值", "预测值", "正确?"], preview_rows)
        return res
    return run


def _tree_kw(op):
    return dict(max_depth=int(op["max_depth"]), min_samples_leaf=int(op["min_samples_leaf"]),
                random_state=int(op["seed"]))


def _fac_tree(op, task):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    cls = task == "classification"
    return DecisionTreeClassifier(criterion="gini", **_tree_kw(op)) if cls else DecisionTreeRegressor(**_tree_kw(op))


def _fac_rf(op, task):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    kw = dict(n_estimators=int(op["n_estimators"]), max_depth=int(op["max_depth"]) or None,
              random_state=int(op["seed"]), n_jobs=-1)
    return RandomForestClassifier(**kw) if task == "classification" else RandomForestRegressor(**kw)


def _fac_svm(op, task):
    from sklearn.svm import SVC, SVR
    kw = dict(C=float(op["C"]), kernel=op["kernel"], gamma="scale")
    return SVC(probability=True, **kw) if task == "classification" else SVR(**kw)


def _fac_knn(op, task):
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    k = int(op["k"])
    kw = dict(n_neighbors=k, weights=op["weights"])
    return KNeighborsClassifier(**kw) if task == "classification" else KNeighborsRegressor(**kw)


def _fac_nb(op, task):
    from sklearn.naive_bayes import GaussianNB
    if task != "classification":
        raise ModelError("朴素贝叶斯仅支持分类任务,请选择定类因变量")
    return GaussianNB()


def _fac_mlp(op, task):
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    kw = dict(hidden_layer_sizes=tuple(int(v) for v in str(op["layers"]).replace("，", ",").split(",") if v.strip()),
              max_iter=int(op["max_iter"]), random_state=int(op["seed"]), early_stopping=True)
    return MLPClassifier(**kw) if task == "classification" else MLPRegressor(**kw)


def _fac_gbdt(op, task):
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    kw = dict(n_estimators=int(op["n_estimators"]), learning_rate=float(op["lr"]),
              max_depth=int(op["max_depth"]), random_state=int(op["seed"]))
    return GradientBoostingClassifier(**kw) if task == "classification" else GradientBoostingRegressor(**kw)


register_model(ModelSpec(
    id="decision_tree", name="决策树", category="机器学习",
    desc="CART 决策树，自动识别分类/回归任务，输出特征重要性、混淆矩阵与预测结果。",
    tags=["决策树", "CART", "可解释"], demo_note="ecommerce",
    slots=[Slot("y", "因变量（目标）", types=(QUANT, CAT, ORDINAL)),
           Slot("xs", "自变量（特征）", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[
        Option("max_depth", "最大深度", "number", default=4, min=1, max=30),
        Option("min_samples_leaf", "叶节点最小样本", "number", default=5, min=1, max=100),
        TEST_RATIO_OPT, SEED_OPT, SAVE_OPT,
    ],
    runner=_ml_runner(_fac_tree, "决策树"),
))

register_model(ModelSpec(
    id="random_forest", name="随机森林", category="机器学习",
    desc="Bootstrap + 随机特征的集成树模型，抗过拟合，输出特征重要性与性能指标。",
    tags=["随机森林", "集成学习"], demo_note="ecommerce",
    slots=[Slot("y", "因变量（目标）", types=(QUANT, CAT, ORDINAL)),
           Slot("xs", "自变量（特征）", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[
        Option("n_estimators", "树数量", "number", default=200, min=10, max=1000),
        Option("max_depth", "最大深度（0=不限）", "number", default=0, min=0, max=30),
        TEST_RATIO_OPT, SEED_OPT, SAVE_OPT,
    ],
    runner=_ml_runner(_fac_rf, "随机森林"),
))

register_model(ModelSpec(
    id="svm", name="支持向量机(SVM)", category="机器学习",
    desc="基于最大间隔的分类/回归模型，支持线性/多项式/RBF 核。",
    tags=["SVM", "支持向量机"], demo_note="ecommerce",
    slots=[Slot("y", "因变量（目标）", types=(QUANT, CAT, ORDINAL)),
           Slot("xs", "自变量（特征）", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[
        Option("kernel", "核函数", "select", default="rbf",
               choices=[{"value": "rbf", "label": "RBF 高斯核"}, {"value": "linear", "label": "线性核"},
                        {"value": "poly", "label": "多项式核"}]),
        Option("C", "惩罚系数 C", "number", default=1.0, min=0.01, step=0.5),
        TEST_RATIO_OPT, SEED_OPT, SAVE_OPT,
    ],
    runner=_ml_runner(_fac_svm, "支持向量机"),
))

register_model(ModelSpec(
    id="knn", name="K近邻(KNN)", category="机器学习",
    desc="按最近 K 个邻居投票/平均的惰性学习算法。",
    tags=["KNN", "近邻"], demo_note="ecommerce",
    slots=[Slot("y", "因变量（目标）", types=(QUANT, CAT, ORDINAL)),
           Slot("xs", "自变量（特征）", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[
        Option("k", "邻居数 K", "number", default=5, min=1, max=50),
        Option("weights", "权重方式", "select", default="uniform",
               choices=[{"value": "uniform", "label": "等权重"}, {"value": "distance", "label": "距离倒数加权"}]),
        TEST_RATIO_OPT, SEED_OPT, SAVE_OPT,
    ],
    runner=_ml_runner(_fac_knn, "K近邻"),
))

register_model(ModelSpec(
    id="naive_bayes", name="朴素贝叶斯", category="机器学习",
    desc="基于贝叶斯定理与特征条件独立假设的分类算法，适合文本/小样本。",
    tags=["朴素贝叶斯", "分类"], demo_note="survey",
    slots=[Slot("y", "因变量（定类，分类任务）", types=(CAT, ORDINAL)),
           Slot("xs", "自变量（特征）", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[TEST_RATIO_OPT, SEED_OPT, SAVE_OPT],
    runner=_ml_runner(_fac_nb, "朴素贝叶斯"),
))

register_model(ModelSpec(
    id="mlp", name="神经网络(MLP)", category="机器学习",
    desc="多层感知机，支持自定义隐藏层结构，自动早停防止过拟合。",
    tags=["神经网络", "MLP", "深度学习"], demo_note="ecommerce",
    slots=[Slot("y", "因变量（目标）", types=(QUANT, CAT, ORDINAL)),
           Slot("xs", "自变量（特征）", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[
        Option("layers", "隐藏层结构（逗号分隔神经元数）", "text", default="64,32"),
        Option("max_iter", "最大迭代轮数", "number", default=300, min=50, max=2000),
        TEST_RATIO_OPT, SEED_OPT, SAVE_OPT,
    ],
    runner=_ml_runner(_fac_mlp, "神经网络"),
))

register_model(ModelSpec(
    id="gbdt", name="梯度提升树(GBDT)", category="机器学习",
    desc="逐步拟合残差的 Boosting 集成方法，通常精度优于单棵树与随机森林。",
    tags=["GBDT", "梯度提升", "集成学习"], demo_note="ecommerce",
    slots=[Slot("y", "因变量（目标）", types=(QUANT, CAT, ORDINAL)),
           Slot("xs", "自变量（特征）", types=(QUANT, CAT, ORDINAL), multiple=True)],
    options=[
        Option("n_estimators", "迭代轮数", "number", default=200, min=10, max=1000),
        Option("lr", "学习率", "number", default=0.1, min=0.01, max=0.5, step=0.05),
        Option("max_depth", "最大深度", "number", default=3, min=1, max=10),
        TEST_RATIO_OPT, SEED_OPT, SAVE_OPT,
    ],
    runner=_ml_runner(_fac_gbdt, "梯度提升树"),
))

# ================================================================ 孤立森林异常检测
def _run_isolation_forest(ds: Dataset, sl: dict, op: dict) -> Result:
    from sklearn.ensemble import IsolationForest
    res = Result()
    cols = list(sl["vars"])
    data = pd.concat([pd.to_numeric(ds.df[c], errors="coerce").rename(c) for c in cols], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 20:
        raise ModelError(f"异常检测需要 ≥20 行完整样本(当前 {len(data)})")
    X = data.values
    contamination = float(op["contamination"])
    model = IsolationForest(n_estimators=int(op["trees"]), contamination=contamination,
                            random_state=int(op["seed"])).fit(X)
    label = model.predict(X)          # -1 异常, 1 正常
    score = -model.score_samples(X)   # 越大越异常
    n_out = int((label == -1).sum())
    res.add_table("检测概要", ["样本量N", "变量数", "异常比例设定", "检出异常数", "异常占比"],
                  [[len(data), len(cols), f"{contamination * 100:.1f}%", n_out, f"{n_out / len(data) * 100:.2f}%"]],
                  note="孤立森林:随机切分特征空间,异常点路径更短、更易被孤立")
    out_idx = data.index[label == -1]
    order = np.argsort(-score[label == -1])[:10]
    res.add_table("异常得分最高的样本(前 10)", ["行号"] + cols + ["异常得分"],
                  [[str(data.index[i]) if not isinstance(data.index[i], int) else str(data.index[i] + 1)]
                   + [fnum(data.iloc[i][c], 3) for c in cols] + [fnum(score[i], 4)]
                   for i in [data.index.get_loc(ix) for ix in out_idx[order]]])
    if op["save_pred"]:
        colname = "是否异常"
        col = pd.Series(np.nan, index=ds.df.index, dtype="object")
        col.loc[data.index] = np.where(label == -1, "异常", "正常")
        ds.write_col(colname, col, "cat")
        res.add_note(f"已将检测结果写回数据集列「{colname}」")
    # 二维可视化:取前两个变量(或 PC1/PC2)
    if len(cols) >= 2:
        px, py = cols[0], cols[1]
        normal_pts = [[round(float(data.iloc[i][px]), 4), round(float(data.iloc[i][py]), 4)]
                      for i in range(len(data)) if label[i] == 1]
        out_pts = [[round(float(data.iloc[i][px]), 4), round(float(data.iloc[i][py]), 4)]
                   for i in range(len(data)) if label[i] == -1]
        res.add_chart("异常点分布", ch.scatter_chart(f"异常检测({px} × {py})",
                     {"正常": normal_pts, "异常": out_pts}, x_name=px, y_name=py, symbol=7))
    else:
        res.add_chart("异常得分分布", ch.hist_chart("异常得分分布", score, x_name="异常得分", normal_fit=False))
    res.add_interp(f"孤立森林在 {len(data)} 个样本中识别出 **{n_out} 个异常点**({n_out / len(data) * 100:.1f}%)。"
                   "异常得分越高越可能是离群点,建议结合业务含义复核(如欺诈、设备故障、录入错误)。")
    return res


register_model(ModelSpec(
    id="isolation_forest", name="异常检测(孤立森林)", category="机器学习",
    runner=_run_isolation_forest,
    desc="无监督异常检测，基于特征随机切分的路径长度识别离群样本，支持写回标记列。",
    tags=["孤立森林", "异常检测", "离群点"], demo_note="ecommerce",
    slots=[Slot("vars", "检测变量（定量，2个以上效果佳）", types=(QUANT,), multiple=True, max_count=15)],
    options=[
        Option("contamination", "异常比例设定", "number", default=0.05, min=0.005, max=0.3, step=0.005),
        Option("trees", "树数量", "number", default=100, min=10, max=500),
        Option("seed", "随机种子", "number", default=42, min=0),
        SAVE_OPT,
    ],
))
