"""降维与聚类:主成分、因子分析、K-means、层次聚类、DBSCAN、判别分析、典型相关、对应分析。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.cluster import hierarchy as hier

from ..core import charts as ch
from ..core.datasets import Dataset, as_numeric, get_series
from ..core.registry import CAT, ORDINAL, QUANT, ModelSpec, Option, Slot, register_model
from ..core.results import ModelError, Result, fnum, fp, judge_p


def _matrix(ds: Dataset, cols: list[str], standardize: bool = True) -> pd.DataFrame:
    data = pd.concat([as_numeric(ds, c).rename(c) for c in cols], axis=1).dropna()
    if len(data) < max(10, len(cols) + 2):
        raise ModelError(f"完整样本不足(当前 {len(data)} 行),降维/聚类类方法通常需要 ≥{max(10, len(cols) + 2)} 行且无缺失")
    if standardize:
        sd = data.std(ddof=1)
        data = (data - data.mean()) / sd.replace(0, 1)
    return data


# ================================================================ 主成分分析
def _run_pca(ds: Dataset, sl: dict, op: dict) -> Result:
    from sklearn.decomposition import PCA
    res = Result()
    cols = sl["vars"]
    data = _matrix(ds, cols, standardize=True)
    n, p = data.shape
    k = op["ncomp"] if op["ncomp"] and 0 < op["ncomp"] <= p else min(p, n - 1)
    pca = PCA(n_components=k).fit(data.values)
    eig = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    # KMO 与 Bartlett
    corr = np.corrcoef(data.values.T)
    inv = np.linalg.pinv(corr)
    partial = np.zeros_like(corr)
    d = np.sqrt(np.diag(inv))
    for i in range(p):
        for j in range(p):
            partial[i, j] = -inv[i, j] / (d[i] * d[j]) if i != j else 0
    r2sum, p2sum = 0.0, 0.0
    for i in range(p):
        for j in range(p):
            if i != j:
                r2sum += corr[i, j] ** 2
                p2sum += partial[i, j] ** 2
    kmo = r2sum / (r2sum + p2sum) if (r2sum + p2sum) else np.nan
    n_obs = len(data)
    chi2_bart = -(n_obs - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(corr)) if np.linalg.det(corr) > 0 else np.nan
    df_bart = p * (p - 1) / 2
    p_bart = sps.chi2.sf(chi2_bart, df_bart) if not np.isnan(chi2_bart) else np.nan
    res.add_table("KMO 和 Bartlett 检验", ["KMO 值", "Bartlett χ²", "df", "p"],
                  [[fnum(kmo, 3), fnum(chi2_bart, 3), int(df_bart), fp(p_bart)]],
                  note=("KMO≥0.8 极适合、" "≥0.7 适合、≥0.6 尚可、<0.5 不适合做主成分/因子分析"))
    cum = np.cumsum(evr)
    rows = []
    for i in range(k):
        rows.append([f"主成分{i + 1}", fnum(eig[i], 3), fnum(evr[i] * 100, 2) + "%", fnum(cum[i] * 100, 2) + "%"])
    res.add_table("方差解释率(特征值)", ["成分", "特征值", "方差解释率", "累积解释率"], rows,
                  note="通常以特征值>1 或累积解释率>80% 确定主成分个数")
    # 载荷矩阵
    loadings = pca.components_.T * np.sqrt(eig)
    lrows = []
    for i, c in enumerate(cols):
        lrows.append([c] + [fnum(loadings[i, j], 3) for j in range(k)])
    res.add_table("载荷矩阵(变量×主成分)", ["变量"] + [f"主成分{i+1}" for i in range(k)], lrows,
                  note="载荷=特征向量×sqrt(特征值),绝对值越大代表与该主成分关系越强")
    scores = pca.transform(data.values)
    if op["save_scores"]:
        for i in range(min(k, 5)):
            col = pd.Series(np.nan, index=ds.df.index, dtype=float)
            col.loc[data.index] = scores[:, i]
            ds.write_col(f"PC{i + 1}", col, "quant")
        res.add_note(f"已将前 {min(k, 5)} 个主成分得分写回数据集(PC1~PC{min(k,5)} 列)")
    res.add_chart("碎石图", ch.line_chart("碎石图", [f"主成分{i+1}" for i in range(k)],
                 {"特征值": [round(float(v), 4) for v in eig]}, x_name="成分", smooth=False))
    res.add_chart("方差解释率", ch.bar_chart("各主成分方差解释率", [f"主成分{i+1}" for i in range(k)],
                 {"方差解释率%": [round(float(v * 100), 2) for v in evr]}, y_name="%"))
    if k >= 2:
        pts = [[float(r[0]), float(r[1])] for r in scores]
        res.add_chart("主成分得分散点图", ch.scatter_chart("主成分得分(PC1 × PC2)", {"样本": pts},
                     x_name="PC1", y_name="PC2", symbol=6))
    if op["weight"]:
        wrows = []
        for i, c in enumerate(cols):
            w = sum(loadings[i, j] ** 2 * evr[j] for j in range(k) if eig[j] > 1)
            wrows.append([c, fnum(w, 4)])
        tot = sum(float(r[1]) for r in wrows)
        if tot > 0:
            res.add_table("主成分权重(信息贡献法)", ["变量", "权重(归一化前)", "最终权重"],
                          [[r[0], r[1], fnum(float(r[1]) / tot * 100, 2) + "%"] for r in wrows])
    res.add_interp(
        f"KMO={fnum(kmo, 3)}({'适合' if kmo >= 0.7 else '勉强适合' if kmo >= 0.6 else '不适合'})做主成分分析;"
        f"Bartlett 球形检验 p={fp(p_bart)}(显著说明变量间存在相关,适合降维)。"
        + (f"前 {k} 个主成分累积解释 **{fnum(cum[k-1] * 100, 1)}%** 的信息"
           f"{';特征值>1 的成分有 ' + str(int((eig > 1).sum())) + ' 个,可作为最终主成分个数' if (eig > 1).any() else ''}。"))
    return res


register_model(ModelSpec(
    id="pca", name="主成分分析(PCA)", category="降维聚类",
    runner=_run_pca,
    desc="将多个相关变量压缩为少数主成分，输出特征值、方差解释率、载荷矩阵、得分与客观权重。",
    tags=["PCA", "降维", "权重"], demo_note="region",
    slots=[Slot("vars", "分析变量（定量，≥2）", types=(QUANT,), multiple=True)],
    options=[
        Option("ncomp", "主成分个数（0=自动）", "number", default=0, min=0, max=30),
        Option("save_scores", "保存得分到数据集", "switch", default=True),
        Option("weight", "输出主成分权重", "switch", default=False),
    ],
))


# ================================================================ 探索性因子分析
def _varimax(loadings: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Kaiser 方差最大化正交旋转"""
    p, k = loadings.shape
    R = np.eye(k)
    d_old = 0
    for _ in range(max_iter):
        L = loadings @ R
        u, s, vt = np.linalg.svd(L.T @ (L ** 3 - L @ np.diag(np.diag(L.T @ L)) / p))
        R = u @ vt
        d = s.sum()
        if d_old != 0 and d < d_old * (1 + tol):
            break
        d_old = d
    return loadings @ R


def _run_factor(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    cols = sl["vars"]
    data = _matrix(ds, cols, standardize=True)
    p = data.shape[1]
    k = op["nfac"] if op["nfac"] and 0 < op["nfac"] <= p else max(2, int(np.ceil(np.sqrt(p))))
    X = data.values
    corr = np.corrcoef(X.T)
    # 主轴法:以 R 的特征向量构造初始载荷
    eig_all, vec_all = np.linalg.eigh(corr)
    order = np.argsort(eig_all)[::-1]
    eig_all = np.clip(eig_all[order], 1e-9, None)
    vec_all = vec_all[:, order]
    if op["nfac"] == 0:
        k = max(1, int((eig_all > 1).sum()))
    load0 = vec_all[:, :k] * np.sqrt(eig_all[:k])
    loadings = _varimax(load0) if op["rotation"] == "varimax" else load0
    # 共同度与特征值
    comm = (loadings ** 2).sum(axis=1)
    ss = (loadings ** 2).sum(axis=0)
    ev = ss
    cum = np.cumsum(ev / p)
    rows = [[f"因子{i+1}", fnum(ev[i], 3), fnum(ev[i] / p * 100, 2) + "%", fnum(cum[i] * 100, 2) + "%"]
            for i in range(k)]
    res.add_table("方差解释", ["因子", "特征值(载荷平方和)", "方差解释率", "累积解释率"], rows)
    lrows = []
    for i, c in enumerate(cols):
        lrows.append([c] + [fnum(loadings[i, j], 3) for j in range(k)] + [fnum(comm[i], 3)])
    res.add_table(f"旋转后因子载荷矩阵({op['rotation']})", ["变量"] + [f"因子{i+1}" for i in range(k)] + ["共同度"],
                  lrows, note="载荷绝对值≥0.5 视为该因子的重要题目;共同度=各因子对该变量解释比例之和")
    # 因子得分(回归法)
    inv = np.linalg.pinv(corr)
    scores = X @ inv @ loadings
    if op["save_scores"]:
        for i in range(min(k, 5)):
            col = pd.Series(np.nan, index=ds.df.index, dtype=float)
            col.loc[data.index] = scores[:, i]
            ds.write_col(f"F{i + 1}", col, "quant")
        res.add_note(f"已保存因子得分列 F1~F{min(k,5)}(回归法)")
    res.add_chart("因子载荷热力图", ch.heatmap_chart("因子载荷热力图", [f"因子{i+1}" for i in range(k)],
                 cols, loadings.tolist(), fmt="{c:.2f}", v_min=-1, v_max=1))
    res.add_chart("碎石图", ch.line_chart("碎石图", [f"因子{i+1}" for i in range(p)],
                 {"特征值": [round(float(v), 4) for v in eig_all]}, x_name="因子", smooth=False))
    res.add_interp(
        f"共提取 {k} 个因子({op['rotation']}旋转),累积解释率 {fnum(cum[k-1] * 100, 1)}%。"
        f"特征值>1 的因子个数为 {int((eig_all > 1).sum())}(碎石图拐点可辅助判断);"
        "载荷绝对值≥0.5 的题目归属对应因子,共同度<0.4 的题目可考虑删除;同因子内题目应具有共同语义以便命名。")
    return res


register_model(ModelSpec(
    id="factor", name="探索性因子分析", category="降维聚类",
    runner=_run_factor,
    desc="提取潜在公因子并旋转，输出载荷矩阵、共同度与因子得分，常用于量表结构效度分析。",
    tags=["因子分析", "效度", "旋转"], demo_note="survey",
    slots=[Slot("vars", "题目变量（定量，≥3）", types=(QUANT,), multiple=True)],
    options=[
        Option("nfac", "因子个数（0=自动）", "number", default=0, min=0, max=30),
        Option("rotation", "旋转方法", "select", default="varimax",
               choices=[{"value": "varimax", "label": "方差最大化（varimax）"}, {"value": "none", "label": "不旋转"}]),
        Option("save_scores", "保存因子得分到数据集", "switch", default=True),
    ],
))


# ================================================================ K-means 聚类
def _run_kmeans(ds: Dataset, sl: dict, op: dict) -> Result:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    res = Result()
    cols = sl["vars"]
    data = _matrix(ds, cols, standardize=op["standardize"])
    k = int(op["k"])
    if k < 2 or k >= len(data):
        raise ModelError("K 需在 2 与样本数之间")
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(data.values)
    labels = km.labels_
    inertia = float(km.inertia_)
    sil = float(silhouette_score(data.values, labels)) if 2 <= k <= len(data) - 1 else np.nan
    res.add_table("聚类模型摘要", ["聚类数K", "样本量N", "簇内平方和(SSE)", "轮廓系数(平均)"],
                  [[k, len(data), fnum(inertia, 3), fnum(sil, 4)]],
                  note="轮廓系数越接近 1 聚类结构越好;肘部法选择 K 时看 SSE 下降拐点")
    rows = []
    series: dict[str, list] = {}
    for c in range(k):
        sub = data.values[labels == c]
        row = [f"类别{c + 1}", int(len(sub))] + [fnum(sub[:, j].mean(), 3) for j in range(len(cols))]
        rows.append(row)
        series[f"类别{c + 1}"] = [round(float(v), 4) for v in sub.mean(axis=0)]
    res.add_table("各类别中心(标准化尺度)", ["类别", "样本量"] + cols, rows)
    # 肘部图
    if op["elbow"]:
        sses = []
        ks = list(range(1, min(9, len(data) - 1) + 1))
        for kk in ks:
            kmk = KMeans(n_clusters=kk, n_init=5, random_state=42).fit(data.values)
            sses.append(round(float(kmk.inertia_), 3))
        res.add_chart("肘部法则图", ch.line_chart("SSE 随 K 变化", [str(x) for x in ks],
                     {"SSE": sses}, x_name="K", smooth=False))
    res.add_chart("各类别中心均值对比", ch.line_chart("类别中心画像", cols, series,
                 x_name="变量", smooth=False))
    if len(cols) >= 2:
        pts_by_cluster: dict[str, list] = {}
        for c in range(k):
            sub = data.values[labels == c]
            pts_by_cluster[f"类别{c + 1}"] = [[float(r[0]), float(r[1])] for r in sub]
        res.add_chart("聚类散点图(前两个变量)", ch.scatter_chart("聚类结果散点图",
                     pts_by_cluster, x_name=cols[0], y_name=cols[1], symbol=7))
    if op["save"]:
        col = pd.Series(np.nan, index=ds.df.index, dtype="object")
        col.loc[data.index] = [f"类别{l + 1}" for l in labels]
        ds.write_col("聚类类别", col, "cat")
        res.add_note("已将聚类类别写回数据集(列:聚类类别)")
    sizes = [int((labels == c).sum()) for c in range(k)]
    res.add_interp(
        f"K-means 将 {len(data)} 个样本聚为 **{k}** 类,各类样本量 {sizes}。"
        f"平均轮廓系数 {fnum(sil, 3)}({'聚类结构良好' if sil > 0.5 else '结构一般' if sil > 0.25 else '结构较弱,建议调整 K 或变量'}),"
        f"可结合肘部图与中心画像解读各类特征。")
    return res


register_model(ModelSpec(
    id="kmeans", name="K-means 聚类", category="降维聚类",
    runner=_run_kmeans,
    desc="基于距离的快速聚类：自动输出类别中心、轮廓系数、肘部图与样本散点图。",
    tags=["聚类", "K-means", "细分"], demo_note="ecommerce",
    slots=[Slot("vars", "聚类变量（定量，≥2）", types=(QUANT,), multiple=True)],
    options=[
        Option("k", "聚类数 K", "number", default=3, min=2, max=20),
        Option("standardize", "聚类前标准化", "switch", default=True),
        Option("elbow", "附肘部法则图", "switch", default=True),
        Option("save", "保存类别到数据集", "switch", default=True),
    ],
))


# ================================================================ 层次聚类
def _run_hier(ds: Dataset, sl: dict, op: dict) -> Result:
    from sklearn.cluster import AgglomerativeClustering
    res = Result()
    cols = sl["vars"]
    data = _matrix(ds, cols, standardize=True)
    method = op["method"]
    metric = "euclidean"
    Z = hier.linkage(data.values, method=method, metric=metric)
    k = int(op["k"])
    labels = hier.fcluster(Z, t=k, criterion="maxclust")
    res.add_table("聚类参数", ["方法", "距离度量", "聚类数K", "样本量N"],
                  [[method, "欧氏距离", k, len(data)]])
    rows = []
    for c in range(1, k + 1):
        sub = data.values[labels == c]
        if len(sub) == 0:
            continue
        rows.append([f"类别{c}", int(len(sub))] + [fnum(sub[:, j].mean(), 3) for j in range(len(cols))])
    res.add_table("各类别中心(标准化尺度)", ["类别", "样本量"] + cols, rows)
    # 树状图 → ECharts tree
    max_show = 40
    if len(data) <= max_show:
        dnd = hier.dendrogram(Z, no_plot=True)
        # 构造 tree 数据
        def build(node):
            if node < len(data):
                return {"name": str(data.index[node])}
            left = int(Z[int(node) - len(data), 0])
            right = int(Z[int(node) - len(data), 1])
            return {"name": "", "children": [build(left), build(right)], "value": int(Z[int(node) - len(data), 2])}
        tree = build(len(data) * 2 - 2)
        option = {
            "title": {"text": "层次聚类树状图", "left": "center", "textStyle": {"fontSize": 13, "fontWeight": 600, "color": "#334155"}},
            "tooltip": {"trigger": "item"},
            "series": [{
                "type": "tree", "data": [tree], "orient": "TB", "top": "2%", "bottom": "12%",
                "symbol": "circle", "symbolSize": 7, "roam": True,
                "expandAndCollapse": False, "initialTreeDepth": -1,
                "label": {"fontSize": 10, "color": "#64748b", "position": "top", "distance": 4},
                "leaves": {"label": {"fontSize": 10, "color": "#334155"}},
                "lineStyle": {"color": "#94a3b8", "width": 1.2},
                "animationDuration": 400,
            }],
        }
        res.add_chart("树状图", option)
    else:
        res.add_note(f"样本量 {len(data)} > {max_show},为保持图表可读性未绘制树状图")
    if op["save"]:
        col = pd.Series(np.nan, index=ds.df.index, dtype="object")
        col.loc[data.index] = [f"类别{l}" for l in labels]
        ds.write_col("层次聚类类别", col, "cat")
        res.add_note("已将类别写回数据集(列:层次聚类类别)")
    res.add_interp(
        f"采用 {op['method']} 联接法将样本聚为 {k} 类,各类样本量 "
        f"{[int((labels == c).sum()) for c in range(1, k + 1)]};"
        "可通过树状图观察合并顺序——树杈越早合并的样本相似度越高。")
    return res


register_model(ModelSpec(
    id="hier_cluster", name="层次聚类", category="降维聚类",
    runner=_run_hier,
    desc="系统聚类（最短/最长/均值/Ward 联接），输出树状图与指定类数的分类结果。",
    tags=["层次聚类", "树状图", "Ward"], demo_note="region",
    slots=[Slot("vars", "聚类变量（定量，≥2）", types=(QUANT,), multiple=True)],
    options=[
        Option("k", "聚类数 K", "number", default=3, min=2, max=20),
        Option("method", "联接方法", "select", default="ward",
               choices=[{"value": "ward", "label": "Ward 方差最小化"}, {"value": "average", "label": "类平均法"},
                        {"value": "complete", "label": "最长距离法"}, {"value": "single", "label": "最短距离法"}]),
        Option("save", "保存类别到数据集", "switch", default=True),
    ],
))


# ================================================================ DBSCAN 密度聚类
def _run_dbscan(ds: Dataset, sl: dict, op: dict) -> Result:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    res = Result()
    cols = sl["vars"]
    data = _matrix(ds, cols, standardize=True)
    model = DBSCAN(eps=float(op["eps"]), min_samples=int(op["min_samples"])).fit(data.values)
    labels = model.labels_
    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters == 0:
        raise ModelError("未能形成任何簇:请增大 eps 或减小 min_samples")
    sil = float(silhouette_score(data.values, labels)) if n_clusters >= 2 else np.nan
    res.add_table("DBSCAN 聚类摘要", ["eps", "min_samples", "簇数", "噪声点数", "轮廓系数"],
                  [[op["eps"], op["min_samples"], n_clusters, n_noise, fnum(sil, 4)]])
    rows = []
    for c in sorted(set(labels)):
        sub = data.values[labels == c]
        nm = "噪声点" if c == -1 else f"簇{c + 1}"
        rows.append([nm, int(len(sub))] + [fnum(sub[:, j].mean(), 3) for j in range(len(cols))])
    res.add_table("各簇中心", ["簇", "样本量"] + cols, rows)
    if len(cols) >= 2:
        series: dict[str, list] = {}
        for c in sorted(set(labels)):
            nm = "噪声点" if c == -1 else f"簇{c + 1}"
            series[nm] = [[float(r[0]), float(r[1])] for r in data.values[labels == c]]
        res.add_chart("密度聚类散点图", ch.scatter_chart("DBSCAN 聚类结果", series,
                     x_name=cols[0], y_name=cols[1], symbol=7))
    if op["save"]:
        col = pd.Series(np.nan, index=ds.df.index, dtype="object")
        col.loc[data.index] = [("噪声" if l == -1 else f"簇{l + 1}") for l in labels]
        ds.write_col("DBSCAN簇", col, "cat")
    res.add_interp(
        f"DBSCAN 识别出 **{n_clusters}** 个密度簇与 {n_noise} 个噪声点"
        f"({n_noise / len(data) * 100:.1f}%)。eps 为邻域半径,min_samples 为核心点最小邻居数;"
        "增大 eps 会合并更多点,减小则产生更多噪声。")
    return res


register_model(ModelSpec(
    id="dbscan", name="DBSCAN 密度聚类", category="降维聚类",
    runner=_run_dbscan,
    desc="基于密度的空间聚类，可自动发现任意形状的簇并识别离群点，无需指定簇数。",
    tags=["DBSCAN", "密度聚类", "离群"], demo_note="ecommerce",
    slots=[Slot("vars", "聚类变量（定量，≥2）", types=(QUANT,), multiple=True)],
    options=[
        Option("eps", "邻域半径 eps", "number", default=0.5, min=0.01, step=0.1),
        Option("min_samples", "核心点最小邻居数", "number", default=5, min=2),
        Option("save", "保存结果到数据集", "switch", default=True),
    ],
))


# ================================================================ 判别分析
def _run_discriminant(ds: Dataset, sl: dict, op: dict) -> Result:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, confusion_matrix
    res = Result()
    xs = sl["vars"]
    gcol = sl["group"]
    data = pd.concat([_matrix(ds, xs, standardize=False), get_series(ds, gcol).astype(str).rename("__g__")], axis=1).dropna()
    X, y = data[xs].values, data["__g__"].values
    levels = pd.unique(y).tolist()
    if len(levels) < 2:
        raise ModelError("分组变量至少需要 2 个水平")
    lda = LinearDiscriminantAnalysis().fit(X, y)
    pred = lda.predict(X)
    acc = accuracy_score(y, pred)
    res.add_table("模型摘要", ["组数", "样本量", "判别准确率(回代)"],
                  [[len(levels), len(y), f"{acc * 100:.2f}%"]])
    cm = confusion_matrix(y, pred, labels=levels)
    res.add_table("混淆矩阵(实际×预测)", ["实际\\预测"] + [str(l) for l in levels] + ["正确率"],
                  [[str(levels[i])] + [int(v) for v in cm[i]] + [f"{cm[i][i] / cm[i].sum() * 100:.1f}%"]
                   for i in range(len(levels))])
    # 结构矩阵:组均值差与标准判别系数
    try:
        Xs = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)
        lda_s = LinearDiscriminantAnalysis().fit(Xs, y)
        coefs = lda_s.scalings_
        ncols = min(coefs.shape[1], 4)
        rows = []
        for i, c in enumerate(xs):
            rows.append([c] + [fnum(coefs[i, j], 3) for j in range(ncols)])
        res.add_table("标准化判别系数(前几个判别函数)", ["变量"] + [f"判别函数{j+1}" for j in range(ncols)], rows,
                      note="绝对值越大,该变量对判别分类的贡献越大")
    except Exception:  # noqa: BLE001
        pass
    res.add_table("各组中心(组均值)", ["变量"] + [str(l) for l in levels],
                  [[c] + [fnum(data.loc[data['__g__'] == l, c].mean(), 3) for l in levels] for c in xs])
    if coefs.shape[1] >= 2:
        scores = lda.transform(X)
        series: dict[str, list] = {}
        for li, l in enumerate(levels):
            series[str(l)] = [[float(r[0]), float(r[1])] for r in scores[y == l]]
        res.add_chart("判别得分空间分布", ch.scatter_chart("判别得分(LD1 × LD2)", series,
                     x_name="LD1", y_name="LD2", symbol=7))
    res.add_interp(
        f"线性判别分析(LDA/Fisher)回代准确率 **{acc * 100:.1f}%**,各类正确率见混淆矩阵;"
        "标准化判别系数显示各变量对分类的贡献大小,可用于判断关键区分变量。")
    return res


register_model(ModelSpec(
    id="discriminant", name="判别分析", category="降维聚类",
    runner=_run_discriminant,
    desc="根据已知组别的训练样本建立 Fisher 线性判别函数，对新样本分类并评估准确率。",
    tags=["判别分析", "LDA", "分类"], demo_note="ecommerce",
    slots=[Slot("group", "分组变量（定类，已知类别）", types=(CAT, ORDINAL)),
           Slot("vars", "判别变量（定量，≥2）", types=(QUANT,), multiple=True)],
    options=[],
))


# ================================================================ 典型相关分析
ALPHA_OPT_CANON = Option("alpha", "显著性水平 α", "select", default=0.05,
                         choices=[{"value": 0.1, "label": "0.1"}, {"value": 0.05, "label": "0.05"}, {"value": 0.01, "label": "0.01"}])


def _run_canonical(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    xs, ys = sl["xset"], sl["yset"]
    data = pd.concat([as_numeric(ds, c) for c in xs + ys], axis=1).dropna()
    if len(data) < len(xs) + len(ys) + 5:
        raise ModelError("样本量不足,典型相关要求 n > 变量总数 + 5")
    X = data[xs].values
    Y = data[ys].values
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    C_all = np.cov(np.hstack([Xc, Yc]).T)
    nx = len(xs)
    Cxx = np.atleast_2d(C_all[:nx, :nx])
    Cyy = np.atleast_2d(C_all[nx:, nx:])
    Cxy = np.atleast_2d(C_all[:nx, nx:])
    inv_xx = np.linalg.pinv(Cxx)
    inv_yy = np.linalg.pinv(Cyy)
    mat = inv_xx @ Cxy @ inv_yy @ Cxy.T
    eig = np.sort(np.real(np.linalg.eigvals(mat)))[::-1]
    eig = np.clip(eig, 0, None)
    cancorr = np.sqrt(eig)
    n = len(data)
    keep = min(len(xs), len(ys))
    rows = []
    for i in range(min(keep, len(cancorr))):
        lam = cancorr[i]
        wilks = float(np.prod(1 - cancorr[i:] ** 2))
        chi2 = -(n - 1 - (len(xs) + len(ys) + 1) / 2) * np.log(max(wilks, 1e-12))
        dfc = (len(xs) - i) * (len(ys) - i)
        pval = sps.chi2.sf(chi2, dfc)
        rows.append([i + 1, fnum(lam, 4), fnum(lam ** 2, 4), fnum(wilks, 4), fnum(chi2, 3), dfc, fp(pval),
                     "显著" if pval < op["alpha"] else "不显著"])
    res.add_table("典型相关系数与显著性检验", ["序号", "典型相关系数", "特征值平方", "Wilks'Λ", "χ²", "df", "p", "显著?"], rows)
    res.add_interp(
        f"第一典型相关系数为 **{fnum(cancorr[0], 4)}**,即「{'、'.join(xs[:2])}…」变量组与「{'、'.join(ys[:2])}…」变量组"
        f"的整体相关强度为 {fnum(cancorr[0], 3)};"
        + (f"第 1 对典型变量显著(p={rows[0][6]}),可继续解读典型载荷。" if rows and rows[0][7] == "显著" else "未见显著典型相关。"))
    return res


register_model(ModelSpec(
    id="canonical", name="典型相关分析", category="降维聚类",
    runner=_run_canonical,
    desc="分析两组变量整体之间的相关结构，输出典型相关系数及其显著性。",
    tags=["典型相关", "两组变量"], demo_note="survey",
    slots=[Slot("xset", "第一组变量", types=(QUANT,), multiple=True, max_count=10),
           Slot("yset", "第二组变量", types=(QUANT,), multiple=True, max_count=10)],
    options=[ALPHA_OPT_CANON],
))


# ================================================================ 对应分析
def _run_correspondence(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    a, b = sl["x"], sl["y"]
    sub = ds.df[[a, b]].dropna().astype(str)
    ct = pd.crosstab(sub[a], sub[b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise ModelError("交叉表维度不足")
    n = ct.values.sum()
    P = ct.values / n
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    E = np.outer(r, c)
    S = (P - E) / np.sqrt(E)
    U, sv, Vt = np.linalg.svd(S, full_matrices=False)
    k = 2
    row_scores = (U[:, :k] * sv[:k]) / np.sqrt(r)[:, None]
    col_scores = (Vt.T[:, :k] * sv[:k]) / np.sqrt(c)[:, None]
    inertia_total = float((sv ** 2).sum())
    dim_rows = [[f"维度{i+1}", fnum(sv[i], 4), fnum(sv[i] ** 2, 4), fnum(sv[i] ** 2 / inertia_total * 100, 2) + "%",
                 fnum(np.cumsum(sv ** 2 / inertia_total)[i] * 100, 2) + "%"] for i in range(min(4, len(sv)))]
    res.add_table("奇异值与维度解释", ["维度", "奇异值", "惯量", "解释比例", "累积比例"], dim_rows)
    res.add_table("行轮廓坐标", ["类别"] + [f"维度{i+1}" for i in range(k)],
                  [[str(idx)] + [fnum(row_scores[i, j], 4) for j in range(k)] for i, idx in enumerate(ct.index)])
    res.add_table("列轮廓坐标", ["类别"] + [f"维度{i+1}" for i in range(k)],
                  [[str(idx)] + [fnum(col_scores[i, j], 4) for j in range(k)] for i, idx in enumerate(ct.columns)])
    series: dict[str, list] = {"行类别": [[float(r[0]), float(r[1])] for r in row_scores],
                               "列类别": [[float(r[0]), float(r[1])] for r in col_scores]}
    res.add_chart("对应分析图(前两维度)", ch.scatter_chart("对应分析双标图", series,
                 x_name="维度1", y_name="维度2", symbol=9))
    res.add_interp(
        f"前两个维度共解释总惯量的 {fnum(np.cumsum(sv ** 2 / inertia_total)[1] * 100, 1)}%"
        f"(总惯量 {fnum(inertia_total, 4)},反映「{a}」与「{b}」的关联强度)。"
        "双标图中距离越近的行列类别关联越强,可据此进行类别定位与品牌/人群画像分析。")
    return res


register_model(ModelSpec(
    id="correspondence", name="对应分析", category="降维聚类",
    runner=_run_correspondence,
    desc="将定类交叉表映射到低维空间的双标图，直观展示行/列类别间的关联与定位。",
    tags=["对应分析", "交叉表", "定位图"], demo_note="survey",
    slots=[Slot("x", "行变量（定类）", types=(CAT, ORDINAL)),
           Slot("y", "列变量（定类）", types=(CAT, ORDINAL))],
    options=[],
))
