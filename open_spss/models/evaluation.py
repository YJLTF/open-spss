"""综合评价:AHP、熵值法、TOPSIS、灰色关联、模糊综合评价、RSR、耦合协调度、CRITIC、DEA、组合权重。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import charts as ch
from ..core.datasets import Dataset, as_numeric, get_series
from ..core.registry import CAT, QUANT, ModelSpec, Option, Slot, register_model
from ..core.results import ModelError, Result, fnum, fp, judge_p

RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.54}


def _parse_matrix(text: str) -> np.ndarray:
    rows = []
    for line in (text or "").strip().splitlines():
        line = line.replace("，", ",").replace(" ", ",").replace("\t", ",")
        vals = [v for v in line.split(",") if v != ""]
        if vals:
            rows.append([float(v) for v in vals])
    if not rows:
        raise ModelError("请填写判断矩阵(每行一行,数值用逗号分隔)")
    n = len(rows)
    if any(len(r) != n for r in rows):
        raise ModelError(f"判断矩阵需为方阵(当前行数 {n},列数不一致)")
    return np.asarray(rows, dtype=float)


def _entropy_weights(mat: np.ndarray, neg_cols: list[int]) -> tuple[np.ndarray, pd.DataFrame]:
    """熵值法:n×m 矩阵 → 权重与归一化表"""
    X = mat.astype(float).copy()
    m = X.shape[1]
    for j in range(m):
        col = X[:, j]
        rng = col.max() - col.min()
        if rng == 0:
            X[:, j] = np.full(len(col), 0.5)
        elif j in neg_cols:
            X[:, j] = (col.max() - col) / rng
        else:
            X[:, j] = (col - col.min()) / rng
    P = X / (X.sum(axis=0, keepdims=True) + 1e-12)
    P = np.clip(P, 1e-12, None)
    k = 1 / np.log(len(X))
    e = -k * (P * np.log(P)).sum(axis=0)
    d = 1 - e
    w = d / d.sum()
    return w, pd.DataFrame(X, columns=[f"指标{j+1}" for j in range(m)])


def _parse_rect(text: str) -> np.ndarray:
    """解析任意行列的数值矩阵(行=行,逗号分隔)"""
    rows = []
    for line in (text or "").strip().splitlines():
        line = line.replace("，", ",").replace(" ", ",").replace("	", ",")
        vals = [v for v in line.split(",") if v != ""]
        if vals:
            rows.append([float(v) for v in vals])
    if not rows:
        raise ModelError("请填写矩阵(每行一行,数值用逗号分隔)")
    ncol = len(rows[0])
    if any(len(r) != ncol for r in rows):
        raise ModelError("矩阵每行列数需一致")
    return np.asarray(rows, dtype=float)


def _parse_neg(op: dict, key: str = "neg_cols") -> list[int]:
    raw = str(op.get(key) or "").replace("，", ",").strip()
    if not raw:
        return []
    out = []
    for v in raw.split(","):
        v = v.strip()
        if v and v.isdigit() and int(v) >= 1:
            out.append(int(v) - 1)
    return out


def _eval_matrix(ds: Dataset, sl: dict, neg: list[int]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """评价矩阵:行=评价对象(每行样本),列=指标"""
    data = pd.concat([as_numeric(ds, c).rename(c) for c in sl["vars"]], axis=1).dropna()
    if len(data) < 2:
        raise ModelError("有效评价对象不足 2 行")
    return data, data.values.astype(float), [f"{i+1}.{c}" for i, c in enumerate(sl["vars"])]


# ================================================================ AHP
def _run_ahp(ds: Dataset, sl: dict, op: dict) -> Result:  # noqa: ARG001
    res = Result()
    A = _parse_matrix(op["matrix"])
    n = A.shape[0]
    eigvals, eigvecs = np.linalg.eig(A)
    idx = int(np.argmax(eigvals.real))
    lmax = float(eigvals.real[idx])
    w = np.abs(eigvecs[:, idx].real)
    w = w / w.sum()
    CI = (lmax - n) / (n - 1) if n > 1 else 0.0
    RI = RI_TABLE.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0
    names = [n.strip() for n in str(op.get("names") or "").replace("，", ",").split(",") if n.strip()]
    if len(names) != n:
        names = [f"指标{i+1}" for i in range(n)]
    order = np.argsort(-w)
    res.add_table("AHP 权重结果(按权重降序)", ["指标", "权重", "权重%"],
                  [[names[i], fnum(w[i], 4), fnum(w[i] * 100, 2) + "%"] for i in order])
    res.add_table("一致性检验", ["λmax", "CI", "RI", "CR", "结论"],
                  [[fnum(lmax, 4), fnum(CI, 4), RI, fnum(CR, 4),
                    "通过(CR<0.1)" if CR < 0.1 else "未通过,请调整判断矩阵"]])
    res.add_chart("权重分布", ch.bar_chart("AHP 权重分布", [names[i] for i in order],
                 {"权重": [round(float(w[i]), 4) for i in order]}, y_name="权重"))
    res.add_interp(
        f"最大特征值 λmax={fnum(lmax, 4)},一致性比率 CR={fnum(CR, 4)} "
        + ("≤0.1,判断矩阵通过一致性检验,权重可用。" if CR < 0.1 else ">0.1,未通过一致性检验,请调整矩阵中互相矛盾的判断。")
        + f"权重最高的指标为「{names[order[0]]}」({fnum(w[order[0]] * 100, 1)}%)。")
    return res


register_model(ModelSpec(
    id="ahp", name="层次分析法(AHP)", category="综合评价",
    runner=_run_ahp,
    desc="输入 n×n 判断矩阵,计算主观权重、最大特征值与一致性比率 CR。",
    tags=["AHP", "主观赋权", "一致性"], demo_note="region",
    slots=[],
    options=[
        Option("matrix", "判断矩阵(每行一行,用逗号分隔)", "textarea", default="1,2,3\n0.5,1,2\n0.333,0.5,1"),
        Option("names", "指标名称(逗号分隔,选填)", "text", default=""),
    ],
))


# ================================================================ 熵值法
def _run_entropy(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    data, mat, labels = _eval_matrix(ds, sl, _parse_neg(op))
    w, _ = _entropy_weights(mat, _parse_neg(op))
    # 输出原指标与方向
    raw_rows = []
    for i in range(len(data)):
        raw_rows.append([str(data.index[i])] + [fnum(v, 3) for v in mat[i]])
    res.add_table("原始数据矩阵", ["对象"] + [sl["vars"][j] for j in range(len(sl["vars"]))], raw_rows)
    neg_set = _parse_neg(op)
    rows = []
    for j in range(len(sl["vars"])):
        rows.append([sl["vars"][j], "负向(成本型)" if j in neg_set else "正向(效益型)",
                     fnum(w[j], 4), fnum(w[j] * 100, 2) + "%"])
    order = np.argsort(-w)
    res.add_table("熵值法权重", ["指标", "指标方向", "权重", "权重%"],
                  [rows[order[i]] for i in range(len(rows))],
                  note="熵值越小、离散程度越大的指标提供的信息越多,权重越高")
    res.add_chart("熵权分布", ch.bar_chart("熵值法权重分布",
                 [rows[order[i]][0] for i in range(len(rows))],
                 {"权重": [round(float(w[order[i]]), 4) for i in range(len(rows))]}, y_name="权重"))
    res.add_interp(
        f"熵值法基于指标数据的离散程度客观赋权:信息熵越小(差异越大)的指标权重越高。"
        f"权重最高的指标为「{rows[order[0]][0]}」({fnum(w[order[0]] * 100, 1)}%)。")
    return res


register_model(ModelSpec(
    id="entropy", name="熵值法(熵权法)", category="综合评价",
    runner=_run_entropy,
    desc="依据指标数据自身的离散程度客观确定权重,常与 TOPSIS 等方法联用。",
    tags=["熵权", "客观赋权"], demo_note="region",
    slots=[Slot("vars", "评价指标(定量)", types=(QUANT,), multiple=True)],
    options=[Option("neg_cols", "负向指标序号(逗号分隔,选填)", "text", default="",
                    tip="如第 2 个指标为成本型(越小越好),填 2")],
))


# ================================================================ TOPSIS
def _run_topsis(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    data, mat, _ = _eval_matrix(ds, sl, _parse_neg(op))
    neg = _parse_neg(op)
    names = list(sl["vars"])
    # 归一化 + 方向统一
    X = mat.astype(float).copy()
    for j in range(X.shape[1]):
        rng = X[:, j].max() - X[:, j].min()
        if rng == 0:
            X[:, j] = 0.5
        else:
            X[:, j] = (X[:, j].max() - X[:, j]) / rng if j in neg else (X[:, j] - X[:, j].min()) / rng
    # 权重
    wname = op["weight_method"]
    if wname == "entropy":
        w, _ = _entropy_weights(mat, neg)
        res.add_table("熵权(TOPSIS 使用)", ["指标", "权重"], [[names[j], fnum(w[j], 4)] for j in range(len(names))])
    elif wname == "ahp":
        wa = _parse_matrix(op["ahp_matrix"])
        if wa.shape[0] != len(names):
            raise ModelError(f"AHP 矩阵维度({wa.shape[0]})与指标数({len(names)})不一致")
        eigvals, eigvecs = np.linalg.eig(wa)
        idx = int(np.argmax(eigvals.real))
        w = np.abs(eigvecs[:, idx].real)
        w = w / w.sum()
    else:
        w = np.full(len(names), 1 / len(names))
    V = X * w
    v_best, v_worst = V.max(axis=0), V.min(axis=0)
    d_plus = np.sqrt(((V - v_best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - v_worst) ** 2).sum(axis=1))
    closeness = d_minus / (d_plus + d_minus + 1e-12)
    rank = pd.Series(closeness).rank(ascending=False).astype(int)
    rows = []
    for i in range(len(data)):
        rows.append([str(data.index[i]), fnum(d_plus[i], 4), fnum(d_minus[i], 4),
                     fnum(closeness[i], 4), int(rank[i]),
                     f"第{int(rank[i])}名"])
    rows.sort(key=lambda r: r[4])
    res.add_table("TOPSIS 评价结果(按贴近度排名)", ["评价对象", "D+(与最优距离)", "D-(与最劣距离)", "相对贴近度 C", "排名", "结论"], rows,
                  note="C 越接近 1 表示综合表现越接近理想解")
    res.add_chart("TOPSIS 贴近度", ch.bar_chart("TOPSIS 相对贴近度排名",
                 [r[0] for r in rows], {"贴近度C": [round(float(r[3]), 4) for r in rows]},
                 y_name="C", horizontal=True))
    wlabel = {"equal": "均分权重", "entropy": "熵权法", "ahp": "AHP 权重"}[wname]
    res.add_interp(
        f"TOPSIS({wlabel})评价完成,"
        f"共 {len(data)} 个评价对象。综合表现最优为「{rows[0][0]}」(C={rows[0][3]}),"
        f"最末为「{rows[-1][0]}」(C={rows[-1][3]})。")
    return res


register_model(ModelSpec(
    id="topsis", name="TOPSIS 优劣解距离法", category="综合评价",
    runner=_run_topsis,
    desc="计算各评价对象与理想解/负理想解的相对贴近度并排序,支持均分/熵权/AHP 权重。",
    tags=["TOPSIS", "综合评价", "排名"], demo_note="region",
    slots=[Slot("vars", "评价指标(定量)", types=(QUANT,), multiple=True)],
    options=[
        Option("weight_method", "权重来源", "select", default="entropy",
               choices=[{"value": "entropy", "label": "熵权法(推荐)"}, {"value": "equal", "label": "均分权重"},
                        {"value": "ahp", "label": "AHP 判断矩阵"}]),
        Option("ahp_matrix", "AHP 判断矩阵(权重选 AHP 时)", "textarea", default=""),
        Option("neg_cols", "负向指标序号(逗号分隔)", "text", default=""),
    ],
))


# ================================================================ 灰色关联分析
def _run_grey(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    data, mat, _ = _eval_matrix(ds, sl, _parse_neg(op))
    neg = _parse_neg(op)
    names = list(sl["vars"])
    X = mat.astype(float)
    # 参考序列:每个指标的最优值(正向取最大,负向取最小)
    ref = np.array([X[:, j].min() if j in neg else X[:, j].max() for j in range(X.shape[1])])
    Y = np.abs(X - ref)
    dmin, dmax = Y.min(), Y.max()
    rho = float(op["rho"])
    xi = (dmin + rho * dmax) / (Y + rho * dmax)
    # 关联度:等权或熵权
    if op["weight_method"] == "entropy":
        w, _ = _entropy_weights(X, neg)
        res.add_table("熵权", ["指标", "权重"], [[names[j], fnum(w[j], 4)] for j in range(len(names))])
    else:
        w = np.full(len(names), 1 / len(names))
    degrees = xi @ w
    rank = pd.Series(degrees).rank(ascending=False).astype(int)
    rows = []
    for i in range(len(data)):
        rows.append([str(data.index[i])] + [fnum(xi[i, j], 4) for j in range(len(names))]
                    + [fnum(degrees[i], 4), int(rank[i])])
    rows.sort(key=lambda r: r[-1])
    res.add_table("灰色关联度(按关联度排名)", ["评价对象"] + [f"ξ({names[j]})" for j in range(len(names))]
                  + ["加权关联度", "排名"], rows,
                  note=f"分辨系数 ρ={rho};参考序列取各指标最优值")
    res.add_chart("灰色关联度排名", ch.bar_chart("灰色关联度排名",
                 [r[0] for r in rows], {"关联度": [round(float(r[-2]), 4) for r in rows]}, y_name="关联度", horizontal=True))
    res.add_interp(
        f"灰色关联分析完成,与理想参考序列关联度最高的是「{rows[0][0]}」(r={rows[0][-2]}),"
        f"说明其与最优目标的整体相似程度最高;各对象与各指标的关联系数见上表。")
    return res


register_model(ModelSpec(
    id="grey", name="灰色关联分析", category="综合评价",
    runner=_run_grey,
    desc="以各指标最优值为参考序列计算关联系数与加权关联度,适合小样本评价。",
    tags=["灰色关联", "GRA", "评价"], demo_note="region",
    slots=[Slot("vars", "评价指标(定量)", types=(QUANT,), multiple=True)],
    options=[
        Option("rho", "分辨系数 ρ", "number", default=0.5, min=0.1, max=1.0, step=0.1),
        Option("weight_method", "权重来源", "select", default="entropy",
               choices=[{"value": "entropy", "label": "熵权"}, {"value": "equal", "label": "均分"}]),
        Option("neg_cols", "负向指标序号(逗号分隔)", "text", default=""),
    ],
))


# ================================================================ 模糊综合评价
def _run_fuzzy(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    data, mat, _ = _eval_matrix(ds, sl, _parse_neg(op))
    neg = _parse_neg(op)
    names = list(sl["vars"])
    m = int(op["grades"])
    X = mat.astype(float)
    n = len(data)
    # 隶属度:将每个指标按升半梯形/降半梯形(以分位数阈值)映射到 m 个等级
    R = np.zeros((n, len(names), m))
    for j in range(len(names)):
        col = X[:, j]
        qs = np.quantile(col, np.linspace(0, 1, m + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        for i in range(n):
            for g in range(m):
                lo, hi = qs[g], qs[g + 1]
                # 三角/梯形隶属:完全落在区间内为 1,相邻区间线性衰减
                mid = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else (qs[1] / 2 if g == 0 else hi)
                if lo <= col[i] <= hi:
                    R[i, j, g] = 1.0
                elif col[i] < lo and g > 0 and np.isfinite(qs[g]):
                    R[i, j, g] = max(0.0, 1 - (qs[g] - col[i]) / max(qs[g] - qs[g - 1], 1e-9))
                elif col[i] > hi and g < m - 1 and np.isfinite(qs[g + 1]):
                    R[i, j, g] = max(0.0, 1 - (col[i] - qs[g + 1]) / max(qs[g + 2] - qs[g + 1], 1e-9) if g + 2 <= m else 0.0)
        if j in neg:
            R[:, j, :] = R[:, j, ::-1]
    R = np.clip(R, 0, None)
    # 权重
    if op["weight_method"] == "entropy":
        w, _ = _entropy_weights(X, neg)
    else:
        w = np.full(len(names), 1 / len(names))
    grade_vec = np.linspace(1, 5, m)  # 等级分值:1~5
    rows = []
    scores = np.zeros(n)
    for i in range(n):
        b = (R[i] * w[:, None]).sum(axis=0)  # M(·,⊕)
        b = b / b.sum() if b.sum() else b
        scores[i] = float(b @ grade_vec)
    rank = pd.Series(scores).rank(ascending=False).astype(int)
    for i in range(n):
        rows.append([str(data.index[i])] + [fnum(b_, 3) for b_ in (R[i] * w[:, None]).sum(axis=0)]
                    + [fnum(scores[i], 3), int(rank[i])])
    rows.sort(key=lambda r: r[-1])
    res.add_table("模糊综合评价结果", ["评价对象"] + [f"等级{g+1}隶属度" for g in range(m)] + ["综合得分(1~5)", "排名"], rows,
                  note="权重算子 M(·,⊕);等级分值取 1~5 等距")
    res.add_chart("模糊综合得分排名", ch.bar_chart("模糊综合评价得分",
                 [r[0] for r in rows], {"综合得分": [round(float(r[-2]), 3) for r in rows]}, y_name="得分", horizontal=True))
    res.add_interp(
        f"以 {m} 级评语集进行模糊综合评价,综合得分最高为「{rows[0][0]}」({rows[0][-2]}),"
        f"最低为「{rows[-1][0]}」({rows[-1][-2]});隶属度列显示各对象对各评价等级的归属程度。")
    return res


register_model(ModelSpec(
    id="fuzzy_eval", name="模糊综合评价", category="综合评价",
    runner=_run_fuzzy,
    desc="构造隶属度矩阵 R 与权重向量 W,按模糊算子合成评价向量,得到等级隶属度与综合得分。",
    tags=["模糊评价", "隶属度"], demo_note="region",
    slots=[Slot("vars", "评价指标(定量)", types=(QUANT,), multiple=True)],
    options=[
        Option("grades", "评语等级数", "select", default=3,
               choices=[{"value": 3, "label": "3 级(差/中/好)"}, {"value": 4, "label": "4 级"}, {"value": 5, "label": "5 级"}]),
        Option("weight_method", "权重来源", "select", default="entropy",
               choices=[{"value": "entropy", "label": "熵权"}, {"value": "equal", "label": "均分"}]),
        Option("neg_cols", "负向指标序号(逗号分隔)", "text", default=""),
    ],
))


# ================================================================ RSR 秩和比
def _run_rsr(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    data, mat, _ = _eval_matrix(ds, sl, _parse_neg(op))
    neg = _parse_neg(op)
    names = list(sl["vars"])
    n, mcols = mat.shape
    R = np.zeros_like(mat, dtype=float)
    for j in range(mcols):
        r = pd.Series(mat[:, j]).rank(ascending=(j not in neg), method="average")
        R[:, j] = r.values
    R = R / n  # 比例秩 R/n
    rsr = R.mean(axis=1)
    rank = pd.Series(rsr).rank(ascending=False).astype(int)
    w = np.full(mcols, 1 / mcols) if not op["use_weights"] else _entropy_weights(mat, neg)[0]
    if op["use_weights"]:
        wrsr = (R * w).sum(axis=1)
    else:
        wrsr = rsr
    rank = pd.Series(wrsr).rank(ascending=False).astype(int)
    rows = []
    for i in range(n):
        rows.append([str(data.index[i])] + [fnum(R[i, j], 3) for j in range(mcols)]
                    + [fnum(wrsr[i], 4), int(rank[i]),
                       "优" if rank[i] <= n / 3 else "中" if rank[i] <= 2 * n / 3 else "差"])
    rows.sort(key=lambda r: r[-2])
    res.add_table("RSR 秩和比评价(按 RSR 降序)", ["评价对象"] + [f"R/{n}({names[j]})" for j in range(mcols)]
                  + ["RSR值", "排名", "分档(三档)"], rows,
                  note="正向指标按升序编秩、负向指标按降序编秩;RSR=ΣR/(n·m)")
    res.add_chart("RSR 排名", ch.bar_chart("RSR 值排名", [r[0] for r in rows],
                 {"RSR": [round(float(r[-3]), 4) for r in rows]}, y_name="RSR", horizontal=True))
    res.add_interp(
        f"RSR 取值越接近 1 综合水平越高。本次评价中「{rows[0][0]}」RSR={rows[0][-3]} 排名第一;"
        f"按三档划分:优 {sum(1 for r in rows if r[-1]=='优')} 个、中 {sum(1 for r in rows if r[-1]=='中')} 个、差 {sum(1 for r in rows if r[-1]=='差')} 个。")
    return res


register_model(ModelSpec(
    id="rsr", name="秩和比(RSR)综合评价", category="综合评价",
    runner=_run_rsr,
    desc="基于秩次的非参数综合评价,对数据分布无要求,输出 RSR 值、排名与分档。",
    tags=["RSR", "秩和比", "分档"], demo_note="region",
    slots=[Slot("vars", "评价指标(定量)", types=(QUANT,), multiple=True)],
    options=[
        Option("use_weights", "使用熵权加权 RSR", "switch", default=False),
        Option("neg_cols", "负向指标序号(逗号分隔)", "text", default=""),
    ],
))


# ================================================================ 耦合协调度
def _run_coupling(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    systems = [s for s in (sl.get("s1"), sl.get("s2"), sl.get("s3"), sl.get("s4")) if s]
    if len(systems) < 2:
        raise ModelError("耦合协调度至少需要 2 个子系统")
    # 每个系统的综合发展指数(熵权 + minmax 正向化)
    us = []
    sys_names = [f"系统{i+1}" for i in range(len(systems))]
    for si, cols in enumerate(systems):
        data = pd.concat([as_numeric(ds, c).rename(c) for c in cols], axis=1).dropna()
        mat = data.values.astype(float)
        w, Xn = _entropy_weights(mat, _parse_neg(op))
        u = (Xn * w).sum(axis=1)
        us.append(pd.Series(u, index=data.index))
        res.add_table(f"系统{si+1} 熵权与指数", ["指标", "权重"],
                      [[cols[j], fnum(w[j], 4)] for j in range(len(cols))])
    U = pd.concat(us, axis=1).dropna()
    U.index = U.index.astype(str)
    U.columns = sys_names
    m = U.shape[1]
    C = (U.prod(axis=1)) ** (1 / m) / (U.sum(axis=1) / m) ** m * m ** (m / (m - 1)) if False else None
    # 标准耦合度 C = m元 (Πu)^(1/m) / Σu/m ... 使用通用公式: C = [Π(u_i)]^(1/m) * m^(1/2) / Σu? 采用常用形式:
    prod = U.prod(axis=1)
    ssum = U.sum(axis=1)
    C = ((prod ** (1 / m)) / ssum.replace(0, np.nan) * m).clip(0, 1)
    T = U.mean(axis=1)  # 等权协调指数
    D = np.sqrt(C * T)
    rank = pd.Series(D).rank(ascending=False).astype(int)
    level = pd.cut(D, [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01],
                   labels=["失调", "濒临失调", "勉强协调", "初级协调", "良好协调", "优质协调"])
    rows = []
    for i in U.index:
        rows.append([str(i)] + [fnum(U.loc[i, c], 4) for c in U.columns]
                    + [fnum(C.loc[i], 4), fnum(T.loc[i], 4), fnum(D.loc[i], 4),
                       str(level.loc[i]), int(rank[i])])
    rows.sort(key=lambda r: r[-1])
    res.add_table("耦合协调度结果", ["评价对象"] + [f"{c}指数" for c in U.columns]
                  + ["耦合度C", "协调指数T", "耦合协调度D", "等级", "排名"], rows,
                  note="D∈[0,1] 越大越协调;C 为系统间耦合度,T 为综合协调指数")
    res.add_chart("耦合协调度排名", ch.bar_chart("耦合协调度 D 排名", [r[0] for r in rows],
                 {"D值": [round(float(r[3 + m]), 4) for r in rows]}, y_name="D", horizontal=True))
    res.add_chart("各系统发展指数对比", ch.line_chart("各系统发展指数",
                 [r[0] for r in rows], {c: [round(float(U.loc[r[0], c]), 4) for r in rows] for c in U.columns},
                 x_name="评价对象", smooth=False))
    res.add_interp(
        f"共 {len(U)} 个评价对象、{m} 个子系统。耦合协调度最高的为「{rows[0][0]}」(D={rows[0][3 + m]},等级:{rows[0][4 + m]});"
        f"整体平均 D={fnum(D.mean(), 3)},处于「{str(level.mode().iloc[0])}」水平。")
    return res


register_model(ModelSpec(
    id="coupling", name="耦合协调度模型", category="综合评价",
    runner=_run_coupling,
    desc="衡量两个及以上子系统(如经济-环境)相互依赖、协调发展水平的 C-T-D 模型。",
    tags=["耦合协调", "系统", "可持续发展"], demo_note="region",
    slots=[Slot("s1", "系统1指标(定量)", types=(QUANT,), multiple=True),
           Slot("s2", "系统2指标(定量)", types=(QUANT,), multiple=True),
           Slot("s3", "系统3指标(选填)", types=(QUANT,), multiple=True, required=False),
           Slot("s4", "系统4指标(选填)", types=(QUANT,), multiple=True, required=False)],
    options=[Option("neg_cols", "负向指标序号(全局,逗号分隔)", "text", default="")],
))


# ================================================================ CRITIC 权重法
def _run_critic(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    data, mat, _ = _eval_matrix(ds, sl, _parse_neg(op))
    X = mat.astype(float)
    m = X.shape[1]
    sd = X.std(axis=0, ddof=1)
    corr = np.corrcoef(X.T)
    conflict = np.array([sd[j] * sum(1 - abs(corr[j, k2]) for k2 in range(m) if k2 != j) for j in range(m)])
    C = conflict
    w = C / C.sum()
    rows = [[sl["vars"][j], fnum(sd[j], 4), fnum(conflict[j], 4), fnum(w[j], 4), fnum(w[j] * 100, 2) + "%"]
            for j in range(m)]
    order = np.argsort(-w)
    res.add_table("CRITIC 权重", ["指标", "标准差(对比强度)", "冲突性", "权重", "权重%"],
                  [rows[order[i]] for i in range(m)],
                  note="CRITIC = 对比强度(标准差) × 冲突性(相关系数和),兼顾指标变异与指标间信息重叠")
    res.add_chart("CRITIC 权重", ch.bar_chart("CRITIC 权重分布",
                 [rows[order[i]][0] for i in range(m)],
                 {"权重": [round(float(w[order[i]]), 4) for i in range(m)]}, y_name="权重"))
    res.add_interp(
        f"CRITIC 法同时考虑指标的变异程度与指标间相关性:标准差越大、与其他指标相关越弱,权重越高。"
        f"权重最高为「{rows[order[0]][0]}」({fnum(w[order[0]] * 100, 1)}%)。")
    return res


register_model(ModelSpec(
    id="critic", name="CRITIC 权重法", category="综合评价",
    runner=_run_critic,
    desc="以对比强度(标准差)与指标间冲突性(相关性)确定客观权重。",
    tags=["CRITIC", "客观赋权"], demo_note="region",
    slots=[Slot("vars", "评价指标(定量)", types=(QUANT,), multiple=True)],
    options=[Option("neg_cols", "负向指标序号(逗号分隔)", "text", default="")],
))


# ================================================================ DEA 数据包络
def _run_dea(ds: Dataset, sl: dict, op: dict) -> Result:
    from scipy.optimize import linprog
    res = Result()
    inputs = list(sl["inputs"])
    outputs = list(sl["outputs"])
    data = pd.concat([as_numeric(ds, c) for c in inputs + outputs], axis=1).dropna()
    X = data[inputs].values.astype(float)
    Y = data[outputs].values.astype(float)
    n = len(data)
    if n < 3:
        raise ModelError("DMU(决策单元)数量不足")
    if (X <= 0).any() or (Y <= 0).any():
        raise ModelError("DEA(CCR)要求投入产出数据均为正数")
    eff = np.zeros(n)
    for o in range(n):
        # min θ  s.t. Σλx ≤ θ·x_o ;  Σλy ≥ y_o ; λ≥0
        c = np.zeros(1 + n)
        c[0] = 1.0
        A_ub, b_ub = [], []
        for r in range(X.shape[1]):
            row = np.zeros(1 + n)
            row[1:] = -X[:, r]
            row[0] = -X[o, r]
            A_ub.append(row)
            b_ub.append(0.0)
        for r in range(Y.shape[1]):
            row = np.zeros(1 + n)
            row[1:] = Y[:, r]
            A_ub.append(row)
            b_ub.append(Y[o, r])
        sp = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                     bounds=[(0, None)] + [(0, None)] * n, method="highs")
        eff[o] = sp.x[0] if sp.success else np.nan
    rank = pd.Series(-eff).rank(ascending=False).astype(int)
    rows = []
    for i in range(n):
        rows.append([str(data.index[i])] + [fnum(X[i, j], 3) for j in range(len(inputs))]
                    + [fnum(Y[i, j], 3) for j in range(len(outputs))]
                    + [fnum(eff[i], 4), "有效(DEA 强有效)" if eff[i] >= 0.999 else "非有效",
                       int(rank[i])])
    rows.sort(key=lambda r: r[-1])
    res.add_table("DEA-CCR 效率评价(投入导向)", ["DMU"] + [f"投入:{c}" for c in inputs] + [f"产出:{c}" for c in outputs]
                  + ["综合效率θ", "有效性", "排名"], rows,
                  note="θ=1 为 DEA 有效;θ<1 说明存在投入冗余,可等比例压缩投入至 θ 倍仍保持产出")
    valid_n = int((eff >= 0.999).sum())
    res.add_chart("DEA 效率值", ch.bar_chart("各 DMU 综合效率 θ", [r[0] for r in rows],
                 {"θ": [round(float(r[-3]), 4) for r in rows]}, y_name="θ", horizontal=True))
    res.add_interp(
        f"共评价 {n} 个决策单元,其中 **{valid_n}** 个 DEA 有效(θ=1),平均效率 {fnum(eff.mean(), 3)};"
        f"效率最低的单元为「{rows[-1][0]}」(θ={rows[-1][-3]}),需按 θ 比例压缩投入。")
    return res


register_model(ModelSpec(
    id="dea", name="数据包络分析(DEA-CCR)", category="综合评价",
    runner=_run_dea,
    desc="投入导向 CCR 模型,计算各决策单元的相对综合效率 θ,用于多投入多产出的效率评价。",
    tags=["DEA", "效率", "CCR"], demo_note="region",
    slots=[Slot("inputs", "投入指标(定量,>0)", types=(QUANT,), multiple=True, max_count=8),
           Slot("outputs", "产出指标(定量,>0)", types=(QUANT,), multiple=True, max_count=8)],
    options=[],
))


# ================================================================ 组合权重
def _run_combo_weight(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    data, mat, _ = _eval_matrix(ds, sl, _parse_neg(op))
    neg = _parse_neg(op)
    names = list(sl["vars"])
    w_e, _ = _entropy_weights(mat, neg)
    wa = _parse_matrix(op["ahp_matrix"])
    if wa.shape[0] != len(names):
        raise ModelError(f"AHP 矩阵维度({wa.shape[0]})与指标数({len(names)})不一致")
    eigvals, eigvecs = np.linalg.eig(wa)
    idx = int(np.argmax(eigvals.real))
    lmax = float(eigvals.real[idx])
    w_a = np.abs(eigvecs[:, idx].real)
    w_a = w_a / w_a.sum()
    n = len(names)
    CI = (lmax - n) / (n - 1) if n > 1 else 0.0
    CR = CI / RI_TABLE.get(n, 1.49)
    mode = op["mode"]
    if mode == "multiplicative":
        w_c = (w_a * w_e) / (w_a * w_e).sum()
    elif mode == "geometric":
        w_c = np.sqrt(w_a * w_e)
        w_c = w_c / w_c.sum()
    else:  # 加权平均
        a = float(op["beta"])
        w_c = a * w_a + (1 - a) * w_e
    rows = [[names[j], fnum(w_a[j], 4), fnum(w_e[j], 4), fnum(w_c[j], 4), fnum(w_c[j] * 100, 2) + "%"]
            for j in range(n)]
    order = np.argsort(-w_c)
    res.add_table("主客观组合权重", ["指标", "AHP 主观权重", "熵权客观权重", "组合权重", "组合权重%"],
                  [rows[order[i]] for i in range(n)],
                  note=f"组合方式:{ {'multiplicative':'乘法合成','geometric':'几何平均','average':'加权平均(β)'}[mode] };"
                       f"AHP 一致性 CR={fnum(CR,4)} {'通过' if CR<0.1 else '未通过'}")
    # 组合权重评分
    Xn = np.zeros_like(mat, dtype=float)
    for j in range(mat.shape[1]):
        rng = mat[:, j].max() - mat[:, j].min()
        if rng == 0:
            Xn[:, j] = 0.5
        elif j in neg:
            Xn[:, j] = (mat[:, j].max() - mat[:, j]) / rng
        else:
            Xn[:, j] = (mat[:, j] - mat[:, j].min()) / rng
    scores = Xn @ w_c
    rank = pd.Series(scores).rank(ascending=False).astype(int)
    srows = sorted([[str(data.index[i]), fnum(scores[i], 4), int(rank[i])] for i in range(len(data))],
                   key=lambda r: r[2])
    res.add_table("组合权重综合得分", ["评价对象", "综合得分", "排名"], srows)
    res.add_chart("三种权重对比", ch.bar_chart("AHP / 熵权 / 组合权重对比", names,
                 {"AHP权重": [round(float(v), 4) for v in w_a],
                  "熵权": [round(float(v), 4) for v in w_e],
                  "组合权重": [round(float(v), 4) for v in w_c]}))
    res.add_interp(
        f"组合权重兼顾专家经验(AHP,CR={'通过' if CR < 0.1 else '未通过'})与数据客观信息(熵权);"
        f"综合得分第一名为「{srows[0][0]}」(得分 {srows[0][1]})。")
    return res


register_model(ModelSpec(
    id="combo_weight", name="组合权重(AHP+熵权)", category="综合评价",
    runner=_run_combo_weight,
    desc="将 AHP 主观权重与熵权客观权重按乘法/几何/加权平均方式合成,并输出综合得分。",
    tags=["组合赋权", "AHP", "熵权"], demo_note="region",
    slots=[Slot("vars", "评价指标(定量)", types=(QUANT,), multiple=True)],
    options=[
        Option("ahp_matrix", "AHP 判断矩阵", "textarea", default="1,2,3\n0.5,1,2\n0.333,0.5,1"),
        Option("mode", "组合方式", "select", default="multiplicative",
               choices=[{"value": "multiplicative", "label": "乘法合成(归一化)"}, {"value": "geometric", "label": "几何平均"},
                        {"value": "average", "label": "加权平均"}]),
        Option("beta", "加权平均时 AHP 占比 β", "number", default=0.5, min=0, max=1, step=0.1),
        Option("neg_cols", "负向指标序号(逗号分隔)", "text", default=""),
    ],
))


# ================================================================ 变异系数权重
def _run_cv_weight(ds: Dataset, sl: dict, op: dict) -> Result:  # noqa: ARG001
    res = Result()
    data, mat, labels = _eval_matrix(ds, sl, [])
    means, stds = mat.mean(axis=0), mat.std(axis=0, ddof=1)
    if (means == 0).any():
        raise ModelError("存在均值为 0 的指标,变异系数无法计算")
    cv = stds / np.abs(means)
    if cv.sum() <= 0:
        raise ModelError("所有指标变异系数为 0,无法计算权重")
    w = cv / cv.sum()
    res.add_table("变异系数权重", ["指标", "均值", "标准差", "变异系数CV", "权重", "权重%"],
                  [[labels[j], fnum(means[j], 3), fnum(stds[j], 3), fnum(cv[j], 4), fnum(w[j], 4), fnum(w[j] * 100, 2) + "%"]
                   for j in range(mat.shape[1])],
                  note="CV 越大表示该指标区分度越高,赋予越大权重")
    res.add_chart("指标权重分布", ch.bar_chart("变异系数法权重", [l.split(".", 1)[-1] for l in labels],
                 {"权重%": [round(float(v) * 100, 2) for v in w]}, y_name="权重%"))
    top = int(np.argmax(w))
    res.add_interp(f"变异系数最大(区分度最高)的指标为「{labels[top]}」(权重 {w[top] * 100:.1f}%),最小为「{labels[int(np.argmin(w))]}」"
                   f"({w.min() * 100:.1f}%)。变异系数法以指标的离散程度客观赋权,常与 AHP/熵值法组合。")
    return res


register_model(ModelSpec(
    id="cv_weight", name="变异系数权重法", category="综合评价",
    runner=_run_cv_weight,
    desc="以各指标的变异系数(离散程度)归一化为权重,客观赋权方法。",
    tags=["变异系数", "客观赋权"], demo_note="region",
    slots=[Slot("vars", "指标变量(定量,≥2)", types=(QUANT,), multiple=True)],
    options=[],
))


# ================================================================ 主成分权重
def _run_pca_weight(ds: Dataset, sl: dict, op: dict) -> Result:
    res = Result()
    data, mat, labels = _eval_matrix(ds, sl, [])
    X = (mat - mat.mean(axis=0)) / mat.std(axis=0, ddof=1).clip(min=1e-12)
    corr = np.corrcoef(X.T)
    eig, vec = np.linalg.eigh(corr)
    order = np.argsort(-eig)
    eig, vec = eig[order], vec[:, order]
    p = len(labels)
    if op["ncomp"] and 0 < op["ncomp"] <= p:
        m = int(op["ncomp"])
    else:
        m = max(1, int((eig > 1).sum()))
    m = min(m, p)
    evr = eig / eig.sum()
    score = vec[:, :m] / np.sqrt(eig[:m])          # 线性组合系数
    F = score @ evr[:m]                             # 综合得分系数
    if np.allclose(F.sum(), 0):
        raise ModelError("综合得分系数接近 0,无法归一化")
    w = np.abs(F) / np.abs(F).sum()
    res.add_table("特征值与方差解释率", ["成分", "特征值", "方差解释率", "累积解释率", "是否入选"],
                  [[f"主成分{i+1}", fnum(eig[i], 3), fnum(evr[i] * 100, 2) + "%",
                    fnum(evr[:i + 1].sum() * 100, 2) + "%", "是" if i < m else "否"] for i in range(p)],
                  note=("按特征值>1 入选" if not op["ncomp"] else f"指定 {m} 个主成分"))
    res.add_table("主成分线性组合系数", ["变量"] + [f"主成分{i+1}" for i in range(m)],
                  [[labels[j].split(".", 1)[-1]] + [fnum(score[j, i], 4) for i in range(m)] for j in range(p)],
                  note="系数=特征向量/√特征值,可写出各主成分得分表达式")
    res.add_table("主成分权重结果", ["变量", "综合得分系数F", "权重", "权重%"],
                  [[labels[j].split(".", 1)[-1], fnum(F[j], 4), fnum(w[j], 4), fnum(w[j] * 100, 2) + "%"] for j in range(p)])
    res.add_chart("主成分权重", ch.bar_chart("主成分分析法权重", [labels[j].split(".", 1)[-1] for j in range(p)],
                 {"权重%": [round(float(v) * 100, 2) for v in w]}, y_name="权重%"))
    top = int(np.argmax(w))
    res.add_interp(f"入选 {m} 个主成分(累积解释 {evr[:m].sum() * 100:.1f}%)。权重最高的指标为「{labels[top]}」"
                   f"({w[top] * 100:.1f}%),最低为「{labels[int(np.argmin(w))]}」({w.min() * 100:.1f}%)。")
    return res


register_model(ModelSpec(
    id="pca_weight", name="主成分权重法", category="综合评价",
    runner=_run_pca_weight,
    desc="用主成分分析的方差信息确定指标客观权重(特征值>1 入选,输出线性组合系数)。",
    tags=["主成分", "客观赋权", "PCA"], demo_note="region",
    slots=[Slot("vars", "指标变量(定量,≥2)", types=(QUANT,), multiple=True)],
    options=[Option("ncomp", "主成分个数(0=特征值>1)", "number", default=0, min=0, max=30)],
))


# ================================================================ 因子分析权重
def _run_factor_weight(ds: Dataset, sl: dict, op: dict) -> Result:
    from .dimreduc import _varimax
    res = Result()
    data, mat, labels = _eval_matrix(ds, sl, [])
    names = [l.split(".", 1)[-1] for l in labels]
    X = (mat - mat.mean(axis=0)) / mat.std(axis=0, ddof=1).clip(min=1e-12)
    corr = np.corrcoef(X.T)
    eig, vec = np.linalg.eigh(corr)
    order = np.argsort(-eig)
    eig, vec = eig[order], vec[:, order]
    p = len(labels)
    m = int(op["ncomp"]) if op["ncomp"] and 0 < int(op["ncomp"]) <= p else max(1, int((eig > 1).sum()))
    evr = eig / eig.sum()
    loadings = vec[:, :m] * np.sqrt(eig[:m])
    if op["rotate"]:
        loadings = _varimax(loadings)
    # 每个因子内部:|载荷|归一化 → 乘以该因子方差贡献 → 求和
    contrib = np.zeros(p)
    for kk in range(m):
        col = np.abs(loadings[:, kk])
        s = col.sum()
        if s > 0:
            contrib += col / s * evr[kk]
    if contrib.sum() <= 0:
        raise ModelError("载荷矩阵退化,无法计算权重")
    w = contrib / contrib.sum()
    res.add_table("因子方差解释", ["因子", "特征值", "方差贡献率"],
                  [[f"因子{i+1}", fnum(eig[i], 3), fnum(evr[i] * 100, 2) + "%"] for i in range(m)],
                  note=f"共入选 {m} 个因子,累积贡献 {evr[:m].sum() * 100:.1f}%" + ("(已方差最大旋转)" if op["rotate"] else ""))
    res.add_table("旋转后载荷矩阵", ["变量"] + [f"因子{i+1}" for i in range(m)],
                  [[names[j]] + [fnum(loadings[j, i], 3) for i in range(m)] for j in range(p)])
    res.add_table("因子分析权重", ["变量", "贡献(归一化前)", "权重", "权重%"],
                  [[names[j], fnum(contrib[j], 4), fnum(w[j], 4), fnum(w[j] * 100, 2) + "%"] for j in range(p)])
    res.add_chart("因子分析权重", ch.bar_chart("因子分析权重", names, {"权重%": [round(float(v) * 100, 2) for v in w]}, y_name="权重%"))
    res.add_interp(f"基于 {m} 个公因子(累积解释 {evr[:m].sum() * 100:.1f}%)计算权重:最高「{names[int(np.argmax(w))]}」"
                   f"({w.max() * 100:.1f}%),最低「{names[int(np.argmin(w))]}」({w.min() * 100:.1f}%)。")
    return res


register_model(ModelSpec(
    id="factor_weight", name="因子分析权重法", category="综合评价",
    runner=_run_factor_weight,
    desc="由因子载荷矩阵确定指标权重,支持方差最大旋转,客观赋权方法。",
    tags=["因子分析", "客观赋权", "旋转载荷"], demo_note="region",
    slots=[Slot("vars", "指标变量(定量,≥2)", types=(QUANT,), multiple=True)],
    options=[
        Option("ncomp", "因子个数(0=特征值>1)", "number", default=0, min=0, max=30),
        Option("rotate", "方差最大旋转(varimax)", "switch", default=True),
    ],
))

# ================================================================ DEMATEL
def _run_dematel(ds: Dataset, sl: dict, op: dict) -> Result:  # noqa: ARG001
    res = Result()
    X = _parse_matrix(op["matrix"])
    n = X.shape[0]
    names = [t.strip() for t in str(op.get("names") or "").replace("，", ",").split(",") if t.strip()]
    if len(names) != n:
        names = [f"因素{i+1}" for i in range(n)]
    if (X < 0).any():
        raise ModelError("直接影响矩阵不能含负数(0~4 标度)")
    dmax = X.sum(axis=1).max()
    if dmax <= 0:
        raise ModelError("直接影响矩阵全为 0")
    G = X / dmax
    T = G @ np.linalg.inv(np.eye(n) - G)
    R = T.sum(axis=1)
    Cc = T.sum(axis=0)
    centrality = R + Cc
    cause = R - Cc
    res.add_table("规范化影响矩阵 G", ["因素"] + names,
                  [[names[i]] + [fnum(G[i, j], 4) for j in range(n)] for i in range(n)])
    res.add_table("综合影响矩阵 T", ["因素"] + names,
                  [[names[i]] + [fnum(T[i, j], 4) for j in range(n)] for i in range(n)])
    order = np.argsort(-centrality)
    res.add_table("中心度与原因度(按中心度降序)",
                  ["因素", "影响度R(行和)", "被影响度C(列和)", "中心度R+C", "原因度R−C", "属性"],
                  [[names[i], fnum(R[i], 4), fnum(Cc[i], 4), fnum(centrality[i], 4), fnum(cause[i], 4),
                    "原因因素" if cause[i] > 0 else "结果因素"] for i in order],
                  note="中心度越大对系统越关键;原因度>0 主动影响其他因素,<0 受其他因素影响")
    pad = float(np.abs(cause).max()) * 0.15 + 0.05
    res.add_chart("原因-结果图", ch.scatter_chart(
        "DEMATEL 因果图(横轴中心度,纵轴原因度)",
        {"原因因素": [[round(float(centrality[i]), 4), round(float(cause[i]), 4)] for i in range(n) if cause[i] > 0],
         "结果因素": [[round(float(centrality[i]), 4), round(float(cause[i]), 4)] for i in range(n) if cause[i] <= 0]},
        x_name="中心度 R+C", y_name="原因度 R−C",
        trend=[[round(float(centrality.min()) - pad, 4), 0], [round(float(centrality.max()) + pad, 4), 0]],
        trend_name="原因度=0 分界线"))
    causes = [names[i] for i in range(n) if cause[i] > 0]
    results = [names[i] for i in range(n) if cause[i] <= 0]
    key = names[order[0]]
    res.add_interp(f"中心度最高(最关键)的因素是「{key}」({centrality[order[0]]:.3f})。"
                   + (f"原因因素({', '.join(causes)})主动影响其他因素,应优先治理;" if causes else "")
                   + (f"结果因素({', '.join(results)})受他因素传导影响。" if results else ""))
    return res


register_model(ModelSpec(
    id="dematel", name="DEMATEL 决策实验室法", category="综合评价",
    runner=_run_dematel,
    desc="由直接影响矩阵求综合影响矩阵,识别原因/结果因素与系统关键因素。",
    tags=["DEMATEL", "影响矩阵", "系统分析"], demo_note="region",
    slots=[],
    options=[
        Option("matrix", "直接影响矩阵(0~4 标度,每行一行)", "textarea",
               default="0,2,3,1\n1,0,2,2\n0,1,0,3\n2,1,1,0"),
        Option("names", "因素名称(逗号分隔,选填)", "text", default=""),
    ],
))

# ================================================================ ISM 解释结构模型
def _run_ism(ds: Dataset, sl: dict, op: dict) -> Result:  # noqa: ARG001
    res = Result()
    A = _parse_matrix(op["matrix"])
    n = A.shape[0]
    names = [t.strip() for t in str(op.get("names") or "").replace("，", ",").split(",") if t.strip()]
    if len(names) != n:
        names = [f"因素{i+1}" for i in range(n)]
    if set(np.unique(A)) - {0, 1}:
        raise ModelError("邻接矩阵只允许 0/1")
    Rm = (A + np.eye(n)) > 0
    changed = True
    while changed:
        nxt = Rm | (Rm @ Rm)
        changed = not np.array_equal(nxt, Rm)
        Rm = nxt
    res.add_table("可达矩阵", ["因素"] + names,
                  [[names[i]] + ["1" if Rm[i, j] else "0" for j in range(n)] for i in range(n)])
    remain = set(range(n))
    levels: list[list[int]] = []
    while remain:
        lv = []
        for i in remain:
            reach = {j for j in remain if Rm[i, j]}
            ant = {j for j in remain if Rm[j, i]}
            if reach <= ant:
                lv.append(i)
        if not lv:
            lv = sorted(remain)
        levels.append(sorted(lv))
        remain -= set(lv)
    lv_of = {}
    for t, lv in enumerate(levels, 1):
        for i in lv:
            lv_of[i] = t
    res.add_table("层级划分结果", ["因素", "可达集R(i)", "前因集A(i)", "层级"],
                  [[names[i],
                    "{" + ",".join(names[j] for j in range(n) if Rm[i, j]) + "}",
                    "{" + ",".join(names[j] for j in range(n) if Rm[j, i]) + "}",
                    f"L{lv_of[i]}"] for i in range(n)],
                  note="同层因素满足可达集⊆前因集;顶层为直接目标,底层为根源因素")
    struct_rows = [[f"L{t}(第{t}层)", "、".join(names[i] for i in lv)] for t, lv in enumerate(levels, 1)]
    res.add_table("递阶结构(自顶向下)", ["层级", "因素"], struct_rows,
                  note="影响方向:低层 → 高层(底层因素逐层向上传导)")
    top = [names[i] for i in levels[0]]
    bottom = [names[i] for i in levels[-1]]
    res.add_interp(f"共划分为 **{len(levels)} 层**。顶层(直接目标/表现):{'、'.join(top)};"
                   f"底层(根源因素):{'、'.join(bottom)},应作为治理的优先切入点。")
    return res


register_model(ModelSpec(
    id="ism", name="ISM 解释结构模型", category="综合评价",
    runner=_run_ism,
    desc="由邻接矩阵求可达矩阵并做层级划分,构建多级递阶结构,理清因素间层次关系。",
    tags=["ISM", "结构模型", "层级分析"], demo_note="region",
    slots=[],
    options=[
        Option("matrix", "邻接矩阵(0/1,A[i][j]=1 表示 i 影响 j,每行一行)", "textarea",
               default="0,1,0,0,0\n0,0,1,0,1\n0,0,0,1,0\n0,0,0,0,1\n0,0,0,0,0"),
        Option("names", "因素名称(逗号分隔,选填)", "text", default=""),
    ],
))

# ================================================================ 德尔菲法
def _run_delphi(ds: Dataset, sl: dict, op: dict) -> Result:  # noqa: ARG001
    res = Result()
    mat = _parse_rect(op["matrix"])
    m, n = mat.shape
    if m < 3:
        raise ModelError("德尔菲法至少需要 3 位专家(行=专家,列=指标)")
    if n < 2:
        raise ModelError("至少需要 2 个指标")
    names = [t.strip() for t in str(op.get("names") or "").replace("，", ",").split(",") if t.strip()]
    if len(names) != n:
        names = [f"指标{i+1}" for i in range(n)]
    means = mat.mean(axis=0)
    sds = mat.std(axis=0, ddof=1)
    cvs = sds / np.abs(means).clip(min=1e-12)
    w = means / means.sum()
    order = np.argsort(-w)
    res.add_table("专家评分统计与权重", ["指标", "均值", "标准差", "变异系数CV", "权重", "权重%", "排序"],
                  [[names[j], fnum(means[j], 3), fnum(sds[j], 3), fnum(cvs[j], 4),
                    fnum(w[j], 4), fnum(w[j] * 100, 2) + "%", int(np.where(order == j)[0][0]) + 1] for j in range(n)],
                  note="行=专家评分,列=指标;权重=均值占比。CV 越小说明专家对该指标意见越集中")
    from scipy import stats as sps
    from scipy.stats import rankdata
    ranks = np.array([rankdata(mat[i]) for i in range(m)])  # m×n,行=专家
    rbar = ranks.mean(axis=0)
    S = float(((rbar - ranks.mean()) ** 2).sum())
    W = 12 * S / (m ** 2 * (n ** 3 - n))
    chi2_w = m * (n - 1) * W
    p_w = float(sps.chi2.sf(chi2_w, n - 1))
    res.add_table("专家意见协调系数", ["Kendall W", "χ²", "df", "p", "一致性结论"],
                  [[fnum(W, 4), fnum(chi2_w, 3), n - 1, fp(p_w),
                    "高度一致" if W >= 0.7 else "较一致" if W >= 0.5 else "一般" if W >= 0.3 else "一致性较差(建议再轮咨询)"]],
                  note="W∈[0,1] 越接近 1 专家意见越一致;p<0.05 表示协调系数显著")
    res.add_chart("德尔菲权重", ch.bar_chart("德尔菲法权重", names,
                 {"权重%": [round(float(v) * 100, 2) for v in w]}, y_name="权重%"))
    res.add_interp(f"{m} 位专家对 {n} 个指标的评分权重:最高「{names[order[0]]}」({w[order[0]] * 100:.1f}%)。"
                   f"协调系数 W={fnum(W, 3)}({'意见一致性可接受' if W >= 0.5 else '意见分歧较大,建议进行下一轮德尔菲咨询'})。")
    return res


register_model(ModelSpec(
    id="delphi", name="德尔菲法(专家权重)", category="综合评价",
    runner=_run_delphi,
    desc="汇总多轮专家评分为主观权重,并给出 Kendall 协调系数检验专家意见一致性。",
    tags=["德尔菲", "专家打分", "Kendall W"], demo_note="region",
    slots=[],
    options=[
        Option("matrix", "专家评分矩阵(行=专家,列=指标,每行一行)", "textarea",
               default="8,6,9,7\n7,7,8,6\n9,6,7,8\n8,7,6,9\n6,8,9,7"),
        Option("names", "指标名称(逗号分隔,选填)", "text", default=""),
    ],
))
