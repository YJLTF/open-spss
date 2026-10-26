"""可复现代码生成:把界面上的分析翻译为等效、可直接运行的 Python 代码。

`/api/run` 按模型 id 在 MODEL_CODE 中查表,命中则把生成的代码放进结果的
`code` 字段;未覆盖的模型不生成,前端也就不展示「查看代码」入口。
模板与对应 runner 的统计过程保持等价(同一检验、同一参数语义)。
"""
from __future__ import annotations

import time
from typing import Callable

from .core.datasets import Dataset


def _q(s) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _cols(xs) -> str:
    return "[" + ", ".join(_q(c) for c in (xs or [])) + "]"


def _header(name: str, mid: str, ds: Dataset, sl: dict) -> str:
    vars_used: list[str] = []
    for v in (sl or {}).values():
        if isinstance(v, str):
            vars_used.append(v)
        elif isinstance(v, list):
            vars_used.extend(v)
    return (f"# OpenSPSS 可复现代码 —— {name}({mid})\n"
            f"# 数据集:{ds.name} · 变量:{('、'.join(dict.fromkeys(vars_used))) or '—'}\n"
            f"# 生成时间:{time.strftime('%Y-%m-%d %H:%M')}\n"
            f"# 说明:df 请替换为你自己的 DataFrame,列名与界面分析保持一致。\n\n")


def _pairs_code(sl: dict) -> str:
    flat = [c for c in (sl.get("pairs") or []) if c]
    pairs = [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]
    return ", ".join(f"({_q(a)}, {_q(b)})" for a, b in pairs)


MODEL_CODE: dict[str, Callable[[Dataset, dict, dict], str]] = {}


def _register(*mids):
    def deco(fn):
        for m in mids:
            MODEL_CODE[m] = fn
        return fn
    return deco


# ---------------------------------------------------------------- 通用统计
@_register("frequency")
def _frequency(ds, sl, op):
    return _header("频数分析", "frequency", ds, sl) + f'''import pandas as pd

for col in {_cols(sl["vars"])}:
    print(f"\\n「{{col}}」频数分布")
    tab = df[col].value_counts(dropna=True).rename("频数").to_frame()
    tab["百分比%"] = (tab["频数"] / tab["频数"].sum() * 100).round(2)
    tab["累积%"] = tab["百分比%"].cumsum().round(2)
    print(tab)
'''


@_register("descriptive")
def _descriptive(ds, sl, op):
    return _header("描述性统计", "descriptive", ds, sl) + f'''import pandas as pd

cols = {_cols(sl["vars"])}
desc = df[cols].describe().T
desc["偏度"] = df[cols].skew()
desc["峰度"] = df[cols].kurt()
print(desc.round(4))
'''


@_register("correlation")
def _correlation(ds, sl, op):
    return _header("相关分析", "correlation", ds, sl) + f'''from scipy import stats as sps

cols = {_cols(sl["vars"])}
method = {_q(op.get("method", "pearson"))}          # pearson / spearman / kendall
fn = {{"pearson": sps.pearsonr, "spearman": sps.spearmanr, "kendall": sps.kendalltau}}[method]
print(f"{{method}} 相关系数矩阵:")
print(df[cols].corr(method=method).round(3))

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        sub = df[[cols[i], cols[j]]].dropna()
        r, p = fn(sub[cols[i]], sub[cols[j]])
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"「{{cols[i]}}」×「{{cols[j]}}」: r={{r:.3f}}{{sig}}, p={{p:.4g}}")
'''


@_register("partial_corr")
def _partial_corr(ds, sl, op):
    return _header("偏相关分析", "partial_corr", ds, sl) + f'''import numpy as np
from scipy import stats as sps

vars_, ctrls = {_cols(sl["vars"])}, {_cols(sl["ctrls"])}
data = df[vars_ + ctrls].dropna()
n = len(data)
Z = np.column_stack([np.ones(n), data[ctrls].values])   # 控制变量(含截距)
resid = {{}}
for c in vars_:
    y = data[c].values
    resid[c] = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]   # 残差:剔除控制变量的影响

keys = list(resid)
dfree = n - len(ctrls) - 2
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        r = float(np.corrcoef(resid[keys[i]], resid[keys[j]])[0, 1])
        t = r * np.sqrt(dfree / (1 - r * r))
        p = 2 * sps.t.sf(abs(t), dfree)
        print(f"「{{keys[i]}}」×「{{keys[j]}}」偏相关: r={{r:.3f}}, t={{t:.3f}}, p={{p:.4g}}")
'''


@_register("normality")
def _normality(ds, sl, op):
    return _header("正态性检验", "normality", ds, sl) + f'''import numpy as np
from scipy import stats as sps
from statsmodels.stats.diagnostic import lilliefors

for c in {_cols(sl["vars"])}:
    x = df[c].dropna()
    if 3 < len(x) <= 5000:
        w, p = sps.shapiro(x)
        print(f"「{{c}}」Shapiro-Wilk: W={{w:.4f}}, p={{p:.4g}}")
    else:
        ks_stat, p = lilliefors(x, dist="norm")
        print(f"「{{c}}」K-S(Lilliefors 校正): stat={{ks_stat:.4f}}, p={{p:.4g}}")
    jb, jbpv, skew, kurt = __import__("statsmodels").stats.stattools.jarque_bera(x)
    print(f"        Jarque-Bera: stat={{jb:.3f}}, p={{jbpv:.4g}}, 偏度={{skew:.3f}}, 峰度={{kurt:.3f}}")
'''


# ---------------------------------------------------------------- 差异检验
@_register("ttest_1sample")
def _ttest1(ds, sl, op):
    return _header("单样本 T 检验", "ttest_1sample", ds, sl) + f'''from scipy import stats as sps

mu = {op.get("popmean", 0)}
for c in {_cols(sl["vars"])}:
    x = df[c].dropna()
    t, p = sps.ttest_1samp(x, popmean=mu)
    d = (x.mean() - mu) / x.std(ddof=1)          # Cohen's d
    print(f"「{{c}}」: 均值={{x.mean():.4f}}, t({{len(x) - 1}})={{t:.4f}}, p={{p:.4g}}, d={{d:.3f}}")
'''


@_register("ttest_indep")
def _ttest_indep(ds, sl, op):
    return _header("独立样本 T 检验", "ttest_indep", ds, sl) + f'''from scipy import stats as sps

g = df[{_q(sl["group"])}].astype(str)
y = {_q(sl["y"])}
levels = sorted(g.dropna().unique())
a = df.loc[g == levels[0], y].dropna()
b = df.loc[g == levels[1], y].dropna()

lev_stat, lev_p = sps.levene(a, b)                 # 方差齐性
equal_var = lev_p >= 0.05
t, p = sps.ttest_ind(a, b, equal_var=equal_var)    # 不齐时自动 Welch
print(f"Levene: stat={{lev_stat:.4f}}, p={{lev_p:.4g}} → 假定{{'齐' if equal_var else '不齐(Welch 校正)'}}")
print(f"「{{levels[0]}}」n={{len(a)}}, M={{a.mean():.4f}}, SD={{a.std(ddof=1):.4f}}")
print(f"「{{levels[1]}}」n={{len(b)}}, M={{b.mean():.4f}}, SD={{b.std(ddof=1):.4f}}")
print(f"t={{t:.4f}}, p={{p:.4g}}")
'''


@_register("ttest_paired")
def _ttest_paired(ds, sl, op):
    return _header("配对样本 T 检验", "ttest_paired", ds, sl) + f'''from scipy import stats as sps

pairs = [{_pairs_code(sl)}]                        # 按界面顺序两两配对
for a_col, b_col in pairs:
    sub = df[[a_col, b_col]].dropna()
    t, p = sps.ttest_rel(sub[b_col], sub[a_col])
    d = (sub[b_col] - sub[a_col]).mean() / (sub[b_col] - sub[a_col]).std(ddof=1)
    print(f"「{{a_col}}」vs「{{b_col}}」: 差值均值={{(sub[b_col] - sub[a_col]).mean():.4f}}, "
          f"t({{len(sub) - 1}})={{t:.4f}}, p={{p:.4g}}, d={{d:.3f}}")
'''


@_register("anova_1way")
def _anova1(ds, sl, op):
    return _header("单因素方差分析", "anova_1way", ds, sl) + f'''from scipy import stats as sps
from statsmodels.stats.multicomp import pairwise_tukeyhsd

y, g = {_q(sl["y"])}, {_q(sl["group"])}
sub = df[[y, g]].dropna()
groups = [v[y].values for _, v in sub.groupby(g)]
F, p = sps.f_oneway(*groups)
print(f"omnibus: F({{len(groups) - 1}}, {{len(sub) - len(groups)}})={{F:.4f}}, p={{p:.4g}}")

posthoc = {_q(op.get("posthoc", "tukey"))}
if posthoc in ("tukey", "scheffe"):
    print(pairwise_tukeyhsd(sub[y], sub[g]))       # Tukey HSD(scheffe 请以界面结果为准)
elif posthoc == "lsd":
    from itertools import combinations
    lvls = sorted(sub[g].unique())
    import numpy as np
    for a, b in combinations(lvls, 2):
        xa, xb = sub.loc[sub[g] == a, y], sub.loc[sub[g] == b, y]
        t, tp = sps.ttest_ind(xa, xb)
        print(f"LSD「{{a}}」vs「{{b}}」: t={{t:.3f}}, p={{tp:.4g}}")
'''


@_register("anova_multi")
def _anova_multi(ds, sl, op):
    factors = list(sl["factors"])
    fx = " * ".join(f"C(Q({_q(f)}), Sum)" for f in factors)
    return _header("多因素方差分析", "anova_multi", ds, sl) + f'''import statsmodels.api as sm
from statsmodels.formula.api import ols

sub = df[{_cols([sl["y"]] + factors)}].dropna()
for f in {factors!r}:
    sub[f] = sub[f].astype(str)
model = ols({_q("Q(" + _q(sl["y"]) + ") ~ " + fx)}, data=sub).fit()
print(sm.stats.anova_lm(model, typ=2))
'''


@_register("anova_rm")
def _anova_rm(ds, sl, op):
    return _header("重复测量方差分析", "anova_rm", ds, sl) + f'''from statsmodels.stats.anova import AnovaRM

wide = df.pivot_table(index={_q(sl["subject"])}, columns={_q(sl["time"])},
                      values={_q(sl["y"])}, aggfunc="mean").dropna()
print(f"完整被试: {{len(wide)}}, 水平: {{list(wide.columns)}}")
print(AnovaRM(df, depvar={_q(sl["y"])}, subject={_q(sl["subject"])},
              within=[{_q(sl["time"])}]).fit())
# Mauchly 球形度检验与 Greenhouse-Geisser 校正:界面结果中已给出,判断标准
# 为 ε(GG)<0.75 时以校正后 p 为准。
'''


@_register("chisq")
def _chisq(ds, sl, op):
    return _header("卡方检验(交叉分析)", "chisq", ds, sl) + f'''import pandas as pd
from scipy import stats as sps

tab = pd.crosstab(df[{_q(sl["x"])}], df[{_q(sl["y"])}])
chi2, p, dof, expected = sps.chi2_contingency(tab, correction={bool(op.get("correction", True))})
print("交叉表:"); print(tab)
print(f"chi2={{chi2:.4f}}, df={{dof}}, p={{p:.4g}}")
print("最小期望频数:", expected.min().round(2), "(<5 占比过高时请谨慎解读或合并水平)")
'''


@_register("mannwhitney", "kruskal")
def _rank_group_tests(ds, sl, op):
    return _header("秩和检验", "rank_tests", ds, sl) + f'''from scipy import stats as sps

y, g = {_q(sl["y"])}, {_q(sl["group"])}
sub = df[[y, g]].dropna()
groups = [v[y].values for _, v in sub.groupby(g)]
alt = {_q(op.get("alternative", "two-sided"))}
if len(groups) == 2:
    U, p = sps.mannwhitneyu(groups[0], groups[1], alternative=alt, method="asymptotic")
    print(f"Mann-Whitney U={{U:.3f}}, p={{p:.4g}}")
else:
    H, p = sps.kruskal(*groups)
    print(f"Kruskal-Wallis H={{H:.4f}}, p={{p:.4g}}")
    # 事后两两比较(Dunn/Bonferroni)见界面结果
'''


@_register("wilcoxon_1")
def _wilcoxon1(ds, sl, op):
    return _header("单样本 Wilcoxon 检验", "wilcoxon_1", ds, sl) + f'''from scipy import stats as sps

med = {op.get("med", 0)}
for c in {_cols(sl["vars"])}:
    x = df[c].dropna() - med
    w, p = sps.wilcoxon(x, alternative={_q(op.get("alternative", "two-sided"))},
                        mode="approx", correction={bool(op.get("correction", True))})
    print(f"「{{c}}」: 中位数差检验 W={{w:.3f}}, p={{p:.4g}}")
'''


@_register("friedman")
def _friedman(ds, sl, op):
    return _header("Friedman 检验", "friedman", ds, sl) + f'''from scipy import stats as sps

cols = {_cols(sl["vars"])}
sub = df[cols].dropna()
chi2, p = sps.friedmanchisquare(*[sub[c] for c in cols])
print(f"Friedman chi2={{chi2:.4f}}, p={{p:.4g}}")
'''


# ---------------------------------------------------------------- 回归建模
@_register("linear")
def _linear(ds, sl, op):
    return _header("多元线性回归", "linear", ds, sl) + f'''import pandas as pd
import statsmodels.api as sm

y = {_q(sl["y"])}
xs = {_cols(sl["xs"])}
cat_cols = [c for c in xs if df[c].dtype == object or str(df[c].dtype).startswith("category")]
X = pd.get_dummies(df[xs], columns=cat_cols, drop_first=True, dtype=float)  # 定类哑变量化
data = pd.concat([df[y], X], axis=1).dropna()
model = sm.OLS(data[y], sm.add_constant(data[X.columns])).fit()
print(model.summary())
# 诊断:VIF 与 Durbin-Watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
Xc = sm.add_constant(data[X.columns])
print("VIF:", {{c: round(variance_inflation_factor(Xc.values, i + 1), 3)
                for i, c in enumerate(X.columns)}})
print("Durbin-Watson:", round(durbin_watson(model.resid), 3))
'''


@_register("logit_binary")
def _logit(ds, sl, op):
    return _header("二元 Logistic 回归", "logit_binary", ds, sl) + f'''import numpy as np
import pandas as pd
import statsmodels.api as sm

y_raw = df[{_q(sl["y"])}]
y = pd.Series(pd.factorize(y_raw)[0], index=df.index)   # 与界面一致:首水平为 0
xs = {_cols(sl["xs"])}
cat_cols = [c for c in xs if df[c].dtype == object or str(df[c].dtype).startswith("category")]
X = pd.get_dummies(df[xs], columns=cat_cols, drop_first=True, dtype=float)
data = pd.concat([y.rename("y"), X], axis=1).dropna()
model = sm.Logit(data["y"], sm.add_constant(data[X.columns])).fit()
print(model.summary2())
print("OR(优势比):")
print(np.exp(model.params).round(4))
'''


def _make_penalized(mid: str):
    name = "岭回归" if mid == "ridge" else "Lasso 回归"
    cls = "Ridge" if mid == "ridge" else "Lasso"

    def _fn(ds, sl, op):
        lam = op.get("lambda", 0)
        body = ("from sklearn.linear_model import RidgeCV" if not lam
                else "from sklearn.linear_model import Ridge")
        fit = (f"model = {cls}CV(alphas=np.logspace(-3, 3, 60))" if not lam
               else f"model = {cls}(alpha={lam}, max_iter=100000)")
        return _header(name, mid, ds, sl) + f'''import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
{body}

y = {_q(sl["y"])}
xs = {_cols(sl["xs"])}
data = df[[y] + xs].dropna()
Z = StandardScaler().fit_transform(data[xs])       # 惩罚回归前先标准化
{fit}
model.fit(Z, data[y])
print("最优 alpha:", getattr(model, "alpha_", {lam}))
print(pd.Series(model.coef_, index=xs, name="标准化系数").round(4))
print("截距:", round(float(model.intercept_), 4), " R2(train):", round(model.score(Z, data[y]), 4))
'''
    return _fn


MODEL_CODE["ridge"] = _make_penalized("ridge")
MODEL_CODE["lasso"] = _make_penalized("lasso")


@_register("mediation")
def _mediation(ds, sl, op):
    covs = [sl["covs"]] if isinstance(sl.get("covs"), str) and sl.get("covs") else (
        list(sl.get("covs") or []))
    return _header("中介效应分析", "mediation", ds, sl) + f'''import numpy as np
import pandas as pd

y, x = {_q(sl["y"])}, {_q(sl["x"])}
ms = {_cols(sl["ms"])}
covs = {_cols(covs)}
n_rep = {int(op.get("n_rep", 1000))}
data = df[[y, x] + ms + covs].apply(pd.to_numeric, errors="coerce").dropna()
Y = data[y].values
X = data[x].values
M = [data[m].values for m in ms]
one = np.ones(len(data))
Xa = np.column_stack([one, X] + [data[c].values for c in covs])
Xr = Xa.copy()
rng = np.random.default_rng(20240001)               # 与界面一致的固定种子

def ols(A, v):
    return np.linalg.lstsq(A, v, rcond=None)[0]

for mi, m in enumerate(ms):
    Xo = np.column_stack([one, X, M[mi]] + [data[c].values for c in covs])
    a, b, cp, c = ols(Xa, M[mi])[1], ols(Xo, Y)[2], ols(Xo, Y)[1], ols(Xr, Y)[1]
    boots = []
    for _ in range(n_rep):                           # case 重抽样百分位 Bootstrap
        idx = rng.integers(0, len(data), len(data))
        a_, b_ = ols(Xa[idx], M[mi][idx])[1], ols(Xo[idx], Y[idx])[2]
        cp_, c_ = ols(Xo[idx], Y[idx])[1], ols(Xr[idx], Y[idx])[1]
        boots.append((a_ * b_, cp_, c_))
    boots = np.array(boots)
    lo, hi = np.percentile(boots[:, 0], [2.5, 97.5])
    p_ind = max(2 * min((boots[:, 0] <= 0).mean(), (boots[:, 0] >= 0).mean()), 1 / (2 * n_rep))
    kind = ("部分中介" if 2 * min((boots[:, 1] <= 0).mean(), (boots[:, 1] >= 0).mean()) < 0.05
            else "完全中介") if p_ind < 0.05 else "未检出中介效应"
    print(f"「{{m}}」: ACME={{a * b:.4f}}(95%CI {{lo:.4f}}~{{hi:.4f}}, p={{p_ind:.4g}}), "
          f"ADE={{cp:.4f}}, 总效应={{c:.4f}} → {{kind}}")
'''


@_register("moderation")
def _moderation(ds, sl, op):
    return _header("调节效应分析", "moderation", ds, sl) + f'''import numpy as np
import statsmodels.api as sm
import pandas as pd
from scipy import stats as sps

y, x, m = {_q(sl["y"])}, {_q(sl["x"])}, {_q(sl["mod"])}
sub = df[[y, x, m]].apply(pd.to_numeric, errors="coerce").dropna()   # 定类调节请先哑变量化
xc = sub[x] - sub[x].mean()
mc = sub[m] - sub[m].mean()
Y = sub[y].values
one = np.ones(len(sub))
M1 = sm.OLS(Y, np.column_stack([one, xc])).fit()
M2 = sm.OLS(Y, np.column_stack([one, xc, mc])).fit()
M3 = sm.OLS(Y, np.column_stack([one, xc, mc, xc * mc])).fit()

def f_change(s, b_):
    df1, df2 = int(b_.df_model - s.df_model), int(b_.df_resid)
    F = (b_.rsquared - s.rsquared) / df1 / ((1 - b_.rsquared) / df2)
    return F, sps.f.sf(F, df1, df2)

F3, p3 = f_change(M2, M3)
print(f"M1 R2={{M1.rsquared:.4f}} → M2 R2={{M2.rsquared:.4f}} → M3 R2={{M3.rsquared:.4f}}")
print(f"交互项系数={{M3.params[3]:.4f}}, F 变化={{F3:.3f}}, p={{p3:.4g}}")
b3, V = M3.params[3], M3.cov_params()
msd = mc.std(ddof=1)
for label, mv in (("M=-1SD", -msd), ("M=均值", 0.0), ("M=+1SD", msd)):
    slope = M3.params[1] + b3 * mv
    se = np.sqrt(V[1, 1] + mv ** 2 * V[3, 3] + 2 * mv * V[1, 3])
    p = 2 * sps.t.sf(abs(slope / se), int(M3.df_resid))
    print(f"简单斜率 {{label}}: {{slope:.4f}}(SE={{se:.4f}}, p={{p:.4g}})")
'''


@_register("mixed_lm")
def _mixed_lm(ds, sl, op):
    re_part = (f"exog_re=data[{_q(sl['group'])}]  if False else None"
               if op.get("re_type") == "ris" else "")
    return _header("线性混合模型", "mixed_lm", ds, sl) + f'''import numpy as np
import pandas as pd
import statsmodels.api as sm

y, g = {_q(sl["y"])}, {_q(sl["group"])}
xs = {_cols(sl["xs"])}
cat_cols = [c for c in xs if df[c].dtype == object or str(df[c].dtype).startswith("category")]
X = pd.get_dummies(df[xs], columns=cat_cols, drop_first=True, dtype=float)
data = pd.concat([df[y], X, df[g].astype(str)], axis=1).dropna()
md = sm.MixedLM(data[y], sm.add_constant(data[X.columns]), groups=data[g])
res = md.fit(reml={op.get("reml", "reml") == "reml"})
print(res.summary())
var_re = float(np.diag(res.cov_re)[0]); resid = float(res.scale)
print(f"ICC = {{var_re / (var_re + resid):.3f}}")
'''


@_register("gee")
def _gee(ds, sl, op):
    return _header("广义估计方程(GEE)", "gee", ds, sl) + f'''import pandas as pd
import statsmodels.api as sm

y, g = {_q(sl["y"])}, {_q(sl["group"])}
xs = {_cols(sl["xs"])}
cat_cols = [c for c in xs if df[c].dtype == object or str(df[c].dtype).startswith("category")]
X = pd.get_dummies(df[xs], columns=cat_cols, drop_first=True, dtype=float)
data = pd.concat([df[y].apply(pd.to_numeric, errors="coerce"), X, df[g].astype(str)],
                 axis=1).dropna()
fam = {{"gaussian": sm.families.Gaussian(), "binomial": sm.families.Binomial(),
        "poisson": sm.families.Poisson()}}[{_q(op.get("family", "gaussian"))}]
cov = {{"exchangeable": sm.cov_struct.Exchangeable(), "ar1": sm.cov_struct.Autoregressive(),
        "independence": sm.cov_struct.Independence(),
        "unstructured": sm.cov_struct.Unstructured()}}[{_q(op.get("corstr", "exchangeable"))}]
res = sm.GEE(data[y], sm.add_constant(data[X.columns]), groups=data[g],
             family=fam, cov_struct=cov).fit()
print(res.summary())
print("作业相关参数 ρ/α:", getattr(cov, "dep_params", None))
'''


# ---------------------------------------------------------------- 降维聚类 / 机器学习
@_register("pca")
def _pca(ds, sl, op):
    return _header("主成分分析(PCA)", "pca", ds, sl) + f'''import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

cols = {_cols(sl["vars"])}
Z = StandardScaler().fit_transform(df[cols].dropna())
k = {int(op.get("ncomp", 0) or 0)}
if k <= 0:                                          # 自动:特征值>1(标准化下=解释方差>1)
    k = int((PCA().fit(Z).explained_variance_ > 1).sum()) or 1
pca = PCA(n_components=k).fit(Z)
load = pca.components_.T * np.sqrt(pca.explained_variance_)   # 载荷 = 特征向量×√特征值
print("方差解释率%:", (pca.explained_variance_ratio_ * 100).round(2))
print(pd.DataFrame(load, index=cols,
                   columns=[f"PC{{i+1}}" for i in range(k)]).round(3))
scores = pd.DataFrame(pca.transform(Z), index=df[cols].dropna().index,
                      columns=[f"PC{{i+1}}" for i in range(k)])   # 主成分得分,可并回原数据
'''


@_register("kmeans")
def _kmeans(ds, sl, op):
    return _header("K-means 聚类", "kmeans", ds, sl) + f'''import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

cols = {_cols(sl["vars"])}
X = df[cols].dropna()
if {bool(op.get("standardize", True))}:
    X = pd.DataFrame(StandardScaler().fit_transform(X), index=X.index, columns=cols)
k = {int(op.get("k", 3) or 3)}
km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
labels = pd.Series(km.labels_, index=X.index, name="聚类类别")
print("各类规模:", labels.value_counts().sort_index().to_dict())
print("轮廓系数:", round(silhouette_score(X, labels), 4))
'''


_ML_SPEC = {
    "decision_tree": ("sklearn.tree", "DecisionTreeClassifier", "DecisionTreeRegressor"),
    "random_forest": ("sklearn.ensemble", "RandomForestClassifier", "RandomForestRegressor"),
    "gbdt": ("sklearn.ensemble", "GradientBoostingClassifier", "GradientBoostingRegressor"),
    "svm": ("sklearn.svm", "SVC", "SVR"),
    "knn": ("sklearn.neighbors", "KNeighborsClassifier", "KNeighborsRegressor"),
    "naive_bayes": ("sklearn.naive_bayes", "GaussianNB", "GaussianNB"),
    "mlp": ("sklearn.neural_network", "MLPClassifier", "MLPRegressor"),
}


def _ml_params(mid: str, op: dict) -> str:
    if mid == "decision_tree":
        return f"max_depth={int(op.get('max_depth', 4) or 4)}, min_samples_leaf={int(op.get('min_samples_leaf', 5) or 5)}, "
    if mid == "random_forest":
        return f"n_estimators={int(op.get('n_estimators', 200) or 200)}, "
    if mid == "svm":
        return f"kernel={_q(op.get('kernel', 'rbf'))}, C={op.get('C', 1.0)}, "
    if mid == "knn":
        return f"n_neighbors={int(op.get('k', 5) or 5)}, "
    if mid == "mlp":
        return f"hidden_layer_sizes=({op.get('layers', '64,32')}), max_iter=500, "
    return ""


def _make_ml(mid: str):
    def _fn(ds, sl, op):
        module, clf, reg = _ML_SPEC[mid]
        xs = list(sl["xs"])
        seed = int(op.get("seed", 42) or 42)
        body = f'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_squared_error, r2_score)
from {module} import {clf}, {reg}

FEATURES = {_cols(xs)}
TARGET = {_q(sl["y"])}

X = pd.get_dummies(df[FEATURES], drop_first=True)   # 定类特征自动哑变量化(与界面一致)
y = df[TARGET]
is_clf = y.dtype == object or str(y.dtype) == "category" or y.nunique() <= 10
if is_clf:
    y = y.astype("category").cat.codes              # 类别编码:按出现顺序 0..k-1

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size={op.get("test_ratio", 0.3)}, random_state={seed},
    stratify=y if is_clf else None)

model = ({clf} if is_clf else {reg})({_ml_params(mid, op)} random_state={seed})
model.fit(X_tr, y_tr)
pred = model.predict(X_te)

if is_clf:
    print("准确率:", round(accuracy_score(y_te, pred), 4))
    print("F1(macro):", round(f1_score(y_te, pred, average="macro"), 4))
    print(pd.DataFrame(confusion_matrix(y_te, pred),
                       index=[["真实"] * len(model.classes_), model.classes_],
                       columns=[["预测"] * len(model.classes_), model.classes_]))
    print(classification_report(y_te, pred))
else:
    print("R2:", round(r2_score(y_te, pred), 4))
    print("RMSE:", round(mean_squared_error(y_te, pred) ** 0.5, 4))

imp = getattr(model, "feature_importances_", None)
if imp is not None:
    print(pd.Series(imp, index=X.columns).sort_values(ascending=False).round(4))
'''
        return _header("机器学习·" + mid, mid, ds, sl) + body
    return _fn


for _mid in _ML_SPEC:
    MODEL_CODE[_mid] = _make_ml(_mid)


# ---------------------------------------------------------------- 医学统计
@_register("roc")
def _roc(ds, sl, op):
    return _header("ROC 曲线", "roc", ds, sl) + f'''import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

y_raw = df[{_q(sl["y"])}]
levels = sorted(y_raw.dropna().astype(str).unique())
y = y_raw.astype(str).map({{levels[0]: 0, levels[1]: 1}})   # 「{{levels[1]}}」视为阳性
for c in {_cols(sl["scores"])}:
    sub = pd.concat([y.rename("y"), df[c]], axis=1).dropna()
    auc = roc_auc_score(sub["y"], sub[c])
    fpr, tpr, thr = roc_curve(sub["y"], sub[c])
    j = int(np.argmax(tpr - fpr))                  # 约登指数最佳截断
    print(f"「{{c}}」: AUC={{auc:.4f}}, 最佳截断={{thr[j]:.4f}}, "
          f"灵敏度={{tpr[j]:.3f}}, 特异度={{1 - fpr[j]:.3f}}")
'''


@_register("cox")
def _cox(ds, sl, op):
    return _header("Cox 比例风险回归", "cox", ds, sl) + f'''import pandas as pd
from lifelines import CoxPHFitter   # pip install lifelines

ev = df[{_q(sl["event"])}].astype(str)
levels = sorted(ev.dropna().unique())
data = pd.concat([
    df[{_q(sl["time"])}].rename("time"),
    ev.map({{levels[0]: 0, levels[1]: 1}}).rename("event"),   # 1=发生事件
    df[{_cols(sl["xs"])}],
], axis=1).dropna()
cph = CoxPHFitter()
cph.fit(data, duration_col="time", event_col="event")
print(cph.summary[["coef", "exp(coef)", "p"]].round(4))       # exp(coef) 即 HR
print("C-index:", round(cph.concordance_index_, 4))
'''


@_register("kaplan_meier")
def _kaplan_meier(ds, sl, op):
    grp = f', group_col={_q(sl["group"])}' if sl.get("group") else ""
    return _header("Kaplan-Meier 生存曲线", "kaplan_meier", ds, sl) + f'''import pandas as pd
from lifelines import KaplanMeierFitter   # pip install lifelines

ev = df[{_q(sl["event"])}].astype(str)
levels = sorted(ev.dropna().unique())
data = pd.concat([
    df[{_q(sl["time"])}].rename("time"),
    ev.map({{levels[0]: 0, levels[1]: 1}}).rename("event"),   # 1=事件
], axis=1).dropna()

def _plot_one(sub, label):
    kmf = KaplanMeierFitter()
    kmf.fit(sub["time"], sub["event"], label=label)
    return kmf

results = []
{f'for label, sub in data.assign(g=df[{sl["group"]!r}].astype(str)).groupby("g"):' if sl.get("group") else 'sub = data\nlabel = "全部样本"\nif True:'}
    kmf = _plot_one(sub, str(label))
    results.append((label, kmf))
for label, kmf in results:
    print(f"{{label}}: 中位生存时间={{kmf.median_survival_time_}}, "
          f"1 年生存率={{kmf.predict(1):.3f}}(按时间单位)")
'''


# ---------------------------------------------------------------- 问卷研究
@_register("reliability")
def _reliability(ds, sl, op):
    return _header("信度分析", "reliability", ds, sl) + f'''import numpy as np

X = df[{_cols(sl["vars"])}].dropna().values
k = X.shape[1]

def cronbach(X):
    return k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / X.sum(axis=1).var(ddof=1))

print(f"总量表 Cronbach alpha={{cronbach(X):.3f}}")
for j, c in enumerate({_cols(sl["vars"])}):
    rest = np.delete(X, j, axis=1).sum(axis=1)
    citc = np.corrcoef(X[:, j], rest)[0, 1]
    a_del = cronbach(np.delete(X, j, axis=1))
    print(f"{{c}}: CITC={{citc:.3f}}, 删除后 alpha={{a_del:.3f}}")
half1, half2 = X[:, :k // 2].sum(axis=1), X[:, k // 2:].sum(axis=1)
r = np.corrcoef(half1, half2)[0, 1]
print(f"折半信度 Spearman-Brown={{2 * r / (1 + r):.3f}}")
'''


@_register("omega")
def _omega(ds, sl, op):
    return _header("McDonald's omega", "omega", ds, sl) + f'''import numpy as np

X = df[{_cols(sl["vars"])}].dropna()
Z = (X - X.mean()) / X.std(ddof=1)                 # 标准化题目
corr = np.corrcoef(Z.T, rowvar=False)
eig, vec = np.linalg.eigh(corr)
order = np.argsort(eig)[::-1]
eig, vec = eig[order], vec[:, order]
load = np.clip(vec[:, 0] * np.sqrt(max(eig[0], 1e-9)), -0.999, 0.999)
if (load < 0).sum() > len(load) / 2:
    load = -load                                   # 载荷方向统一
theta = 1 - load ** 2
omega = load.sum() ** 2 / (load.sum() ** 2 + theta.sum())
print("标准化载荷:", dict(zip(X.columns, load.round(3))))
print(f"McDonald's omega={{omega:.3f}}")
'''


# ---------------------------------------------------------------- 统计绘图
@_register("plot_hist")
def _plot_hist(ds, sl, op):
    return _header("直方图", "plot_hist", ds, sl) + f'''import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

for c in {_cols(sl["vars"])}:
    x = df[c].dropna()
    plt.figure()
    plt.hist(x, bins="auto", color="#818cf8", edgecolor="white")
    mu, sd = x.mean(), x.std(ddof=1)
    xs = np.linspace(x.min(), x.max(), 200)
    plt.plot(xs, sps.norm.pdf(xs, mu, sd) * len(x) * (x.max() - x.min()) / 12,
             color="#ef4444", lw=2)                    # 正态拟合参考线
    plt.title(f"「{{c}}」直方图(均值={{mu:.2f}}, 标准差={{sd:.2f}})")
    plt.tight_layout()
plt.show()
'''


@_register("plot_box")
def _plot_box(ds, sl, op):
    if sl.get("y") and sl.get("x"):
        inner = f'''sub = df[{_cols([sl["y"], sl["x"]])}].dropna()
groups = [g[{_q(sl["y"])}].values for _, g in sub.groupby(sub.columns[-1], observed=True)]
labels = [str(k) for k, _ in sub.groupby(sub.columns[-1], observed=True)]
plt.boxplot(groups, labels=labels)'''
    else:
        inner = f'''plt.boxplot([df[c].dropna() for c in {_cols(sl["vars"])}],
            labels={_cols(sl["vars"])})'''
    return _header("箱线图", "plot_box", ds, sl) + f'''import matplotlib.pyplot as plt
{inner}
plt.show()
'''


@_register("plot_scatter")
def _plot_scatter(ds, sl, op):
    return _header("散点图", "plot_scatter", ds, sl) + f'''import matplotlib.pyplot as plt
from scipy import stats as sps

sub = df[{_cols([sl["x"], sl["y"]])}].dropna()
r, p = sps.pearsonr(sub[{_q(sl["x"])}], sub[{_q(sl["y"])}])
plt.scatter(sub[{_q(sl["x"])}], sub[{_q(sl["y"])}], alpha=0.7, s=22, color="#6366f1")
coef = sps.linregress(sub[{_q(sl["x"])}], sub[{_q(sl["y"])}])
xs = sub[{_q(sl["x"])}]
plt.plot(sorted(xs), coef.intercept + coef.slope * sorted(xs), "--", color="#ef4444",
         label=f"r={{r:.3f}}, p={{p:.3g}}")
plt.xlabel({_q(sl["x"])}); plt.ylabel({_q(sl["y"])}); plt.legend()
plt.show()
'''


# ---------------------------------------------------------------- 入口
def code_for(mid: str, ds: Dataset, sl: dict, op: dict) -> str | None:
    """按模型 id 生成可复现代码;未覆盖的模型返回 None。"""
    fn = MODEL_CODE.get(mid)
    if fn is None:
        return None
    try:
        code = fn(ds, sl, op)
    except Exception:  # noqa: BLE001  代码生成失败不阻断分析主流程
        return None
    return code or None
