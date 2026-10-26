"""内置示例数据集:覆盖各分析模块的演示与自测数据。"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

def _rng(seed: int) -> np.random.Generator:
    """每个示例数据集独立固定种子:同一示例每次生成完全一致,不随调用顺序漂移。"""
    return np.random.default_rng(seed)


def _survey(n: int = 300) -> pd.DataFrame:
    rng = _rng(2028)
    gender = rng.choice(["男", "女"], n, p=[0.48, 0.52])
    dept = rng.choice(["技术部", "市场部", "人事部", "财务部"], n, p=[0.4, 0.25, 0.2, 0.15])
    age = rng.normal(32, 7, n).clip(22, 55).round(0)
    tenure = (age - 24 - rng.gamma(2, 1.2, n)).clip(0.5, 20).round(1)
    salary = (5000 + tenure * 800 + rng.normal(0, 1500, n) + np.where(dept == "技术部", 2000, 0)).round(0)
    latent = 0.5 * (salary / salary.max()) + 0.3 * (tenure / tenure.max()) + rng.normal(0, 0.3, n)
    q = {}
    for i in range(1, 6):
        base = latent * 2 + rng.normal(0, 0.8, n)
        q[f"Q{i}满意度"] = np.clip(np.round(base + 3), 1, 5).astype(int)
        # 配对的重要性评分(供 IPA 分析),与满意度弱相关但均值更高
        qi = np.clip(np.round((latent * 2 + rng.normal(0, 0.8, n)) * 0.6 + 3.6), 1, 5).astype(int)
        q[f"Q{i}重要性"] = qi
    overall = np.clip(np.round(pd.DataFrame(q).mean(axis=1)), 1, 5).astype(int)
    repurchase = np.where((overall >= 4) | (rng.random(n) < 0.2), "是", "否")
    visits = rng.poisson(np.clip(overall * 2 + 2, 1, 15))
    focus = {f"关注点_{k}": (rng.random(n) < p).astype(int)
             for k, p in [("价格", 0.55), ("质量", 0.62), ("服务", 0.40), ("品牌", 0.35), ("物流", 0.50)]}
    return pd.DataFrame({
        "编号": [f"S{i:03d}" for i in range(1, n + 1)],
        "性别": gender, "部门": dept, "年龄": age.astype(int), "工龄": tenure,
        "月薪": salary.astype(int),
        **q,
        "总体满意度": overall, "是否复购": repurchase, "月使用次数": visits, **focus,
    })


def _scores(n: int = 120) -> pd.DataFrame:
    rng = _rng(2029)
    cls = rng.choice(["1班", "2班", "3班"], n)
    gender = rng.choice(["男", "女"], n)
    base = rng.normal(70, 10, n) + np.where(cls == "1班", 6, np.where(cls == "3班", -5, 0))
    math = np.clip(rng.normal(base + 3, 8), 30, 100).round(1)
    english = np.clip(rng.normal(base - 2, 9) + np.where(gender == "女", 5, 0), 30, 100).round(1)
    physics = np.clip(rng.normal(base, 10) + 0.3 * (math - 70), 30, 100).round(1)
    chem = np.clip(0.6 * physics + 0.4 * math + rng.normal(0, 5, n), 30, 100).round(1)
    skew_scores = np.clip(rng.exponential(8, n) + 55, 30, 100).round(1)  # 非正态列,供非参数检验
    return pd.DataFrame({
        "学号": [f"ST{i:03d}" for i in range(1, n + 1)],
        "班级": cls, "性别": gender,
        "数学": math, "英语": english, "物理": physics, "化学": chem,
        "竞赛得分": skew_scores,
    })


def _ecommerce(n: int = 200) -> pd.DataFrame:
    rng = _rng(2030)
    seg = rng.choice(3, n, p=[0.4, 0.35, 0.25])
    recency = np.where(seg == 0, rng.gamma(2, 8, n), np.where(seg == 1, rng.gamma(3, 20, n), rng.gamma(4, 40, n))).round(0)
    freq = np.where(seg == 0, rng.poisson(14, n), np.where(seg == 1, rng.poisson(6, n), rng.poisson(2, n))).clip(1)
    amount = np.where(seg == 0, rng.normal(800, 150, n), np.where(seg == 1, rng.normal(420, 100, n), rng.normal(150, 60, n))).clip(30).round(0)
    income = (3000 + freq * 120 + rng.normal(0, 1200, n)).clip(1500).round(0)
    member = np.where(seg < 2, "是", np.where(rng.random(n) < 0.4, "是", "否"))
    satisfaction = np.clip(rng.normal(3.6 + (seg == 0) * 0.6, 0.7, n), 1, 5).round(0)
    return pd.DataFrame({
        "客户ID": [f"C{i:04d}" for i in range(1, n + 1)],
        "最近购买天数": recency.astype(int), "购买频次": freq.astype(int),
        "消费金额": amount.astype(int), "月收入": income.astype(int),
        "会员": member, "满意度": satisfaction.astype(int),
    })


def _quality(n_sub: int = 25, size: int = 5) -> pd.DataFrame:
    rng = _rng(2031)
    rows = []
    for b in range(1, n_sub + 1):
        shift = rng.normal(0, 0.08)
        vals = rng.normal(0.50 + shift, 0.05, size)
        for v in vals:
            rows.append([f"B{b:02d}", round(float(v), 4)])
    df = pd.DataFrame(rows, columns=["子组", "直径"])
    defect = rng.binomial(100, np.clip(0.05 + rng.normal(0, 0.015, n_sub), 0.01, 0.15))
    sizes = rng.integers(80, 121, n_sub)
    defect = np.minimum(defect, sizes)
    p_df = pd.DataFrame({"批次": [f"P{i:02d}" for i in range(1, n_sub + 1)],
                         "检查数": sizes.tolist(), "不合格数": defect.tolist()})
    out = pd.concat([df.assign(批次=np.nan, 检查数=np.nan, 不合格数=np.nan),
                     p_df.assign(子组=np.nan, 直径=np.nan)], ignore_index=True)
    for c in ('检查数', '不合格数', '直径'):
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out


def _timeseries(n: int = 48) -> pd.DataFrame:
    rng = _rng(2032)
    t = np.arange(n)
    trend = 120 + t * 2.2
    season = 18 * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, 8, n)
    sales = (trend + season + noise).round(1)
    profit = (sales * 0.28 + rng.normal(0, 4, n)).round(1)
    months = pd.period_range("2022-01", periods=n, freq="M").astype(str).tolist()
    return pd.DataFrame({"月份": months, "销量": sales, "利润": profit})


def _region(n_city: int = 12) -> pd.DataFrame:
    rng = _rng(2033)
    cities = ["江城市", "湖州市", "山阳市", "临海市", "云川市", "宁远市",
              "平川县", "望城县", "清河市", "温泉县", "长城市", "曲水县"]
    base = rng.normal(0, 1, n_city)
    gdp = (5000 + base * 1500 + rng.normal(0, 400, n_city)).clip(1000).round(0)
    income = (28000 + base * 5000 + rng.normal(0, 1500, n_city)).round(0)
    edu = np.clip(rng.normal(60 + base * 12, 6, n_city), 20, 100).round(1)      # 每万人大学生数
    medical = np.clip(rng.normal(55 + base * 10, 8, n_city), 20, 100).round(1)  # 每千人床位数
    environment = np.clip(rng.normal(75 - base * 8, 5, n_city), 40, 100).round(1)  # 空气优良率%
    pollution = np.clip(rng.normal(3 + base * 1.5, 0.6, n_city), 0.3, None).round(2)  # 负向指标
    traffic = np.clip(rng.normal(50 + base * 12, 7, n_city), 10, 100).round(1)
    return pd.DataFrame({
        "城市": cities[:n_city], "GDP亿元": gdp.astype(int), "人均可支配收入": income.astype(int),
        "每万人大学生数": edu, "每千人床位数": medical,
        "空气优良率%": environment, "污染指数": pollution, "交通便利度": traffic,
    })


def _clinical(n: int = 150) -> pd.DataFrame:
    rng = _rng(3002)
    treat = rng.choice(["新药", "安慰剂"], n, p=[0.5, 0.5])
    stage = rng.choice(["I期", "II期", "III期"], n, p=[0.4, 0.35, 0.25])
    age = rng.normal(58, 11, n).clip(30, 80).round(0)
    marker = rng.normal(5 + np.where(stage == "III期", 3, 0) + np.where(treat == "安慰剂", 1.2, 0), 1.8, n).clip(0.5).round(2)
    hazard = 0.03 + 0.004 * (age - 58) / 10 + np.where(treat == "安慰剂", 0.02, 0) + np.where(stage == "III期", 0.04, 0)
    time = rng.exponential(1 / hazard).clip(0.5, 60).round(1)
    event = ((time < 60) | (rng.random(n) < 0.15)).astype(int)
    time = np.where(event == 0, 60.0, time)
    baseline = (age * 20 + rng.normal(0, 300, n)).round(0)
    outcome = np.clip(np.round(4.5 - 0.02 * (time) + np.where(treat == "新药", 0.8, 0) + rng.normal(0, 0.8, n)), 1, 5).astype(int)
    rating1 = np.where(outcome >= 3, "有效", "无效")
    rating2 = np.where((outcome >= 3) ^ (rng.random(n) < 0.12), "有效", "无效")  # 供 McNemar/Kappa
    grade2 = np.where(outcome >= 4, "显效", np.where(outcome >= 3, "有效", rating1))
    return pd.DataFrame({
        "患者ID": [f"P{i:03d}" for i in range(1, n + 1)],
        "治疗组": treat, "肿瘤分期": stage, "年龄": age.astype(int),
        "肿瘤标志物": marker, "生存月数": time, "是否死亡": np.where(event == 1, "是", "否"),
        "基线评分": baseline.astype(int), "疗效评分": outcome,
        "医生A评价": rating1, "医生B评价": rating2, "疗效等级": grade2,
    })


def _panel(n: int = 40) -> pd.DataFrame:
    """重复测量干预研究:n 名被试 × 3 个时点(前/中/后测),供重复测量 ANOVA、混合模型、GEE 使用。"""
    rng = _rng(2035)
    rows = []
    group = rng.choice(["干预组", "对照组"], n, p=[0.5, 0.5])
    base = rng.normal(50, 8, n).clip(20, 90)
    for i in range(n):
        effect = 8 if group[i] == "干预组" else 1.5
        scores = [base[i] + rng.normal(0, 3),
                  base[i] - effect * 0.5 + rng.normal(0, 3),
                  base[i] - effect + rng.normal(0, 3)]
        for t, s in zip(["前测", "中测", "后测"], scores):
            rows.append({"被试编号": f"S{i + 1:03d}", "时点": t, "组别": group[i],
                         "焦虑得分": round(float(np.clip(s, 10, 100)), 1)})
    rng.shuffle(rows)
    return pd.DataFrame(rows)


SAMPLES = {
    "survey": ("员工满意度调查(300条)", _survey),
    "scores": ("学生成绩数据(120条)", _scores),
    "ecommerce": ("电商客户行为数据(200条)", _ecommerce),
    "quality": ("产品质量检测数据(SPC)", _quality),
    "timeseries": ("月度销量时间序列(48期)", _timeseries),
    "region": ("地区发展指标(12城市)", _region),
    "clinical": ("临床试验数据(150例)", _clinical),
    "panel": ("重复测量干预研究(120条)", _panel),
}


def build(key: str) -> pd.DataFrame:
    if key not in SAMPLES:
        raise KeyError(key)
    return SAMPLES[key][1]()


# 等级型变量提示:这些列在加载数据集时应设为「定级」类型
ORDINAL_HINTS: dict[str, list[str]] = {
    "survey": ["Q1满意度", "Q2满意度", "Q3满意度", "Q4满意度", "Q5满意度", "总体满意度"],
    "ecommerce": ["满意度"],
    "clinical": ["疗效评分", "疗效等级"],
}


def create(key: str, store=None):
    """构建示例数据集并注册到 store,返回 Dataset"""
    from .core.datasets import STORE as default_store
    store = store or default_store
    ds = store.add(SAMPLES[key][0], build(key), source="sample")
    for col in ORDINAL_HINTS.get(key, []):
        if col in ds.df.columns:
            ds.set_dtype(col, "ordinal")
    return ds
