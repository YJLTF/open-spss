"""全模型冒烟测试:每个模型用配套示例数据、按槽位类型自动挑选变量跑通一遍。"""
from __future__ import annotations

import pytest

from open_spss.core.datasets import STORE
from open_spss.core.registry import CAT, ORDINAL, QUANT, REGISTRY
from open_spss.models import CATEGORIES  # noqa: F401
from open_spss.samples import SAMPLES, create

_cache: dict[str, str] = {}


def ds_id(key: str) -> str:
    """每个测试用全新的数据集实例,避免 save 回写造成跨用例污染。"""
    _cache[key] = create(key).id
    return _cache[key]


NOISE = {"编号", "学号", "客户ID", "患者ID", "月份"}
GROUP_PREF = ["部门", "班级", "肿瘤分期", "治疗组", "性别", "会员", "是否复购", "是否死亡",
              "医生A评价", "医生B评价", "疗效等级"]
BIN_PREF = ["是否复购", "是否死亡", "会员", "医生A评价", "医生B评价", "性别", "治疗组"]
ORD_PREF = ["疗效等级", "总体满意度", "疗效评分", "满意度", "肿瘤分期", "Q1满意度", "Q2满意度",
            "Q3满意度", "Q4满意度", "Q5满意度", "性别", "是否复购", "是否死亡"]
QUAL_PREF = ["直径"]
MODEL_PREF = {
    "mcnemar": {"x": ["医生A评价", "是否复购", "会员"], "y": ["医生B评价", "是否死亡", "性别"]},
    "negbin": {"y": ["购买频次"]},
    "rsm": {"y": ["生存月数"], "xs": ["肿瘤标志物", "年龄"]},
    "rfm": {"cust": ["客户ID"], "r": ["最近购买天数"], "f": ["购买频次"], "m": ["消费金额"]},
    "ipa": {"imp": ["Q1重要性", "Q2重要性", "Q3重要性"], "perf": ["Q1满意度", "Q2满意度", "Q3满意度"]},
}


def _pref(name: str, prefs: list[str]) -> int:
    return prefs.index(name) if name in prefs else len(prefs)


def default_slots(spec, dataset) -> dict:
    cols = dataset.columns
    slots: dict = {}
    used: set[str] = set()
    for s in spec.slots:
        cand = [c for c in cols
                if c["name"] not in used and c["name"] not in NOISE and s.accepts(c["dtype"])]
        if not cand:
            if s.required:
                pytest.skip(f"{spec.id}: 无可用变量填充槽位 {s.key}")
            slots[s.key] = [] if s.multiple else ""
            continue
        label = s.label
        prefs_extra = MODEL_PREF.get(spec.id, {}).get(s.key, [])
        if s.multiple:
            if prefs_extra:
                cand.sort(key=lambda c: _pref(c["name"], prefs_extra))
            elif "量表题目" in label or "配对" in label:
                cand.sort(key=lambda c: (_pref(c["name"], ORD_PREF), c["name"]))
            elif "选项列" in label:
                cand.sort(key=lambda c: (0 if c["name"].startswith("关注点") else 1, c["name"]))
            elif QUANT in s.types and CAT not in s.types and ORDINAL not in s.types:
                cand.sort(key=lambda c: (c["dtype"] != "quant", c["name"]))
            else:
                cand.sort(key=lambda c: (c["dtype"] != "cat", c["name"]))
        else:
            if prefs_extra:
                cand.sort(key=lambda c: (_pref(c["name"], prefs_extra),))
            elif "二分类" in label:
                cand.sort(key=lambda c: (_pref(c["name"], BIN_PREF),))
            elif "质量特性" in label:
                cand.sort(key=lambda c: (_pref(c["name"], QUAL_PREF),))
            elif "量表" in label or "等级" in label and s.key == "y":
                cand.sort(key=lambda c: (_pref(c["name"], ORD_PREF),))
            elif s.key == "group" or "分组" in label:
                cand.sort(key=lambda c: (_pref(c["name"], GROUP_PREF), len(c.get("levels") or [])))
            elif CAT in s.types and QUANT not in s.types:
                cand.sort(key=lambda c: (_pref(c["name"], ORD_PREF), len(c.get("levels") or [])))
            elif CAT in s.types or ORDINAL in s.types:
                cand.sort(key=lambda c: (c["dtype"] not in ("cat", "ordinal"),
                                         len(c.get("levels") or [])))
            else:
                cand.sort(key=lambda c: c["name"])
        n_pick = min(len(cand), min(s.max_count, 3) if s.max_count else (3 if s.multiple else 1))
        picked = [c["name"] for c in cand[:n_pick]]
        used.update(picked)
        slots[s.key] = picked if s.multiple else picked[0]
    return slots


ALL_MODELS = [m for cat in CATEGORIES for m in cat["models"]]


@pytest.mark.parametrize("m", ALL_MODELS, ids=[m["id"] for m in ALL_MODELS])
def test_model_runs(m):
    spec = REGISTRY.get(m["id"])
    key = spec.demo_note or "survey"
    assert key in SAMPLES, f"{spec.id}: demo_note 指向未知示例 {key}"
    dataset = STORE.get(ds_id(key))
    slots = default_slots(spec, dataset)
    opts = {o.key: o.default for o in spec.options}
    slots, opts = REGISTRY.validate(spec.id, slots, opts)
    result = spec.runner(dataset, slots, opts)
    d = result.to_dict()
    assert d["tables"] or d["charts"], f"{spec.id}: 未输出任何结果"
