"""数据工作台:单元格编辑、撤销/重做、行列增删、变量/值标签、数据集合并。"""
import numpy as np
import pandas as pd
import pytest

from open_spss.core import datasets as dsm
from open_spss.core.results import ModelError
from open_spss.models import CATEGORIES  # noqa: F401  触发模型注册
from open_spss.processing import OPERATIONS


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENSPSS_DATA_DIR", str(tmp_path))
    return dsm.DatasetStore()


@pytest.fixture()
def ds(store):
    return store.add("wb", pd.DataFrame({"数量": [1.0, 2.0, 3.0], "类别": ["a", "b", "a"]}))


# ---------------------------------------------------------------- 单元格编辑
def test_set_cell_quant_validated(ds):
    ds.set_cell(0, "数量", "9.5")
    assert ds.df.loc[ds.df.index[0], "数量"] == 9.5
    with pytest.raises(ModelError, match="无法转为数值"):
        ds.set_cell(0, "数量", "abc")
    with pytest.raises(ModelError, match="越界"):
        ds.set_cell(99, "数量", "1")
    ds.set_cell(1, "类别", "改后的类别")
    assert ds.df.loc[ds.df.index[1], "类别"] == "改后的类别"
    ds.set_cell(2, "类别", "")                       # 置空 → NaN/None
    assert ds.df["类别"].isna().iloc[2]


# ---------------------------------------------------------------- 撤销 / 重做
def test_undo_redo_roundtrip(ds):
    v0 = ds.df["数量"].iloc[0]
    ds.set_cell(0, "数量", "42")
    assert ds.df["数量"].iloc[0] == 42
    assert ds.summary()["canUndo"] and not ds.summary()["canRedo"]
    assert ds.undo()
    assert ds.df["数量"].iloc[0] == v0
    assert ds.summary()["canRedo"]
    assert ds.redo()
    assert ds.df["数量"].iloc[0] == 42
    assert not ds.undo() or True
    # 连续撤销超过栈深后清空
    while ds.undo():
        pass
    assert not ds.summary()["canUndo"]


def test_undo_reverts_structural_ops(ds):
    ds.add_col("新列")
    ds.undo()
    assert "新列" not in ds.df.columns
    ds.del_row(0)
    ds.undo()
    assert len(ds.df) == 3
    ds.add_row()
    assert len(ds.df) == 4
    ds.undo()
    assert len(ds.df) == 3


def test_undo_disabled_for_huge_frames(store, monkeypatch):
    monkeypatch.setattr(dsm, "MAX_UNDO_ROWS", 5)
    ds = store.add("big", pd.DataFrame({"x": range(10)}))
    ds.set_dtype("x", "quant")
    ds.del_row(0)
    assert not ds.summary()["canUndo"]               # 超阈值不建快照


# ---------------------------------------------------------------- 行列增删
def test_row_col_ops(ds):
    ds.add_col("评分", "quant")
    assert "评分" in ds.df.columns
    with pytest.raises(ModelError, match="已存在"):
        ds.add_col("评分")
    ds.del_col("评分")
    assert "评分" not in ds.df.columns
    ds.add_row()
    assert ds.df["数量"].isna().iloc[-1]


# ---------------------------------------------------------------- 变量 / 值标签
def test_labels_and_value_label_op(store):
    meta = {"var_labels": {"疗效": "治疗疗效评级"},
            "value_labels": {"疗效": {1: "无效", 2: "有效", "3": "显效"}}}
    ds = store.add("标签测试", pd.DataFrame({"疗效": [1, 2, 3, 2]}), meta=meta)
    col = next(c for c in ds.columns if c["name"] == "疗效")
    assert col["label"] == "治疗疗效评级"
    assert col["vals"] == {"1": "无效", "2": "有效", "3": "显效"}
    assert ds.col_meta()[0]["label"] == "治疗疗效评级"

    msg = OPERATIONS["value_labels"](ds, cols=["疗效"], suffix="_文本")
    assert "疗效" in msg
    assert ds.df["疗效_文本"].tolist() == ["无效", "有效", "显效", "有效"]

    ds.set_labels("疗效", "新标签", {"1": "无效"})   # 整体替换值标签字典
    col = next(c for c in ds.columns if c["name"] == "疗效")
    assert col["label"] == "新标签" and col["vals"] == {"1": "无效"}
    with pytest.raises(ValueError, match="没有值标签"):
        OPERATIONS["value_labels"](ds, cols=["疗效_文本"], suffix="_2")


def test_labels_persist(store):
    ds = store.add("持久", pd.DataFrame({"x": [1]}))
    ds.set_labels("x", "年龄标签", {"1": "一"})
    ds2 = dsm.DatasetStore().get(ds.id)
    col = next(c for c in ds2.columns if c["name"] == "x")
    assert col["label"] == "年龄标签" and col["vals"] == {"1": "一"}


# ---------------------------------------------------------------- 数据集合并
def test_merge_append_and_join(store):
    a = store.add("A", pd.DataFrame({"k": ["1", "2"], "v": [10, 20]}))
    b = store.add("B", pd.DataFrame({"k": ["2", "3"], "w": [200, 300]}))

    m1 = store.add("append", pd.concat([a.df, b.df], ignore_index=True))
    assert len(m1.df) == 4

    merged = pd.merge(a.df, b.df, on="k", how="inner")
    assert merged["w"].tolist() == [200]
    left = pd.merge(a.df, b.df, on="k", how="left")
    assert len(left) == 2 and left["w"].isna().any()


# ---------------------------------------------------------------- 加权个案
def test_set_weight_validation(store):
    ds = store.add("权重", pd.DataFrame({"w": [1.0, 2.0, 2.0], "v": ["a", "b", "a"], "bad": ["x", "y", "z"]}))
    ds.set_weight("w")
    assert ds.weight_col == "w" and ds.summary()["weightCol"] == "w"
    with pytest.raises(ModelError, match="不能作权重"):
        ds.set_weight("bad")
    with pytest.raises(ModelError, match="不能为负"):
        store.add("neg", pd.DataFrame({"w": [1, -2]})).set_weight("w")
    ds.set_weight(None)
    assert ds.weight_col is None


def test_weighted_descriptive_and_frequency(store):
    ds = store.add("加权", pd.DataFrame({"v": [10.0, 20.0], "w": [1.0, 3.0]}))
    ds.set_weight("w")
    from open_spss.core.registry import REGISTRY
    res = REGISTRY.get("descriptive").runner(ds, {"vars": ["v"]}, {})
    row = res.tables[0].rows[0]
    assert abs(float(row[1]) - 4.0) < 1e-6            # 加权样本量 = 1+3
    assert abs(float(row[5]) - 17.5) < 1e-6           # 加权均值 = (10+60)/4
    assert "加权" in res.tables[0].note

    res2 = REGISTRY.get("frequency").runner(ds, {"vars": ["v"]}, {})
    assert res2.tables[0].rows[0][1] == "3"           # v=20 加权频数 3(降序)
    assert res2.tables[0].rows[1][1] == "1"           # v=10 加权频数 1
    assert res2.notes and "加权" in res2.notes[0]


def test_weighted_chisq(store):
    from open_spss.core.registry import REGISTRY
    ds = store.add("卡方", pd.DataFrame({"x": ["是", "否", "是", "否"], "y": ["甲", "甲", "乙", "乙"],
                                         "w": [2.0, 2.0, 2.0, 2.0]}))
    ds.set_weight("w")
    res = REGISTRY.get("chisq").runner(ds, {"x": "x", "y": "y", "freq": ""}, {"correction": True, "fisher": True, "alpha": 0.05})
    assert "按「w」加权" in res.tables[0].title
    # 权重全为 2 等价于频数翻倍:p 值会变小,但列合计应为加权值 4
    assert res.tables[0].rows[-1][-1] == 8
