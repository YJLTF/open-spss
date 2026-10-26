"""数据集落盘:保存/恢复/删除,类型与标签元数据一并持久化。"""
import pandas as pd

from open_spss.core import datasets as dsm


def test_persist_and_restore(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENSPSS_DATA_DIR", str(tmp_path))
    store = dsm.DatasetStore()
    ds = store.add("落盘测试", pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
    ds.set_dtype("a", "cat")
    ds.note("改了类型")
    assert (tmp_path / f"{ds.id}.dspss").exists()

    # 模拟重启:新 store 实例从同一目录恢复
    store2 = dsm.DatasetStore()
    ds2 = store2.get(ds.id)
    assert ds2.name == "落盘测试"
    assert ds2.df.shape == (3, 2)
    assert {c["name"]: c["dtype"] for c in ds2.columns}["a"] == "cat"
    assert ds2.log and "改了类型" in ds2.log[-1]
    assert [c["dtype"] for c in ds2.columns] == ["cat", "cat"]  # b 自动推断也是 cat


def test_delete_removes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENSPSS_DATA_DIR", str(tmp_path))
    store = dsm.DatasetStore()
    ds = store.add("待删除", pd.DataFrame({"x": [1]}))
    path = tmp_path / f"{ds.id}.dspss"
    assert path.exists()
    store.delete(ds.id)
    assert not path.exists()


def test_corrupt_file_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENSPSS_DATA_DIR", str(tmp_path))
    (tmp_path / "bad.dspss").write_bytes(b"not-a-pickle")
    store = dsm.DatasetStore()  # 不应抛异常
    assert store.list() == []
