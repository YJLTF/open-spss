"""数据集落盘:本地数据目录的解析、保存与启动恢复。

目录规则(可用环境变量 OPENSPSS_DATA_DIR 覆盖):
- 打包 exe:%APPDATA%/OpenSPSS/datasets
- 开发运行:仓库 data/datasets(已加入 .gitignore)
每个数据集一个文件 <id>.dspss,pickle 存储 {"meta": 元数据, "df": DataFrame};
写盘先写 .tmp 再原子替换,损坏文件在启动恢复时自动跳过。
"""
from __future__ import annotations

import os
import pickle
import sys

EXT = ".dspss"


def data_dir() -> str:
    env = os.environ.get("OPENSPSS_DATA_DIR")
    if env:
        d = env
    elif getattr(sys, "frozen", False):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        d = os.path.join(base, "OpenSPSS", "datasets")
    else:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        d = os.path.join(root, "data", "datasets")
    os.makedirs(d, exist_ok=True)
    return d


def save_dataset(ds) -> None:
    """原子写入数据集(数据帧 + 类型/标签等元数据)。"""
    path = os.path.join(data_dir(), f"{ds.id}{EXT}")
    payload = {
        "meta": {"id": ds.id, "name": ds.name, "source": ds.source,
                 "created": ds.created, "columns": ds.columns, "log": ds.log[-200:]},
        "df": ds.df,
    }
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except OSError as e:
        print(f"[storage] save {ds.id} failed: {e}", file=sys.stderr)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def remove_dataset(ds_id: str) -> None:
    try:
        os.unlink(os.path.join(data_dir(), f"{ds_id}{EXT}"))
    except OSError:
        pass


def load_all(store) -> int:
    """启动时恢复本地数据集到 store,返回加载数量;损坏文件跳过。"""
    from .datasets import Dataset

    n = 0
    try:
        files = [f for f in os.listdir(data_dir()) if f.endswith(EXT)]
    except OSError:
        return 0
    items = []
    for fn in files:
        try:
            with open(os.path.join(data_dir(), fn), "rb") as f:
                payload = pickle.load(f)
            meta, df = payload["meta"], payload["df"]
            items.append((meta, df))
        except Exception as e:  # noqa: BLE001
            print(f"[storage] skip corrupt {fn}: {e}", file=sys.stderr)
    items.sort(key=lambda it: it[0].get("created", 0))
    for meta, df in items:
        ds = Dataset(name=meta["name"], df=df, source=meta.get("source", "upload"))
        ds.id = meta["id"]
        ds.created = meta.get("created", ds.created)
        saved_cols = meta.get("columns") or []
        names = [c["name"] for c in saved_cols]
        # 元数据与数据帧结构一致才恢复(否则保留重新推断的结果)
        if names and list(map(str, df.columns)) == names:
            ds.columns = saved_cols
        ds.log = meta.get("log", [])
        store.datasets[ds.id] = ds
        store.order.append(ds.id)
        n += 1
    return n
