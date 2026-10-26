"""数据集管理:上传、类型推断、预览、元数据维护、落盘持久化。"""
from __future__ import annotations

import io
import sys
import time
import uuid

import numpy as np
import pandas as pd

from . import storage
from .results import ModelError

MAX_DATASETS = 30

DTYPES = {"quant": "定量", "cat": "定类", "ordinal": "定级", "text": "文本"}


class Dataset:
    def __init__(self, name: str, df: pd.DataFrame, source: str = "upload"):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.df = df
        self.source = source
        self.created = time.time()
        self.log: list[str] = []
        self._infer_all()

    # ---------- 类型推断 ----------
    def _infer_all(self) -> None:
        self.columns: list[dict] = []
        for col in self.df.columns:
            self.columns.append({"name": str(col), "dtype": self._infer_col(self.df[col]), "labels": {}})

    @staticmethod
    def _infer_col(s: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(s):
            return "cat"
        if pd.api.types.is_numeric_dtype(s):
            nn = s.dropna()
            if nn.nunique() <= 2:          # 0/1 型二分类
                return "cat"
            return "quant"
        if pd.api.types.is_datetime64_any_dtype(s):
            return "cat"
        return "cat"

    # ---------- 查询 ----------
    def col_meta(self) -> list[dict]:
        out = []
        for c in self.columns:
            info = dict(c)
            try:
                s = self.df[c["name"]]
                nn = s.dropna()
                if c["dtype"] == "quant":
                    info["range"] = [None if pd.isna(v) else float(v) for v in (nn.min(), nn.max())] if len(nn) else []
                else:
                    vals = nn.value_counts().head(12)
                    info["levels"] = [{"v": _safe(v), "n": int(n)} for v, n in vals.items()]
            except Exception:
                pass
            out.append(info)
        return out

    def summary(self) -> dict:
        return {
            "id": self.id, "name": self.name, "source": self.source,
            "rows": int(len(self.df)), "cols": int(self.df.shape[1]),
            "created": int(self.created), "log": self.log[-12:],
            "columns": self.col_meta(),
        }

    def preview(self, page: int = 1, page_size: int = 50) -> dict:
        total = len(self.df)
        start = max(0, (page - 1) * page_size)
        chunk = self.df.iloc[start:start + page_size]
        rows = [[_safe(v) for v in row] for row in chunk.itertuples(index=False)]
        return {"columns": [c["name"] for c in self.columns], "rows": rows,
                "total": total, "page": page, "pageSize": page_size}

    # ---------- 变更 ----------
    def save(self) -> None:
        """落盘(数据帧 + 类型/标签元数据)。失败不阻断分析主流程。"""
        try:
            storage.save_dataset(self)
        except OSError:
            pass

    def set_dtype(self, col: str, dtype: str) -> None:
        if dtype not in DTYPES:
            raise ModelError(f"未知变量类型: {dtype}")
        for c in self.columns:
            if c["name"] == col:
                if dtype == "quant":
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                c["dtype"] = dtype
                self.save()
                return
        raise ModelError(f"未找到列: {col}")

    def rename(self, old: str, new: str) -> None:
        if old not in self.df.columns:
            raise ModelError(f"未找到列: {old}")
        new = new.strip()
        if not new or new in self.df.columns:
            raise ModelError("新列名为空或已存在")
        self.df = self.df.rename(columns={old: new})
        for c in self.columns:
            if c["name"] == old:
                c["name"] = new
        self.save()

    def drop_cols(self, cols: list[str]) -> None:
        self.df = self.df.drop(columns=[c for c in cols if c in self.df.columns])
        self.columns = [c for c in self.columns if c["name"] not in cols]

    def note(self, text: str) -> None:
        self.log.append(f"{time.strftime('%H:%M')} {text}")
        self.save()


def _safe(v):
    """numpy/pandas 标量 → JSON 安全值"""
    if v is None:
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(v, pd.Timestamp):
        return str(v.date()) if hasattr(v, "date") else str(v)
    if isinstance(v, bytes):
        return v.decode("utf-8", "ignore")
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v) if not isinstance(v, (int, str)) else v


class DatasetStore:
    def __init__(self) -> None:
        self.datasets: dict[str, Dataset] = {}
        self.order: list[str] = []
        try:
            n = storage.load_all(self)
            if n:
                print(f"[storage] restored {n} datasets", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[storage] load failed: {e}", file=sys.stderr)

    # ---------- 读取 ----------
    def read_upload(self, filename: str, content: bytes) -> pd.DataFrame:
        name = filename.lower()
        try:
            if name.endswith((".csv", ".txt", ".tsv")):
                sep = "\t" if name.endswith(".tsv") else None
                for enc in ("utf-8-sig", "gbk", "utf-8", "latin1"):
                    try:
                        return pd.read_csv(io.BytesIO(content), sep=sep, engine="python",
                                           encoding=enc, skip_blank_lines=True)
                    except UnicodeDecodeError:
                        continue
                raise ModelError("无法识别文件编码,请另存为 UTF-8 后重试")
            if name.endswith((".xlsx", ".xls", ".xlsm")):
                return pd.read_excel(io.BytesIO(content), sheet_name=0)
            if name.endswith(".sav"):
                import pyreadstat
                df, meta = pyreadstat.read_sav(io.BytesIO(content))
                return df
            if name.endswith(".json"):
                return pd.read_json(io.BytesIO(content))
            if name.endswith(".parquet"):
                return pd.read_parquet(io.BytesIO(content))
        except ModelError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ModelError(f"文件解析失败: {e}") from e
        raise ModelError("不支持的文件格式,支持 csv / xlsx / xls / sav / tsv / json / parquet")

    # ---------- 管理 ----------
    def add(self, name: str, df: pd.DataFrame, source: str = "upload") -> Dataset:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        if df.shape[1] == 0:
            raise ModelError("未检测到任何列")
        ds = Dataset(name=name, df=df, source=source)
        if len(self.datasets) >= MAX_DATASETS:
            evict = self.order.pop(0)
            self.datasets.pop(evict, None)
            storage.remove_dataset(evict)
        self.datasets[ds.id] = ds
        self.order.append(ds.id)
        ds.save()
        return ds

    def get(self, ds_id: str) -> Dataset:
        ds = self.datasets.get(ds_id)
        if ds is None:
            raise ModelError("数据集不存在或已被清理,请重新上传")
        return ds

    def list(self) -> list[dict]:
        out = []
        for i in self.order:
            ds = self.datasets[i]
            out.append({"id": ds.id, "name": ds.name, "rows": len(ds.df),
                        "cols": ds.df.shape[1], "source": ds.source, "created": int(ds.created)})
        return out

    def delete(self, ds_id: str) -> None:
        self.datasets.pop(ds_id, None)
        if ds_id in self.order:
            self.order.remove(ds_id)
        storage.remove_dataset(ds_id)


STORE = DatasetStore()


def numeric_cols(ds: Dataset, cols: list[str]) -> list[str]:
    """校验所选列存在且可数值化(自动 to_numeric),返回实际列名。"""
    out = []
    for c in cols:
        if c not in ds.df.columns:
            raise ModelError(f"列「{c}」不存在")
        out.append(c)
    return out


def as_numeric(ds: Dataset, col: str) -> pd.Series:
    if col not in ds.df.columns:
        raise ModelError(f"列「{col}」不存在")
    s = pd.to_numeric(ds.df[col], errors="coerce")
    if s.notna().sum() == 0:
        raise ModelError(f"列「{col}」无法转换为数值")
    return s


def get_series(ds: Dataset, col: str) -> pd.Series:
    if col not in ds.df.columns:
        raise ModelError(f"列「{col}」不存在")
    return ds.df[col]


def clean_numeric(ds: Dataset, cols: list[str], how: str = "listwise") -> pd.DataFrame:
    """取数值列并按策略清理缺失。how: listwise(删行) / pairwise(保留NaN)"""
    data = pd.concat([as_numeric(ds, c).rename(c) for c in cols], axis=1)
    if how == "listwise":
        data = data.dropna()
        if len(data) < 3:
            raise ModelError("有效样本不足(去除缺失后少于 3 行),请检查数据")
    return data
