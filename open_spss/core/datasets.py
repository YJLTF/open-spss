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


# 撤销快照:数据行数超过该阈值时禁用(内存保护),栈深上限 20
UNDO_LIMIT = 20
MAX_UNDO_ROWS = 200_000


class Dataset:
    def __init__(self, name: str, df: pd.DataFrame, source: str = "upload",
                 meta: dict | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.df = df
        self.source = source
        self.created = time.time()
        self.log: list[str] = []
        self.weight_col: str | None = None
        self._undo: list[tuple] = []
        self._redo: list[tuple] = []
        self._infer_all()
        self._apply_meta(meta or {})

    # ---------- 类型推断 ----------
    def _infer_all(self) -> None:
        self.columns: list[dict] = []
        for col in self.df.columns:
            self.columns.append({"name": str(col), "dtype": self._infer_col(self.df[col]),
                                 "label": "", "vals": {}})

    def _apply_meta(self, meta: dict) -> None:
        """应用上传时携带的元数据(sav 变量标签/值标签)。"""
        for c in self.columns:
            if c["name"] in meta.get("var_labels", {}):
                c["label"] = str(meta["var_labels"][c["name"]])
            vl = meta.get("value_labels", {}).get(c["name"])
            if vl:
                c["vals"] = {str(k): str(v) for k, v in vl.items()}

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
            "canUndo": bool(self._undo), "canRedo": bool(self._redo),
            "weightCol": self.weight_col,
        }

    # ---------- 加权个案 ----------
    def set_weight(self, column: str | None) -> None:
        if column:
            self._col_meta_of(column)
            s = pd.to_numeric(self.df[column], errors="coerce")
            if s.isna().any():
                raise ModelError(f"「{column}」存在无法转为数值的取值,不能作权重")
            if (s < 0).any():
                raise ModelError("权重不能为负")
        self.weight_col = column or None
        self.save()

    def weights(self) -> pd.Series | None:
        """加权个案:返回与数据框对齐的数值权重;未设置返回 None。"""
        if not self.weight_col:
            return None
        w = pd.to_numeric(self.df[self.weight_col], errors="coerce").fillna(0).astype(float)
        return w.rename("__w__")

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

    # ---------- 撤销 / 重做 ----------
    def _push_snapshot(self) -> None:
        if len(self.df) > MAX_UNDO_ROWS:
            return
        self._undo.append((self.df.copy(), [dict(c) for c in self.columns]))
        if len(self._undo) > UNDO_LIMIT:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append((self.df.copy(), [dict(c) for c in self.columns]))
        self.df, self.columns = self._undo.pop()
        self.save()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append((self.df.copy(), [dict(c) for c in self.columns]))
        self.df, self.columns = self._redo.pop()
        self.save()
        return True

    # ---------- 单元格 / 行列编辑 ----------
    def _col_meta_of(self, column: str) -> dict:
        meta = next((c for c in self.columns if c["name"] == column), None)
        if meta is None:
            raise ModelError(f"未找到列: {column}")
        return meta

    def set_cell(self, row: int, column: str, value) -> None:
        if row < 0 or row >= len(self.df):
            raise ModelError(f"行号越界: {row + 1}(共 {len(self.df)} 行)")
        meta = self._col_meta_of(column)
        text = "" if value is None else str(value).strip()
        ridx = self.df.index[row]
        if text == "" or text.lower() in ("na", "nan", "null", "n/a"):
            self._push_snapshot()
            self.df.loc[ridx, column] = None
        elif meta["dtype"] == "quant":
            try:
                num = float(text)
            except ValueError:
                raise ModelError(f"「{column}」是定量变量,「{text}」无法转为数值") from None
            self._push_snapshot()
            self.df.loc[ridx, column] = num
        else:
            self._push_snapshot()
            self.df.loc[ridx, column] = text
        self.save()

    def add_row(self) -> None:
        self._push_snapshot()
        nxt = int(self.df.index.max()) + 1 if len(self.df) else 0   # 索引可能非连续
        self.df.loc[nxt] = [None] * self.df.shape[1]
        self.save()

    def del_row(self, row: int) -> None:
        if row < 0 or row >= len(self.df):
            raise ModelError(f"行号越界: {row + 1}")
        self._push_snapshot()
        self.df = self.df.drop(self.df.index[row]).reset_index(drop=True)
        self.save()

    def add_col(self, name: str, dtype: str = "cat") -> None:
        name = name.strip()
        if not name or name in self.df.columns:
            raise ModelError("新列名为空或已存在")
        if dtype not in DTYPES:
            raise ModelError(f"未知变量类型: {dtype}")
        self._push_snapshot()
        self.df[name] = None
        self.columns.append({"name": name, "dtype": dtype, "label": "", "vals": {}})
        self.save()

    def write_col(self, name: str, values, dtype: str = "quant") -> None:
        """模型结果写回:整列赋值并同步列元数据(同名覆盖,元数据保留首次类型),落盘持久化"""
        if dtype not in DTYPES:
            raise ModelError(f"未知变量类型: {dtype}")
        self.df[name] = values
        if not any(c["name"] == name for c in self.columns):
            self.columns.append({"name": name, "dtype": dtype, "label": "", "vals": {}})
        self.save()

    def del_col(self, name: str) -> None:
        self._col_meta_of(name)
        self._push_snapshot()
        self.drop_cols([name])
        self.save()

    # ---------- 变量标签 / 值标签 ----------
    def set_labels(self, column: str, label: str, vals: dict | None = None) -> None:
        meta = self._col_meta_of(column)
        meta["label"] = str(label or "").strip()
        if vals is not None:
            meta["vals"] = {str(k): str(v) for k, v in vals.items() if str(k) != ""}
        self.save()

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
    def read_upload(self, filename: str, content: bytes) -> tuple[pd.DataFrame, dict]:
        """解析上传文件,返回 (DataFrame, 元数据);元数据目前含 sav 的变量/值标签。"""
        name = filename.lower()
        try:
            if name.endswith((".csv", ".txt", ".tsv")):
                sep = "\t" if name.endswith(".tsv") else None
                for enc in ("utf-8-sig", "gbk", "utf-8", "latin1"):
                    try:
                        return pd.read_csv(io.BytesIO(content), sep=sep, engine="python",
                                           encoding=enc, skip_blank_lines=True), {}
                    except UnicodeDecodeError:
                        continue
                raise ModelError("无法识别文件编码,请另存为 UTF-8 后重试")
            if name.endswith((".xlsx", ".xls", ".xlsm")):
                return pd.read_excel(io.BytesIO(content), sheet_name=0), {}
            if name.endswith(".sav"):
                import pyreadstat
                df, meta = pyreadstat.read_sav(io.BytesIO(content))
                extra = {
                    "var_labels": dict(getattr(meta, "variable_labels", None) or {}),
                    "value_labels": {k: dict(v) for k, v in
                                     (getattr(meta, "variable_value_labels", None) or {}).items()},
                }
                return df, extra
            if name.endswith(".json"):
                return pd.read_json(io.BytesIO(content)), {}
            if name.endswith(".parquet"):
                return pd.read_parquet(io.BytesIO(content)), {}
        except ModelError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ModelError(f"文件解析失败: {e}") from e
        raise ModelError("不支持的文件格式,支持 csv / xlsx / xls / sav / tsv / json / parquet")

    # ---------- 管理 ----------
    def add(self, name: str, df: pd.DataFrame, source: str = "upload",
            meta: dict | None = None) -> Dataset:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        if df.shape[1] == 0:
            raise ModelError("未检测到任何列")
        ds = Dataset(name=name, df=df, source=source, meta=meta)
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
