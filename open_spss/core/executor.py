"""模型执行核心:校验与运行,供单次分析(/api/run)与方案批量重放共用。"""
from __future__ import annotations

from .datasets import STORE, Dataset
from .registry import ModelSpec, REGISTRY
from .results import ModelError, Result

_DTYPE_NAME = {"quant": "定量", "cat": "定类", "ordinal": "定级"}


def execute(model_id: str, ds_id: str, slots: dict, options: dict
            ) -> tuple[ModelSpec, Dataset, dict, dict, Result]:
    """校验并运行单个模型,返回 (spec, ds, 规范化slots, 规范化opts, Result)。

    未知模型抛 KeyError;数据集缺失、变量类型不符等业务错误抛 ModelError。
    """
    spec = REGISTRY.get(model_id)
    ds = STORE.get(ds_id)
    norm_slots, opts = REGISTRY.validate(model_id, slots, options)
    dtype_map = {c["name"]: c["dtype"] for c in ds.columns}
    for s in spec.slots:
        val = norm_slots.get(s.key)
        cols = val if isinstance(val, list) else ([val] if val else [])
        for c in cols:
            if c not in dtype_map:
                raise ModelError(f"变量「{c}」不在当前数据集中")
            if not s.accepts(dtype_map[c]):
                tname = _DTYPE_NAME.get(dtype_map[c], dtype_map[c])
                raise ModelError(f"「{s.label}」不接收{tname}变量「{c}」，请在数据页修改变量类型或换选变量")
    return spec, ds, norm_slots, opts, spec.runner(ds, norm_slots, opts)
