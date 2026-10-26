"""模型注册表:以声明式元数据描述每个分析模型,前端据此自动渲染配置表单。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .results import ModelError, zh_punct

# 变量类型:quant=定量 cat=定类 ordinal=定级/有序 any=任意
QUANT = "quant"
CAT = "cat"
ORDINAL = "ordinal"
ANY = "any"


@dataclass
class Slot:
    """变量槽位。multiple=True 时可拖入多个变量。"""
    key: str
    label: str
    types: tuple[str, ...] = (QUANT,)
    multiple: bool = False
    required: bool = True
    max_count: int | None = None
    tip: str = ""

    def accepts(self, dtype: str) -> bool:
        return ANY in self.types or dtype in self.types

    def to_dict(self) -> dict:
        return {
            "key": self.key, "label": self.label, "types": list(self.types),
            "multiple": self.multiple, "required": self.required,
            "maxCount": self.max_count, "tip": self.tip,
        }


@dataclass
class Option:
    """标量配置项。kind: select / number / switch / text / textarea"""
    key: str
    label: str
    kind: str = "select"
    default: Any = None
    choices: list[dict] | None = None   # [{"value":..., "label":...}]
    min: float | None = None
    max: float | None = None
    step: float | None = None
    tip: str = ""
    show_if: dict | None = None         # {"optKey": value} 满足时才显示

    def to_dict(self) -> dict:
        return {
            "key": self.key, "label": self.label, "kind": self.kind,
            "default": self.default, "choices": self.choices,
            "min": self.min, "max": self.max, "step": self.step,
            "tip": self.tip, "showIf": self.show_if,
        }


@dataclass
class ModelSpec:
    id: str
    name: str
    category: str
    desc: str = ""
    tags: list[str] = field(default_factory=list)
    slots: list[Slot] = field(default_factory=list)
    options: list[Option] = field(default_factory=list)
    runner: Callable | None = None      # runner(df, slots, opts) -> Result
    demo_note: str = ""                 # 推荐搭配的示例数据集 id

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "category": self.category,
            "desc": self.desc, "tags": self.tags,
            "slots": [s.to_dict() for s in self.slots],
            "options": [o.to_dict() for o in self.options],
            "demoNote": self.demo_note,
        }


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> ModelSpec:
        if spec.id in self._models:
            raise ModelError(f"模型 id 重复: {spec.id}")
        # UI 文案统一中文标点:描述/槽位/选项/标签在注册口归一,
        # 目录展示与校验错误提示(引用 slot.label)自动一致
        spec.name = zh_punct(spec.name)
        spec.desc = zh_punct(spec.desc)
        spec.tags = [zh_punct(t) for t in spec.tags]
        for s in spec.slots:
            s.label = zh_punct(s.label)
            s.tip = zh_punct(s.tip)
        for o in spec.options:
            o.label = zh_punct(o.label)
            o.tip = zh_punct(o.tip)
            if o.choices:
                o.choices = [{**c, "label": zh_punct(c["label"])} for c in o.choices]
        self._models[spec.id] = spec
        return spec

    def get(self, model_id: str) -> ModelSpec:
        return self._models[model_id]

    def all(self) -> list[ModelSpec]:
        return list(self._models.values())

    def categories(self) -> list[dict]:
        cats: dict[str, dict] = {}
        order: list[str] = []
        for m in self._models.values():
            if m.category not in cats:
                order.append(m.category)
                cats[m.category] = {"name": m.category, "models": []}
            cats[m.category]["models"].append(m.to_dict())
        return [cats[c] for c in order]

    def validate(self, model_id: str, slots: dict, opts: dict) -> tuple[dict, dict]:
        """校验前端提交的槽位/选项,返回规范化的 (slots, opts)。"""
        spec = self.get(model_id)
        norm_slots: dict[str, list[str] | str] = {}
        for s in spec.slots:
            v = slots.get(s.key)
            if s.multiple:
                cols = [c for c in (v or []) if isinstance(c, str) and c]
                if s.required and not cols:
                    raise ModelError(f"请为「{s.label}」选择变量")
                if s.max_count and len(cols) > s.max_count:
                    raise ModelError(f"「{s.label}」最多选择 {s.max_count} 个变量")
                norm_slots[s.key] = cols
            else:
                if not isinstance(v, str) or not v:
                    if s.required:
                        raise ModelError(f"请为「{s.label}」选择变量")
                    v = ""
                norm_slots[s.key] = v
        norm_opts: dict[str, Any] = {}
        for o in spec.options:
            val = opts.get(o.key, o.default)
            if val is None:
                val = o.default
            if o.kind == "select" and o.choices:
                allowed = [c["value"] for c in o.choices]
                if val not in allowed:
                    val = o.default
            if o.kind == "switch":
                val = bool(val)
            if o.kind == "number" and val is not None:
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = o.default
                if o.min is not None and val < o.min:
                    val = o.min
                if o.max is not None and val > o.max:
                    val = o.max
                if val == int(val) and (o.step in (1, None)):
                    val = int(val)
            norm_opts[o.key] = val
        return norm_slots, norm_opts


REGISTRY = ModelRegistry()


def register_model(spec: ModelSpec) -> ModelSpec:
    return REGISTRY.register(spec)
