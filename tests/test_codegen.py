"""代码生成:覆盖的模型 id 必须存在,生成的代码必须是合法 Python。"""
from __future__ import annotations

import ast

from open_spss.codegen import MODEL_CODE, code_for
from open_spss.core.registry import REGISTRY
from open_spss.models import CATEGORIES  # noqa: F401
from open_spss.samples import create


def test_codegen_ids_exist_in_registry():
    known = {s.id for s in REGISTRY.all()}
    unknown = [m for m in MODEL_CODE if m not in known]
    assert not unknown, f"codegen 覆盖了不存在的模型: {unknown}"


def test_codegen_min_coverage():
    # 规划要求 v1.1.0 至少覆盖 20 个高频模型
    assert len(MODEL_CODE) >= 20


def test_all_templates_generate_valid_python():
    from test_smoke_all_models import default_slots  # pytest prepend 模式下的同目录导入

    failures = []
    for mid in MODEL_CODE:
        spec = REGISTRY.get(mid)
        ds = create(spec.demo_note or "survey")
        slots = default_slots(spec, ds)
        slots, opts = REGISTRY.validate(mid, slots, {})
        code = code_for(mid, ds, slots, opts)
        if not code:
            failures.append((mid, "empty"))
            continue
        try:
            ast.parse(code)
        except SyntaxError as e:
            failures.append((mid, str(e)[:80]))
    assert not failures, f"代码模板生成失败: {failures}"
