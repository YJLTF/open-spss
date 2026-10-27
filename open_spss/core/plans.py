"""分析方案:分析步骤快照的可持久化存储、管理与批量重放。

方案只存「怎么算」(模型 + 数据集引用 + 槽位/选项),不存结果——重放时现场计算,
避免落盘膨胀。数据集按 id 解析,被删除的步骤标记跳过,可改绑到现有数据集。
"""
from __future__ import annotations

import time
import traceback
import uuid

from . import storage
from ..codegen import code_for
from .datasets import STORE
from .registry import REGISTRY
from .results import ModelError

MAX_STEPS = 100
MAX_NAME = 60


def _clean_step(raw: dict) -> dict:
    """白名单化步骤字段并截断超长文本;slots/options 原样保留(JSON 值)。"""
    if not isinstance(raw, dict):
        raise ModelError("方案步骤格式不正确")
    model = str(raw.get("model") or "").strip()
    if not model:
        raise ModelError("方案步骤缺少模型 id")
    try:
        REGISTRY.get(model)
    except KeyError:
        pass  # 允许保存来自更高版本的模型:展示不受影响,重放时标记跳过
    return {
        "model": model[:80],
        "modelName": str(raw.get("modelName") or model)[:80],
        "dsId": str(raw.get("dsId") or "")[:24],
        "dsName": str(raw.get("dsName") or "")[:80],
        "slots": raw.get("slots") if isinstance(raw.get("slots"), dict) else {},
        "options": raw.get("options") if isinstance(raw.get("options"), dict) else {},
        "slotsText": str(raw.get("slotsText") or "")[:200],
    }


def _clean_name(name: str, what: str = "方案名称") -> str:
    name = str(name or "").strip()
    if not name:
        raise ModelError(f"{what}不能为空")
    if len(name) > MAX_NAME:
        raise ModelError(f"{what}过长（不超过 {MAX_NAME} 字符）")
    return name


class PlanStore:
    def __init__(self) -> None:
        self.plans: dict[str, dict] = {}
        for doc in storage.load_plans():
            self.plans[doc["id"]] = doc

    # ---------- 查询 / 管理 ----------
    def list(self) -> list[dict]:
        out = sorted(self.plans.values(), key=lambda p: p.get("updated", 0), reverse=True)
        return [{"id": p["id"], "name": p["name"],
                 "created": p.get("created", 0), "updated": p.get("updated", 0),
                 "steps": [{"model": s["model"], "modelName": s.get("modelName", ""),
                            "dsId": s.get("dsId", ""), "dsName": s.get("dsName", "")}
                           for s in p.get("steps", [])]}
                for p in out]

    def get(self, plan_id: str) -> dict:
        plan = self.plans.get(plan_id)
        if plan is None:
            raise ModelError("分析方案不存在或已被删除")
        return plan

    def create(self, name: str, steps: list) -> dict:
        doc = {"v": 1, "id": uuid.uuid4().hex[:12], "name": _clean_name(name),
               "created": time.time(), "updated": time.time(),
               "steps": self._clean_steps(steps)}
        self._save(doc)
        return doc

    def update(self, plan_id: str, name: str | None = None, steps: list | None = None) -> dict:
        doc = self.get(plan_id)
        if name is not None:
            doc["name"] = _clean_name(name)
        if steps is not None:
            doc["steps"] = self._clean_steps(steps)
        doc["updated"] = time.time()
        self._save(doc)
        return doc

    def delete(self, plan_id: str) -> None:
        self.get(plan_id)
        self.plans.pop(plan_id, None)
        storage.remove_plan(plan_id)

    @staticmethod
    def _clean_steps(steps: list) -> list[dict]:
        cleaned = [_clean_step(s) for s in (steps or [])]
        if len(cleaned) > MAX_STEPS:
            raise ModelError(f"单个方案最多 {MAX_STEPS} 个步骤")
        return cleaned

    def _save(self, doc: dict) -> None:
        try:
            storage.save_plan(doc)
        except OSError as e:
            raise ModelError(f"方案写入失败: {e}") from e
        self.plans[doc["id"]] = doc

    # ---------- 批量重放 ----------
    def run(self, plan_id: str) -> dict:
        """顺序重放各步,单步失败/跳过不中断后续;返回逐条状态与结果。"""
        plan = self.get(plan_id)
        outcomes = [self._run_step(i, step) for i, step in enumerate(plan.get("steps", []))]
        return {"plan": plan, "outcomes": outcomes}

    @staticmethod
    def _run_step(index: int, step: dict) -> dict:
        # 延迟导入避免循环依赖(executor → datasets/registry,plans → executor)
        from .executor import execute

        out = {"index": index, "model": step.get("model", ""), "modelName": step.get("modelName", ""),
               "dsId": step.get("dsId", ""), "dsName": step.get("dsName", ""), "status": "ok"}
        if step.get("dsId") and step["dsId"] not in STORE.datasets:
            out["status"] = "skipped"
            out["error"] = "原数据集已被删除，可改绑到现有数据集后重试"
            return out
        try:
            try:
                spec, ds, slots, opts, result = execute(
                    step["model"], step["dsId"], step.get("slots") or {}, step.get("options") or {})
            except KeyError:
                out["status"] = "skipped"
                out["error"] = "该模型在当前版本中不存在"
                return out
            except ModelError as e:
                out["status"] = "error"
                out["error"] = str(e)
                return out
            res = result.to_dict()
            res["model"] = {"id": spec.id, "name": spec.name}
            code = code_for(spec.id, ds, slots, opts)
            if code:
                res["code"] = code
            out["result"] = res
        except RecursionError:
            out["status"] = "error"
            out["error"] = "数据规模过大导致计算中断，请抽样后重试"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            out["status"] = "error"
            out["error"] = f"计算出错: {type(e).__name__}: {e}"
        return out


PLAN_STORE = PlanStore()
