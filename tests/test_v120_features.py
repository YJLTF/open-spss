"""v1.2.0 新特性:分析方案落盘管理与批量重放、示例数据集删除辅助(sample_key/重置)、
数据集重命名与导出、容量驱逐提示。"""
import pandas as pd


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENSPSS_DATA_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from open_spss.app import app
    return TestClient(app)


def _plan_steps(ds_id: str) -> list:
    return [
        {"model": "descriptive", "modelName": "描述性统计", "dsId": ds_id, "dsName": "成绩表",
         "slots": {"vars": ["数学", "英语"]}, "options": {}, "slotsText": "分析变量:数学、英语"},
        {"model": "descriptive", "modelName": "描述性统计", "dsId": "missing123", "dsName": "已删数据集",
         "slots": {"vars": ["数学"]}, "options": {}, "slotsText": ""},
        {"model": "correlation", "modelName": "相关分析", "dsId": ds_id, "dsName": "成绩表",
         "slots": {"vars": ["数学", "化学"]}, "options": {}, "slotsText": ""},
    ]


# ---------------------------------------------------------------- 分析方案
def test_plan_crud_and_persistence(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    ds = c.post("/api/datasets/sample/scores").json()

    plan = c.post("/api/plans", json={"name": "成绩研究", "steps": _plan_steps(ds["id"])}).json()
    assert plan["name"] == "成绩研究" and len(plan["steps"]) == 3
    assert plan["steps"][0]["slots"] == {"vars": ["数学", "英语"]}

    lst = c.get("/api/plans").json()["plans"]
    assert [p["id"] for p in lst] == [plan["id"]]
    assert len(lst[0]["steps"]) == 3          # 列表带步骤概要,便于抽屉展示

    assert c.put(f"/api/plans/{plan['id']}", json={"name": "成绩研究 v2"}).json()["name"] == "成绩研究 v2"
    shorter = c.put(f"/api/plans/{plan['id']}", json={"steps": _plan_steps(ds["id"])[:1]}).json()
    assert len(shorter["steps"]) == 1

    # 模拟重启:新实例从落盘 JSON 恢复
    from open_spss.core.plans import PlanStore
    reloaded = PlanStore().get(plan["id"])
    assert reloaded["name"] == "成绩研究 v2" and len(reloaded["steps"]) == 1

    assert c.delete(f"/api/plans/{plan['id']}").json()["ok"] is True
    assert c.get(f"/api/plans/{plan['id']}").status_code == 400


def test_plan_validation(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    assert c.post("/api/plans", json={"name": "  ", "steps": []}).status_code == 400
    assert c.post("/api/plans", json={"name": "x" * 61, "steps": []}).status_code == 400
    assert c.post("/api/plans", json={"name": "坏步骤", "steps": [{"modelName": "没有模型 id"}]}).status_code == 400
    assert c.post("/api/plans", json={"name": "超长", "steps": [{"model": "descriptive"}] * 101}).status_code == 400
    assert c.put("/api/plans/nonexistent", json={"name": "x"}).status_code == 400
    assert c.delete("/api/plans/nonexistent").status_code == 400


def test_plan_run_batch(monkeypatch, tmp_path):
    """批量重放:ok / 跳过(数据集已删) / 业务错误逐条返回,单步不中断。"""
    c = _client(monkeypatch, tmp_path)
    ds = c.post("/api/datasets/sample/scores").json()
    plan = c.post("/api/plans", json={"name": "批量", "steps": _plan_steps(ds["id"])}).json()

    run = c.post(f"/api/plans/{plan['id']}/run").json()
    assert [o["status"] for o in run["outcomes"]] == ["ok", "skipped", "ok"]
    ok = run["outcomes"][0]
    assert ok["result"]["tables"] and "数学" in str(ok["result"]["tables"][0]["rows"][0])
    assert ok["result"]["model"] == {"id": "descriptive", "name": "描述性统计"}
    assert run["outcomes"][1]["error"]         # 跳过原因可读
    assert run["outcomes"][2]["result"]["charts"]


def test_plan_run_step_error(monkeypatch, tmp_path):
    """槽位不满足的步骤标记失败,信息为友好业务文案。"""
    c = _client(monkeypatch, tmp_path)
    ds = c.post("/api/datasets/sample/scores").json()
    plan = c.post("/api/plans", json={"name": "失败步", "steps": [
        {"model": "descriptive", "dsId": ds["id"], "slots": {}, "options": {}}]}).json()
    run = c.post(f"/api/plans/{plan['id']}/run").json()
    assert run["outcomes"][0]["status"] == "error"
    assert "选择变量" in run["outcomes"][0]["error"]


# ---------------------------------------------------------------- 示例数据集:sample_key / 重置
def test_sample_key_and_reset(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    ds = c.post("/api/datasets/sample/scores").json()
    assert ds["sampleKey"] == "scores"

    lst = c.get("/api/datasets").json()["datasets"]
    assert [d["sampleKey"] for d in lst if d["id"] == ds["id"]] == ["scores"]

    # 重置:行数与初始一致,保留 id
    r = c.post(f"/api/datasets/{ds['id']}/reset-sample")
    assert r.status_code == 200 and r.json()["rows"] == ds["rows"]
    assert r.json()["id"] == ds["id"]

    # sample_key 落盘恢复
    from open_spss.core.datasets import DatasetStore
    assert DatasetStore().get(ds["id"]).sample_key == "scores"

    # 非示例数据集不能重置
    from open_spss.core.datasets import STORE
    other = STORE.add("自建", pd.DataFrame({"a": [1, 2]}))
    assert c.post(f"/api/datasets/{other.id}/reset-sample").status_code == 400


# ---------------------------------------------------------------- 数据集重命名 / 导出
def test_dataset_rename(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    ds = c.post("/api/datasets/sample/scores").json()
    r = c.put(f"/api/datasets/{ds['id']}/name", json={"name": "  高三成绩  "})
    assert r.json()["name"] == "高三成绩"
    assert c.put(f"/api/datasets/{ds['id']}/name", json={"name": "   "}).status_code == 400

    from open_spss.core.datasets import DatasetStore
    assert DatasetStore().get(ds["id"]).name == "高三成绩"   # 重命名落盘


def test_dataset_export(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    ds = c.post("/api/datasets/sample/scores").json()

    r = c.get(f"/api/datasets/{ds['id']}/export?format=csv")
    assert r.status_code == 200
    assert r.content.decode("utf-8-sig").splitlines()[0].startswith("学号")
    assert "filename*=UTF-8''" in r.headers["content-disposition"]

    r = c.get(f"/api/datasets/{ds['id']}/export?format=xlsx")
    assert r.status_code == 200 and r.content[:2] == b"PK"
    assert c.get(f"/api/datasets/{ds['id']}/export?format=pdf").status_code == 400
    assert c.get("/api/datasets/missing/export?format=csv").status_code == 400


# ---------------------------------------------------------------- 容量驱逐提示
def test_eviction_reported(monkeypatch, tmp_path):
    monkeypatch.setattr("open_spss.core.datasets.MAX_DATASETS", 3)
    c = _client(monkeypatch, tmp_path)
    from open_spss.core.datasets import STORE
    STORE.datasets.clear()      # TestClient 复用模块级单例,清掉其他用例残留保证驱逐点确定
    STORE.order.clear()
    STORE.take_evicted()
    ids = []
    for i in range(4):
        r = c.post("/api/upload", files={"file": (f"d{i}.csv", f"x\n{i}".encode())})
        assert r.status_code == 200
        ids.append(r.json())
    assert ids[0]["id"] not in [d["id"] for d in c.get("/api/datasets").json()["datasets"]]
    assert ids[-1]["evicted"] == "d0"          # 最旧的被驱逐且随响应告知前端
    assert STORE.take_evicted() is None        # 取一次即清空


# ---------------------------------------------------------------- /api/run 回归
def test_run_endpoint_regression(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    ds = c.post("/api/datasets/sample/scores").json()
    r = c.post("/api/run", json={"model": "descriptive", "dataset": ds["id"],
                                 "slots": {"vars": ["数学"]}, "options": {}})
    assert r.status_code == 200 and r.json()["model"]["id"] == "descriptive"
    assert c.post("/api/run", json={"model": "nope", "dataset": ds["id"],
                                    "slots": {}, "options": {}}).status_code == 404
