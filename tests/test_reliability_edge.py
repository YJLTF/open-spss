"""信度分析含常数题目(零方差,CITC 为 NaN)时不应崩溃。"""
from open_spss.core.datasets import Dataset
from open_spss.models.reliability import _run_reliability
from open_spss.samples import create


def test_reliability_with_constant_item():
    df = create("survey").df.copy()
    df["常数题"] = 3.0
    ds = Dataset("常数题测试", df)
    res = _run_reliability(ds, {"vars": ["Q1满意度", "Q2满意度", "常数题"]}, {})
    d = res.to_dict()
    t = d["tables"][0]
    citc_row = [r for r in t["rows"] if r[0] == "常数题"][0]
    assert citc_row[3] == "—"          # NaN 显示为 —,而非空串
    assert citc_row[5] == "建议检查"
    # 图表数据中 NaN 项为 None(ECharts 空档),不再有 float("") 崩溃
    assert any(v is None for v in d["charts"][0]["option"]["series"][0]["data"])
    assert "常数题" in d["interpretation"][0]
