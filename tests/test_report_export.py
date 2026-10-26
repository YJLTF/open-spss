"""报告导出:Word 与 PDF 生成合法文件且内容完整。"""
from __future__ import annotations

import base64
import io
import zipfile

from open_spss.report_export import ReportChart, ReportItem, ReportPayload, ReportTable, build_docx, build_pdf

import struct
import zlib


def _make_png() -> str:
    """程序化生成一张 8×8 的合法 PNG(纯色),模拟图表导出。"""
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    sig = bytes([0x89]) + b"PNG" + bytes([0x0D, 0x0A, 0x1A, 0x0A])
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
    raw = b"".join(bytes([0]) + bytes([0x70, 0x6B, 0xFF]) * 8 for _ in range(8))
    return base64.b64encode(sig + ihdr + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")).decode()


PNG_1PX = _make_png()


def _payload() -> ReportPayload:
    return ReportPayload(
        title="数据分析报告",
        generated="生成时间:2026-10-26 · 共 1 项分析",
        items=[ReportItem(
            title="1. 相关分析",
            meta="通用统计 · 数据集「员工满意度调查」",
            slots="分析变量:月薪、年龄",
            paragraphs=["相关系数为 **0.52**,呈中等相关。", "结论:`p<0.01` 显著。"],
            tables=[ReportTable(title="相关系数矩阵", headers=["变量", "月薪", "年龄"],
                                rows=[["月薪", "1", "0.52"], ["年龄", "0.52", "1"]], note="*** p<0.01")],
            charts=[ReportChart(title="散点图", png="data:image/png;base64," + PNG_1PX)],
        )],
    )


def test_build_docx_valid_zip():
    data = build_docx(_payload())
    assert data[:2] == b"PK"                              # docx 是 zip 容器
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert "word/document.xml" in z.namelist()
    assert len(data) > 4000


def test_build_pdf_valid():
    data = build_pdf(_payload())
    assert data[:5] == b"%PDF-"
    assert b"EOF" in data[-32:]
    assert len(data) > 2000


def test_build_docx_dariong_theme():
    p = _payload()
    p.dark = True
    assert build_docx(p)[:2] == b"PK"                     # 暗色标记不影响生成


def test_export_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENSPSS_DATA_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from open_spss.app import app

    c = TestClient(app)
    r = c.post("/api/export/report/docx", json=_payload().model_dump())
    assert r.status_code == 200 and r.content[:2] == b"PK"
    r = c.post("/api/export/report/pdf", json=_payload().model_dump())
    assert r.status_code == 200 and r.content[:5] == b"%PDF-"


def test_model_error_returns_friendly_400(tmp_path, monkeypatch):
    """回归:业务错误(ModelError)必须是 400 + 友好 detail,而非 500(自 v1.0.0 起处理器一直有误)。"""
    monkeypatch.setenv("OPENSPSS_DATA_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from open_spss.app import app

    c = TestClient(app)
    ds = c.post("/api/datasets/sample/survey").json()
    r = c.post("/api/run", json={"model": "mediation", "dataset": ds["id"], "slots": {}, "options": {}})
    assert r.status_code == 400
    assert "选择变量" in r.json()["detail"]
