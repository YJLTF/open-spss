"""文档一致性:README 的模型计数与清单必须与注册表同步(CI 阻断)。"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_readme_models_in_sync():
    r = subprocess.run([sys.executable, os.path.join("scripts", "sync_models.py"), "--check"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr or "README 与模型注册表不同步"
