"""从 CHANGELOG.md 提取指定版本的发布说明(供 CI 创建 GitHub Release)。

用法:python scripts/release_notes.py v1.1.0   # 版本可带或不带 v 前缀
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHANGELOG = os.path.join(ROOT, "CHANGELOG.md")


def extract(version: str) -> str:
    want = version.lower().lstrip("v")
    with open(CHANGELOG, encoding="utf-8") as f:
        lines = f.read().splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("## ") and ln[3:].strip().lower().lstrip("v").startswith(want):
            start = i + 1
            break
    if start is None:
        raise SystemExit(f"CHANGELOG 中未找到版本 {version}")
    out = []
    for ln in lines[start:]:
        if ln.startswith("## "):
            break
        out.append(ln)
    return "\n".join(out).strip()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("用法: python scripts/release_notes.py <版本>")
    print(extract(sys.argv[1]))
