"""README 模型清单与计数的单一来源:从模型注册表生成。

用法:
    python scripts/sync_models.py            # 原地更新 README
    python scripts/sync_models.py --check    # 仅校验,不同步时退出码 1(CI 用)
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
sys.path.insert(0, ROOT)                     # 未 pip install -e 时也可直接运行

BEGIN, END = "<!-- MODELS:BEGIN -->", "<!-- MODELS:END -->"


def build_section() -> str:
    from open_spss.core.registry import REGISTRY  # noqa: F401
    from open_spss.models import CATEGORIES  # noqa: F401  触发注册

    total = len(REGISTRY.all())
    lines = [f"## 模型清单({total} 个)", "", "| 大类 | 模型 |", "|---|---|"]
    for cat in CATEGORIES:
        names = "、".join(m["name"] for m in cat["models"])
        lines.append(f"| **{cat['name']}** | {names} |")
    return "\n".join(lines)


def sync(text: str) -> str:
    section = build_section()
    if BEGIN in text and END in text:
        pre = text[:text.index(BEGIN) + len(BEGIN)]
        post = text[text.index(END):]
        return pre + "\n" + section + "\n" + post
    # 首次运行:给现有「模型清单」节打上标记
    start = text.index("## 模型清单")
    end = text.index("## 使用指南")
    replaced = text[:start] + BEGIN + "\n" + section + "\n" + END + "\n\n" + text[end:]
    return replaced


def sync_counts(text: str) -> str:
    from open_spss.core.registry import REGISTRY
    from open_spss.models import CATEGORIES  # noqa: F401

    total, cats = len(REGISTRY.all()), len(CATEGORIES)
    text = re.sub(rf"\d+\+? 个分析模型", f"{total} 个分析模型", text)
    text = re.sub(rf"\d+ 大类", f"{cats} 大类", text)
    text = re.sub(rf"含 {total - 100} 个", f"含 {total - 100} 个", text)  # 兼容位,保持幂等
    return text


def main() -> int:
    with open(README, encoding="utf-8") as f:
        text = f.read()
    updated = sync_counts(sync(text))
    if updated == text:
        print("README 已与模型注册表同步")
        return 0
    if "--check" in sys.argv:
        print("README 与模型注册表不同步,请运行 python scripts/sync_models.py", file=sys.stderr)
        return 1
    with open(README, "w", encoding="utf-8", newline="\n") as f:
        f.write(updated)
    print("README 已更新")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
