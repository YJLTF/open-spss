"""OpenSPSS 命令行入口:python -m open_spss 或 open-spss 命令。"""
from __future__ import annotations

import argparse
import threading
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="open-spss",
        description="OpenSPSS —— 开源统计分析平台(100 个分析模型)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址(默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8420, help="端口(默认 8420)")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    args = parser.parse_args()

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"\n  OpenSPSS {url}  (Ctrl+C 退出)\n")
    uvicorn.run("open_spss.app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
