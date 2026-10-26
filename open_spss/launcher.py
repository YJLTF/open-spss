r"""PyInstaller 打包入口(无控制台窗口):

点击 exe → Bootloader 立即显示品牌过渡图(Splash,覆盖自解压等待期)
→ Python 启动后选定可用端口(默认 8420,被占用时自动后移;
  若占用者本身就是 OpenSPSS,则直接打开浏览器复用该实例)
→ 后台线程拉起 uvicorn → 健康接口就绪的瞬间关闭 Splash 并打开浏览器。

窗口模式下没有控制台,异常写入 %TEMP%/OpenSPSS.log 便于排查。
"""
import json
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
import urllib.request


def _log_dir() -> str:
    return os.environ.get("TEMP") or os.path.dirname(os.path.abspath(sys.executable))


def _log(msg: str) -> None:
    try:
        with open(os.path.join(_log_dir(), "OpenSPSS.log"), "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except OSError:
        pass


def _close_splash() -> None:
    try:
        import pyi_splash  # 仅在以 Splash 打包时存在
        pyi_splash.close()
    except Exception:  # noqa: BLE001
        pass


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) != 0


def _our_health(port: int, timeout: float = 1.0) -> bool:
    """端口上是否已是 OpenSPSS 服务(健康接口返回含 models 字段)。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=timeout) as r:
            return "models" in json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return False


def _wait_our_health(port: int, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _our_health(port, timeout=1.0):
            return True
        time.sleep(0.2)
    return False


def main() -> None:
    try:
        # 窗口模式(无控制台)下 stdout/stderr 为 None,重定向到日志避免任何库写日志时报错
        if sys.stdout is None or sys.stderr is None:
            try:
                logf = open(os.path.join(_log_dir(), "OpenSPSS.log"), "a", buffering=1)
                sys.stdout = sys.stdout or logf
                sys.stderr = sys.stderr or logf
            except OSError:
                pass
        import uvicorn

        from open_spss.app import app  # 直接引用,保证 PyInstaller 模块图完整

        port = int(os.environ.get("OPENSPSS_PORT", "8420"))
        # 端口已被占用:若是 OpenSPSS 实例则复用并打开浏览器;否则向后找一个空闲端口
        if not _port_free(port):
            if _our_health(port):
                _log(f"reuse existing instance at {port}")
                _close_splash()
                webbrowser.open(f"http://127.0.0.1:{port}")
                return
            for cand in range(port + 1, port + 21):
                if _port_free(cand):
                    _log(f"port {port} busy(foreign), fallback to {cand}")
                    port = cand
                    break
        host = "127.0.0.1"
        url = f"http://{host}:{port}"

        server = threading.Thread(
            target=lambda: uvicorn.run(app, host=host, port=port, log_level="warning"),
            daemon=True,
        )
        server.start()

        if _wait_our_health(port, timeout=90):
            _close_splash()
            webbrowser.open(url)
            _log(f"served at {url}")
        else:
            _close_splash()
            _log(f"ERROR: our health check never passed for {url}")
        server.join()
    except Exception:  # noqa: BLE001
        _log("FATAL:\n" + traceback.format_exc())
        _close_splash()
        raise


if __name__ == "__main__":
    main()
