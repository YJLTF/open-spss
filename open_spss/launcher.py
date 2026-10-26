r"""PyInstaller 打包入口(无控制台窗口):

点击 exe → Bootloader 立即显示品牌过渡图(Splash,覆盖自解压等待期)
→ Python 启动后通过 IPC 实时刷新过渡页文本:按真实进度依次报告
  加载引擎 → 启动服务 → 打开浏览器,并驱动字符进度条
→ 端口选定(默认 8420,被占用时自动后移;若占用者本身就是 OpenSPSS,
  则直接打开浏览器复用该实例)→ 健康接口就绪后关闭过渡页并打开浏览器。

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

_BAR_SLOTS = 20


class SplashReporter:
    """与过渡页(Tcl)通信:协议只支持 update_text 一条命令,断开连接即关闭过渡页。

    端口经环境变量 _PYI_SPLASH_IPC 传入;未打包运行时该变量不存在,全部为空操作。
    文案一律中文标点;消息内不得出现半角括号(Tcl 侧按首尾括号截取参数)。
    """

    def __init__(self) -> None:
        port = os.environ.pop("_PYI_SPLASH_IPC", "")
        self.sock = None
        if port and port != "0":
            try:
                self.sock = socket.create_connection(("127.0.0.1", int(port)), timeout=1.0)
                self.sock.settimeout(None)
            except OSError:
                self.sock = None

    def stage(self, label: str, pct: int) -> None:
        if not self.sock:
            return
        pct = max(0, min(100, int(pct)))
        filled = round(_BAR_SLOTS * pct / 100)
        bar = "█" * filled + "░" * (_BAR_SLOTS - filled)
        self._send(f"update_text({label}  {bar}  {pct}%)")

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def _send(self, cmd: str) -> None:
        try:
            self.sock.sendall(cmd.encode("utf-8") + b"\x0d")
        except OSError:
            self.sock = None


def _log_dir() -> str:
    return os.environ.get("TEMP") or os.path.dirname(os.path.abspath(sys.executable))


def _log(msg: str) -> None:
    try:
        with open(os.path.join(_log_dir(), "OpenSPSS.log"), "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except OSError:
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
    splash = None
    try:
        # 窗口模式(无控制台)下 stdout/stderr 为 None,重定向到日志避免任何库写日志时报错
        if sys.stdout is None or sys.stderr is None:
            try:
                logf = open(os.path.join(_log_dir(), "OpenSPSS.log"), "a", buffering=1)
                sys.stdout = sys.stdout or logf
                sys.stderr = sys.stderr or logf
            except OSError:
                pass

        # 过渡页通信就绪;自解压期间显示 spec 中的 text_default,此后由各阶段实时刷新
        splash = SplashReporter()
        splash.stage("正在启动 OpenSPSS", 8)

        import uvicorn

        splash.stage("正在加载统计分析引擎", 32)
        from open_spss.app import app  # 首次导入加载 pandas/sklearn/statsmodels,耗时大头

        splash.stage("正在启动本地分析服务", 64)

        port = int(os.environ.get("OPENSPSS_PORT", "8420"))
        # 端口已被占用:若是 OpenSPSS 实例则复用并打开浏览器;否则向后找一个空闲端口
        if not _port_free(port):
            if _our_health(port):
                _log(f"reuse existing instance at {port}")
                splash.stage("检测到已启动的实例，正在打开浏览器", 100)
                splash.close()
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

        splash.stage("等待服务就绪", 86)
        if _wait_our_health(port, timeout=90):
            splash.stage("正在打开浏览器", 100)
            splash.close()
            webbrowser.open(url)
            _log(f"served at {url}")
        else:
            splash.close()
            _log(f"ERROR: our health check never passed for {url}")
        server.join()
    except Exception:  # noqa: BLE001
        _log("FATAL:\n" + traceback.format_exc())
        if splash:
            splash.close()
        raise


if __name__ == "__main__":
    main()
