# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置:构建单文件 exe(含前端静态资源)。
用法:.venv/Scripts/python.exe -m PyInstaller openspss.spec --noconfirm
"""
import os
import subprocess
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# 构建前重新生成过渡图:模型数与版本号取自注册表/包元数据,新增模型或升版本无需改图
subprocess.run(
    [sys.executable, os.path.join("scripts", "make_splash.py")],
    cwd=SPECPATH, check=True,
)

hiddenimports = (
    collect_submodules("uvicorn")
    + collect_submodules("statsmodels")
    + [
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.loops.asyncio",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.protocols.websockets.wsproto_impl",
        "uvicorn.lifespan.on",
        "uvicorn.lifespan.asyncio",
        "sklearn.utils._weight_vector",
        "sklearn.utils._cyblas",
    ]
)

a = Analysis(
    ["open_spss/launcher.py"],
    pathex=[],
    binaries=[],
    datas=[("open_spss/web", "open_spss/web")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PIL", "PyQt5", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)

# 过渡图:Bootloader 在自解压期间立即显示;launcher 通过 update_text IPC
# 按真实启动阶段刷新文本(含字符进度条),断开 IPC 连接即关闭过渡页。
splash = Splash(
    "open_spss/splash.png",
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(50, 246),                    # 动态文本行左下角:自解压期默认文案,之后是阶段+进度条
    text_size=10,
    # 注意:text_font 不能用含空格的字族名——模板按裸词拼接 Tcl,-family 会拆成多参数致脚本中断;
    # 缺省用 TkDefaultFont(中文 Windows 即微软雅黑),CJK 与 █░ 字符都能渲染
    text_color="#DCE4FF",
    text_default="正在准备运行环境……",
    always_on_top=True,
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    splash,               # Splash 归档目标(位置参数,与 pyi-makespec 生成的模板一致)
    splash.binaries,      # 过渡页所需的 Tcl/Tk 运行时(缺失则过渡页静默不显示)
    [],
    name="OpenSPSS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon="open_spss/logo.ico",   # 资源管理器/任务栏图标(scripts/make_logo.py 生成)
)
