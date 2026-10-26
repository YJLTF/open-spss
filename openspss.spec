# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置:构建单文件 exe(含前端静态资源)。
用法:.venv/Scripts/python.exe -m PyInstaller openspss.spec --noconfirm
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

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

# 过渡图:Bootloader 在自解压期间立即显示,launcher 在端口就绪后调用 pyi_splash.close()
splash = Splash(
    "open_spss/splash.png",
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,          # 不绘制进度文本,图片自带提示
    always_on_top=True,
    scale=False,
    width=520,
    height=320,
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
    icon=None,
)
