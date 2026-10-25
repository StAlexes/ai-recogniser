# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for the AI Recogniser desktop shell.

One-folder build: ``dist/AI Recogniser/AI Recogniser.exe``. The shell starts
the core by re-running the same exe with ``--core-worker``; a one-folder build
avoids extracting the runtime twice and keeps startup fast.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).resolve().parent

datas = [
    (str(ROOT / "ui" / "dist"), "ui/dist"),
    (str(ROOT / "shell" / "assets"), "assets"),
    (str(ROOT / "models" / "manifest.json"), "models"),
]
binaries = []
hiddenimports = [
    "clr",
    "webview.platforms.edgechromium",
    "pystray._win32",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "anyio._backends._asyncio",
    "airec.api.app",
    "airec.api.rag",
    "airec.asr.audio",
    "airec.asr.manager",
    "airec.asr.service",
    "airec.asr.vad",
    "airec.rag.client",
    "multipart",
    "python_multipart",
]

# pywebview ships WebView2 .NET assemblies and helper DLLs; pythonnet/clr_loader
# load them dynamically, so PyInstaller cannot discover them by static imports.
# The ASR stack is also dynamic: onnx-asr imports onnxruntime at runtime, and
# imageio-ffmpeg contains the ffmpeg executable as package data.
for package in (
    "webview",
    "clr_loader",
    "pythonnet",
    "pystray",
    "onnx_asr",
    "onnxruntime",
    "imageio_ffmpeg",
    "multipart",
    "python_multipart",
):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("onnx_asr")

a = Analysis(
    [str(ROOT / "shell" / "main.py")],
    pathex=[str(ROOT / "app"), str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "pytest", "setuptools", "pip"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AI Recogniser",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "shell" / "assets" / "icon.ico"),
    version=str(ROOT / "packaging" / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AI Recogniser",
)
