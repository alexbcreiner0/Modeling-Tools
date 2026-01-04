# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.building.datastruct import Tree
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(__file__).resolve().parent

APP_NAME = "CrossDualDynamicRoPRelease"
ENTRYPOINT = str(ROOT / "main.py")

# Exclude big / irrelevant stuff from being bundled
EXCLUDES = [
    "dist",
    "build",
    ".git",
    ".github",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".DS_Store",
    "*.pyc",
    "*.pyo",
]

datas = []

# Bundle *everything in your repo root* (config.yml + all folders/files),
# so long as it’s not excluded above.
datas.append(
    Tree(
        str(ROOT),
        prefix=".",
        excludes=EXCLUDES,
    )
)

# Package-data needed by scienceplots (your earlier error)
datas += collect_data_files("scienceplots")

hiddenimports = []
hiddenimports += collect_submodules("scienceplots")

a = Analysis(
    [ENTRYPOINT],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=APP_NAME,
    debug=False,
    strip=False,
    upx=True,
    console=True,   # set False for GUI-only (no terminal)
)
