# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

from PyInstaller.building.build_main import Analysis, PYZ, EXE
from PyInstaller.utils.hooks import collect_submodules


APP_NAME = "CrossDualDynamicRoPRelease"

# IMPORTANT: run pyinstaller from the repo root.
ROOT = Path(os.getcwd()).resolve()
ENTRYPOINT = str(ROOT / "main.py")

# Exclude junk / dev dirs from being bundled as data
EXCLUDE_DIRS = {
    "dist", "build", ".git", ".github",
    ".venv", "venv", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
}
EXCLUDE_FILES = {".DS_Store"}

def should_skip_path(p: Path) -> bool:
    parts = set(p.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if p.name in EXCLUDE_FILES:
        return True
    if p.suffix in {".pyc", ".pyo"}:
        return True
    return False

def add_tree_as_datas(root_dir: Path, prefix: str = ".") -> list[tuple[str, str]]:
    """
    Return datas as a list of (src_file, dest_dir) pairs.
    dest_dir is a directory relative to the bundle root.
    """
    out: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dp = Path(dirpath)

        # prune excluded directories (prevents descending)
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and d != "__pycache__"]

        if should_skip_path(dp):
            continue

        for fn in filenames:
            src = dp / fn
            if should_skip_path(src):
                continue

            rel_parent = src.parent.relative_to(ROOT)  # relative directory inside repo
            # target directory inside bundle:
            dest_dir = str((Path(prefix) / rel_parent).as_posix())
            if dest_dir == "":
                dest_dir = "."

            out.append((str(src), dest_dir))
    return out

datas = []
# Bundle your whole repo (minus excludes) so all yaml/other folders exist inside the bundle
datas += add_tree_as_datas(ROOT, prefix=".")

# --- Bundle scienceplots non-python assets (styles etc) WITHOUT collect_data_files ---
# This avoids 2-tuple/3-tuple incompatibilities across PyInstaller versions.
try:
    import scienceplots  # available in build env
    sp_root = Path(scienceplots.__file__).resolve().parent
    # Bundle everything under scienceplots/ except .py and caches (safe but slightly bigger)
    for dirpath, dirnames, filenames in os.walk(sp_root):
        dp = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            src = dp / fn
            if src.suffix in {".py", ".pyc", ".pyo"}:
                continue
            rel_parent = src.parent.relative_to(sp_root)
            dest_dir = str((Path("scienceplots") / rel_parent).as_posix())
            datas.append((str(src), dest_dir))
except Exception:
    # If scienceplots isn't installed in the build env, you’ll get an import error.
    # In that case install it (pip install scienceplots) before building.
    pass

hiddenimports = []
hiddenimports += collect_submodules("scienceplots")

a = Analysis(
    [ENTRYPOINT],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,           # NOTE: list of 2-tuples only
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
    console=True,  # set False if you want GUI-only (no terminal)
)
