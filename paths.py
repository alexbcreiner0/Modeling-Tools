# paths.py
from pathlib import Path
import sys

def bundle_root() -> Path:
    """
    Root directory for bundled resources.
    Works in dev, onedir, and onefile.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent

def rpath(*parts: str) -> Path:
    return bundle_root().joinpath(*parts)
