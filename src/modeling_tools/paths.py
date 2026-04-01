# paths.py
from pathlib import Path
from platformdirs import user_config_dir, user_data_dir, user_cache_dir
import sys
import os
from dataclasses import dataclass

APP_NAME = "Modeling-Tools"
APP_AUTHOR = False

# the path of the containing folder
APP_DIR = Path(__file__).resolve().parent

CACHE_DIR = Path(user_cache_dir(APP_NAME, APP_AUTHOR))
USER_APP_DIR = Path.home() / "Documents" / "Modeling-Tools"
MODELS_DIR = USER_APP_DIR / "models"
DATA_DIR = Path(user_data_dir(APP_NAME, APP_AUTHOR))
CONFIG_DIR = Path(user_config_dir(APP_NAME, APP_AUTHOR, roaming= True))

CONFIG_FILE = CONFIG_DIR / "config.yml"
LOG_DIR = USER_APP_DIR / "logs"

def ensure_dirs():
    """ create any missing directory """
    for d in [CONFIG_DIR, DATA_DIR, CACHE_DIR, LOG_DIR, MODELS_DIR]:
        d.mkdir(parents= True, exist_ok = True)

def app_path(*parts):
    return APP_DIR.joinpath(*parts)

def bundle_root() -> Path:
    """
    Root directory for bundled resources.
    Works in dev, onedir, and onefile.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent

def assets_path(*parts: str) -> Path:
    return bundle_root().parent.parent.joinpath("assets", *parts)

def defaults_path(*parts: str) -> Path:
    # defaults lives one level above the package if you keep it at repo top level
    return bundle_root().parent.parent.joinpath("defaults", *parts)

def rpath(*parts: str) -> Path:
    """ very simple helper function to create paths throughout the app """
    return bundle_root().joinpath(*parts)

if __name__ == "__main__":
    print(f"{CONFIG_DIR=}")
    print(f"{DATA_DIR=}")
    print(f"{CACHE_DIR=}")
    print(f"{LOG_DIR=}")
    print(f"{MODELS_DIR=}")
    print(f"{APP_DIR}")

