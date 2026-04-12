from setuptools import setup
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

APP = [ str(HERE / "main.py") ]

OPTIONS = {
    "argv_emulation": False,
    "iconfile": str(ROOT / "assets" / "icon.icns"),
    "plist": {
        "CFBundleName": "Modeling Tools",
        "CFBundleDisplayName": "Modeling Tools",
        "CFBundleIdentifier": "com.alexcreiner.modeling-tools",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "LSMinimumSystemVersion": "12.0",
    },
    "packages": [
        # TODO: This was the minimal set of packages to get it working properly. Scipy, mesa, and networkx were needed
        #       to get the modules loaded during startup. In the future I need to refactor that so that modules can be 
        #       loaded independently of whether or not the packages are installed in the app venv.
        "modeling_tools",
        # NEEDED BY APP
        # "PyQt6",
        # "matplotlib",
        # "numpy",
        "scienceplots",
        # "pyyaml",
        # "platformdirs",
        # NEEDED BY SPECIFIC MODELS
        "scipy",
        "mesa",
        "networkx"
    ],
    "includes": [
        "sip",
    ],
    "resources": [
        str(ROOT / "assets"),
        str(ROOT / "defaults"),
    ],
    # "frameworks": [],
    "matplotlib_backends": ["QtAgg"],
    "arch": "arm64",
}

setup(
    app=APP,
    name="Modeling Tools",
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
