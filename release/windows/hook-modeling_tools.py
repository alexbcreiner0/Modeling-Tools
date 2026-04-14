# hook-modeling_tools.py
from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files(
    "modeling_tools",
    include_py_files=True,
    subdir="defaults/models"
)