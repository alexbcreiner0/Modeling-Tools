from PyQt6 import QtWidgets as qw
from PyQt6 import QtCore as qc
import re
import os
import tempfile
from pathlib import Path
import yaml

class FormSection(qw.QGroupBox):
    """A tidy groupbox with a built-in form layout."""
    def __init__(self, title: str):
        super().__init__(title)
        self.form = qw.QFormLayout(self)
        self.form.setFieldGrowthPolicy(qw.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.form.setLabelAlignment(qc.Qt.AlignmentFlag.AlignRight)

def make_shortname(display_name: str) -> str:
    s = display_name.lower()
    # replace spaces and punctuation with underscores
    s = re.sub(r"[^a-z0-9]+", "_", s)
    # collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s

def atomic_write(path: Path, new_text: str | dict) -> None:
    bak = path.with_suffix(".bak")

    # backup current on-disk bytes/text
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            old = f.read()
        with open(bak, "w", encoding="utf-8") as f:
            f.write(old)

    # atomic replace
    d = os.path.dirname(path) or "."
    base = os.path.basename(path)
    fd, tmp = tempfile.mkstemp(prefix=base + ".", suffix=".tmp", dir=d, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            if isinstance(new_text, dict):
                yaml.safe_dump(new_text, f, sort_keys= False, allow_unicode= True)
            else:
                f.write(new_text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    if os.path.exists(bak):
        os.remove(bak)
