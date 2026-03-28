from PyQt6 import QtWidgets as qw
from PyQt6 import QtCore as qc
import re
import os
import tempfile
from pathlib import Path
import yaml
from .HelpFormLayout import HelpFormLayout

class FormSection(qw.QGroupBox):
    """A tidy groupbox with a built-in form layout."""
    def __init__(self, title: str):
        super().__init__(title)
        self.form = HelpFormLayout(self)
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

