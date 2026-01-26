from PyQt6 import QtWidgets as qw
import yaml
from paths import rpath

from widgets.common import FormSection

class GlobalSettingsTab(qw.QWidget):

    def __init__(self, parent):

        super().__init__(parent)
        layout = qw.QVBoxLayout(self)

        sec = FormSection("Global settings")

        with open(rpath("config.yml"), "r") as f:
            global_settings = yaml.safe_load(f).get("global_settings")
            save_dir = global_settings.get("default_save_dir", "")
        self.edit_default_save_dir = qw.QLineEdit(save_dir)
        self.btn_browse_save_dir = qw.QToolButton()
        self.btn_browse_save_dir.setText("…")

        self.window = self.window()

        save_dir_row = qw.QHBoxLayout()
        save_dir_row.setContentsMargins(0, 0, 0, 0)
        save_dir_row.addWidget(self.edit_default_save_dir, 1)
        save_dir_row.addWidget(self.btn_browse_save_dir, 0)

        sec.form.addRow("Default image save directory:", self._wrap_layout(save_dir_row))

        # TODO: make apply/save actually work this way
        # hint = qw.QLabel(
        #     "Tip: use Apply to test changes without closing.\n"
        #     "Save writes the config file (you implement save_data())."
        # )
        # hint.setWordWrap(True)
        # hint.setStyleSheet("opacity: 0.85;")

        layout.addWidget(sec, 0)
        # layout.addWidget(hint, 0)
        layout.addStretch(1)

        # Browse button placeholder
        self.btn_browse_save_dir.clicked.connect(self._browse_save_dir)

    def _wrap_layout(self, layout: qw.QLayout) -> qw.QWidget:
        w = qw.QWidget()
        w.setLayout(layout)
        return w

    def _browse_save_dir(self) -> None:
        # Optional convenience; you can remove if you don't want file dialogs
        path = qw.QFileDialog.getExistingDirectory(self, "Choose default save directory")
        if path:
            self.edit_default_save_dir.setText(path)
            self.window.status.show("Note: To actually apply the changes, you must first click Apply.", 2000)
