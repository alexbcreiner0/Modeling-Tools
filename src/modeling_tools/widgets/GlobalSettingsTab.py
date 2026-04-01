from PyQt6 import QtWidgets as qw
import yaml
from .FilePicker import FilePicker
from pathlib import Path

from .common import FormSection

class GlobalSettingsTab(qw.QWidget):

    def __init__(self, env, parent):

        super().__init__(parent)
        self.env = env
        layout = qw.QVBoxLayout(self)

        sec = FormSection("Global settings")

        with open(self.env.config_dir / "config.yml", "r") as f:
            global_settings = yaml.safe_load(f).get("global_settings")
            image_save_dir = global_settings.get("default_save_dir", str(Path.home()))
            save_name = global_settings.get("default_save_name", "figure")
            run_on_startup = global_settings.get("run_on_startup", True)
            autosave_axis_settings = global_settings.get("autosave_axis_settings", False)
            user_model_dir = global_settings.get("user_models_dir", str(env.models_dir))
            user_logs_dir = global_settings.get("user_logs_dir", str(env.log_dir))

        # self.edit_default_save_dir = qw.QLineEdit(save_dir)
        # self.btn_browse_save_dir = qw.QToolButton()
        # self.btn_browse_save_dir.setText("…")

        self.edit_default_save_dir = FilePicker()
        self.edit_models_dir = FilePicker()
        self.edit_logs_dir = FilePicker()
        self.edit_default_save_dir.setText(image_save_dir)
        self.edit_models_dir.setText(user_model_dir)
        self.edit_logs_dir.setText(user_logs_dir)

        self.window = self.window()

        # save_dir_row = qw.QHBoxLayout()
        # save_dir_row.setContentsMargins(0, 0, 0, 0)
        # save_dir_row.addWidget(self.edit_default_save_dir, 1)
        # save_dir_row.addWidget(self.btn_browse_save_dir, 0)

        self.save_name = qw.QLineEdit(save_name)
        self.run_on_startup = qw.QCheckBox("Auto-run simulation on startup")
        self.autosave_axis_settings = qw.QCheckBox("Auto-save axis settings")
        self.checkbox_row = qw.QWidget()
        self.run_on_startup.setChecked(run_on_startup)
        self.autosave_axis_settings.setChecked(autosave_axis_settings)
        self.checkbox_row_lay = qw.QHBoxLayout(self.checkbox_row)
        self.checkbox_row_lay.addWidget(self.run_on_startup)
        self.checkbox_row_lay.addWidget(self.autosave_axis_settings)

        sec.form.addRow("Default image save directory:", self.edit_default_save_dir)
        sec.form.addRow("User models directory:", self.edit_models_dir)
        sec.form.addRow("Logging directory:", self.edit_logs_dir)
        help_text = "A template string for your save name to default to. Examples: \n \
                    'my_pic' will result in the save name defaulting to my_pic.png. \n \
                    'my_pic {a} {b}' will attempt to replace {a} and {b} with the values of the parameter named a and b in your model. \n \
                    'my_pic {a=}' will attempt to replace {a=} with the string 'a=<value of a>. So same as above except it titles the parameter with its name. \n \
                    'If a is not the name of a parameter in either of the above cases, then {a} will just be replaced with a in the name."
        sec.form.addRow("Default image save name:", self.save_name, help_text= help_text)
        sec.form.addRow('', self.checkbox_row)

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
        # self.btn_browse_save_dir.clicked.connect(self._browse_save_dir)

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

    def get_settings_for_config(self):
        settings = {
            "save_name": self.save_name.text(),
            "run_on_startup": self.run_on_startup.isChecked(),
            "autosave_axis_settings": self.autosave_axis_settings.isChecked(),
        }
        if self.edit_models_dir.text() != self.env.models_dir and self.edit_models_dir.text():
            settings["user_models_dir"] = self.edit_models_dir.text()
        if self.edit_logs_dir.text() != self.env.log_dir and self.edit_logs_dir.text():
            settings["user_logs_dir"] = self.edit_logs_dir.text()
        if self.edit_default_save_dir.text() != str(Path.home()) and self.edit_default_save_dir.text():
            settings["default_save_dir"] = self.edit_default_save_dir.text()

        return settings
