import sys
from PyQt6 import (
    QtCore as qc,
    QtWidgets as qw,
    QtGui as qg
)
from matplotlib import pyplot as plt
from widgets.EntryBlock import EntryBlock
from widgets.DropdownChoices import DropdownChoices
from widgets.MatrixEntry import MatrixEntry
from dataclasses import asdict
import math
# import scienceplots
# plt.style.use(["grid", "notebook"])

class ControlPanel(qw.QWidget):
    paramChanged = qc.pyqtSignal(str, object)
    plotChoiceChanged = qc.pyqtSignal(int)
    checkStateChanged = qc.pyqtSignal()

    def __init__(self, params, dropdown_choices, dropdown_tooltips, panel_data, plotting_data):
        # print(f"Loaded params: {asdict(params)}")
        super().__init__()
        layout = qw.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.dropdown_widget = DropdownChoices()
        self.dropdown_widget.addItems(dropdown_choices)
        self.dropdown_widget.infoBoxHovered.connect(self.get_tooltip)
        self.dropdown_widget.currentIndexChanged.connect(self.new_selection)
        self.dropdown_widget.checkStateChanged.connect(self.checkbox_change)
        self.dropdown_widget.setSizePolicy(qw.QSizePolicy.Policy.Preferred, qw.QSizePolicy.Policy.Maximum)

        self.dropdown_tooltips = dropdown_tooltips

        wrap = qw.QWidget()
        wlay = qw.QVBoxLayout(wrap); wlay.setContentsMargins(0,0,0,0); wlay.addWidget(self.dropdown_widget)
        wrap.setSizePolicy(qw.QSizePolicy.Policy.Preferred, qw.QSizePolicy.Policy.Maximum)
        layout.addWidget(wrap, alignment= qc.Qt.AlignmentFlag.AlignTop, stretch= 0)

        for dropdown_choice in plotting_data:
            choice_dict = plotting_data[dropdown_choice]
            plots = choice_dict["plots"]
            for plot in plots:
                plot_dict = plots[plot]
                if "checkbox_name" in plot_dict:
                    self.dropdown_widget.add_checkbox(choice_dict["name"], plot_dict["checkbox_name"], plot_dict["toggled"])

        matrix_widget1 = qw.QWidget()
        matrix_widget1.setContentsMargins(0,0,0,0)
        matrix_layout1 = qw.QHBoxLayout(matrix_widget1)
        matrix_layout1.setContentsMargins(0,0,0,0)
        matrix_layout1.setSpacing(0)

        self.entry_blocks = {}
        self.row_wrappers = []

        for row in panel_data:
            wrap = qw.QWidget()
            wlay = qw.QHBoxLayout(wrap)
            wlay.setContentsMargins(0,0,0,0)
            wlay.setSpacing(0)
            wrap.setSizePolicy(qw.QSizePolicy.Policy.Preferred, qw.QSizePolicy.Policy.Maximum)

            self.row_wrappers.append(wrap)

            for entry in panel_data[row]:
                info = panel_data[row][entry]
                param_name, label, tooltip = info["param_name"], info["label"], info["tooltip"]
                init_val = getattr(params, param_name)
                # print(getattr(params, param_name))

                if info["type"] == "scalar":
                    scalar_range, scalar_type = tuple(info["range"]), info["scalar_type"]
                    widget = EntryBlock(param_name, label, scalar_range, init_val, tooltip, scalar_type)
                    self.entry_blocks[param_name] = {"widget": widget, "is_matrix": False}
                    widget.setSizePolicy(qw.QSizePolicy.Policy.Preferred, qw.QSizePolicy.Policy.Maximum)
                    wlay.addWidget(widget, alignment= qc.Qt.AlignmentFlag.AlignTop, stretch=0)
                    widget.valueChanged.connect(self.update_plot)

                elif info["type"] == "matrix":
                    dim = tuple(info["dim"])
                    widget = MatrixEntry(param_name, label, dim, init_val, tooltip)
                    widget.textChanged.connect(self.update_plot)
                    wlay.addWidget(widget, alignment= qc.Qt.AlignmentFlag.AlignTop, stretch=0)
                    self.entry_blocks[param_name] = {"widget": widget, "is_matrix": True}
                    
                elif info["type"] == "vector":
                    dim1 = info["dim"]
                    dim = (dim1, 1)
                    widget = MatrixEntry(param_name, label, dim, init_val.reshape(-1,1), tooltip)
                    widget.textChanged.connect(self.update_plot)
                    wlay.addWidget(widget, alignment= qc.Qt.AlignmentFlag.AlignTop, stretch=0)
                    self.entry_blocks[param_name] = {"widget": widget, "is_matrix": True}
                
                else:
                    print(f"Unrecognized type: {info["type"]}. Options for type are scalar, vector, and matrix.")

        for wrapper in self.row_wrappers:
            layout.addWidget(wrapper, alignment= qc.Qt.AlignmentFlag.AlignTop, stretch= 0)

    def new_selection(self, index):
        self.plotChoiceChanged.emit(index)

    def checkbox_change(self):
        self.checkStateChanged.emit()

    def get_data(self, index):
        data = {}
        for widget in self.entry_blocks:
            data[widget.name] = widget.get()
        return data

    def update_plot(self, name, new_val):
        self.paramChanged.emit(name, new_val)

    def get_tooltip(self):
        if self.dropdown_tooltips[self.dropdown_widget.dropdown_choices.currentText()] != "":
            self.dropdown_widget.setToolTip(self.dropdown_tooltips[self.dropdown_widget.dropdown_choices.currentText()])
        else:
            self.dropdown_widget.setToolTip("No notes")

    def load_new_params(self, params):
        params_dict = asdict(params)
        for param in params_dict:
            if param in self.entry_blocks:
                if self.entry_blocks[param]["is_matrix"]:
                    self.entry_blocks[param]["widget"].change_values(params_dict[param])
                else:
                    self.entry_blocks[param]["widget"].entry.setText(str(params_dict[param]))
