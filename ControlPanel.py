from PyQt6 import (
    QtCore as qc,
    QtWidgets as qw,
    QtGui as qg
)
from matplotlib import pyplot as plt
from EntryBlock import EntryBlock
from DropdownChoices import DropdownChoices
from MatrixEntry import MatrixEntry
import math
# import scienceplots
# plt.style.use(["grid", "notebook"])

class ControlPanel(qw.QWidget):
    paramChanged = qc.pyqtSignal(str, object)
    plotChoiceChanged = qc.pyqtSignal(int)
    checkStateChanged = qc.pyqtSignal()

    def __init__(self, entry_boxes, params, dropdown_choices, dropdown_tooltips):
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

        # thing keeps getting too fucking big so wrapping it in this
        wrap = qw.QWidget()
        wlay = qw.QVBoxLayout(wrap); wlay.setContentsMargins(0,0,0,0); wlay.addWidget(self.dropdown_widget)
        wrap.setSizePolicy(qw.QSizePolicy.Policy.Preferred, qw.QSizePolicy.Policy.Maximum)
        layout.addWidget(wrap, alignment= qc.Qt.AlignmentFlag.AlignTop, stretch= 0)

        # layout.addWidget(wrap, alignment= qc.Qt.AlignmentFlag.AlignTop, stretch=0)
        self.dropdown_widget.add_checkbox("Prices vs Values", "Toggle prices", True)
        self.dropdown_widget.add_checkbox("Prices vs Values", "Toggle values", True)
        self.dropdown_widget.add_checkbox("Prices vs Values", "Toggle equilibrium prices")
        self.dropdown_widget.add_checkbox("Prices vs Equilibrium Prices", "Toggle prices", True)
        self.dropdown_widget.add_checkbox("Prices vs Equilibrium Prices", "Toggle values")
        self.dropdown_widget.add_checkbox("Prices vs Equilibrium Prices", "Toggle equilibrium prices", True)

        self.dropdown_widget.add_checkbox("Output and Supply", "Toggle output", True)
        self.dropdown_widget.add_checkbox("Output and Supply", "Toggle supply", True)

        self.dropdown_widget.add_checkbox("Wages and Employment", "Toggle wage rate", True)
        self.dropdown_widget.add_checkbox("Wages and Employment", "Toggle employment", True)
        self.dropdown_widget.add_checkbox("Wages and Employment", "Toggle reserve army size", True)
        self.dropdown_widget.add_checkbox("Wages and Employment", "Toggle labor demand")
        self.dropdown_widget.add_checkbox("Wages and Employment", "Toggle rate of exploitation")

        self.dropdown_widget.add_checkbox("Money Rates of Profit", "Toggle value RoP")
        self.dropdown_widget.add_checkbox("Money Rates of Profit", "Toggle sectoral RoPs", True)
        self.dropdown_widget.add_checkbox("Money Rates of Profit", "Toggle rate of exploitation")
        self.dropdown_widget.add_checkbox("Money Rates of Profit", "Toggle interest rate")
        self.dropdown_widget.add_checkbox("Money Rates of Profit", "Toggle equilibrium RoP", True)
        self.dropdown_widget.add_checkbox("Money Rates of Profit", "Toggle money rate of accumulation")
        # self.dropdown_widget.add_checkbox("Money Rates of Profit", "Toggle value rate of accumulation")

        matrix_widget1 = qw.QWidget()
        matrix_widget1.setContentsMargins(0,0,0,0)
        matrix_layout1 = qw.QHBoxLayout(matrix_widget1)
        matrix_layout1.setContentsMargins(0,0,0,0)
        matrix_layout1.setSpacing(0)

        self.A_entry = MatrixEntry("A", r"$A= $", (3,3), params.A, tooltip= "The requirements matrix. Rows and columns (for this and the vectors next to this) represent \n the first, second and third commodities circulating (for sake of example corn, iron, and sugar). \n The (i,j) entry of this matrix represents the number of units of commodity i needed to produce a unit of commodity j. \n So the bottom right entry is the amount of 'sugar' needed to make 'sugar'.")
        self.A_entry.textChanged.connect(self.update_plot)
        matrix_layout1.addWidget(self.A_entry)

        self.l_entry = MatrixEntry("l", r"$\mathbf{l}= $", (3,1), params.l.reshape(-1,1), tooltip= "The living labor vector. The ith entry of this is the number of hours of labor required to produce a unit of the ith commodity.")
        self.l_entry.textChanged.connect(self.update_plot)
        matrix_layout1.addWidget(self.l_entry)

        self.p_entry = MatrixEntry("p0", r"$\mathbf{p}(0)= $", (3,1), params.p0.reshape(-1,1), tooltip= "The initial unit price vector. Entry i is the price of a unit of i at the start of the simulation.")
        self.p_entry.textChanged.connect(self.update_plot)
        matrix_layout1.addWidget(self.l_entry)

        matrix_widget2 = qw.QWidget()
        matrix_layout2 = qw.QHBoxLayout(matrix_widget2)
        matrix_layout2.setContentsMargins(0,0,0,0)
        matrix_layout2.setSpacing(0)

        self.b_entry = MatrixEntry("b_bar", r"$\mathbf{b}= $", (3,1), params.b_bar.reshape(-1,1), tooltip= "Workers will spend a proportion of their savings each period on a means of subsistence bundle amounting to a scalar multiple of this vector. By default, we have workers consuming corn and sugar, but mostly corn.")
        self.b_entry.textChanged.connect(self.update_plot)
        matrix_layout2.addWidget(self.b_entry)

        self.c_entry = MatrixEntry("c_bar", r"$\mathbf{c}= $", (3,1), params.c_bar.reshape(-1,1), tooltip= "Workers will spend a proportion of their savings each period on a consumption bundle amounting to a scalar multiple of this vector. By default, we have capitalists consuming corn and sugar, but mostly sugar.")
        self.c_entry.textChanged.connect(self.update_plot)
        matrix_layout2.addWidget(self.c_entry)

        self.q_entry = MatrixEntry("q0", r"$\mathbf{q}(0)= $", (3,1), params.q0.reshape(-1,1), tooltip= "Initial output vector. This one is much less consequential than the initial supply vector found next to this.")
        self.q_entry.textChanged.connect(self.update_plot)
        matrix_layout2.addWidget(self.q_entry)

        self.s_entry = MatrixEntry("s0", r"$\mathbf{s}(0)= $", (3,1), params.s0.reshape(-1,1), tooltip= "Initial supply vector. Default example has an initial shortage of 'iron'. Try increasing this amount (the second entry) and see what changes.")
        self.s_entry.textChanged.connect(self.update_plot)
        matrix_layout2.addWidget(self.s_entry)

        layout.addWidget(matrix_widget1)
        layout.addWidget(matrix_widget2)
        matrix_widget1.setSizePolicy(qw.QSizePolicy.Policy.Preferred, qw.QSizePolicy.Policy.Maximum)
        matrix_widget2.setSizePolicy(qw.QSizePolicy.Policy.Preferred, qw.QSizePolicy.Policy.Maximum)

        self.entry_blocks = []
        wrap = qw.QWidget()
        wrappers = [qw.QWidget() for i in range(math.ceil(len(entry_boxes) / 2))]
        wlays = [qw.QHBoxLayout(wrapper) for wrapper in wrappers]
        for wlay in wlays: 
            wlay.setContentsMargins(0,0,0,0)
            wlay.setSpacing(0)
        # wlay = qw.QHBoxLayout(wrap); wlay.setContentsMargins(0,0,0,0)
        for wrapper in wrappers:
            wrapper.setSizePolicy(qw.QSizePolicy.Policy.Preferred, qw.QSizePolicy.Policy.Maximum)
        i = 0
        j = 0

        for name, entry in entry_boxes.items():
            widget = EntryBlock(name, entry["label"], entry["range"], entry["init_val"], entry["tooltip"], entry["num_type"])
            self.entry_blocks.append(widget)
            widget.setSizePolicy(qw.QSizePolicy.Policy.Preferred, qw.QSizePolicy.Policy.Maximum)
            wlays[j].addWidget(widget, alignment= qc.Qt.AlignmentFlag.AlignTop, stretch=0)
            widget.valueChanged.connect(self.update_plot)
            i += 1
            if i % 2 == 0: j += 1

        for wrapper in wrappers:
            layout.addWidget(wrapper, alignment= qc.Qt.AlignmentFlag.AlignTop, stretch= 0)

        matrix_widget3 = qw.QWidget()
        matrix_layout3 = qw.QHBoxLayout(matrix_widget3)
        matrix_layout3.setContentsMargins(0,0,0,0)
        matrix_layout3.setSpacing(0)

        self.eta_entry = MatrixEntry("eta", r"$\mathbf{\eta}= $", (3,1), params.eta.reshape(-1,1))
        self.eta_entry.textChanged.connect(self.update_plot)
        matrix_layout3.addWidget(self.eta_entry)

        self.kappa_entry = MatrixEntry("kappa", r"$\mathbf{\kappa}= $", (3,1), params.kappa.reshape(-1,1))
        self.c_entry.textChanged.connect(self.update_plot)
        matrix_layout3.addWidget(self.kappa_entry)

        layout.addWidget(matrix_widget3)
        layout.insertStretch(-1,1)

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

