from PyQt6 import (
    QtCore as qc,
    QtWidgets as qw,
)
from functools import partial
from numpy import ndarray, zeros
from LatexLabel import LatexLabel

class MatrixEntry(qw.QWidget):
    textChanged = qc.pyqtSignal(str, ndarray)

    def __init__(self, name, label, dim, initial, tooltip= ""):
        super().__init__()
        root = qw.QHBoxLayout(self)
        self.name = name
        self.dim = dim

        left_entries = qw.QWidget()
        left_layout = qw.QVBoxLayout(left_entries)

        label_widget = LatexLabel()
        font = label_widget.font()
        font.setPointSize(7)
        label_widget.setFont(font)
        label_widget.setText(label)

        tooltip_widget = qw.QToolButton()
        tooltip_widget.setText("?")
        tooltip_widget.setToolTip(tooltip)

        left_layout.addWidget(label_widget, alignment = qc.Qt.AlignmentFlag.AlignLeft, stretch=3)
        left_layout.addWidget(tooltip_widget, alignment= qc.Qt.AlignmentFlag.AlignLeft, stretch=0)
        root.addWidget(left_entries, alignment= qc.Qt.AlignmentFlag.AlignLeft)

        matrix_entries = qw.QWidget()
        matrix_layout = qw.QGridLayout(matrix_entries)

        self.entries = []
        for i in range(dim[0]):
            self.entries.append([])
            for j in range(dim[1]):
                entry = qw.QLineEdit()
                self.entries[i].append(entry)
                matrix_layout.addWidget(entry, i, j, alignment= qc.Qt.AlignmentFlag.AlignLeft)
                entry.setText(str(initial[i][j]))
                entry.textChanged.connect(partial(self._on_text_change, i, j))

        root.addWidget(matrix_entries, alignment= qc.Qt.AlignmentFlag.AlignLeft)

    def _on_text_change(self, i, j, text):
        new_matrix = zeros(self.dim)
        try:
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    new_matrix[i][j] = float(self.entries[i][j].text())
            if self.dim[0] != self.dim[1]:
                new_matrix = new_matrix.reshape(1,-1)[0]
            self.textChanged.emit(self.name, new_matrix)
        except ValueError:
            pass


