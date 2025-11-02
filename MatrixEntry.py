from PyQt6 import (
    QtCore as qc,
    QtWidgets as qw,
)
from functools import partial
from numpy import ndarray, zeros
from LatexLabel import LatexLabel
from HelpButton import HelpButton

class MatrixEntry(qw.QWidget):
    textChanged = qc.pyqtSignal(str, ndarray)

    def __init__(self, name, label, dim, initial, tooltip= ""):
        super().__init__()
        root = qw.QHBoxLayout(self)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)
        self.name = name
        self.dim = dim

        left_entries = qw.QWidget()
        left_layout = qw.QVBoxLayout(left_entries)

        label_widget = LatexLabel()
        font = label_widget.font()
        font.setPointSize(7)
        label_widget.setFont(font)
        label_widget.setText(label)

        tooltip_widget = HelpButton("?", tooltip)

        left_layout.addWidget(label_widget, alignment = qc.Qt.AlignmentFlag.AlignLeft, stretch=3)
        left_layout.addWidget(tooltip_widget, alignment= qc.Qt.AlignmentFlag.AlignLeft, stretch=0)
        root.addWidget(left_entries, alignment= qc.Qt.AlignmentFlag.AlignLeft, stretch= 0)

        matrix_entries = qw.QWidget()
        matrix_layout = qw.QGridLayout(matrix_entries)

        self.debounce_timer = qc.QTimer(self)
        self.debounce_timer.setSingleShot(True)

        self.entries = []
        for i in range(dim[0]):
            self.entries.append([])
            for j in range(dim[1]):
                entry = qw.QLineEdit()
                self.entries[i].append(entry)
                matrix_layout.addWidget(entry, i, j)
                entry.setText(str(initial[i][j]))

                timer = qc.QTimer(entry)
                timer.setSingleShot(True)
                entry.textChanged.connect(lambda _=None, t= timer: t.start(300))
                timer.timeout.connect(partial(self._on_text_change, i, j))

                # entry.textChanged.connect(partial(self._on_text_change, i, j))

        root.addWidget(matrix_entries, alignment= qc.Qt.AlignmentFlag.AlignLeft, stretch= 3)

    def _on_text_change(self, i, j):
        new_matrix = zeros(self.dim)
        try:
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    new_matrix[i][j] = float(self.entries[i][j].text())
            if self.dim[0] != self.dim[1]:
                new_matrix = new_matrix.reshape(1,-1)[0]
            print(f"Emitting: {new_matrix}")
            self.textChanged.emit(self.name, new_matrix)
        except ValueError:
            pass


