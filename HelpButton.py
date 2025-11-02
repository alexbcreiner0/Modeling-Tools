from PyQt6 import (
    QtCore as qc,
    QtGui as qg,
    QtWidgets as qw,
)

class HelpButton(qw.QToolButton):
    hovered = qc.pyqtSignal()

    def __init__(self, text, tooltip= None):
        super().__init__()
        self.text = text
        self.tooltip = tooltip

        self.setText("?")
        self.setToolTip(tooltip)

        self.clicked.connect(self.show_tip)

    def show_tip(self):
        pos = qg.QCursor.pos()
        qw.QToolTip.showText(pos, self.tooltip, self)

    def enterEvent(self, a0):
        self.hovered.emit()
        super().enterEvent(a0)


