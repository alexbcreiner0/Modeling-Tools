from PyQt6 import (
    QtCore as qc,
    QtWidgets as qw,
)
from HelpButton import HelpButton

class DropdownChoices(qw.QWidget):
    infoBoxHovered = qc.pyqtSignal()
    currentIndexChanged = qc.pyqtSignal(int)
    checkStateChanged = qc.pyqtSignal()

    def __init__(self, parent= None, items_per_row= 3):
        super().__init__(parent)
        root = qw.QVBoxLayout(self) # passing self as parent means don't have to call self.setLayout!
        top = qw.QHBoxLayout()
       
        self.dropdown_choices = qw.QComboBox()
        self.info = HelpButton("?")
        self.items_per_row = items_per_row

        self.stack = qw.QStackedWidget()
        root.addWidget(self.stack)
        top.addWidget(self.dropdown_choices)
        top.addWidget(self.info)
        root.addLayout(top)

        self.pages = {}
        self.grids = {}
        self.boxes = {}

        # self.dropdown_choices.currentTextChanged.connect(self._on_selection_changed)
        self.dropdown_choices.currentIndexChanged.connect(self._on_selection_changed)
        self.info.hovered.connect(self._on_info_hovered)

    def addItem(self, name):
        """Adds the dropdown option like normal, but also creates the necessary accompanying checkbox area for that option."""
        if name in self.pages:
            return

        page = qw.QWidget()
        grid = qw.QGridLayout(page)
        grid.setContentsMargins(0,0,0,0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        self.pages[name] = page
        self.grids[name] = grid
        self.boxes[name] = []

        self.stack.addWidget(page)
        self.dropdown_choices.addItem(name)

        if self.dropdown_choices.count() == 1:
            self.stack.setCurrentWidget(page)

    def addItems(self, names):
        for name in names: self.addItem(name)

    def add_checkbox(self, option, label, checked= False):
        if option not in self.pages:
            self.addItem(option)

        box = qw.QCheckBox(label)
        box.setChecked(checked)
        box.checkStateChanged.connect(self._on_checkstate_changed)

        grid = self.grids[option]
        boxes = self.boxes[option]
        index = len(boxes)
        row = index // self.items_per_row
        col = index % self.items_per_row

        grid.addWidget(box, row, col)
        boxes.append(box)
        return box

    def _on_checkstate_changed(self):
        self.checkStateChanged.emit()

    def _on_selection_changed(self, index):
        page = self.pages.get(self.dropdown_choices.currentText())
        if page is not None:
            self.stack.setCurrentWidget(page)
        self.currentIndexChanged.emit(index)

    def _on_info_hovered(self):
        self.infoBoxHovered.emit()

    def setToolTip(self, a0):
        self.info.setToolTip(a0)

    def get_current_checked_boxes(self):
        page = self.pages[self.dropdown_choices.currentText()]
        boxes = []
        for checkbox in page.findChildren(qw.QCheckBox):
            if checkbox.isChecked(): boxes.append(checkbox.text())
        return boxes


