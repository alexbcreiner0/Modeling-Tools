from PyQt6 import QtWidgets as qw, QtCore as qc

class AxesControlWidget(qw.QWidget):
    """
    Composite widget:
      [Save] [Load]   X: [x_min] to [x_max]   Y: [y_min] to [y_max]

    - settingsChanged: emitted whenever the current limits should be applied
                       (editingFinished on any line edit OR after Load).
    """
    settingsChanged = qc.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        outer = qw.QHBoxLayout(self)
        outer.setContentsMargins(0,0,8,0)
        outer.setSpacing(10)

        entries = qw.QVBoxLayout()
        entries.setContentsMargins(0,1,0,1)
        entries.setSpacing(0)

        top_entries = qw.QHBoxLayout()
        top_entries.setContentsMargins(0, 1, 0, 1)
        top_entries.setSpacing(8)

        self.save_button = qw.QPushButton("Save Current Axes")
        self.load_button = qw.QPushButton("Load Saved Axes")

        self.xmin_edit = qw.QLineEdit()
        self.xmax_edit = qw.QLineEdit()
        self.ymin_edit = qw.QLineEdit()
        self.ymax_edit = qw.QLineEdit()

        for edit in (self.xmin_edit, self.xmax_edit, self.ymin_edit, self.ymax_edit):
            # edit.setMaximumWidth(70)
            edit.textChanged.connect(self._on_editing_finished)

        # Layout: [Save] [Load]   X: [ .. ] to [ .. ]   Y: [ .. ] to [ .. ]

        top_entries.addSpacing(8)
        top_entries.addWidget(qw.QLabel("X-axis from:"), stretch= 0)
        top_entries.addWidget(self.xmin_edit, alignment= qc.Qt.AlignmentFlag.AlignLeft)
        top_entries.addWidget(qw.QLabel("to"), stretch= 0)
        top_entries.addWidget(self.xmax_edit, alignment= qc.Qt.AlignmentFlag.AlignLeft)

        self.xmin_edit.setSizePolicy(qw.QSizePolicy.Policy.Expanding, qw.QSizePolicy.Policy.Preferred)
        self.ymin_edit.setSizePolicy(qw.QSizePolicy.Policy.Expanding, qw.QSizePolicy.Policy.Preferred)

        self.xmax_edit.setSizePolicy(qw.QSizePolicy.Policy.Expanding, qw.QSizePolicy.Policy.Preferred)
        self.ymax_edit.setSizePolicy(qw.QSizePolicy.Policy.Expanding, qw.QSizePolicy.Policy.Preferred)

        entries.addLayout(top_entries)

        bottom_entries = qw.QHBoxLayout()
        bottom_entries.setContentsMargins(0,0,0,0)
        bottom_entries.setSpacing(8)

        bottom_entries.addSpacing(8)
        bottom_entries.addWidget(qw.QLabel("Y-axis from:"), stretch= 0)
        bottom_entries.addWidget(self.ymin_edit, alignment = qc.Qt.AlignmentFlag.AlignLeft)
        bottom_entries.addWidget(qw.QLabel("to"), stretch= 0)
        bottom_entries.addWidget(self.ymax_edit, alignment = qc.Qt.AlignmentFlag.AlignLeft)

        entries.addLayout(bottom_entries)

        coords = qw.QVBoxLayout()
        coords.setContentsMargins(0,0,0,0)
        coords.setSpacing(2)

        top_coords = qw.QHBoxLayout()
        top_coords.setContentsMargins(0,0,0,0)
        top_coords.setSpacing(2)

        bottom_coords = qw.QHBoxLayout()
        bottom_coords.setContentsMargins(0,0,0,0)
        bottom_coords.setSpacing(2)

        top_coords.addWidget(self.save_button)
        bottom_coords.addWidget(self.load_button)

        self.saved_x_label = qw.QLabel("Saved X: -")
        self.saved_y_label = qw.QLabel("Saved Y: -")
        top_coords.addWidget(self.saved_x_label, alignment= qc.Qt.AlignmentFlag.AlignRight)
        bottom_coords.addWidget(self.saved_y_label, alignment= qc.Qt.AlignmentFlag.AlignRight)
        # bottom_entries.addStretch(1)

        coords.addLayout(top_coords)
        coords.addLayout(bottom_coords)

        outer.addLayout(entries)
        outer.addLayout(coords)

        # per-widget saved limits
        self._saved_xlim = None
        self._saved_ylim = None
        self._update_saved_labels()

        self.save_button.clicked.connect(self._on_save_clicked)
        self.load_button.clicked.connect(self._on_load_clicked)

    def get_limits(self):
        """
        Returns (xlim, ylim) where each is a (min, max) tuple of floats,
        or None if parsing fails.
        """
        try:
            x0 = float(self.xmin_edit.text())
            x1 = float(self.xmax_edit.text())
            y0 = float(self.ymin_edit.text())
            y1 = float(self.ymax_edit.text())
        except ValueError:
            return None
        return (x0, x1), (y0, y1)

    def set_limits(self, xlim, ylim):
        """
        Programmatically update the line edits to match given limits.
        """
        (x0, x1), (y0, y1) = xlim, ylim

        edits = (
            (self.xmin_edit, x0),
            (self.xmax_edit, x1),
            (self.ymin_edit, y0),
            (self.ymax_edit, y1),
        )
        for edit, val in edits:
            edit.blockSignals(True)
            edit.setText(f"{val:g}")
            edit.blockSignals(False)

    # ---- internal handlers ----
    def _update_saved_labels(self):
        if self._saved_xlim is None or self._saved_ylim is None:
            self.saved_x_label.setText("Saved X: –")
            self.saved_y_label.setText("Saved Y: –")
        else:
            x0, x1 = self._saved_xlim
            y0, y1 = self._saved_ylim
            self.saved_x_label.setText(f"Saved X: ({x0:g}, {x1:g})")
            self.saved_y_label.setText(f"Saved Y: ({y0:g}, {y1:g})")

    def _on_editing_finished(self):
        # Whenever user finishes editing any box, tell the outside world
        self.settingsChanged.emit()

    def _on_save_clicked(self):
        limits = self.get_limits()
        if limits is None:
            return
        self._saved_xlim, self._saved_ylim = limits
        self._update_saved_labels()
        # optional: also emit settingsChanged so that whatever is currently
        # in the boxes definitely gets applied to the plot.
        self.settingsChanged.emit()

    def _on_load_clicked(self):
        if self._saved_xlim is None or self._saved_ylim is None:
            return
        self.set_limits(self._saved_xlim, self._saved_ylim)
        # After loading from saved limits, apply them to the plot:
        self.settingsChanged.emit()
