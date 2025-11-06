from PyQt6 import (
    QtCore as qc,
    QtWidgets as qw,
)
from .LatexLabel import LatexLabel
from .FloatSlider import FloatSlider
from .HelpButton import HelpButton

class EntryBlock(qw.QWidget):
    valueChanged = qc.pyqtSignal(str, object)

    def __init__(self, name, var_label, slider_range, initial, tooltip, num_type= "float"):
        super().__init__()
        layout = qw.QVBoxLayout(self)
        layout.setSpacing(0)

        self.entry_text_changed = False

        self.name = name
        self.top_row_layout = qw.QHBoxLayout()
        if num_type != "float":
            if num_type == "int":
                self.num_type = num_type
                self.current_val = int(initial)
            else:
                print("Unrecognized num_type argument. Valid arguments are 'int' and 'float'. Defaulting to float.")
                self.num_type = "float"
        else:
            self.num_type = "float"
            self.current_val = initial
        self.range = slider_range
        
        self.label = LatexLabel()
        font = self.label.font()
        font.setPointSize(5)
        self.label.setFont(font)
        self.label.setText(var_label)
        
        self.entry = qw.QLineEdit()
        self.entry.setText(str(self.current_val))
        self.debounce_timer = qc.QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.entry.textChanged.connect(lambda _: self.debounce_timer.start(300))
        self.debounce_timer.timeout.connect(self.entry_change)

        self.info_button = HelpButton("?", tooltip)

        self.top_row_layout.addWidget(self.label)
        self.top_row_layout.addWidget(self.entry)
        self.top_row_layout.addWidget(self.info_button)

        self.top_row_widget = qw.QWidget()
        self.setSizePolicy(
            qw.QSizePolicy.Policy.Preferred,
            qw.QSizePolicy.Policy.Fixed
        )
        self.top_row_widget.setLayout(self.top_row_layout)

        self.bottom_row_layout = qw.QHBoxLayout()
        self.lhlabel = qw.QLabel(str(self.range[0]))
        self.rhlabel = qw.QLabel(str(self.range[1]))

        self.slider = FloatSlider(orientation= "h", float_range= self.range, init_val= self.current_val)
        self.slider.sliderReleased.connect(self.slider_change)
        self.slider.valueChanged.connect(self.slider_change_prelude)

        self.bottom_row_layout.addWidget(self.lhlabel)
        self.bottom_row_layout.addWidget(self.slider)
        self.bottom_row_layout.addWidget(self.rhlabel)

        self.bottom_row_widget = qw.QWidget()
        self.bottom_row_widget.setLayout(self.bottom_row_layout)

        layout.addWidget(self.top_row_widget)
        layout.addWidget(self.bottom_row_widget)

    def entry_change(self):
        try:
            val = float(self.entry.text())
        except ValueError:
            self.entry_text_change = False
            return
        
        if val < self.range[0]:
            if self.num_type == "int":
                self.current_val = int(self.range[0])
            else:
                self.current_val = self.range[0]
        elif val > self.range[1]:
            if self.num_type == "int":
                self.current_val = int(self.range[1])
            else:
                self.current_val = self.range[1]
        else:
            if self.num_type == "int":
                self.current_val = int(val)
            else:
                self.current_val = val

        self.entry_text_changed = True
        self.slider.change_value(self.current_val)
        self.entry_text_changed = False
        self.valueChanged.emit(self.name, self.current_val)

    def slider_change_prelude(self):
        if not self.slider.isSliderDown(): self.slider_change()

    def slider_change(self):
        if self.entry_text_changed:
            return
        new_val = self.slider.get_current_val()
        if self.num_type == "int":
            self.current_val = int(new_val)
        else:
            self.current_val = new_val
        self.entry.setText(str(self.current_val))

    def get(self):
        return self.current_val


