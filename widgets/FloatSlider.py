from PyQt6 import (
    QtWidgets as qw,
    QtCore as qc
)
import sys

class FloatSlider(qw.QSlider):
    def __init__(self, orientation= "h", float_range= None, init_val= None):
        try: 
            if orientation == "v":
                super().__init__(qc.Qt.Orientation.Vertical)
            elif orientation == "h":
                super().__init__(qc.Qt.Orientation.Horizontal)
            else:
                raise ValueError
        except ValueError:
            print("Invalid orientation choice for FloatSlider. Valid choices are 'h' and 'v'.")
            sys.exit()

        self.current_val = init_val
        self.float_range = float_range
        self.slider_range = (0,1000)
        self.setMinimum(self.slider_range[0])
        self.setMaximum(self.slider_range[1])
        self.setTickInterval(10)

        self.tic_pos = self.compute_tic_pos(self.current_val)
        self.setValue(self.tic_pos)
        self.valueChanged.connect(self.update_value)

    def compute_tic_pos(self, val):
        tic_pos = int((val / self.float_range[1])*1000)
        return tic_pos

    def compute_value(self, tic_pos):
        advance = (tic_pos / self.slider_range[1])
        new_val = self.slider_range[0]+(self.float_range[1] - self.float_range[0])*advance
        return new_val

    def get_current_val(self):
        advance = (self.value() / self.slider_range[1])
        new_val = self.slider_range[0]+(self.float_range[1] - self.float_range[0])*advance
        return new_val

    def update_value(self, tic_pos):
        new_val = self.compute_value(tic_pos)
        self.current_val = new_val

    def change_value(self, val):
        if val > self.float_range[1]:
            new_val = self.float_range[1]
        elif val < self.float_range[0]:
            new_val = self.float_range[0]
        else:
            new_val = val
        tic_pos = self.compute_tic_pos(new_val)
        self.setValue(tic_pos)


