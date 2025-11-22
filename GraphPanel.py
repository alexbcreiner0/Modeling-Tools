import sys
from PyQt6 import (
    QtCore as qc,
    QtWidgets as qw,
)
from matplotlib import (
    pyplot as plt,
    rcParams
)
import numpy as np
from matplotlib.backend_bases import cursors
import scienceplots
plt.style.use(["grid", "notebook"])
rcParams["figure.constrained_layout.use"] = False
rcParams["figure.autolayout"] = False
rcParams["figure.constrained_layout.h_pad"] = float(0)
rcParams["figure.constrained_layout.hspace"] = float(0)
with open("log.txt", "w") as f:
    print(rcParams, file=f)

class GraphPanel(qw.QWidget):
    saved_lims_changed = qc.pyqtSignal(tuple, tuple)

    def __init__(self, init_traj, init_t, dropdown_choices, T,
                 plotting_data, canvas, figure, axis, toolbar,
                 entries, buttons, status_bar):
        super().__init__()
        self.entries = entries
        self.buttons = buttons
        self.start_up = True
        self.data = plotting_data
        layout = qw.QVBoxLayout()
        self.dropdown_choices = dropdown_choices
        self.canvas = canvas
        self.figure, self.axis = figure, axis
        self.toolbar = toolbar
        self.status_bar = status_bar

        self._orig_canvas_set_cursor = self.canvas.set_cursor
        def custom_set_cursor(cursor):
            if cursor == cursors.MOVE:
                self.canvas.unsetCursor()
            else:
                self._orig_canvas_set_cursor(cursor)

        self.canvas.set_cursor = custom_set_cursor

        try:
            with open("dimensions.txt", "r") as f:
                xlim_str = f.readline().strip().strip('()').split(',')
                ylim_str = f.readline().strip().strip('()').split(',')
                self.xlim = tuple([float(i) for i in xlim_str])
                self.ylim = tuple([float(i) for i in ylim_str])
        except FileNotFoundError:
            with open("dimensions.txt", "w") as f:
                print(str((0,50)),"\n",str((0,50)), file= f)
            self.xlim, self.ylim = (0,50), (0,50)
        self.saved_xlim, self.saved_ylim = self.xlim, self.ylim
        self.xlower_entry, self.xupper_entry = entries[0], entries[2]
        self.ylower_entry, self.yupper_entry = entries[1], entries[3]
        self.xlower_entry.setText(f"{self.xlim[0]:g}")
        self.xupper_entry.setText(f"{self.xlim[1]:g}")
        self.ylower_entry.setText(f"{self.ylim[0]:g}")
        self.yupper_entry.setText(f"{self.ylim[1]:g}")
        for entry in entries: 
            entry.setSizePolicy(qw.QSizePolicy.Policy.Fixed,qw.QSizePolicy.Policy.Fixed)
            entry.setFixedWidth(70)
            entry.textChanged.connect(self.edit_axes) 
        self.save_button, self.load_button = buttons[0], buttons[1] 
        self.save_button.clicked.connect(self.save_axes)
        self.load_button.clicked.connect(self.load_axes)
        
        self._init_snap_artists()
        self.dragging = False

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
      
        self.camera_controls = qw.QWidget()
        # self.toolbar.pan()

        layout.addWidget(self.canvas, stretch=5)
        self.setLayout(layout)
        self.T = T
        self.edit_axes()
        try:
            self.make_plot(init_traj, init_t, 0, {})
        except Exception as e:
            print(f"Problem? {e}")
            sys.exit()

        self.start_up = False

        self._connect_axis_callbacks()
        self._block_axis_callback = False

    def on_scroll(self, event):
        if event.inaxes is not self.axis:
            return
        if event.xdata is None or event.ydata is None:
            return

        base_scale = 1.2

        if event.button == "up":
            scale_factor = 1 / base_scale
        elif event.button == "down":
            scale_factor = base_scale
        else:
            return

        xdata, ydata = event.xdata, event.ydata

        x_min, x_max = self.axis.get_xlim()
        y_min, y_max = self.axis.get_ylim()

        dx_left = xdata - x_min
        dx_right = x_max - xdata
        dy_bottom = ydata - y_min
        dy_top = y_max - ydata

        new_x_min = xdata - dx_left * scale_factor
        new_x_max = xdata + dx_right * scale_factor
        new_y_min = ydata - dy_bottom * scale_factor
        new_y_max = ydata + dy_top * scale_factor

        if new_x_max == new_x_min or new_y_max == new_y_min:
            return

        self._block_axis_callback = True
        self.axis.set_xlim(new_x_min, new_x_max)
        self.axis.set_ylim(new_y_min, new_y_max)
        self._block_axis_callback = False

        self.canvas.draw_idle()

    def _on_axis_limits_changed(self, ax):
        if self._block_axis_callback:
            return

        # Read current limits from the axes
        self.xlim = ax.get_xlim()
        self.ylim = ax.get_ylim()

        # Update the QLineEdits without re-triggering edit_axes
        widgets_and_values = [
            (self.xlower_entry, self.xlim[0]),
            (self.xupper_entry, self.xlim[1]),
            (self.ylower_entry, self.ylim[0]),
            (self.yupper_entry, self.ylim[1]),
        ]

        for w, v in widgets_and_values:
            w.blockSignals(True)
            text = f"{v:g}"
            w.setText(text)
            w.blockSignals(False)

    def _connect_axis_callbacks(self):
        self.axis.callbacks.connect("xlim_changed", self._on_axis_limits_changed)
        self.axis.callbacks.connect("ylim_changed", self._on_axis_limits_changed)

    def _init_snap_artists(self):
        self.snap_marker, = self.axis.plot([], [], "o", ms=6)
        self.snap_marker.set_visible(False)

        self.snap_annot = self.axis.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=0.5),
        )
        self.snap_annot.set_visible(False)

    def edit_axes(self):
        try:
            new_xlim = (float(self.xlower_entry.text()), float(self.xupper_entry.text()))
            new_ylim = (float(self.ylower_entry.text()), float(self.yupper_entry.text()))

            self._block_axis_callback = True
            self.axis.set_xlim(new_xlim)
            self.axis.set_ylim(new_ylim)
            self.canvas.draw_idle()
            self._block_axis_callback = False

            self.xlim, self.ylim = new_xlim, new_ylim

        except ValueError:
            self._block_axis_callback = False

    def save_axes(self):
        # xlim_str = f"({self.xlim[0]:g}, {self.xlim[1]:g})"
        # ylim_str = f"({self.ylim[0]:g}, {self.ylim[1]:g})"
        self.saved_xlim, self.saved_ylim = self.xlim, self.ylim
        # self.saved_xlim, self.saved_ylim = (float(f"{float(self.xlim[0]):.3f}"), float(f"{float(self.xlim[1]):.3f}")), (float(f"{float(self.ylim[0]):.3f}"), float(f"{float(self.ylim[1]):.3f}"))
        self.saved_lims_changed.emit(self.saved_xlim, self.saved_ylim)

    def load_axes(self):
        self.xlim, self.ylim = self.saved_xlim, self.saved_ylim

        
        widgets_and_values = [
            (self.xlower_entry, self.xlim[0]),
            (self.xupper_entry, self.xlim[1]),
            (self.ylower_entry, self.ylim[0]),
            (self.yupper_entry, self.ylim[1]),
        ]

        for w, v in widgets_and_values:
            w.blockSignals(True)
            w.setText(f"{v:g}")   # or str(v)
            w.blockSignals(False)

        self.edit_axes()

    def make_plot(self, traj, t, dropdown_choice, options):
        current_xlim = self.axis.get_xlim()
        current_ylim = self.axis.get_ylim()
        self.axis.clear()
        self.axis.set_xlabel("Time [t]")

        dropdown_list = list(self.data.keys())
        plots = self.data[dropdown_list[dropdown_choice]]["plots"]

        for plot in plots:
            plot_dict = plots[plot]
            n = len(plot_dict["labels"])
            if "checkbox_name" in plot_dict:
                if plot_dict["checkbox_name"] in options or (self.start_up and "on_startup" in plot_dict):
                    if len(plot_dict["labels"]) == 1:
                        if "linestyle" in plot_dict: 
                            linestyle = plot_dict["linestyle"] 
                        else: 
                            linestyle = "solid"
                        self.axis.plot(t, traj[plot_dict["traj_key"]], color= plot_dict["colors"][0], linestyle= linestyle, label= plot_dict["labels"][0])
                    else:
                        n = len(plot_dict["labels"])
                        if "linestyle" in plot_dict:
                             linestyle = plot_dict["linestyle"] 
                        else: 
                            linestyle = "solid"
                        for i in range(n):
                            self.axis.plot(t, traj[plot_dict["traj_key"]][:,i], color= plot_dict["colors"][i], linestyle= linestyle, label= plot_dict["labels"][i])
            else:
                if len(plot_dict["labels"]) == 1:
                    if "linestyle" in plot_dict: 
                        linestyle = plot_dict["linestyle"] 
                    else: 
                        linestyle = "solid"
                    self.axis.plot(t, traj[plot_dict["traj_key"]], color= plot_dict["colors"][0], linestyle= linestyle, label= plot_dict["labels"][0])
                else:
                    n = len(plot_dict["labels"])
                    if "linestyle" in plot_dict:
                         linestyle = plot_dict["linestyle"] 
                    else: 
                        linestyle = "solid"
                    for i in range(n):
                        self.axis.plot(t, traj[plot_dict["traj_key"]][:,i], color= plot_dict["colors"][i], linestyle= linestyle, label= plot_dict["labels"][i])

        self.axis.set_xlim(current_xlim)
        self.axis.set_ylim(current_ylim)
        self.axis.legend(fontsize= 10)
        self._init_snap_artists()
        self._connect_axis_callbacks()
        self.canvas.draw()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def on_press(self, event):
        if event.button != 1:
            return
        if event.inaxes is not self.axis:
            return
    
        self.dragging = True
        self._update_snap(event)

    def _update_snap(self, event):
        if event.xdata is None or event.ydata is None:
            return
        
        ex, ey = event.x, event.y
        best_line = None
        best_idx = None
        best_dist = np.inf

        trans = self.axis.transData
        
        for line in self.axis.lines:
            if not line.get_visible():
                continue

            xdata = np.asarray(line.get_xdata(), dtype= float)
            ydata = np.asarray(line.get_ydata(), dtype= float)
            if xdata.size == 0:
                continue
            
            pts = np.column_stack((xdata, ydata))
            disp = trans.transform(pts)
            dx = disp[:,0] - ex
            dy = disp[:,1] - ey
            dist = dx**2 + dy**2

            idx = int(np.argmin(dist))
            d = dist[idx]

            if d < best_dist:
                best_dist = d
                best_line = line
                best_idx = idx

        if best_line is None:
            self.snap_marker.set_visible(False)
            self.snap_annot.set_visible(False)
            self.canvas.draw_idle()
            return

        color = best_line.get_color()
        self.snap_marker.set_color(color)
        x_near = best_line.get_xdata()[best_idx]
        y_near = best_line.get_ydata()[best_idx]

        self.snap_marker.set_data([x_near], [y_near])
        self.snap_marker.set_visible(True)

        self.snap_annot.xy = (x_near, y_near)
        self.snap_annot.set_text(f"({x_near:g}, {y_near:g})")
        self.snap_annot.set_visible(True)

        self.canvas.draw_idle()

    def on_motion(self, event):
        if not self.dragging:
            return
        if event.inaxes is not self.axis:
            return

        self._update_snap(event)

    def on_release(self, event):
        if event.button != 1:
            return
        self.dragging = False
        self.snap_marker.set_visible(False)
        self.snap_annot.set_visible(False)
        self.canvas.draw_idle()

