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
rcParams["savefig.bbox"] = "tight"
rcParams["savefig.pad_inches"] = 0.05  # or 0.0 if you want absolutely no padding

class GraphPanel(qw.QWidget):
    saved_lims_changed = qc.pyqtSignal(tuple, tuple)
    slot_axes_limits_changed = qc.pyqtSignal(int, tuple, tuple)
    slot_title_changed = qc.pyqtSignal(int, str)

    def __init__(self, init_traj, init_t, dropdown_choices, T,
                 plotting_data, canvas, figure, axis, toolbar, status_bar):
        super().__init__()
        self.start_up = True
        self.data = plotting_data
        layout = qw.QVBoxLayout()
        self.dropdown_choices = dropdown_choices
        self.canvas = canvas
        self.figure, self.axis = figure, axis
        self.toolbar = toolbar
        self.status_bar = status_bar
        self.slot_choice_key = {}
        self.legend_label_overrides = {}

        self.traj = init_traj

        self.axes_rows = 1
        self.axes_cols = 1
        self.axes = [self.axis]

        self._cid_draw = self.canvas.mpl_connect(
            "draw_event", self._on_canvas_draw
        )

        self.font_vals = {(1,1): 10, (1,2): 8, (1,3): 6, (2,1): 8, (2,2): 8, (3,2): 6, (2,3): 6, (3,3): 0}

        self.snap_artists = {}
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
       
        self._init_snap_artists()
        self.dragging = False

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
      
        self.camera_controls = qw.QWidget()

        layout.addWidget(self.canvas, stretch=5)
        self.setLayout(layout)
        self.T = T
        self.make_plot(init_traj, init_t, 0, {})

        self.start_up = False

        self._connect_axis_callbacks()

        # Compute and store the base box aspect (height / width) from
        # the initial single-axes layout so we can reuse it for grids.
        try:
            fig_w, fig_h = self.figure.get_size_inches()
            dpi = self.figure.dpi
            fig_w_px = fig_w * dpi
            fig_h_px = fig_h * dpi

            pos = self.axis.get_position()  # in figure-relative coordinates
            width_px = pos.width * fig_w_px
            height_px = pos.height * fig_h_px

            self._base_box_aspect = height_px / width_px if width_px else 1.0
        except Exception:
            # fallback: a reasonable wide-ish plot
            self._base_box_aspect = 0.6

       # Optional: also enforce it on the initial single axes
        try:
            self.axis.set_box_aspect(self._base_box_aspect)
        except AttributeError:
            # set_box_aspect requires Matplotlib >= 3.3
            pass

        self._block_axis_callback = False

    def on_scroll(self, event):
        ax = event.inaxes

        if ax is None or ax not in self.axes:
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

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

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
        ax.set_xlim(new_x_min, new_x_max)
        ax.set_ylim(new_y_min, new_y_max)
        self._block_axis_callback = False

        self._on_axis_limits_changed(ax)

        self.canvas.draw_idle()

    def _on_axis_limits_changed_bak(self, ax):
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

    def _on_axis_limits_changed(self, ax):
        if getattr(self, "_block_axis_callback", False):
            return

        # Read current limits from the axes
        self.xlim = ax.get_xlim()
        self.ylim = ax.get_ylim()

        try:
            slot_index = self.axes.index(ax)
        except ValueError:
            slot_index = 0

        self.slot_axes_limits_changed.emit(slot_index, self.xlim, self.ylim)

    def _connect_axis_callbacks(self):
        for ax in self.axes:
            ax.callbacks.connect("xlim_changed", self._on_axis_limits_changed)
            ax.callbacks.connect("ylim_changed", self._on_axis_limits_changed)

    def _init_snap_artists(self):
        self.snap_artists = {}

        for ax in self.axes:
            marker, = ax.plot([], [], "o", ms= 6)
            marker.set_visible(False)

            annot = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=0.5),
            )
            annot.set_visible(False)

            self.snap_artists[ax] = (marker, annot)

    def set_axes_layout(self, rows, cols):
        """ Rebuild the figure with a rows x cols grid of axes """
        if rows < 1 or cols < 1:
            return

        if rows == self.axes_rows and cols == self.axes_cols:
            return

        self.axes_rows = rows
        self.axes_cols = cols

        # Snapshot existing per-slot limits (one (xlim, ylim) per axes)
        old_limits = []
        try:
            for ax in self.axes:
                old_limits.append((ax.get_xlim(), ax.get_ylim()))
        except Exception:
            old_limits = []

        # Fallback: if we somehow have no per-slot limits, use the primary axis
        fallback_xlim = None
        fallback_ylim = None
        if old_limits:
            fallback_xlim, fallback_ylim = old_limits[0]
        else:
            try:
                fallback_xlim = self.axis.get_xlim()
                fallback_ylim = self.axis.get_ylim()
            except Exception:
                fallback_xlim = fallback_ylim = None

        self.figure.clear()
        axes_array = self.figure.subplots(rows, cols, squeeze=False)

        self.axes = [ax for row in axes_array for ax in row]
        self.axis = self.axes[0]

        self._init_snap_artists()
        self._connect_axis_callbacks()

        self._block_axis_callback = True

        if old_limits:
            last_xlim, last_ylim = old_limits[-1]
        else:
            last_xlim = fallback_xlim
            last_ylim = fallback_ylim

        for i, ax in enumerate(self.axes):
            if old_limits and i < len(old_limits):
                xlim, ylim = old_limits[i]
            else:
                xlim = last_xlim
                ylim = last_ylim

            if xlim is not None and ylim is not None:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

            try:
                ax.set_box_aspect(self._base_box_aspect)
            except AttributeError:
                pass

        self._block_axis_callback = False

        self.canvas.draw_idle()

    def edit_slot_axes(self, slot_index, xlim, ylim):
        """ Apply (xlim, ylim) to the axes corresponding to the slot_index """
        if slot_index < 0 or slot_index >= len(self.axes):
            return

        ax = self.axes[slot_index]

        self._block_axis_callback = True
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        self._block_axis_callback = False

        self.xlim = xlim
        self.ylim = ylim

        self.canvas.draw_idle()

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
        self.saved_xlim, self.saved_ylim = self.xlim, self.ylim
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

    def _plot_on_axis(self, axis, slot_index, choice_name, traj, t, dropdown_choice, options):

        dropdown_list = list(self.data.keys())
        choice = self.data[dropdown_list[dropdown_choice]]
        plots = choice["plots"]

        for plot in plots:
            plot_dict = plots[plot]
            n = len(plot_dict["labels"])

            # deciding whether to plot
            should_draw = True
            if "checkbox_name" in plot_dict:
                name = plot_dict["checkbox_name"]
                if name not in options and not (self.start_up and "on_startup" in plot_dict):

                    should_draw = False

            if not should_draw: continue
        
            try:
                # plotting
                # special cases
                if "special" in plot_dict and plot_dict["special"] == "scatter":
                    axis.scatter(
                        traj[plot_dict["traj_key_x"]],
                        traj[plot_dict["traj_key_y"]],
                        color= plot_dict.get("color", "k"),
                        label= plot_dict["labels"][0],
                    )
                    continue
                
                if n == 1:
                    # scalar trajectory case
                    default_label = plot_dict["labels"][0]
                    gid = f"{choice_name}::{plot}::0"

                    over = self.legend_label_overrides.get((slot_index, choice_name), {})
                    label = over.get(gid, default_label)

                    line = axis.plot(
                        t, traj[plot_dict["traj_key"]],
                        color= plot_dict["colors"][0],
                        linestyle= plot_dict.get("linestyle", "solid"),
                        label= label
                    )[0]
                    line.set_gid(gid)
                else:
                    # vector trajectory case
                    over = self.legend_label_overrides.get((slot_index, choice_name), {})
                    for i in range(n):
                        default_label = plot_dict["labels"][i]
                        gid = f"{choice_name}::{plot}::{i}"
                        label = over.get(gid, default_label)

                        line = axis.plot(
                            t, traj[plot_dict["traj_key"]][:,i],
                            color= plot_dict["colors"][i],
                            linestyle= plot_dict.get("linestyle", "solid"),
                            label= label
                        )[0]
                        line.set_gid(gid)
            except KeyError:
                print("Error, no key found")
            except ValueError as e:
                if str(e)[0:7] == "x and y":
                    print(e)
                    print("Truncating the plot to the length of the x-axis")
                    plot = traj[plot_dict["traj_key"]][0:len(t)]
                    if n == 1:
                        # scalar trajectory case
                        default_label = plot_dict["labels"][0]
                        gid = f"{choice_name}::{plot}::0"

                        over = self.legend_label_overrides.get((slot_index, choice_name), {})
                        label = over.get(gid, default_label)

                        line = axis.plot(
                            t, plot,
                            color= plot_dict["colors"][0],
                            linestyle= plot_dict.get("linestyle", "solid"),
                            label= label
                        )[0]
                        line.set_gid(gid)
                    else:
                        # vector trajectory case
                        over = self.legend_label_overrides.get((slot_index, choice_name), {})

                        for i in range(n):
                            default_label = plot_dict["labels"][i]
                            gid = f"{choice_name}::{plot}::{i}"
                            label = over.get(gid, default_label)

                            line = axis.plot(
                                t, plot[:,i],
                                color= plot_dict["colors"][i],
                                linestyle= plot_dict.get("linestyle", "solid"),
                                label= label
                            )[0]
                            line.set_gid(gid)


    def make_plot(self, traj, t, dropdown_choice, options):
        self.traj = traj
        self.t = t

        for i,ax in enumerate(self.axes):
            ax.clear()

            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()

            dropdown_list = list(self.data.keys())
            choice_name = dropdown_list[dropdown_choice]

            # TODO: replace time with a plotting file specification

            self._plot_on_axis(ax, i, choice_name, traj, t, dropdown_choice, options)

            ax.set_xlim(current_xlim)
            ax.set_ylim(current_ylim)
            ax.legend(fontsize= 10)

        self._init_snap_artists()
        self._connect_axis_callbacks()
        self.canvas.draw()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _on_canvas_draw(self, event):
        """
        Called after Matplotlib redraws the figure.
        Capture titles so they persist across app-driven redraws.
        """
        if not hasattr(self, "title_overrides"):
            self.title_overrides = {}

        for i, ax in enumerate(self.axes):
            title = ax.get_title()
            if title and title.strip():
                self.title_overrides[i] = title
            else:
                self.title_overrides.pop(i, None)

        # capture legend/line labels after toolbar Apply/OK
        for slot_index, ax in enumerate(self.axes):
            choice_name = self.slot_choice_key.get(slot_index)
            if not choice_name:
                continue

            bucket = self.legend_label_overrides.setdefault((slot_index, choice_name), {})
            for line in ax.lines:
                gid = line.get_gid()
                if not gid:
                    continue
                lab = line.get_label()
                if lab and str(lab).strip():
                    bucket[gid] = str(lab)

    def plot_slot(self, slot_index, dropdown_choice, options, slot_config= None):
        """ Apply a plot to a slot """
        dropdown_list = list(self.data.keys())
        choice_name = dropdown_list[dropdown_choice]
        self.slot_choice_key[slot_index] = choice_name
        
        if self.traj is None or self.t is None: return
        if slot_index < 0 or slot_index >= len(self.axes): return

        ax = self.axes[slot_index]
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()

        try:
            default_font = self.font_vals[(self.axes_rows, self.axes_cols)]
        except KeyError:
            default_font = 10

        ax.clear()

        self._plot_on_axis(ax, slot_index, choice_name, self.traj, self.t, dropdown_choice, options)

        self._block_axis_callback = True
        ax.set_xlim(current_xlim)
        ax.set_ylim(current_ylim)
        self._block_axis_callback = False

        if slot_config is None:
            ax.legend(fontsize= default_font)
        else:
            legend_visible = slot_config.get("legend_visible", True)
            fontsize = slot_config.get("legend_fontsize", default_font)
            loc = slot_config.get("legend_loc", "upper right")
            legend_title = slot_config.get("legend_title", None)

            if legend_visible:
                handles, labels = ax.get_legend_handles_labels()

                overrides = slot_config.get("legend_label_overrides", slot_config.get("legend_labels", None))
                if overrides:
                    if isinstance(overrides, dict):
                        labels = [str(overrides.get(lbl, lbl)) for lbl in labels]
                    elif isinstance(overrides, (list, tuple)):
                        labels = [str(overrides[i]) if i < len(overrides) else lbl for i, lbl in enumerate(labels)]

                ax.legend(handles, labels, fontsize=fontsize, loc=loc, title=legend_title)
            else:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()

            # if slot_config.get("legend_visible", True):
            #     fontsize = slot_config.get("legend_fontsize", default_font)
            #     loc = slot_config.get("legend_loc", "upper right")
            #     ax.legend(fontsize= fontsize, loc= loc)
            # else:
            #     leg = ax.get_legend()
            #     if leg is not None:
            #         leg.remove()

            display_title = slot_config.get("title", False)
            display_x_title = slot_config.get("xlabel", True)
            display_y_title = slot_config.get("ylabel", False)

            dropdown_list = list(self.data.keys())
            choice = self.data[dropdown_list[dropdown_choice]]

            title = choice.get("title") or choice["name"]
            x_title = choice.get("x_label") or "Time [t]"
            y_title = choice.get("y_label") or ""

            title_fs = default_font+8
            label_fs = default_font+4

            if display_title: 
                override = self.title_overrides.get(slot_index)
                if override:
                    ax.set_title(override)
                else:
                    ax.set_title(title)
            if display_x_title: ax.set_xlabel(x_title)
            if display_y_title: ax.set_ylabel(y_title)

        self._init_snap_artists()
        self.canvas.draw_idle()

    def on_press(self, event):
        if event.button != 1:
            return
        ax = event.inaxes
        if ax is None or ax not in self.axes:
            return
    
        self.dragging = True
        self._update_snap(event, ax)

    def _update_snap(self, event, ax):
        if ax is None or ax not in self.axes: return
        if event.xdata is None or event.ydata is None: return

        ex, ey = event.x, event.y
        best_line = None
        best_idx = None
        best_dist = np.inf

        trans = ax.transData

        for line in ax.lines:
            if not line.get_visible(): continue

            xdata = np.asarray(line.get_xdata(), dtype= float)
            ydata = np.asarray(line.get_ydata(), dtype= float)
            if xdata.size == 0: continue

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

        marker, annot = self.snap_artists.get(ax, (None, None))
        if marker is None or annot is None:
            return

        if best_line is None:
            marker.set_visible(False)
            annot.set_visible(False)
            self.canvas.draw_idle()
            return

        color = best_line.get_color()
        marker.set_color(color)

        x_near = best_line.get_xdata()[best_idx]
        y_near = best_line.get_ydata()[best_idx]

        marker.set_data([x_near], [y_near])
        marker.set_visible(True)

        annot.xy = (x_near, y_near)
        annot.set_text(f"({x_near:g}, {y_near:g})")
        annot.set_visible(True)

        self.canvas.draw_idle()

    def on_motion(self, event):
        if not self.dragging:
            return
        ax = event.inaxes
        if ax is None or ax not in self.axes:
            return

        self._update_snap(event, ax)

        active_tool = getattr(self.toolbar, "mode", None)
        if active_tool == "pan/zoom":
            self._on_axis_limits_changed(ax)

    def on_release(self, event):
        if event.button != 1:
            return
        self.dragging = False
    
        for marker, annot in self.snap_artists.values():
            marker.set_visible(False)
            annot.set_visible(False)

        self.canvas.draw_idle()

