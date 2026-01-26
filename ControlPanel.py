import sys
from PyQt6 import (
    QtCore as qc,
    QtWidgets as qw,
    QtGui as qg
)
from matplotlib import pyplot as plt
from widgets.SectionDivider import SectionDivider
from widgets.EntryBlock import EntryBlock
from widgets.HelpButton import HelpButton
from widgets.LatexLabel import LatexLabel
from widgets.DropdownChoices import DropdownChoices
from widgets.MatrixEntry import MatrixEntry
from widgets.AxesControlWidget import AxesControlWidget
from widgets.SlotControlsWidget import SlotControlsWidget
from dataclasses import asdict
import math, importlib, inspect
# import scienceplots
# plt.style.use(["grid", "notebook"])

class VScrollArea(qw.QScrollArea):
    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        w = self.widget()
        if w:
            # Lock content width to the viewport width
            w.setFixedWidth(self.viewport().width())

class ControlPanel(qw.QWidget):
    paramChanged = qc.pyqtSignal(str, object)
    layoutChanged = qc.pyqtSignal(int, int)
    slotPlotChoiceChanged = qc.pyqtSignal(int)
    slotOptionsChanged = qc.pyqtSignal(int)
    slotAxesChanged = qc.pyqtSignal(int)
    paramsReplaced = qc.pyqtSignal(object)

    def __init__(self, params, dropdown_choices, dropdown_tooltips, panel_data, plotting_data, sim_model, demo, current_tab= 0):
        # print(f"Loaded params: {asdict(params)}")
        super().__init__()
        self.params = params
        self.block_signals = True
        self.sim_model = sim_model
        self.plotting_data = plotting_data
        self.panel_data = panel_data
        self.dropdown_choices = dropdown_choices
        self.demo = demo
        self.constructing = True

        self.content = qw.QTabWidget()

        scroll_main = VScrollArea()
        scroll_main.setWidgetResizable(True)
        scroll_main.setHorizontalScrollBarPolicy(qc.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_main.setVerticalScrollBarPolicy(qc.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
   
        scroll_plot = VScrollArea()
        scroll_plot.setWidgetResizable(True)
        scroll_plot.setHorizontalScrollBarPolicy(qc.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_plot.setVerticalScrollBarPolicy(qc.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        # --- Sim Controls Tab ---
        main_controls = qw.QWidget()
        scroll_main.setWidget(main_controls)
        self.content.addTab(scroll_main, "Simulation Controls")

        main_controls.setSizePolicy(
            qw.QSizePolicy.Policy.Expanding,
            qw.QSizePolicy.Policy.Preferred   # or Expanding
        )

        main_control_layout = qw.QVBoxLayout(main_controls)
        main_control_layout.setContentsMargins(0, 0, 0, 0)
        main_control_layout.setSpacing(0)

        plot_controls = qw.QWidget()
        self.content.addTab(scroll_plot, "Plot Controls")
        plot_controls.setSizePolicy(
            qw.QSizePolicy.Policy.Expanding,
            qw.QSizePolicy.Policy.Preferred
        )

        scroll_plot.setWidget(plot_controls)

        plot_control_layout = qw.QVBoxLayout(plot_controls)
        plot_control_layout.setContentsMargins(0,0,0,0)
        plot_control_layout.setSpacing(0)

        grid_widget = qw.QWidget()
        grid_widget.setSizePolicy(qw.QSizePolicy.Policy.Expanding, qw.QSizePolicy.Policy.Fixed)
        settings_row = qw.QHBoxLayout(grid_widget)
        settings_row.setContentsMargins(8,8,8,8)
        settings_row.setSpacing(10)

        wrap = qw.QWidget()
        wlay = qw.QVBoxLayout(wrap)
        wlay.setContentsMargins(0,0,0,0)
        wlay.setSpacing(0)
        wlay.addWidget(SectionDivider("Overall Settings"), alignment= qc.Qt.AlignmentFlag.AlignTop, stretch= 0)
        wlay.addWidget(grid_widget)
        wrap.setSizePolicy(qw.QSizePolicy.Policy.Expanding, qw.QSizePolicy.Policy.Fixed)
        # layout.addWidget(wrap, alignment= qc.Qt.AlignmentFlag.AlignTop, stretch= 0)

        grid_layout = qw.QHBoxLayout()

        grid_layout.addWidget(qw.QLabel("Rows:"))
        self.rows_spinner = qw.QSpinBox()
        self.rows_spinner.setRange(1,3)
        self.rows_spinner.setValue(1)
        self.rows_spinner.setSizePolicy(qw.QSizePolicy.Policy.Preferred, qw.QSizePolicy.Policy.Fixed)
        grid_layout.addWidget(self.rows_spinner)

        grid_layout.addSpacing(6)
        grid_layout.addWidget(qw.QLabel("Columns:"))
        self.cols_spinner = qw.QSpinBox()
        self.cols_spinner.setRange(1,3)
        self.cols_spinner.setValue(1)
        self.cols_spinner.setSizePolicy(qw.QSizePolicy.Policy.Preferred, qw.QSizePolicy.Policy.Fixed)
        grid_layout.addWidget(self.cols_spinner)

        settings_row.addLayout(grid_layout)

        self.rows_spinner.valueChanged.connect(self._emit_plot_dim_change)
        self.cols_spinner.valueChanged.connect(self._emit_plot_dim_change)

        # plot_control_layout.addLayout(grid_row) # THIS EXISTS?!
        plot_control_layout.addWidget(wrap, alignment= qc.Qt.AlignmentFlag.AlignTop, stretch= 0)
        plot_control_layout.setContentsMargins(0,0,0,0)
        plot_control_layout.setSpacing(0)

        self.slot_controls_container = qw.QWidget()
        self.slot_controls_layout = qw.QVBoxLayout(self.slot_controls_container)
        self.slot_controls_container.setSizePolicy(qw.QSizePolicy.Policy.Expanding, qw.QSizePolicy.Policy.Fixed)
        self.slot_controls_layout.setContentsMargins(0,0,0,0)
        self.slot_controls_layout.setSpacing(0)

        plot_control_layout.addWidget(self.slot_controls_container, qc.Qt.AlignmentFlag.AlignTop)
        plot_control_layout.addStretch(1)
        
        self.slot_dropdowns = []
        self.slot_options = []
        self.slot_axes_controls = []
        self.slot_titles = {}

        self.content.setCurrentIndex(current_tab)

        outer_layout = qw.QVBoxLayout(self)
        outer_layout.addWidget(self.content)

        self.dropdown_tooltips = dropdown_tooltips

        self.entry_blocks = {}
        self.dropdowns = {}
        self.row_wrappers = []

        for row in panel_data:
            wrap = qw.QWidget()
            wlay = qw.QHBoxLayout(wrap)
            wlay.setContentsMargins(0,0,0,0)
            wlay.setSpacing(0)
            wrap.setSizePolicy(qw.QSizePolicy.Policy.Expanding, qw.QSizePolicy.Policy.Fixed)

            self.row_wrappers.append(wrap)

            if row[0:7] == "divider":
                if "side" in panel_data[row]:
                    wlay.addWidget(SectionDivider(panel_data[row]["title"], panel_data[row]["side"]))
                else:
                    wlay.addWidget(SectionDivider(panel_data[row]["title"]))
                continue

            for entry in panel_data[row]:
                info = panel_data[row][entry]
                widget = self.make_widget(info, params)
                wlay.addWidget(widget, stretch= 1, alignment= qc.Qt.AlignmentFlag.AlignTop)


        for wrapper in self.row_wrappers:
            main_control_layout.addWidget(wrapper, alignment= qc.Qt.AlignmentFlag.AlignTop, stretch= 0)

        main_control_layout.addStretch(1)

        self._rebuild_slot_dropdowns(self.rows_spinner.value(), self.cols_spinner.value())

        for i in range(len(self.slot_dropdowns)):
            self.get_tooltip(i)

        self.block_signals = False
        self.constructing = False

    def set_slot_dropdown_index(self, slot_index: int, idx: int):
        if 0 <= slot_index < len(self.slot_dropdowns):
            self.slot_dropdowns[slot_index].dropdown_choices.setCurrentIndex(idx)

    def set_slot_axes_limits(self, slot_index: int, xlim, ylim):
        """ Update the axes for a given slot """
        if 0 <= slot_index < len(self.slot_axes_controls):
            self.slot_axes_controls[slot_index].set_limits(xlim, ylim)

    def set_slot_title(self, slot_index: int, title: str) -> None:
        if title is None or str(title).strip() == "":
            self.slot_titles.pop(slot_index, None)
        else:
            self.slot_titles[slot_index] = str(title)
        self.slotOptionsChanged.emit(slot_index)

    def _rebuild_slot_dropdowns(self, rows, cols, old_limits= None, old_dropdown_indices= None, old_checked= None, old_slot_settings= None):
        """ Destroy and rebuild all control widgets for individual plots (or build for the first time) """
        for i in reversed(range(self.slot_controls_layout.count())):
            item = self.slot_controls_layout.takeAt(i)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self.slot_dropdowns.clear()
        self.slot_options.clear()
        self.slot_axes_controls.clear()

        num_slots = rows * cols
        for slot_index in range(num_slots):

            # magic
            r = slot_index // cols
            c = slot_index % cols
            section_divider = SectionDivider(f"Axis ({r+1},{c+1})")
            # label = qw.QLabel(f"Plot ({r+1},{c+1}): ")
            # label.setMinimumWidth(70)

            dropdown = DropdownChoices()
            dropdown.setSizePolicy(qw.QSizePolicy.Policy.Expanding, qw.QSizePolicy.Policy.Fixed)
            dropdown.addItems(self.dropdown_choices)
            dropdown_layout = qw.QVBoxLayout()

            for dropdown_choice in self.plotting_data:
                choice_dict = self.plotting_data[dropdown_choice]
                plots = choice_dict["plots"]
                for plot in plots:
                    plot_dict = plots[plot]
                    if "checkbox_name" in plot_dict:
                        dropdown.add_checkbox(choice_dict["name"], plot_dict["checkbox_name"], plot_dict["toggled"])

            default_font = self._auto_fontsize(rows, cols)
            options_widget = SlotControlsWidget()
            options_widget.legend_size_spin.setValue(default_font)
            axes_widget = AxesControlWidget()

            if self.constructing: self._set_initial_plot_params(axes_widget)

            self.slot_controls_layout.addWidget(section_divider)
            self.slot_controls_layout.addWidget(SectionDivider("Settings", alignment= "left"))
            self.slot_controls_layout.addWidget(axes_widget)
            self.slot_controls_layout.addWidget(options_widget)
            self.slot_controls_layout.addWidget(dropdown)

            self.slot_dropdowns.append(dropdown)
            self.slot_options.append(options_widget)
            self.slot_axes_controls.append(axes_widget)

            dropdown.currentIndexChanged.connect(lambda idx, s=slot_index: self._on_dropdown_changed(idx, s))
            dropdown.checkStateChanged.connect(lambda s=slot_index: self.slotPlotChoiceChanged.emit(s))
            dropdown.infoBoxHovered.connect(lambda s=slot_index: self._on_info_hovered(s))
            options_widget.settingsChanged.connect(lambda s=slot_index: self.slotOptionsChanged.emit(s))
            axes_widget.settingsChanged.connect(lambda s=slot_index: self.slotAxesChanged.emit(s))
    
        if old_limits:
            for i, lims in enumerate(old_limits):
                if i >= len(self.slot_axes_controls):
                    break
                if lims is None:
                    continue
                xlim, ylim = lims
                if xlim is None or ylim is None:
                    continue
                self.slot_axes_controls[i].set_limits(xlim, ylim)

            last_valid = None
            for lims in reversed(old_limits):
                if lims is not None and lims[0] is not None and lims[1] is not None:
                    last_valid = lims
                    break

            if last_valid is not None:
                xlim0, ylim0 = last_valid
                for i in range(len(old_limits), len(self.slot_axes_controls)):
                    self.slot_axes_controls[i].set_limits(xlim0, ylim0)

        if old_dropdown_indices:
            # 1) Restore existing slots
            for i, idx in enumerate(old_dropdown_indices):
                if i >= len(self.slot_dropdowns):
                    break
                if idx is None or idx < 0:
                    continue
                self.slot_dropdowns[i].dropdown_choices.setCurrentIndex(idx)

            # 2) For new slots, copy from last existing choice
            last_idx = None
            for idx in reversed(old_dropdown_indices):
                if idx is not None and idx >= 0:
                    last_idx = idx
                    break

            if last_idx is not None:
                for i in range(len(old_dropdown_indices), len(self.slot_dropdowns)):
                    self.slot_dropdowns[i].dropdown_choices.setCurrentIndex(last_idx)

        # restore checkbox choices
        if old_checked:
            for i, checked in enumerate(old_checked):
                if i >= len(self.slot_dropdowns):
                    break
                if checked is None:
                    continue
                dd = self.slot_dropdowns[i]
                if hasattr(dd, "set_checked_boxes"):
                    try:
                        dd.set_checked_boxes(checked)
                        continue
                    except Exception:
                        pass
                # Otherwise, try common internal shapes: dict name->QCheckBox
                for attr_name in ("checkboxes", "check_boxes", "checkbox_widgets"):
                    box_map = getattr(dd, attr_name, None)
                    if isinstance(box_map, dict):
                        for name, box in box_map.items():
                            try:
                                box.blockSignals(True)
                                box.setChecked(name in checked)
                                box.blockSignals(False)
                            except Exception:
                                pass
                        break

            # for new slots, copy from last existing choice
            last_checked = None
            for checked in reversed(old_checked):
                if checked is not None:
                    last_checked = checked
                    break

            if last_checked is not None:
                for i in range(len(old_checked), len(self.slot_dropdowns)):
                    dd = self.slot_dropdowns[i]
                    if hasattr(dd, "set_checked_boxes"):
                        try:
                            dd.set_checked_boxes(last_checked)
                            continue
                        except Exception:
                            pass
                    for attr_name in ("checkboxes", "check_boxes", "checkbox_widgets"):
                        box_map = getattr(dd, attr_name, None)
                        if isinstance(box_map, dict):
                            for name, box in box_map.items():
                                try:
                                    box.blockSignals(True)
                                    box.setChecked(name in last_checked)
                                    box.blockSignals(False)
                                except Exception:
                                    pass
                            break

            if old_slot_settings:
                for i, settings in enumerate(old_slot_settings):
                    if i >= len(self.slot_options):
                        break
                    if settings is None:
                        continue

                    w = self.slot_options[i]
                    # Try a generic setter first if present
                    if hasattr(w, "set_settings"):
                        try:
                            w.set_settings(settings)
                            continue
                        except Exception:
                            pass

                    # Otherwise, set individual controls defensively
                    s = self._normalize_slot_settings(settings)

                    # legend
                    if hasattr(w, "legend_visible_check"):
                        try:
                            w.legend_visible_check.blockSignals(True)
                            w.legend_visible_check.setChecked(bool(s.get("legend_visible", True)))
                            w.legend_visible_check.blockSignals(False)
                        except Exception:
                            pass

                    if hasattr(w, "legend_size_spin"):
                        try:
                            w.legend_size_spin.blockSignals(True)
                            w.legend_size_spin.setValue(int(s.get("legend_fontsize", 10)))
                            w.legend_size_spin.blockSignals(False)
                        except Exception:
                            pass

                    if hasattr(w, "legend_loc_dropdown"):
                        try:
                            loc = s.get("legend_loc", "upper right")
                            combo = w.legend_loc_dropdown
                            combo.blockSignals(True)
                            # match by text
                            idx = combo.findText(str(loc))
                            if idx >= 0:
                                combo.setCurrentIndex(idx)
                            combo.blockSignals(False)
                        except Exception:
                            pass

                    # title/x/y toggles
                    for key, attr in (("title", "title_check"), ("xlabel", "xlabel_check"), ("ylabel", "ylabel_check")):
                        if hasattr(w, attr):
                            try:
                                cb = getattr(w, attr)
                                cb.blockSignals(True)
                                cb.setChecked(bool(s.get(key)))
                                cb.blockSignals(False)
                            except Exception:
                                pass


    def _on_info_hovered(self, slot_index: int):
        self.get_tooltip(slot_index)


    def _normalize_slot_settings(self, settings: dict) -> dict:
        if not settings:
            return {}

        out = dict(settings)

        # Legend settings
        if "visible" in out:
            out["legend_visible"] = out.pop("visible")

        if "fontsize" in out:
            out["legend_fontsize"] = out.pop("fontsize")

        if "loc" in out:
            out["legend_loc"] = out.pop("loc")

        return out

    def _on_dropdown_changed(self, idx: int, slot_index: int):
        self.get_tooltip(slot_index)
        self.slotPlotChoiceChanged.emit(slot_index)

    def _set_initial_plot_params(self, axes_widget):
        if "starting_lims" in self.demo["details"]:
            lims = self.demo["details"]["starting_lims"]
            try:
                xlim = tuple(lims[0])
                ylim = tuple(lims[1])
            except ValueError:
                return

            axes_widget.set_limits(xlim, ylim)

    def get_slot_axes_limits(self, slot_index: int):
        """ return (xlim, ylim) for a given slot """
        if slot_index < 0 or slot_index >= len(self.slot_axes_controls):
            return None
        return self.slot_axes_controls[slot_index].get_limits()

    def _emit_plot_dim_change(self):
        rows = self.rows_spinner.value()
        cols = self.cols_spinner.value()

        old_limits = [w.get_limits() for w in self.slot_axes_controls]
        old_dropdown_indices = [w.dropdown_choices.currentIndex() for w in self.slot_dropdowns]
        old_checked = [w.get_current_checked_boxes() for w in self.slot_dropdowns]
        old_slot_settings = []
        for w in self.slot_options:
            try:
                old_slot_settings.append(w.get_settings())
            except Exception:
                old_slot_settings.append(None)

        self.layoutChanged.emit(rows, cols)
        self._rebuild_slot_dropdowns(
            rows, cols, 
            old_limits= old_limits, 
            old_dropdown_indices= old_dropdown_indices,
            old_checked=old_checked,
            old_slot_settings=old_slot_settings
        )

        self.layoutChanged.emit(rows, cols)

    def make_widget(self, info, params):
        control_type = info["control_type"]

        if control_type == "checkbox":
            param_name, label, tooltip = (
                info["param_name"], info['label'], info['tooltip']
            )
            widget = qw.QCheckBox(label)
            widget.setToolTip(tooltip)
            widget.stateChanged.connect(self.update_plot)

        if control_type == "dropdown":
            outer_widget = qw.QWidget()
            # outer_widget.setSizePolicy(qw.QSizePolicy.Policy.Preferred, qw.QSizePolicy.Policy.Maximum)
            outer_layout = qw.QVBoxLayout(outer_widget)
            outer_layout.setContentsMargins(0, 0, 0, 0)
            outer_layout.setSpacing(2)

            param_name, label, names, values, tooltip_plain = (
                info["param_name"], info["label"], info["names"], info["values"], info["tooltip"]
            )
            tooltip = f"""{tooltip_plain}"""

            label_widget = qw.QLabel(label)
            label_widget.setSizePolicy(
                qw.QSizePolicy.Policy.Expanding,
                qw.QSizePolicy.Policy.Preferred
            )
            outer_layout.addWidget(label_widget, alignment = qc.Qt.AlignmentFlag.AlignCenter)

            top_row = qw.QWidget()
            # top_row.setSizePolicy(qw.QSizePolicy.Policy.Preferred, qw.QSizePolicy.Policy.Maximum)
            row_layout = qw.QHBoxLayout(top_row)
            row_layout.setContentsMargins(5, 0, 5, 0)
            row_layout.setSpacing(0)

            dropdown = qw.QComboBox()
            dropdown.setSizePolicy(qw.QSizePolicy.Policy.Expanding, qw.QSizePolicy.Policy.Fixed)

            def no_wheel(event):
                event.ignore()
            dropdown.wheelEvent = no_wheel

            for name in names:
                dropdown.addItem(name)

            init_val = getattr(params, param_name)
            dropdown.setCurrentIndex(values.index(init_val))
            dropdown.currentIndexChanged.connect(
                lambda idx, pn=param_name, vals=values: self.update_plot(pn, vals[idx])
            )

            row_layout.addWidget(dropdown, stretch=1, alignment= qc.Qt.AlignmentFlag.AlignTop)
            row_layout.addWidget(HelpButton("?", tooltip), stretch=0)

            outer_layout.addWidget(top_row, alignment= qc.Qt.AlignmentFlag.AlignTop)

            self.dropdowns[param_name] = {"widget": dropdown, "values": values}
            return outer_widget
        
        elif control_type == "button_group":
            widget = qw.QWidget()
            if info["display"] == "horizontal":
                button_layout = qw.QHBoxLayout(widget)
            else:
                button_layout = qw.QVBoxLayout(widget)
            
            names, functions = info["names"], info["functions"]
            for i,name in enumerate(names):
                button = qw.QPushButton(name)

                extra_functions_module = importlib.import_module(f"models.{self.sim_model}.simulation.extra_functions")

                functions_dict = dict(inspect.getmembers(extra_functions_module, inspect.isfunction))
                try:
                    function = functions_dict[functions[i]]
                    def outer_func(_checked= False):
                        new_params = None
                        sector_names = None
                        try:
                            new_params, sector_names = function(self.params)
                        except Exception as e:
                            print(f"Error: {e}")

                        if new_params is None: return

                        self.params = new_params
                        self.load_new_params(new_params)
                        self.paramsReplaced.emit((new_params, sector_names))

                        # self.load_new_params(output)

                    button.clicked.connect(outer_func)
                    button_layout.addWidget(button)

                except ValueError:
                    print(f"Error loading function: {functions[i]}. Skipping button")
                    continue

                button_layout.addWidget(button)
                
            return widget

        elif control_type == "entry_block": 
            param_name, label, tooltip_plain = info["param_name"], info["label"], info["tooltip"]
            tooltip = f"""{tooltip_plain}"""
            init_val = getattr(params, param_name)
            # print(getattr(params, param_name))

            if info["type"] == "scalar":
                scalar_range, scalar_type = tuple(info["range"]), info["scalar_type"]
                widget = EntryBlock(param_name, label, scalar_range, init_val, tooltip, scalar_type)
                self.entry_blocks[param_name] = {"widget": widget, "is_matrix": False}
                widget.setSizePolicy(qw.QSizePolicy.Policy.Expanding, qw.QSizePolicy.Policy.Fixed)
                widget.valueChanged.connect(self.update_plot)

            elif info["type"] == "matrix":
                dim = tuple(info["dim"])
                widget = MatrixEntry(param_name, label, dim, init_val, tooltip)
                widget.textChanged.connect(self.update_plot)
                if "vsize_policy" in info:
                    if info["vsize_policy"] == "expanding":
                        widget.setSizePolicy(qw.QSizePolicy.Policy.Expanding, qw.QSizePolicy.Policy.Minimum)
                else:
                    widget.setSizePolicy(qw.QSizePolicy.Policy.Expanding, qw.QSizePolicy.Policy.Preferred)
                self.entry_blocks[param_name] = {"widget": widget, "is_matrix": True}
                
            elif info["type"] == "vector":
                dim1 = info["dim"]
                dim = (dim1, 1)
                widget = MatrixEntry(param_name, label, dim, init_val.reshape(-1,1), tooltip)
                widget.textChanged.connect(self.update_plot)
                widget.setSizePolicy(qw.QSizePolicy.Policy.Expanding, qw.QSizePolicy.Policy.Fixed)
                self.entry_blocks[param_name] = {"widget": widget, "is_matrix": True}
            else:
                print(f"Unrecognized type: {info["type"]}. Options for type are scalar, vector, and matrix.")
                return qw.QWidget()

            return widget

        elif control_type[0:10] == "vsub_panel":

            widget = qw.QWidget()
            vlayout = qw.QVBoxLayout(widget)
            widget.setSizePolicy(qw.QSizePolicy.Policy.Expanding, qw.QSizePolicy.Policy.Fixed)

            subentries = info["entries"]
            for entry in subentries:
                subinfo = subentries[entry]
                subwidget = self.make_widget(subinfo, params)
                vlayout.addWidget(subwidget)

            return widget

        elif control_type[0:10] == "hsub_panel":
            widget = qw.QWidget()
            widget.setSizePolicy(qw.QSizePolicy.Policy.Expanding, qw.QSizePolicy.Policy.Fixed)
            vlayout = qw.QHBoxLayout(widget)

            subentries = info["entries"]
            for entry in subentries:
                subinfo = subentries[entry]
                subwidget = self.make_widget(subinfo, params)
                vlayout.addWidget(subwidget)

            return widget

        else:
            print("Unrecognized control type.")
            return qw.QWidget()

    def _auto_fontsize(self, rows: int, cols: int) -> int:
        font_vals = {
            (1,1): 10, (1,2): 8, (1,3): 6,
            (2,1): 8,  (2,2): 8, (3,2): 6,
            (2,3): 6,  (3,3): 0
        }
        return font_vals.get((rows, cols), 10)
    
    def get_slot_config(self, slot_index: int):
        """ Return the current dropdown index, checkbox options, and legend settings for a slot """
        if slot_index < 0 or slot_index >= len(self.slot_dropdowns): return None

        plot_widget = self.slot_dropdowns[slot_index]
        dropdown_index = plot_widget.dropdown_choices.currentIndex()
        options = plot_widget.get_current_checked_boxes()

        if 0 <= slot_index < len(self.slot_options):
            raw_settings = self.slot_options[slot_index].get_settings()
            slot_settings = self._normalize_slot_settings(raw_settings)
        else:
            slot_settings = {"legend_visible": True, "legend_fontsize": 10, "legend_loc": "upper right"}

        return dropdown_index, options, slot_settings

    def get_data(self, index):
        data = {}
        for widget in self.entry_blocks:
            data[widget.name] = widget.get()
        return data

    def update_plot(self, name, new_val):
        if not self.block_signals:
            self.paramChanged.emit(name, new_val)

    def get_tooltip(self, slot_index: int= 0) -> str:
        """ When user hovers their mouse on the tooltip button by a dropdown menu of plots,
            the DropdownChoices widget emits an infoBoxHovered signal to the ControlPanel, which
            calls this function to return the string which is given as input to the 
            DropdownChoices.setToolTip method for displaying. 
        """
        if not (0 <= slot_index < len(self.slot_dropdowns)):
            return "No notes"

        wrapper = self.slot_dropdowns[slot_index]
        text = wrapper.dropdown_choices.currentText()
        tooltip_plain = self.dropdown_tooltips.get(text, "No notes")
        tooltip = f"""{tooltip_plain}"""

        wrapper.info.setToolTip(tooltip)

        wrapper.setToolTip(tooltip)
        return tooltip

    def load_new_params(self, params):
        self.block_signals = True
        params_dict = asdict(params)
        for param in params_dict:
            if param in self.entry_blocks:
                widget_info = self.entry_blocks[param]
                widget = widget_info["widget"]
                value = params_dict[param]

                if widget_info["is_matrix"]:
                    widget.blockSignals(True)
                    widget.change_values(value)
                    widget.blockSignals(False)
                else:
                    try:
                        v_float = float(value)
                        text = f"{v_float:.8g}"   
                    except (TypeError, ValueError):
                        text = str(value)

                    widget.entry.blockSignals(True)
                    widget.entry.setText(text)
                    widget.entry.blockSignals(False)

            if param in self.dropdowns:
                info = self.dropdowns[param]
                dropdown = info["widget"]
                values = info["values"]
                new_val = params_dict[param]

                try:
                    idx = values.index(new_val)
                except ValueError:
                    continue

                dropdown.blockSignals(True)
                dropdown.setCurrentIndex(idx)
                dropdown.blockSignals(False)

        self.block_signals = False

