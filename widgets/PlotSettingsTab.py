from __future__ import annotations
from pathlib import Path
from paths import rpath
import os
import yaml
import copy
from widgets.common import make_shortname
from tools.modelling_tools import FlowSeq, flowseq_representer
import logging
from .common import atomic_write

from PyQt6 import (
    QtWidgets as qw,
    QtCore as qc,
    QtGui as qg
)

yaml.add_representer(FlowSeq, flowseq_representer, Dumper=yaml.SafeDumper)
logger = logging.getLogger(__name__)

def list_subdirs(path):
    return [
            p.name
            for p in Path(path).iterdir()
            if p.is_dir()
        ]

def _safe_load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

class ColorLineEdit(qw.QLineEdit):
    def set_hex(self, hex_color: str) -> None:
        hex_color = (hex_color or "").strip()
        self.setText(hex_color)
        self._update_swatch()

    def _update_swatch(self) -> None:
        txt = self.text().strip()
        c = qg.QColor(txt)
        if c.isValid():
            self.setStyleSheet(f"QLineEdit {{ background-color: {c.name()}; }}")
        else:
            self.setStyleSheet("")

class LabelColorRow(qw.QWidget):
    removed = qc.pyqtSignal(object)

    def __init__(self, label: str = "", color: str = "", parent=None):
        super().__init__(parent)
        lay = qw.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.label_edit = qw.QLineEdit(label)
        self.label_edit.setPlaceholderText("Label")

        self.color_edit = ColorLineEdit()
        self.color_edit.setPlaceholderText("#RRGGBB")
        self.color_edit.setMaximumWidth(120)
        self.color_edit.set_hex(color)

        self.pick_btn = qw.QToolButton()
        self.pick_btn.setText("🎨")
        self.pick_btn.setToolTip("Pick a color")

        self.del_btn = qw.QToolButton()
        self.del_btn.setText("✕")
        self.del_btn.setToolTip("Remove this row")

        lay.addWidget(self.label_edit, 1)
        lay.addWidget(self.color_edit, 0)
        lay.addWidget(self.pick_btn, 0)
        lay.addWidget(self.del_btn, 0)

        self.pick_btn.clicked.connect(self._pick_color)
        self.del_btn.clicked.connect(lambda: self.removed.emit(self))
        self.color_edit.textChanged.connect(self.color_edit._update_swatch)

    def _pick_color(self) -> None:
        initial = qg.QColor(self.color_edit.text().strip())
        c = qw.QColorDialog.getColor(initial, self, "Choose color")
        if c.isValid():
            self.color_edit.set_hex(c.name())

    def get_pair(self) -> tuple[str, str]:
        return (self.label_edit.text().strip(), self.color_edit.text().strip())

class LabelColorListEditor(qw.QWidget):
    changed = qc.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = qw.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.rows_layout = qw.QVBoxLayout()
        self.rows_layout.setSpacing(6)
        root.addLayout(self.rows_layout)

        btn_row = qw.QHBoxLayout()
        self.add_btn = qw.QPushButton("+ Add series")
        btn_row.addWidget(self.add_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self.add_btn.clicked.connect(self.add_row)
        root.addStretch(1)

    def clear_rows(self) -> None:
        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _emit_changed(self, *args) -> None:
        # Emit a single 'changed' signal for any modification, unless we're in a load
        if self.signalsBlocked():
            return
        self.changed.emit()

    def add_row(self, stupid_truth_val= False, label: str = "", color: str = "") -> None:
        row = LabelColorRow(label=label, color=color)
        row.removed.connect(self._remove_row)

        # Any field edits should mark the list as changed
        row.label_edit.textChanged.connect(self._emit_changed)
        row.color_edit.textChanged.connect(self._emit_changed)

        self.rows_layout.addWidget(row)
        self._emit_changed()

    def _remove_row(self, row_widget: LabelColorRow) -> None:
        row_widget.setParent(None)
        row_widget.deleteLater()
        self._emit_changed()

    def set_pairs(self, labels: list[str], colors: list[str]) -> None:
        self.clear_rows()
        n = max(len(labels), len(colors))
        for i in range(n):
            self.add_row(
                label=labels[i] if i < len(labels) else "",
                color=colors[i] if i < len(colors) else "",
            )
        if n == 0:
            self.add_row()

    def get_pairs(self) -> tuple[list[str], list[str]]:
        labels: list[str] = []
        colors: list[str] = []
        for i in range(self.rows_layout.count()):
            w = self.rows_layout.itemAt(i).widget()
            if isinstance(w, LabelColorRow):
                lab, col = w.get_pair()
                if lab == "" and col == "":
                    continue
                labels.append(lab)
                colors.append(col)
        return labels, colors

class PlotSettingsTab(qw.QWidget):
    ROLE = qc.Qt.ItemDataRole.UserRole  # store payload tuples here

    def __init__(self, model= None, parent=None):
        super().__init__(parent)

        root = qw.QVBoxLayout(self)
        self.window = self.window()

        # --- Model selection row ---
        top = qw.QHBoxLayout()
        top.addWidget(qw.QLabel("Model:"))
        self.model_combo = qw.QComboBox()
        self.model_combo.setMinimumWidth(260)
        top.addWidget(self.model_combo, 1)

        self.reload_btn = qw.QPushButton("Reload")
        top.addWidget(self.reload_btn, 0)
        root.addLayout(top)

        # --- Main splitter: [tree] | [editor] ---
        splitter = qw.QSplitter(qc.Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # ========= Left column: Category/Plot Tree =========
        tree_panel = qw.QWidget()
        tree_layout = qw.QVBoxLayout(tree_panel)
        tree_layout.addWidget(qw.QLabel("Categories / Plots"))

        self.tree = qw.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(qw.QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setUniformRowHeights(True)
        self.tree.setDragDropMode(qw.QAbstractItemView.DragDropMode.InternalMove)
        self.tree.setDefaultDropAction(qc.Qt.DropAction.MoveAction)
        self.tree.setDropIndicatorShown(True)
        self.tree.model().rowsMoved.connect(self._on_tree_rows_moved)
        self.tree.setExpandsOnDoubleClick(True)
        tree_layout.addWidget(self.tree, 1)

        btns = qw.QVBoxLayout()
        btns.setContentsMargins(0,0,0,0)
        btns_top = qw.QHBoxLayout()
        btns_bot = qw.QHBoxLayout()
        self.add_cat_btn = qw.QPushButton("+ Category")
        self.add_plot_btn = qw.QPushButton("+ Plot")
        self.del_cat_btn = qw.QPushButton("Delete")

        self.dup_plot_btn = qw.QPushButton("Duplicate in")
        self.dup_plot_dropdown = qw.QComboBox()

        self.add_cat_btn.clicked.connect(self._on_add_cat)
        self.add_plot_btn.clicked.connect(self._on_add_plot)
        self.del_cat_btn.clicked.connect(self._on_del_clicked)
        self.dup_plot_btn.clicked.connect(self._on_dup_plot_clicked)

        btns_top.addWidget(self.add_cat_btn)
        btns_top.addWidget(self.add_plot_btn)
        btns_top.addWidget(self.del_cat_btn)

        btns_bot.addWidget(self.dup_plot_btn)
        btns_bot.addWidget(self.dup_plot_dropdown, 1)
        btns.addLayout(btns_top)
        btns.addLayout(btns_bot)
        tree_layout.addLayout(btns)

        splitter.addWidget(tree_panel)

        # ========= Right column: Editor (unchanged) =========
        editor_panel = qw.QWidget()
        editor_layout = qw.QVBoxLayout(editor_panel)

        self.form = qw.QFormLayout()
        self.form.setLabelAlignment(qc.Qt.AlignmentFlag.AlignRight)

        self.lbl_internal_name = qw.QLabel()
        self.name_edit = qw.QLineEdit()
        self.toggled_check = qw.QCheckBox("Initially toggled")
        self.name_edit.textChanged.connect(self._update_internal_name)

        self.form.addRow("Name:", self.name_edit)
        self.form.addRow("Internal Name:", self.lbl_internal_name)
        self.form.addRow("", self.toggled_check)

        self.plot_type_combo = qw.QComboBox()
        self.plot_type_combo.addItems(["Curve", "Histogram", "Scatter"])
        self.form.addRow("Plot type:", self.plot_type_combo)

        editor_layout.addLayout(self.form)

        self.type_stack = qw.QStackedWidget()
        self.type_stack.addWidget(self._build_cat_panel()) # 0 
        self.type_stack.addWidget(self._build_curve_panel()) # 1
        self.type_stack.addWidget(self._build_hist_panel()) # 2
        self.type_stack.addWidget(self._build_scatter_panel()) # 3
        self.type_stack_dict = {0: "cat", 1: "curve", 2: "hist", 3: "scatter"}
        editor_layout.addWidget(self.type_stack, 1)

        action_row = qw.QHBoxLayout()
        action_row.addStretch(1)
        self.save_changes_btn = qw.QPushButton("Save Changes")
        self.save_changes_btn.setVisible(False)
        action_row.addWidget(self.save_changes_btn)
        self.save_changes_btn.clicked.connect(self._save_changes)
        editor_layout.addLayout(action_row)

        splitter.addWidget(editor_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # --- state ---
        self._current_model: str | None = None
        self._plotting_data: dict = {}

        self._loading_editor: bool = False

        # --- wiring ---
        self.reload_btn.clicked.connect(self.reload_current_model)

        self.tree.currentItemChanged.connect(self._on_tree_selection_changed)

        self.plot_type_combo.currentIndexChanged.connect(self._on_plot_type_changed)

        self._refresh_models()
        self._current_model = self.model_combo.currentText().strip() or None
        if model is not None:
            models = [self.model_combo.itemText(i) for i in range(self.model_combo.count())]
            self.model_combo.setCurrentIndex(models.index(model))
            self._current_model = model

        self._working_plot_data = {}
        self._original_plot_data = {}
        if not self._current_model:
            return
        try:
            with open(rpath("models", self._current_model, "data", "plotting_data.yml"), "r") as f:
                data = yaml.safe_load(f) or {}
            self._working_plot_data[self._current_model] = copy.deepcopy(data)
            self._original_plot_data[self._current_model] = copy.deepcopy(data)
        except FileNotFoundError as e:
            self.window.status.show("plotting_data.yml not found. Creating one automatically.")
            try:
                with open(rpath("models", self._current_model, "data", "plotting_data.yml"), "w") as f:
                    pass
            except FileNotFoundError as e:
                self.window.status.show(f"Error opening the data folder of {self._current_model}. Have you ran the new model creation wizard yet?")
            self._original_plot_data[self._current_model] = {}
            self._working_plot_data[self._current_model] = {}
        except Exception as e:
            self.window.status.show(f"Error opening plotting_data.yml: {e}")
            self._original_plot_data[self._current_model] = {}
            self._working_plot_data[self._current_model] = {}

        self.model_combo.currentIndexChanged.connect(self._on_model_changed)

        self._wire_autosave_signals()
        self._refresh_tree()

    def _update_internal_name(self, text):
        self.lbl_internal_name.setText(make_shortname(text))

    def _block_editor_signals(self, block: bool) -> None:
        widgets = [
            self.name_edit, self.toggled_check, self.plot_type_combo,
            self.cat_title, self.x_label, self.y_label, self.tooltip,
            self.curve_traj_key, self.curve_linestyle, self.curve_series_editor,
            self.hist_traj_key, self.hist_bins, self.hist_density, self.hist_color,
            self.scatter_traj_key_x, self.scatter_traj_key_y, self.scatter_label, self.scatter_color,
        ]
        for w in widgets:
            try:
                w.blockSignals(block)
            except Exception:
                pass

    def _wire_autosave_signals(self) -> None:
        # General
        self.name_edit.textChanged.connect(self._save_changes)
        self.toggled_check.stateChanged.connect(self._save_changes)

        # Category
        self.cat_title.textChanged.connect(self._save_changes)
        self.x_label.textChanged.connect(self._save_changes)
        self.y_label.textChanged.connect(self._save_changes)
        self.tooltip.textChanged.connect(self._save_changes)

        # Curve
        self.curve_traj_key.textChanged.connect(self._save_changes)
        self.curve_linestyle.currentIndexChanged.connect(self._save_changes)
        self.curve_series_editor.changed.connect(self._save_changes)

        # Histogram
        self.hist_traj_key.textChanged.connect(self._save_changes)
        self.hist_bins.valueChanged.connect(self._save_changes)
        self.hist_density.stateChanged.connect(self._save_changes)
        self.hist_color.textChanged.connect(self._save_changes)

        # Scatter
        self.scatter_traj_key_x.textChanged.connect(self._save_changes)
        self.scatter_traj_key_y.textChanged.connect(self._save_changes)
        self.scatter_label.textChanged.connect(self._save_changes)
        self.scatter_color.textChanged.connect(self._save_changes)

    def _on_plot_type_changed(self, idx: int) -> None:
        stack_idx = {0: 1, 1: 2, 2: 3}.get(idx, 1)
        self.type_stack.setCurrentIndex(stack_idx)
        self._save_changes()

    def _save_changes(self):
        if self._loading_editor:
            return
        if not self._current_model:
            return
        item = self.tree.currentItem()
        if item is None:
            return
        data = item.data(0, self.ROLE)
        if not data:
            return

        res = self._get_plot_data()
        if not res:
            return
        inter_name, new_data = res
        inter_name = (inter_name or "").strip()
        if not inter_name:
            return

        if data[0] == "category":
            old_cat_name = data[1]

            model_dict = self._working_plot_data[self._current_model]

            # Preserve plots + metadata order
            self._replace_key_preserve_order(model_dict, old_cat_name, inter_name, new_data)

            self._refresh_tree()
            self._select_cat(inter_name)
            return

        plot_type = self.plot_type_combo.currentText()
        old_cat_name = data[1]
        old_plot_name = data[2]
        plots_dict = self._working_plot_data[self._current_model][old_cat_name]["plots"]

        if inter_name == old_plot_name:
            plots_dict[old_plot_name] = new_data

            display = (new_data or {}).get("checkbox_name") or old_plot_name
            if item.text(0) != display:
                item.setText(0, display)
            return

        self._replace_key_preserve_order(plots_dict, old_plot_name, inter_name, new_data)
        self._refresh_tree()
        self._select_plot(old_cat_name, inter_name)

    def _on_add_cat(self):
        self._working_plot_data[self._current_model]["new_category"] = {
            "name": "New Category",
            "tooltip": None,
            "plots": {}
        }
        self._refresh_tree()
        self._select_cat("new_category")

    def _replace_key_preserve_order(self, d: dict, old_key: str, new_key: str, new_val) -> None:
        """Replace old_key with new_key (and new_val) keeping existing iteration order."""
        if old_key == new_key:
            d[old_key] = new_val
            return

        new_d = {}
        for k, v in d.items():
            if k == old_key:
                new_d[new_key] = new_val
            else:
                new_d[k] = v
        d.clear()
        d.update(new_d)

    def _on_dup_plot_clicked(self) -> None:
        item = self.tree.currentItem()
        if not item:
            return

        payload = item.data(0, self.ROLE)
        if not payload or payload[0] != "plot":
            return

        _, src_cat, src_plot = payload
        target_cat = self.dup_plot_dropdown.currentData()
        if not target_cat:
            return

        model = self._current_model
        model_dict = self._working_plot_data.get(model)
        if not model_dict:
            return

        src_plots = model_dict[src_cat]["plots"]
        if src_plot not in src_plots:
            return

        # --- deep copy the plot spec ---
        import copy
        new_plot = copy.deepcopy(src_plots[src_plot])

        # --- generate a unique internal name ---
        base = src_plot
        plots_in_target = model_dict[target_cat]["plots"]

        if base not in plots_in_target:
            new_key = base
        else:
            i = 2
            while f"{base}_{i}" in plots_in_target:
                i += 1
            new_key = f"{base}_{i}"

        # --- adjust display name if present ---
        if "checkbox_name" in new_plot:
            new_plot["checkbox_name"] = f'{new_plot["checkbox_name"]}'

        # --- insert ---
        plots_in_target[new_key] = new_plot

        # --- refresh + select ---
        self._refresh_tree()
        self._select_plot(target_cat, new_key)

    def _refresh_dup_targets(self) -> None:
        self.dup_plot_dropdown.blockSignals(True)
        self.dup_plot_dropdown.clear()

        # No model or no data → disable
        if not self._current_model:
            self.dup_plot_dropdown.setEnabled(False)
            self.dup_plot_btn.setEnabled(False)
            self.dup_plot_dropdown.blockSignals(False)
            return

        item = self.tree.currentItem()
        if not item:
            self.dup_plot_dropdown.setEnabled(False)
            self.dup_plot_btn.setEnabled(False)
            self.dup_plot_dropdown.blockSignals(False)
            return

        payload = item.data(0, self.ROLE)
        if not payload or payload[0] != "plot":
            # Only meaningful for plots
            self.dup_plot_dropdown.setEnabled(False)
            self.dup_plot_btn.setEnabled(False)
            self.dup_plot_dropdown.blockSignals(False)
            return

        _, src_cat, _ = payload

        categories = self._working_plot_data.get(self._current_model, {})
        for cat_key, cat in categories.items():
            if cat_key == src_cat:
                continue  # don’t duplicate into same category
            display = cat.get("name") or cat_key
            self.dup_plot_dropdown.addItem(display, cat_key)

        enabled = self.dup_plot_dropdown.count() > 0
        self.dup_plot_dropdown.setEnabled(enabled)
        self.dup_plot_btn.setEnabled(enabled)

        self.dup_plot_dropdown.blockSignals(False)

    def _on_add_plot(self):
        current_sel = self.tree.currentItem()
        if current_sel is None:
            self.window.status.show("Please first select/create a category which the plot will belong to.", 4000)
            return
        data = current_sel.data(0, self.ROLE)
        if data[0] == "category":
            cat_name = data[1]
        else:
            parent_item = current_sel.parent()
            if parent_item is None:
                self.window.status.show("Error: Selected plot doesn't belong to a category? Is this even possible?", 4000)
                return
            par_data = parent_item.data(0, self.ROLE)
            cat_name = par_data[1]
        self._working_plot_data[self._current_model][cat_name]["plots"]["new_plot"] = {
            "checkbox_name": "New Plot",
            "labels": [""],
        }
        self._refresh_tree()
        self._select_plot(cat_name, "new_plot")

    def _select_cat(self, category_name):
        for i in range(self.tree.topLevelItemCount()):
            category_item = self.tree.topLevelItem(i)
            data = category_item.data(0, self.ROLE)

            if data[1] != category_name:
                continue

            self.tree.setCurrentItem(category_item)

    def _select_plot(self, category_name, plot_name):
        for i in range(self.tree.topLevelItemCount()):
            category_item = self.tree.topLevelItem(i)
            data = category_item.data(0, self.ROLE)

            if data[1] != category_name:
                continue

            self.tree.expandItem(category_item)

            for j in range(category_item.childCount()):
                plot_item = category_item.child(j)
                plot_payload = plot_item.data(0, self.ROLE)

                if plot_payload[2] == plot_name:
                    self.tree.setCurrentItem(plot_item)
                    return

    def _on_del_clicked(self):
        data = self.tree.currentItem().data(0, self.ROLE)
        if data[0] == "plot":
            cat_name, plot_name = data[1], data[2]
            del self._working_plot_data[self._current_model][cat_name]["plots"][plot_name]
        if data[0] == "category":
            cat_name = data[1]
            del self._working_plot_data[self._current_model][cat_name]
        self._refresh_tree()

    # ---------- build type panels ----------
    def _build_cat_panel(self) -> qw.QWidget:
        w = qw.QWidget()
        layout = qw.QVBoxLayout(w)

        form_widget = qw.QWidget()
        form_layout = qw.QFormLayout(form_widget)

        self.cat_title = qw.QLineEdit()
        self.x_label = qw.QLineEdit()
        self.y_label = qw.QLineEdit()
        self.tooltip = qw.QTextEdit()
        self.hint = qw.QLabel(
            "Hint: The only necessary field here is the title. \n"
       "All others can be left blank. x-axis defaults to 'Time [s]'. \n"
            "The other fields default to nothing."
        )
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("opacity: 0.85;")

        form_layout.addRow("Title: ", self.cat_title)
        form_layout.addRow("x-Axis Label: ", self.x_label)
        form_layout.addRow("y-Axis Label: ", self.y_label)
        form_layout.addRow("Tooltip Info: ", self.tooltip)

        layout.addWidget(form_widget, stretch= 1)
        layout.addWidget(self.hint, stretch= 0)

        return w

    def _build_curve_panel(self) -> qw.QWidget:
        w = qw.QWidget()
        layout = qw.QFormLayout(w)

        self.curve_traj_key = qw.QLineEdit()
        self.curve_linestyle = qw.QComboBox()
        self.curve_linestyle.addItem("Solid", "solid")
        self.curve_linestyle.addItem("Dashed", "dashed")
        self.curve_linestyle.addItem("Dotted", "dotted")

        self.curve_series_editor = LabelColorListEditor()

        # autosave wiring is done in _wire_autosave_signals

        layout.addRow("Trajectory Key:", self.curve_traj_key)
        layout.addRow("Line Style:", self.curve_linestyle)
        layout.addRow("Curves (label + color):", self.curve_series_editor)
        return w

    def _build_hist_panel(self) -> qw.QWidget:
        w = qw.QWidget()
        layout = qw.QFormLayout(w)

        self.hist_traj_key = qw.QLineEdit()
        self.hist_bins = qw.QSpinBox()
        self.hist_bins.setRange(1, 10_000)
        self.hist_bins.setValue(50)

        self.hist_density = qw.QCheckBox("Density (normalize)")
        self.hist_color = qw.QLineEdit()
        self.hist_color.setPlaceholderText("e.g. blue or #3366ff")

        layout.addRow("traj_key:", self.hist_traj_key)
        layout.addRow("bins:", self.hist_bins)
        layout.addRow("", self.hist_density)
        layout.addRow("color:", self.hist_color)
        return w

    def _build_scatter_panel(self) -> qw.QWidget:
        w = qw.QWidget()
        layout = qw.QFormLayout(w)

        self.scatter_traj_key_x = qw.QLineEdit()
        self.scatter_traj_key_y = qw.QLineEdit()
        self.scatter_label = qw.QLineEdit()
        self.scatter_color = qw.QLineEdit()
        self.scatter_color.setPlaceholderText("e.g. orange or #ff8800")

        layout.addRow("x-Axis Key:", self.scatter_traj_key_x)
        layout.addRow("y-Axis Key:", self.scatter_traj_key_y)
        layout.addRow("Label:", self.scatter_label)
        layout.addRow("Color:", self.scatter_color)
        return w

    # ---------- model loading ----------
    def _refresh_models(self) -> None:
        models = list_subdirs(rpath("models"))
        self.model_combo.clear()
        self.model_combo.addItems(models)

    def reload_current_model(self) -> None:
        if not self._current_model:
            return
        self._get_new_plotting_data(self._current_model)
        self._refresh_tree()
        self._refresh_dup_targets()

    def _on_model_changed(self) -> None:
        self._current_model = self.model_combo.currentText().strip() or None
        if not self._current_model:
            return
        if self._current_model not in self._original_plot_data:
            self._get_new_plotting_data(self._current_model)
        self._refresh_tree()
        self._refresh_dup_targets()

    def _get_new_plotting_data(self, model: str) -> None:
        try:
            with open(rpath("models", model, "data", "plotting_data.yml"), "r") as f:
                data = yaml.safe_load(f) or {}
            self._original_plot_data[self._current_model] = copy.deepcopy(data)
            self._working_plot_data[self._current_model] = copy.deepcopy(data)
        except FileNotFoundError as e:
            self.window.status.show("plotting_data.yml not found. Creating one automatically.")
            try:
                with open(rpath("models", model, "data", "plotting_data.yml"), "w") as f:
                    pass
            except FileNotFoundError as e:
                self.window.status.show(f"Error opening the data folder of {self._current_model}. Have you ran the new model creation wizard yet?")
            self._original_plot_data[self._current_model] = {}
            self._working_plot_data[self._current_model] = {}
            return
        except Exception as e:
            self.window.status.show(f"Error opening plotting_data.yml: {e}")
            self._original_plot_data[self._current_model] = {}
            self._working_plot_data[self._current_model] = {}
            return

    # ---------- tree refresh + selection ----------
    def _refresh_tree(self) -> None:
        selected = self._selected_payload()

        self.tree.blockSignals(True)
        self.tree.clear()

        for cat_key, cat in (self._working_plot_data[self._current_model] or {}).items():
            cat_display = (cat or {}).get("name") or cat_key
            cat_item = qw.QTreeWidgetItem([cat_display])
            cat_item.setFlags(cat_item.flags()
                              | qc.Qt.ItemFlag.ItemIsDragEnabled
                              | qc.Qt.ItemFlag.ItemIsDropEnabled)
            cat_item.setData(0, self.ROLE, ("category", cat_key))
            self.tree.addTopLevelItem(cat_item)

            plots = (cat or {}).get("plots") or {}
            for plot_key, plot_spec in plots.items():
                plot_display = (plot_spec or {}).get("checkbox_name") or plot_key
                plot_item = qw.QTreeWidgetItem([plot_display])
                plot_item.setFlags(plot_item.flags()
                                   | qc.Qt.ItemFlag.ItemIsDragEnabled)
                plot_item.setData(0, self.ROLE, ("plot", cat_key, plot_key))
                cat_item.addChild(plot_item)

            cat_item.setExpanded(True)

        self.tree.blockSignals(False)

        # restore selection if possible
        if selected:
            found = self._select_payload(selected)
            if found:
                return

        # else select first plot if exists, else first category
        if self.tree.topLevelItemCount() > 0:
            first_cat = self.tree.topLevelItem(0)
            if first_cat.childCount() > 0:
                self.tree.setCurrentItem(first_cat.child(0))
            else:
                self.tree.setCurrentItem(first_cat)
        else:
            self._clear_editor()

        self._refresh_dup_targets()

    def _on_tree_rows_moved(self, *args) -> None:
        try:
            self._rebuild_plot_data_from_tree()
        except Exception as e:
            return
        self._refresh_tree()

    def _rebuild_plot_data_from_tree(self) -> None:
        model = self._current_model
        if not model:
            return

        old_data = self._working_plot_data.get(model, {})
        new_data = {}

        for i in range(self.tree.topLevelItemCount()):
            cat_item = self.tree.topLevelItem(i)
            cat_payload = cat_item.data(0, self.ROLE)
            if not cat_payload or cat_payload[0] != "category":
                continue

            _, cat_key = cat_payload
            old_cat = old_data.get(cat_key, {})
            new_cat = {}

            # preserve category metadata
            for k in ("name", "title", "tooltip", "x_label", "y_label"):
                if k in old_cat:
                    new_cat[k] = old_cat[k]

            new_cat["plots"] = {}

            for j in range(cat_item.childCount()):
                plot_item = cat_item.child(j)
                plot_payload = plot_item.data(0, self.ROLE)
                if not plot_payload or plot_payload[0] != "plot":
                    continue

                _, old_cat_key, plot_key = plot_payload
                try:
                    new_cat["plots"][plot_key] = copy.deepcopy(
                        old_data[old_cat_key]["plots"][plot_key]
                    )
                except KeyError:
                    pass

            new_data[cat_key] = new_cat

        self._working_plot_data[model] = new_data

    def _selected_payload(self):
        it = self.tree.currentItem()
        return it.data(0, self.ROLE) if it else None

    def _select_payload(self, payload) -> bool:
        """Find and select a node matching payload."""
        kind = payload[0]
        for i in range(self.tree.topLevelItemCount()):
            cat_item = self.tree.topLevelItem(i)
            cat_payload = cat_item.data(0, self.ROLE)
            if not cat_payload:
                continue

            if kind == "category" and cat_payload == payload:
                self.tree.setCurrentItem(cat_item)
                return True

            # search children for plot
            for j in range(cat_item.childCount()):
                ch = cat_item.child(j)
                if ch.data(0, self.ROLE) == payload:
                    self.tree.setCurrentItem(ch)
                    return True
        return False

    def _on_tree_selection_changed(self, current, previous) -> None:
        if not current:
            self._clear_editor()
            return

        payload = current.data(0, self.ROLE)
        if not payload:
            self._clear_editor()
            return

        kind = payload[0]
        if kind == "category":
            self._clear_editor()
            self.form.setRowVisible(self.toggled_check, False)
            self.form.setRowVisible(self.plot_type_combo, False)

            name = self._working_plot_data[self._current_model][payload[1]].get("name", "")
            title = self._working_plot_data[self._current_model][payload[1]].get("title", "")
            x_label = self._working_plot_data[self._current_model][payload[1]].get("x_label", "")
            y_label = self._working_plot_data[self._current_model][payload[1]].get("y_label", "")
            tooltip = self._working_plot_data[self._current_model][payload[1]].get("tooltip", "")

            self._loading_editor = True
            self._block_editor_signals(True)
            try:
                self.name_edit.setText(name)
                self.lbl_internal_name.setText(payload[1])
                self.cat_title.setText(title)
                self.x_label.setText(x_label)
                self.y_label.setText(y_label)
                self.tooltip.setText(tooltip)
                self.type_stack.setCurrentIndex(0)
            finally:
                self._block_editor_signals(False)
                self._loading_editor = False
            return

        # plot
        self.form.setRowVisible(self.toggled_check, True)
        self.form.setRowVisible(self.plot_type_combo, True)

        _, cat_key, plot_key = payload
        cat = self._working_plot_data[self._current_model].get(cat_key, {}) if self._working_plot_data[self._current_model] else {}
        plot = ((cat or {}).get("plots") or {}).get(plot_key, {}) or {}

        self._loading_editor = True
        self._block_editor_signals(True)
        try:
            self._load_plot_into_editor(plot_key, plot)
        finally:
            self._block_editor_signals(False)
            self._loading_editor = False

        self._refresh_dup_targets()

    # ---------- existing editor methods (unchanged) ----------
    def _clear_editor(self) -> None:
        self._loading_editor = True
        self._block_editor_signals(True)
        self.lbl_internal_name.clear()
        self.name_edit.clear()
        self.toggled_check.setChecked(False)

        self.curve_traj_key.blockSignals(True)
        self.curve_traj_key.clear()
        self.curve_traj_key.blockSignals(False)
        idx = self.curve_linestyle.findData("solid")
        if idx >= 0:
            self.curve_linestyle.setCurrentIndex(idx)
        self.curve_series_editor.set_pairs([], [])

        self.hist_traj_key.clear()
        self.hist_bins.setValue(50)
        self.hist_density.setChecked(False)
        self.hist_color.clear()

        self.scatter_traj_key_x.clear()
        self.scatter_traj_key_y.clear()
        self.scatter_label.clear()
        self.scatter_color.clear()

        self.plot_type_combo.setCurrentText("Curve")
        self.type_stack.setCurrentIndex(1)

        self._block_editor_signals(False)
        self._loading_editor = False

    def _load_plot_into_editor(self, plot_key: str, plot: dict) -> None:
        self.lbl_internal_name.setText(plot_key)
        self.name_edit.setText(plot.get("checkbox_name", "") or "")
        self.toggled_check.setChecked(bool(plot.get("toggled", False)))

        special = (plot.get("special") or "curve").strip().lower()
        if special == "scatter" or ("traj_key_x" in plot and "traj_key_y" in plot):
            self.plot_type_combo.setCurrentText("Scatter")
            self.type_stack.setCurrentIndex(3)
        elif special == "hist":
            self.plot_type_combo.setCurrentText("Histogram")
            self.type_stack.setCurrentIndex(2)
        elif special == "curve":
            self.plot_type_combo.setCurrentText("Curve")
            self.type_stack.setCurrentIndex(1)

        if self.plot_type_combo.currentText() == "Curve":
            self.curve_traj_key.blockSignals(True)
            self.curve_traj_key.setText(plot.get("traj_key", "") or "")
            self.curve_traj_key.blockSignals(False)
            val = (plot.get("linestyle") or "solid").strip().lower()
            idx = self.curve_linestyle.findData(val)
            if idx >= 0:
                self.curve_linestyle.setCurrentIndex(idx)
            else:
                self.curve_linestyle.setCurrentIndex(self.curve_linestyle.findData("solid"))
            labels = plot.get("labels", []) or []
            colors = plot.get("colors", []) or []
            self.curve_series_editor.set_pairs(labels, colors)

        elif self.plot_type_combo.currentText() == "Histogram":
            self.hist_traj_key.setText(plot.get("traj_key", "") or "")

        else:  # Scatter
            self.scatter_traj_key_x.setText(plot.get("traj_key_x", "") or "")
            self.scatter_traj_key_y.setText(plot.get("traj_key_y", "") or "")
            self.scatter_label.setText(plot.get("label", "") or "")
            self.scatter_color.setText(plot.get("color", "") or "")

    def _get_plot_data(self):
        data_type = self.type_stack_dict.get(self.type_stack.currentIndex(), "")
        if data_type == "":
            return
        
        item = self.tree.currentItem()
        data = item.data(0, self.ROLE)
        new_data = {}

        if data[0] == "category":
            old_dict = self._working_plot_data[self._current_model][data[1]]
            new_data["name"] = self.name_edit.text()
            inter_name = self.lbl_internal_name.text()
            if self.cat_title.text():
                new_data["title"] = self.cat_title.text()
            if self.x_label.text():
                new_data["x_label"] = self.x_label.text()
            if self.y_label.text():
                new_data["y_label"] = self.y_label.text()
            if self.tooltip.toPlainText():
                new_data["tooltip"] = self.tooltip.toPlainText()
            new_data["plots"] = old_dict["plots"]
            return inter_name, new_data

        if data_type == "curve":
            if self.name_edit.text(): new_data["checkbox_name"] = self.name_edit.text()
            new_data["linestyle"] = (self.curve_linestyle.currentData() or "solid")
            labels, colors = self.curve_series_editor.get_pairs()
            new_data["labels"] = labels
            new_data["colors"] = colors
            new_data["traj_key"] = self.curve_traj_key.text()
            if self.name_edit.text(): new_data["toggled"] = self.toggled_check.isChecked()
            inter_name = self.lbl_internal_name.text()

            return inter_name, new_data

        if data_type == "scatter":
            new_data["special"] = "scatter"
            if self.name_edit.text(): new_data["checkbox_name"] = self.name_edit.text()
            if self.name_edit.text(): new_data["toggled"] = self.toggled_check.isChecked()
            new_data["traj_key_x"] = self.scatter_traj_key_x.text()
            new_data["traj_key_y"] = self.scatter_traj_key_y.text()
            new_data["label"] = self.scatter_label.text()
            new_data["color"] = self.scatter_color.text()
            inter_name = self.lbl_internal_name.text()

            return inter_name, new_data

        if data_type == "hist":
            new_data["special"] = "hist"
            if self.name_edit.text():
                new_data["checkbox_name"] = self.name_edit.text()
                new_data["toggled"] = self.toggled_check.isChecked()

            new_data["traj_key"] = self.hist_traj_key.text()
            new_data["bins"] = int(self.hist_bins.value())
            new_data["density"] = bool(self.hist_density.isChecked())
            if self.hist_color.text().strip():
                new_data["color"] = self.hist_color.text().strip()

            inter_name = self.lbl_internal_name.text()
            return inter_name, new_data

    def on_apply_clicked(self) -> None:
        self._rebuild_plot_data_from_tree()
        self._normalize_flowseqs_for_dump(self._working_plot_data)
        self._normalize_flowseqs_for_dump(self._original_plot_data)

        try:
            for model, _ in self._original_plot_data.items():
                path = rpath("models", model, "data", "plotting_data.yml")
                model_dict = self._working_plot_data[model]
                atomic_write(path, model_dict)
        except Exception as e:
            self.window.status.show(f"Error writing changes: {e}", 8000)
            logger.log(logger.ERROR, "Error writing changes", exc_info= e)
        else:
            for model, _ in self._original_plot_data.items():
                new_dict = self._working_plot_data[model]
                self._original_plot_data[model] = copy.deepcopy(new_dict)
            
    # def on_apply_clicked(self):
    #     self._rebuild_plot_data_from_tree()
    #     self._normalize_flowseqs_for_dump(self._working_plot_data)
    #     self._normalize_flowseqs_for_dump(self._original_plot_data)

    #     try: 
    #         for model_name, model_dict in self._original_plot_data.items():
    #             with open(rpath("models", model_name, "data", "plotting_data.yml.bak"), "w") as f:
    #                 yaml.safe_dump(model_dict, f, sort_keys= False, allow_unicode= True)

    #         for model_name, model_dict in self._working_plot_data.items():
    #             with open(rpath("models", model_name, "data", "plotting_data.yml"), "w") as f:
    #                 yaml.safe_dump(model_dict, f, sort_keys= False, allow_unicode= True)
    #     except OSError as e:
    #         self.window.status.show(f"Error saving your plotting files: {e}. Backups should be available in their respective folders for anything deleted.")
    #         logger.error(f"Error applying changes.", )
    #     finally:
    #         try:
    #             os.remove(bak)
    #         except OSError as e:
    #             self.window.status.show("Error removing backup, you should check your directory.", 5000)
    #             logger.log(warning, f"Error removing directory {bak}", exc_info= e)

            
    #     self._get_new_plotting_data(self._current_model)
        
    def _normalize_flowseqs_for_dump(self, data: dict) -> dict:
        for model_name, model_dict in data.items():
            for cat_name, cat_dict in model_dict.items():
                plots_dict = cat_dict["plots"]
                for plot_name, plot_dict in plots_dict.items():
                    labels, colors = plot_dict.get("labels"), plot_dict.get("colors")
                    if isinstance(labels, (list, tuple)):
                        plot_dict["labels"] = FlowSeq(labels)
                    if isinstance(colors, (list, tuple)):
                        plot_dict["colors"] = FlowSeq(colors)

        demos = data.get("demos", {})
        for _k, demo in demos.items():
            if not isinstance(demo, dict):
                continue
            details = demo.get("details")
            if not isinstance(details, dict):
                continue

            lims = details.get("starting_lims")
            if not lims:
                continue

            if (
                isinstance(lims, (list, tuple))
                and len(lims) == 2
                and all(isinstance(row, (list, tuple)) and len(row) == 2 for row in lims)
            ):
                x = [float(lims[0][0]), float(lims[0][1])]
                y = [float(lims[1][0]), float(lims[1][1])]
                details["starting_lims"] = FlowSeq([FlowSeq(x), FlowSeq(y)])

        return data

    def set_model(self, model_name: str):
        idx = self.model_combo.findText(model_name)
        if idx >= 0 and idx != self.model_combo.currentIndex():
            self.model_combo.setCurrentIndex(idx)
