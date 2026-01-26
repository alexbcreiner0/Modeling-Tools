import logging
logger = logging.getLogger(__name__)

from PyQt6 import (
    QtWidgets as qw,
    QtGui as qg,
    QtCore as qc
)
from tools.qt_tools import recolor_icon
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from widgets.CustomNavigationToolbar import CustomNavigationToolbar
from matplotlib import pyplot as plt
import sys, importlib, yaml, math, inspect
from ControlPanel import ControlPanel
from GraphPanel import GraphPanel
from tools.loader import load_presets, _dump_to_yaml, to_plain, params_from_mapping, format_plot_config
from paths import rpath

# from simulation.parameters import params_from_mapping, to_plain
from widgets.Dialogs import SaveDialog, DescDialog, NewModelDialog
from widgets.EditConfigDialog import EditConfigDialog

def _is_text_input_widget(w: qw.QWidget | None) -> bool:
    if w is None:
        return False

    # If you click inside a QLineEdit embedded in another control,
    # this catches it as well.
    if isinstance(w, (qw.QLineEdit, qw.QTextEdit, qw.QPlainTextEdit)):
        return True

    # Editable combo box has an internal QLineEdit; clicking it should not clear focus
    if isinstance(w, qw.QComboBox) and w.isEditable():
        return True

    # Spin boxes have their own internal line edits
    if isinstance(w, (qw.QSpinBox, qw.QDoubleSpinBox)):
        return True

    return False

class SimWorker(qc.QObject):
    progress = qc.pyqtSignal(object, object)          # traj, t
    finished = qc.pyqtSignal(object, object, object)  # traj, t, e

    def __init__(self, params, stream_func, *, yield_every=1):
        super().__init__()
        self.params = params
        self.stream_func = stream_func
        self.yield_every = yield_every
        self._stop = False
        self._pause = False

    @qc.pyqtSlot()
    def request_stop(self):
        self._stop = True

    @qc.pyqtSlot()
    def toggle_pause(self):
        self._pause = not self._pause

    def _should_stop(self) -> bool:
        return self._stop or qc.QThread.currentThread().isInterruptionRequested()

    @qc.pyqtSlot()
    def run(self):
        e = None
        latest_traj, latest_t = None, None
        animating = True

        try:
            result = self.stream_func(self.params, should_stop= self._should_stop, yield_every= self.yield_every)

            # if it's a normal function output (i.e. the user is not animating)
            if isinstance(result, tuple) and len(result) == 2:
                animating = False
                traj, t = result
                latest_traj, latest_t = traj, t
                self.progress.emit(traj, t)

            else:
                # output can now be assumed to be an iterator object
                for frame in result:
                    # stop receiving new outputs if sim is paused
                    while self._pause and not self._should_stop():
                        qc.QThread.msleep(25) # recheck every 25 ms

                    if not (isinstance(frame, tuple) and len(frame) == 2):
                        raise TypeError(f"Streaming sim must yield (traj, t) tuples. Got {type(frame)} {frame!r}")

                    latest_traj, latest_t = frame
                    self.progress.emit(latest_traj, latest_t)

                    if self._should_stop():
                        break

        except Exception as ex:
            latest_t_val = latest_t[-1] if latest_t is not None else None
            extra = {
                "Sim function": self.stream_func.__name__,
                "Animating from generator": animating,
                "latest t value": latest_t_val
            }
            info = (extra, ex)
            self.finished.emit(latest_traj, latest_t, info)
            return

        self.finished.emit(latest_traj, latest_t, e)

class MainWindow(qw.QMainWindow):

    def __init__(self, config):
        super().__init__()
        self.setWindowTitle("Tex's Modeling Tools")

        self.status_bar = self.statusBar()

        self.config = config
        self.settings = config["global_settings"]
        self.demos = config["demos"]
        self.thread = qc.QThread()
        self.thread.finished.connect(self.on_thread_finished)
        self.live_animation = True
        self._run_id = 0

        # short term fix
        self._pending_traj = None
        self._pending_t = None
        self._anim_timer = qc.QTimer(self)
        self._anim_timer.setInterval(30)  # ~30fps
        self._anim_timer.timeout.connect(self._apply_next_frame)

        self.current_demo_name, self.current_demo = self._find_default(self.demos)
        self.sim_model = self.current_demo["details"]["simulation_model"]
        self.params, self.get_trajectories, self.presets, panel_data, plotting_data, self.functions, self.default_dir = self._get_data(self.settings, self.current_demo)

        self.model_label = qw.QLabel(f"Model: {self.current_demo.get("name", "")}")
        self.status_bar.addPermanentWidget(self.model_label)

        model_settings = config["model_specific_settings"][self.sim_model]
        
        if model_settings is not None:
            if "commodity_names" in model_settings:
                com_names = model_settings["commodity_names"]
                plotting_data = format_plot_config(plotting_data, com_names)

        # Create top bar menu
        self.presets_submenu = self._make_menu(self.presets, self.demos, self.functions)
        self.sim_actions[self.get_trajectories.__name__].setChecked(True)

        # make status bar
        # saved_infos = [qw.QLabel("Saved x-axis: "), qw.QLabel("Saved y-axis: ")]
        # self.saved_labels = [qw.QLabel(), qw.QLabel()]

        # make matplotlib stuff, need toolbar for below
        self.figure, self.axis = plt.subplots()
        self.figure.set_constrained_layout(True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = CustomNavigationToolbar(self.canvas, parent= self, default_dir= self.default_dir)
        self.removeToolBar(self.toolbar)
        
        # make top toolbar
        self.nav_toolbar = self._make_nav_toolbar()
        self.addToolBar(qc.Qt.ToolBarArea.TopToolBarArea, self.nav_toolbar)
        
        # Load and perform initial simulation, get trajectories
        # self.traj, self.t, e = self.get_trajectories(self.params)
        self.traj, self.t = None, None

        self.graph_panel, self.control_panel, self.dropdown_choices = self._make_panels(plotting_data, panel_data, self.current_demo)
        self._set_graph_lims(self.current_demo, plotting_data)

        self.graph_panel.toolbar.pan()

        num_slots = len(self.graph_panel.axes)
        for slot_index in range(num_slots):
            cfg = self.control_panel.get_slot_config(slot_index)
            if cfg is None:
                continue
            dropdown_index, options, slot_cfg = cfg
            self.graph_panel.plot_slot(slot_index, dropdown_index, options, slot_cfg)

        self.main_splitter = qw.QSplitter(qc.Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.control_panel)
        self.main_splitter.addWidget(self.graph_panel)

        self.main_splitter.setStretchFactor(0,3)
        self.main_splitter.setStretchFactor(1,5)
        
        self.main_splitter.setCollapsible(1, False)
        self.graph_panel.setMinimumWidth(50) # matplotlib crashes at 0

        total = max(1, self.main_splitter.width())
        left = int(total * 3 / 8)   # 3:(3+5)
        right = total - left
        self.main_splitter.setSizes([left, right])

        self.update_figure_background(use_window_background=True)

        qc.QCoreApplication.instance().installEventFilter(self)

        # main_container = qw.QWidget()
        # main_container.setLayout(self.main_layout)

        self.setCentralWidget(self.main_splitter)
        qc.QTimer.singleShot(0, lambda: (self.graph_panel.canvas.draw_idle(), self.tight_layout()))

        self._install_focus_clear_filter()
        self.update_plot()

    def _install_focus_clear_filter(self) -> None:
        # Put it on the window and also on the central widget / scroll areas if needed
        self.installEventFilter(self)
        cw = self.centralWidget()
        if cw:
            cw.installEventFilter(self)

    # MainWindow.py
    def show_partial_results(self, traj, t):
        if traj is None or t is None:
            return

        self.traj, self.t = traj, t
        self.graph_panel.traj = traj
        self.graph_panel.t = t

        num_slots = len(self.graph_panel.axes)
        for slot_index in range(num_slots):
            cfg = self.control_panel.get_slot_config(slot_index)
            if cfg is None:
                continue
            dropdown_index, options, slot_cfg = cfg
            self.graph_panel.update_slot_frame(slot_index, dropdown_index, options, slot_cfg)

    def toggle_pause(self):
        if self.worker:
            self.worker.toggle_pause()

    def update_figure_background(self, use_window_background: bool) -> None:
        """
        If use_transparent is True:
            - Figure and axes are fully transparent.
            - Legend background mimics the window background color (fake transparency).
        If use_transparent is False:
            - Figure, axes, and legend backgrounds are pure white.
        """
        axes = getattr(self.graph_panel, "axes", [self.axis])

        if use_window_background:
            # True transparency for figure & axes
            fig_fc = (0, 0, 0, 0)
            ax_fc = (0, 0, 0, 0)

            plt.rcParams["figure.facecolor"] = fig_fc
            plt.rcParams["axes.facecolor"] = ax_fc

            # Legend default for *new* legends: match Qt window background
            bg = self.palette().color(qg.QPalette.ColorRole.Window)
            legend_rgba = (bg.redF(), bg.greenF(), bg.blueF(), bg.alphaF())
            plt.rcParams["legend.facecolor"] = legend_rgba

            # Canvas: let parent shine through
            self.canvas.setStyleSheet("background: transparent;")

            # Apply to current figure / axes
            self.figure.patch.set_facecolor(fig_fc)
            self.figure.patch.set_alpha(0.0)

            for ax in axes:
                ax.set_facecolor(ax_fc)
                ax.patch.set_alpha(0.0)

                leg = ax.get_legend()
                if leg is not None:
                    frame = leg.get_frame()

                    frame = leg.get_frame()
                    frame.set_facecolor(legend_rgba)  # fake transparency
                    frame.set_alpha(1.0)
                    # Optional: make sure border is visible
                    # frame.set_edgecolor("black")

        else:
            # Solid white everywhere
            fig_fc = "white"
            ax_fc = "white"
            legend_fc = "white"

            plt.rcParams["figure.facecolor"] = fig_fc
            plt.rcParams["axes.facecolor"] = ax_fc
            plt.rcParams["legend.facecolor"] = legend_fc

            self.canvas.setStyleSheet("")  # reset to default

            self.figure.patch.set_facecolor(fig_fc)
            self.figure.patch.set_alpha(1.0)

            for ax in axes:
                ax.set_facecolor(ax_fc)
                ax.patch.set_alpha(1.0)

                leg = ax.get_legend()
                if leg is not None:
                    frame = leg.get_frame()
                    frame.set_facecolor(legend_fc)
                    frame.set_alpha(1.0)
                    # frame.set_edgecolor("black")

        self.canvas.draw_idle()

    def _set_graph_lims(self, demo, plotting_data):
        lims = self.control_panel.get_slot_axes_limits(0)
        xlim = ylim = None
        if lims is not None:
            xlim, ylim = lims

        # Fallback to demo["details"]["starting_lims"] if needed
        if xlim is None or ylim is None:
            if "starting_lims" in demo["details"]:
                try:
                    sx, sy = demo["details"]["starting_lims"]
                    xlim = tuple(sx)
                    ylim = tuple(sy)
                except Exception:
                    xlim = (0, 100)
                    ylim = (0, 100)
            else:
                xlim = (0, 100)
                ylim = (0, 100)

            # Ensure slot 0's control widget reflects these fallback limits
            self.control_panel.set_slot_axes_limits(0, xlim, ylim)

        # Apply these limits to all current axes in the GraphPanel
        num_slots = len(self.graph_panel.axes)
        for slot_index in range(num_slots):
            self.graph_panel.edit_slot_axes(slot_index, xlim, ylim)

        # Handle initial dropdown selection from starting_plots, if present
        if "starting_plots" in demo["details"]:
            choice = demo["details"]["starting_plots"]
            if choice in plotting_data:
                name = plotting_data[choice]["name"]
                if name in self.dropdown_choices:
                    idx = self.dropdown_choices.index(name)
                    self.control_panel.set_slot_dropdown_index(0, idx)

    def _make_panels(self, plotting_data, panel_data, demo):
        self.current_dropdown_choice = 0
        dropdown_choices, dropdown_tooltips = self._get_dropdown_choices(plotting_data)

        graph_panel = GraphPanel(
            self.traj, self.t, dropdown_choices, plotting_data, self.canvas, 
            self.figure, self.axis, self.toolbar, self.status_bar
        )
        # graph_panel.saved_lims_changed.connect(self.update_saved_lims)
        graph_panel.slot_axes_limits_changed.connect(self.on_slot_axes_limits_changed)
        # xlim, ylim = graph_panel.xlim, graph_panel.ylim
        # self.update_saved_lims(xlim, ylim)

        control_panel = ControlPanel(
            self.params, dropdown_choices, 
            dropdown_tooltips, panel_data, 
            plotting_data, self.sim_model, demo
        )
        control_panel.paramChanged.connect(self.update_plot)
        control_panel.layoutChanged.connect(self.on_layout_changed)
        control_panel.slotPlotChoiceChanged.connect(self.on_slot_plot_choice_changed)
        control_panel.slotOptionsChanged.connect(self.on_slot_options_changed)
        control_panel.slotAxesChanged.connect(self.on_slot_axes_changed)
        control_panel.paramsReplaced.connect(self._on_params_replaced)

        return graph_panel, control_panel, dropdown_choices

    def _on_params_replaced(self, data):
        new_params, new_sector_names = data
        self.params = new_params
        self._update_sector_names(new_sector_names)
        self.update_plot()

    def _update_sector_names(self, names):

        with open(rpath("models", self.sim_model, "data", "plotting_data.yml")) as f:
            plotting_data = yaml.safe_load(f)
        formatted = format_plot_config(plotting_data, names)

        self.graph_panel.apply_plotting_data(formatted)

        for slot_index in range(len(self.graph_panel.axes)):
            cfg = self.control_panel.get_slot_config(slot_index)
            if cfg is None:
                continue
            dropdown_index, options, slot_cfg = cfg
            self.graph_panel.plot_slot(slot_index, dropdown_index, options, slot_cfg)

        self.control_panel.plotting_data = formatted

    def _apply_next_frame(self):
        if self._pending_traj is None:
            return
        self.show_partial_results(self._pending_traj, self._pending_t)

    def on_slot_axes_limits_changed(self, slot_index: int, xlim: tuple, ylim: tuple):
        """ Method to update the axis entries of a plot control widget when user pans a plot """
        self.control_panel.set_slot_axes_limits(slot_index, xlim, ylim)

    def on_slot_axes_changed(self, slot_index: int):
        lims = self.control_panel.get_slot_axes_limits(slot_index)
        if lims is None: return

        xlim, ylim = lims
        self.graph_panel.edit_slot_axes(slot_index, xlim, ylim)


    def on_slot_plot_choice_changed(self, slot_index: int):
        # print(f"[DEBUG] Slot {slot_index} dropdown changed to index {_dropdown_index}")
        if not hasattr(self, "traj") or self.traj is None: return

        cfg = self.control_panel.get_slot_config(slot_index)
        if cfg is None: return

        dropdown_index, options, legend_cfg = cfg

        self.graph_panel.plot_slot(slot_index, dropdown_index, options, legend_cfg)

    def on_slot_options_changed(self, slot_index: int):
        """Options changed for a specific slot."""
        if not hasattr(self, "traj") or self.traj is None:
            return

        cfg = self.control_panel.get_slot_config(slot_index)
        if cfg is None:
            return

        dropdown_index, options, slot_cfg = cfg
        self.graph_panel.plot_slot(slot_index, dropdown_index, options, slot_cfg)

    def on_layout_changed(self, rows, cols):
        """ Propagate plot dimension change to the graph panel """
        self.graph_panel.set_axes_layout(rows, cols)

        if hasattr(self, "traj") and self.traj is not None:
            self.graph_panel.traj = self.traj
            self.graph_panel.t = self.t

            num_slots = len(self.graph_panel.axes)
            for slot_index in range(num_slots):
                cfg = self.control_panel.get_slot_config(slot_index)
                if cfg is None:
                    continue
                dropdown_index, options, slot_cfg = cfg
                self.graph_panel.plot_slot(slot_index, dropdown_index, options, slot_cfg)

        qc.QTimer.singleShot(0, self.tight_layout)

    def _find_default(self, demos):

        for demo in demos:
            if "default" in demos[demo]:
                if demos[demo]["default"]: return demo, demos[demo]
        return next(iter(demos)), demos[next(iter(demos))]

    def _get_dropdown_choices(self, plotting_data):

        dropdown_choices = [plotting_data[dropdown_choice]["name"] for dropdown_choice in plotting_data]
        dropdown_tooltips = {plotting_data[choice]["name"]: plotting_data[choice].get("tooltip", "No notes") for choice in plotting_data}
        for choice in dropdown_tooltips:
            if not dropdown_tooltips[choice]: dropdown_tooltips[choice] = "No notes"

        return dropdown_choices, dropdown_tooltips

    def on_thread_finished(self):

        self.thread.deleteLater()
        self.thread = None
        self.worker = None

    def _make_nav_toolbar(self):

        nav_toolbar = qw.QToolBar("Navigation")

        for i, action in enumerate(self.toolbar.actions()):
            if i == 10: continue
            nav_toolbar.addAction(action)

        nav_toolbar.addSeparator()
        style = self.style()
        bug_icon = style.standardIcon(qw.QStyle.StandardPixmap.SP_LineEditClearButton)
        pause_icon = style.standardIcon(qw.QStyle.StandardPixmap.SP_MediaPause)

        request_pause = qg.QAction(pause_icon, "Pause simulation", self)
        request_pause.triggered.connect(self.toggle_pause)
        nav_toolbar.addAction(request_pause)

        request_threadkill = qg.QAction(bug_icon, "Kill thread (currently doesn't work)", self)
        request_threadkill.triggered.connect(self.request_threadkill)
        nav_toolbar.addAction(request_threadkill)

        self.grab_entry = qw.QLineEdit()
        self.grab_entry.setMaximumWidth(80)
    
        grab_as_initial_button = qw.QPushButton("Grab as Initial: ")
        grab_as_initial_button.setToolTip("Attempt to grab the state of the system at the given point in time, and then apply the relevant parameters at that moment as initial conditions. Useful for getting steady states.")
        grab_as_initial_button.clicked.connect(self.grab_as_initial)
        nav_toolbar.addWidget(grab_as_initial_button)
        nav_toolbar.addWidget(self.grab_entry)


        spacer = qw.QWidget()
        spacer.setSizePolicy(
            qw.QSizePolicy.Policy.Expanding,
            qw.QSizePolicy.Policy.Preferred,
        )
        nav_toolbar.addWidget(spacer)

        figure_background_checkbox = qw.QCheckBox("Transparent Background")
        figure_background_checkbox.setChecked(True)
        figure_background_checkbox.stateChanged.connect(self._on_figure_background_checkbox_changed)
        nav_toolbar.addWidget(figure_background_checkbox)

        animate_sim_checkbox = qw.QCheckBox("Animate Sim")
        animate_sim_checkbox.setChecked(True)
        animate_sim_checkbox.stateChanged.connect(lambda v: setattr(self, "live_animation", v))
        nav_toolbar.addWidget(animate_sim_checkbox)

        catch_icon = style.standardIcon(qw.QStyle.StandardPixmap.SP_DialogHelpButton)
        tight_layout_action = qg.QAction(catch_icon, "Make the plots adapt to their space better", self)
        tight_layout_action.triggered.connect(self.tight_layout)
        nav_toolbar.addAction(tight_layout_action)

        nav_toolbar.addSeparator()

        return nav_toolbar #, entries, buttons

    def _on_figure_background_checkbox_changed(self, state: int) -> None:
        use_window = True if state == 2 else False
        self.update_figure_background(use_window)

    def tight_layout(self):
        self.figure.tight_layout()
        self.figure.canvas.draw_idle()

    def grab_as_initial(self):

        try:
            time = float(self.grab_entry.text())
        except ValueError:
            return

        closest = 0
        for i, t in enumerate(self.t):
            if math.fabs(t - time) < math.fabs(t - self.t[closest]):
                closest = i

        for name in vars(self.params):
            if name in self.traj:
                new_val = self.traj[name][closest]
                setattr(self.params, name, new_val)

        # self.update_plot()
        self.control_panel.load_new_params(self.params)

    # Doesn't work usually
    def request_threadkill(self):
        if self.thread and self.thread.isRunning():
            self.thread.requestInterruption()
            if self.worker:
                self.worker.request_stop()
            self.status_bar.showMessage("Stop requested...", msecs=2000)

    def _make_menu(self, presets, demos, functions):

        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        params_menu = menu.addMenu("Parameters")
        demo_menu = menu.addMenu("Demos")
        # view_menu = menu.addMenu("View")
        functions_menu = menu.addMenu("Sim Functions")
        self.sim_choice = qg.QActionGroup(self)
        self.sim_choice.setExclusive(True)

        save_preset_action = qg.QAction("Save parameter settings", self)
        save_preset_action.triggered.connect(self.save_preset)
        file_menu.addAction(save_preset_action)
        self._create_presets_submenus(presets, params_menu)
    
        rerun_button = qg.QAction("Rerun Simulation", self)
        
        file_menu.addAction(rerun_button)
        rerun_button.triggered.connect(self.update_plot)

        edit_settings_action = qg.QAction("Settings", self)
        file_menu.addAction(edit_settings_action)
        edit_settings_action.triggered.connect(lambda _checked= False, tab= 0: self.open_settings(tab))

        edit_models_action = qg.QAction("Models", self)
        file_menu.addAction(edit_models_action)
        edit_models_action.triggered.connect(lambda _checked= False, tab= 1: self.open_settings(tab))
        
        edit_parameters_action = qg.QAction("Parameters", self)
        file_menu.addAction(edit_parameters_action)
        edit_parameters_action.triggered.connect(lambda _checked= False, tab= 2: self.open_settings(tab))

        edit_presets_action = qg.QAction("Presets", self)
        file_menu.addAction(edit_presets_action)
        edit_presets_action.triggered.connect(lambda _checked= False, tab= 3: self.open_settings(tab))

        edit_controls_action = qg.QAction("Control Panel", self)
        file_menu.addAction(edit_controls_action)
        edit_controls_action.triggered.connect(lambda _checked= False, tab= 4: self.open_settings(tab))

        edit_plots_action = qg.QAction("Plots", self)
        file_menu.addAction(edit_plots_action)
        edit_plots_action.triggered.connect(lambda _checked= False, tab= 5: self.open_settings(tab))

        edit_demos_action = qg.QAction("Demos", self)
        file_menu.addAction(edit_demos_action)
        edit_demos_action.triggered.connect(lambda _checked= False, tab= 6: self.open_settings(tab))

        quit_button = qg.QAction("Quit", self)
        file_menu.addAction(quit_button)
        quit_button.triggered.connect(qw.QApplication.quit)

        for demo in demos:
            name = demos[demo]["name"]
            demo_options_submenu = demo_menu.addMenu(name)
            demo_options_submenu.setProperty("demo_id", demo)
            load_action = qg.QAction("Load demo", self)
            view_desc_action = qg.QAction("View description", self)
            demo_options_submenu.addAction(load_action)
            demo_options_submenu.addAction(view_desc_action)
            load_action.triggered.connect(lambda _checked= False, name= demo: self.load_demo(name))
            view_desc_action.triggered.connect(lambda _checked= False, name= demo: self.view_demo_desc(name))

        self.sim_submenus = {}
        self.sim_actions = {}

        for name, func in functions.items():
            function_submenu = functions_menu.addMenu(name)
            self.sim_submenus[name] = function_submenu

            parent_action = function_submenu.menuAction()
            parent_action.setCheckable(True)
            self.sim_choice.addAction(parent_action)
            self.sim_actions[name] = parent_action

            load_action = qg.QAction("Load function", self)
            view_desc_action = qg.QAction("View description", self)
            function_submenu.addAction(load_action)
            function_submenu.addAction(view_desc_action)
            load_action.triggered.connect(lambda _checked= False, func= func, name= name: self.load_sim(func, name))

        return params_menu

    def _create_presets_submenus(self, presets, presets_submenu):

        for preset in presets:
            name = presets[preset]["name"]
            preset_options_submenu = presets_submenu.addMenu(name)
            preset_options_submenu.setProperty("preset_id", preset)
            load_action = qg.QAction("Load preset", self)
            delete_action = qg.QAction("Delete preset", self)
            rename_action = qg.QAction("Rename preset", self)
            view_desc_action = qg.QAction("View description", self)
            preset_options_submenu.addAction(load_action)
            preset_options_submenu.addAction(delete_action)
            preset_options_submenu.addAction(rename_action)
            preset_options_submenu.addAction(view_desc_action)
            load_action.triggered.connect(lambda _checked= False, name= preset: self.load_preset(name))
            delete_action.triggered.connect(lambda _checked= False, name= preset: self.delete_preset(name))
            rename_action.triggered.connect(lambda _checked= False, name= preset: self.rename_preset(name))
            view_desc_action.triggered.connect(lambda _checked= False, name= preset: self.view_desc(name))

    # def update_saved_lims(self, xlim, ylim):

    #     xlim_str = f"({xlim[0]:g}, {xlim[1]:g})"
    #     ylim_str = f"({ylim[0]:g}, {ylim[1]:g})"
    #     self.saved_labels[0].setText(xlim_str)
    #     self.saved_labels[1].setText(ylim_str)


    def update_plot(self, name=None, new_val=None):
        self._run_id += 1
        if name not in (None, False):
            setattr(self.params, name, new_val)

        # If running: stop and rerun after finish
        if self.thread is not None and self.thread.isRunning():
            self._rerun_pending = True
            if self.worker:
                self.worker.request_stop()
            self.thread.requestInterruption()
            self.status_bar.showMessage("Stop requested… will rerun.", 2000)
            return

        self._rerun_pending = False

        self.thread = qc.QThread(self)

        self.worker = SimWorker(self.params, self.get_trajectories, yield_every=1)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        # self.worker.progress.connect(self.show_partial_results)
        self.worker.progress.connect(self._on_worker_progress)

        if self.live_animation:
            self._anim_timer.start()

        self.worker.finished.connect(self.show_results)
        self.worker.finished.connect(self.thread.quit)

        # cleanup
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._on_sim_thread_finished)

        self.graph_panel.set_sim_run_id(self._run_id)
        self.thread.start()

    def _on_worker_progress(self, traj, t):
        self._pending_traj = traj
        self._pending_t = t

    def _on_sim_thread_finished(self):
        self.thread = None
        self.worker = None
        if getattr(self, "_rerun_pending", False):
            self._rerun_pending = False
            self.update_plot()

    # def update_plot(self, name= None, new_val= None):

    #     if name != None and name != False:
    #         setattr(self.params, name, new_val)

    #     if self.thread is not None and self.thread.isRunning():
    #         return

    #     self.status_bar.showMessage("Computing trajectories...")

    #     self.thread = qc.QThread(self)
    #     self.worker = SimWorker(self.params, self.get_trajectories)
    #     self.worker.moveToThread(self.thread)
    #     self.worker.progress.connect(self.show_partial_results)
    #     self.thread.started.connect(self.worker.run)
    #     self.worker.finished.connect(self.show_results)
    #     self.worker.finished.connect(self.thread.quit)
    #     # self.thread.finished.connect(self.thread.deleteLater)
    #     self.thread.start()

    def show_results(self, traj, t, e):
        self._anim_timer.stop()
        self._pending_traj = None
        self._pending_t = None

        self.traj, self.t = traj, t
        if e != None:
            extra, ex = e
            extra["Model"] = self.sim_model
            extra["Parameters"] = self.params
            self.status_bar.showMessage(f"Simulation failed to complete. Exception caught: {str(ex)}", msecs= 5000)
            logger.log(logging.ERROR, "Simulation failed.", extra= extra, exc_info= ex)
        else:
            self.status_bar.clearMessage()

        # new stuff
        self.graph_panel.traj = traj
        self.graph_panel.t = t

        num_slots = len(self.graph_panel.axes)
        for slot_index in range(num_slots):
            cfg = self.control_panel.get_slot_config(slot_index)
            if cfg is None:
                continue
            dropdown_index, options, legend_cfg = cfg
            self.graph_panel.plot_slot(slot_index, dropdown_index, options, legend_cfg)

    def keyPressEvent(self, a0):
        if a0 is not None:
            if a0.key() == 16777268: # F5
                self.update_plot()
            if a0.key() == 16777269: # F6
                self.tight_layout()
            if a0.key() == 16777270: # F7
                self.reload_current_demo()
            if a0.key() == 16777271: # F8?
                self.refresh_control_panel_and_plots()
            if a0.key() == 32: # space
                self.toggle_pause()

            if a0.key() == 16777216: # ESC
                qw.QApplication.quit()

    def load_preset(self, preset):
        try:
            self.params = params_from_mapping(self.presets[preset]["params"], rpath("models",self.sim_model,"simulation","parameters.py"))
            self.control_panel.load_new_params(self.params)
        except Exception as e:
            self.status_bar.showMessage(f"Failed to load preset {preset}: {e}")
            extras = {
                "Model": self.sim_model,
                "Preset": preset
            }
            logger.log(logging.ERROR, f"Failed to load preset {preset}: {e}", extra= extras, exc_info= e)

    def reload_current_demo(self):
        """
        Reload simulation.py / parameters.py for the current demo
        without changing which demo is active.
        """
        try:
            demo = self.current_demo

            (
                self.params,
                self.get_trajectories,
                self.presets,
                panel_data,
                plotting_data,
                self.functions,
                self.default_dir,
            ) = self._get_data(self.settings, demo)

            # Re-apply model-specific formatting
            model_settings = self.config["model_specific_settings"].get(self.sim_model)
            if model_settings and "commodity_names" in model_settings:
                plotting_data = format_plot_config(
                    plotting_data, model_settings["commodity_names"]
                )

            # Push updated data into existing panels
            self.graph_panel.traj = None
            self.graph_panel.t = None
            self.graph_panel.data = plotting_data
            self.control_panel.plotting_data = plotting_data
            self.control_panel.load_new_params(self.params)
        except Exception as e:
            self.status_bar.showMessage(f"Failed to reload current demo: {e}", msecs= 5000)
            extra = {
                "demo": self.current_demo,
                "params": self.params,
                "sim function": self.get_trajectories,
            }
            logger.log(logging.ERROR, f"Failed to reload current demo: {e}", extra= extra, exc_info= e)

        # Re-run simulation
        self.update_plot()

    def _get_data(self, settings, demo):

        try:
            sim_model = demo["details"]["simulation_model"]
            sim_function_name = demo["details"]["simulation_function"]
            default_preset = demo["details"]["default_preset"]
            default_dir = settings["default_save_dir"]

            presets = load_presets(sim_model)
            trajectories_module = importlib.import_module(f"models.{sim_model}.simulation.simulation")
            trajectories_module = importlib.reload(trajectories_module)
        except Exception as e:
            self.status_bar.showMessage(f"Failed to load data for {demo}, check logs for more info.")
            logger.log(logging.ERROR, f"Failed to load data for {demo}", exc_info= e)
            return

        functions = {}
        for name, obj in inspect.getmembers(trajectories_module, inspect.isfunction):
            if obj.__module__ == trajectories_module.__name__:
                functions[name] = obj

        sim_function = getattr(trajectories_module, sim_function_name)

        if len(sys.argv) == 1:
            try:
                params_dict = presets[default_preset]
            except StopIteration:
                with open(rpath("models",sim_model,"data","extra_data.yml"), 'r') as f:
                    default_presets = yaml.safe_load(f)
                _dump_to_yaml(default_presets, sim_model)
                params_dict = default_presets[next(iter(default_presets))]
        else:
            try:
                params_dict = presets[sys.argv[1]]
            except KeyError:
                logger.log(logging.INFO, f"Preset {sys.argv[1]} not found, loading the first thing in params.yaml.")
                params_dict = presets[next(iter(presets))]

        params_module_name = f"models.{sim_model}.simulation.parameters"

        if params_module_name in sys.modules:
            importlib.reload(sys.modules[params_module_name])
        params = params_from_mapping(params_dict["params"], rpath("models",self.sim_model,"simulation","parameters.py"))

        with open(rpath("models",sim_model,"data","plotting_data.yml")) as f:
            plotting_data = yaml.safe_load(f)

        with open(rpath("models",sim_model,"data","control_panel_data.yml")) as f:
            panel_data = yaml.safe_load(f)

        return params, sim_function, presets, panel_data, plotting_data, functions, default_dir

    def load_demo(self, demo_name):

        try:
            demo = self.demos[demo_name]
            self.sim_model = demo["details"]["simulation_model"]
            self.params, self.get_trajectories, self.presets, panel_data, plotting_data, functions, self.default_dir  = self._get_data(self.settings, demo)

            model_settings = self.config["model_specific_settings"][self.sim_model]
        except Exception as e:
            self.status_bar.showMessage(f"Failed to load data for demo: {e}. Check diagnostics in the settings for more info.", msecs= 5000)
            extra = {
                "Demo name": demo_name,
            }
            logger.log(logging.ERROR, "Failed to load demo.", extra= extra, exc_info= e)
            return
        
        if model_settings is not None:
            if "commodity_names" in model_settings:
                com_names = model_settings["commodity_names"]
                plotting_data = format_plot_config(plotting_data, com_names)

        self.traj, self.t = None, None

        self.presets_submenu.clear()
        self._create_presets_submenus(self.presets, self.presets_submenu)


        saved_state = self.main_splitter.saveState()
        if hasattr(self, "main_splitter") and self.main_splitter is not None:
            for w in (self.control_panel, self.graph_panel):
                try:
                    w.setParent(None)
                    w.deleteLater()
                except Exception:
                    pass

        # --- Reset the figure to a clean single-plot layout ---
        self.figure.clear()
        self.axis = self.figure.add_subplot(1, 1, 1)

        self.graph_panel, self.control_panel, self.dropdown_choices = self._make_panels(plotting_data, panel_data, demo)
        self.current_demo_name = demo_name
        self.current_demo = self.demos[self.current_demo_name]

        self._set_graph_lims(demo, plotting_data)
        self.main_splitter.addWidget(self.control_panel)
        self.main_splitter.addWidget(self.graph_panel)

        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 5)

        self.main_splitter.restoreState(saved_state)

        self.model_label.setText(f"Model: {demo["name"]}")
        qc.QTimer.singleShot(0, lambda: (self.graph_panel.canvas.draw_idle(), self.tight_layout()))

        self.update_plot()

    def load_sim(self, func, name):

        print(f"Setting sim function = {func}")
        self.get_trajectories = func
        action = self.sim_actions[name]
        action.setChecked(True)
        self.update_plot()

    def new_model(self):
        NewModelDialog(self).bootstrap()

    def open_settings(self, tab= None):
        dlg = EditConfigDialog(self.sim_model, tab, self)
        dlg.configApplied.connect(self._on_config_applied)
        dlg.bootstrap()

    def _on_config_applied(self):
        # 1) reload config.yml
        with open(rpath("config.yml"), "r") as f:
            self.config = yaml.safe_load(f)
        self.settings = self.config["global_settings"]
        self.demos = self.config["demos"]

        # 2) rebuild the top menus so demo list / global settings changes appear immediately
        self.menuBar().clear()
        self.presets_submenu = self._make_menu(self.presets, self.demos, self.functions)
        # self.sim_actions[self.get_trajectories.__name__].setChecked(True)

    def refresh_plots(self) -> None:
        """Reload plotting_data.yml for the current model and apply it to the live UI."""
        try:
            with open(rpath("models", self.sim_model, "data", "plotting_data.yml"), "r") as f:
                plotting_data = yaml.safe_load(f) or {}
        except Exception as e:
            self.status_bar.showMessage(f"Failed to reload plotting_data.yml: {e}", msecs= 4000)
            logger.log(logging.ERROR, "Failed to reload plotting_data.yml", exc_info= e)
            return

        # If we have sector/commodity names, apply them to plot labels.
        names = None
        try:
            names = self.current_demo.get("details", {}).get("commodity_names")
        except Exception:
            names = None
        if not names:
            names = getattr(self, "sector_names", None)

        try:
            formatted = format_plot_config(plotting_data, names) if names else plotting_data
        except Exception:
            formatted = plotting_data

        # Update live references
        self.graph_panel.data = formatted
        self.control_panel.plotting_data = formatted

        dropdown_choices, dropdown_tooltips = self._get_dropdown_choices(formatted)
        self.dropdown_choices = dropdown_choices

        # Update GraphPanel dropdown mapping, if present
        if hasattr(self.graph_panel, "dropdown_choices"):
            self.graph_panel.dropdown_choices = dropdown_choices

        # Update ControlPanel dropdowns in-place (so user selections don't get wiped)
        if hasattr(self.control_panel, "dropdown_tooltips"):
            self.control_panel.dropdown_tooltips = dropdown_tooltips

        if hasattr(self.control_panel, "slot_dropdowns"):
            for widget in self.control_panel.slot_dropdowns:
                combo = widget.dropdown_choices
                if combo is None:
                    continue
                prev = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(dropdown_choices)
                for i, tip in enumerate(dropdown_tooltips):
                    combo.setItemData(i, tip, qc.Qt.ItemDataRole.ToolTipRole)
                if prev in dropdown_choices:
                    combo.setCurrentIndex(dropdown_choices.index(prev))
                combo.blockSignals(False)

        # Redraw using the current slot configs (new plotting_data may change expressions, labels, etc.)
        try:
            num_slots = len(self.graph_panel.axes)
        except Exception:
            num_slots = 1

        for slot_index in range(num_slots):
            cfg = self.control_panel.get_slot_config(slot_index)
            if cfg is None:
                continue
            dropdown_index, options, slot_cfg = cfg
            self.graph_panel.plot_slot(slot_index, dropdown_index, options, slot_cfg)

        try:
            self.graph_panel.canvas.draw_idle()
        except Exception:
            pass

    def refresh_control_panel_and_plots(self):
        self.refresh_control_panel()
        self.refresh_plots()
        self.status_bar.showMessage("Reloaded panel settings.", msecs= 4000)

    def refresh_control_panel(self) -> None:
        """Rebuild the ControlPanel from control_panel_data.yml (and current plotting_data.yml)."""
        # Snapshot UI state we can reasonably preserve.
        old_cp = getattr(self, "control_panel", None)
        old_current_tab = self.control_panel.content.currentIndex()
        old_sizes = None
        try:
            old_sizes = self.main_splitter.sizes()
        except Exception:
            old_sizes = None

        old_rows = old_cols = None
        old_slot_dropdown_texts = []
        old_slot_axes = []

        try:
            old_rows = int(old_cp.rows_spinner.value())
            old_cols = int(old_cp.cols_spinner.value())
        except Exception:
            pass

        if old_cp is not None:
            try:
                for combo in getattr(old_cp, "slot_dropdowns", []):
                    old_slot_dropdown_texts.append(combo.currentText() if combo is not None else "")
            except Exception:
                old_slot_dropdown_texts = []

            try:
                for i in range(len(getattr(self.graph_panel, "axes", []))):
                    old_slot_axes.append(old_cp.get_slot_axes_limits(i))
            except Exception:
                old_slot_axes = []

        # Reload yaml sources
        try:
            with open(rpath("models", self.sim_model, "data", "plotting_data.yml"), "r") as f:
                plotting_data = yaml.safe_load(f) or {}
        except Exception as e:
            self.status_bar.showMessage(f"Failed to reload plotting_data.yml: {e}", msecs= 4000)
            logger.log(logging.ERROR, "Failed to reload plotting_data.yml", exc_info= e)
            return

        try:
            with open(rpath("models", self.sim_model, "data", "control_panel_data.yml"), "r") as f:
                panel_data = yaml.safe_load(f) or {}
        except Exception as e:
            self.status_bar.showMessage(f"Failed to reload control_panel_data.yml: {e}", msecs= 4000)
            logger.log(logging.ERROR, "Failed to reload control_panel_data.yml", exc_info= e)
            return

        # Apply sector/commodity names to plotting labels (if available)
        names = None
        try:
            names = self.current_demo.get("details", {}).get("commodity_names")
        except Exception:
            names = None
        if not names:
            names = getattr(self, "sector_names", None)

        try:
            formatted = format_plot_config(plotting_data, names) if names else plotting_data
        except Exception:
            formatted = plotting_data

        dropdown_choices, dropdown_tooltips = self._get_dropdown_choices(formatted)

        # Build new panel and wire signals exactly like _make_panels()
        new_cp = ControlPanel(
            self.params,
            dropdown_choices,
            dropdown_tooltips,
            panel_data,
            formatted,
            self.sim_model,
            self.current_demo,
            old_current_tab
        )
        new_cp.paramChanged.connect(self.update_plot)
        new_cp.layoutChanged.connect(self.on_layout_changed)
        new_cp.slotPlotChoiceChanged.connect(self.on_slot_plot_choice_changed)
        new_cp.slotOptionsChanged.connect(self.on_slot_options_changed)
        new_cp.slotAxesChanged.connect(self.on_slot_axes_changed)
        new_cp.paramsReplaced.connect(self._on_params_replaced)

        # Swap it into the splitter
        try:
            self.main_splitter.replaceWidget(0, new_cp)
        except Exception:
            # Fallback: remove/re-add
            try:
                self.main_splitter.widget(0).setParent(None)
            except Exception:
                pass
            self.main_splitter.insertWidget(0, new_cp)

        if old_cp is not None:
            old_cp.setParent(None)
            old_cp.deleteLater()

        self.control_panel = new_cp
        self.dropdown_choices = dropdown_choices

        # Best-effort restore of rows/cols + per-slot dropdown selections/axes limits
        if old_rows is not None and old_cols is not None:
            try:
                self.control_panel.rows_spinner.setValue(old_rows)
                self.control_panel.cols_spinner.setValue(old_cols)
            except Exception:
                pass

        try:
            for i, txt in enumerate(old_slot_dropdown_texts):
                if not txt:
                    continue
                if txt in dropdown_choices:
                    self.control_panel.set_slot_dropdown_index(i, dropdown_choices.index(txt))
        except Exception:
            pass

        try:
            for i, lims in enumerate(old_slot_axes):
                if lims is None:
                    continue
                xlim, ylim = lims
                self.control_panel.set_slot_axes_limits(i, xlim, ylim)
        except Exception:
            pass

        # Keep splitter size as-is
        if old_sizes:
            try:
                self.main_splitter.setSizes(old_sizes)
            except Exception:
                pass

        # Finally, re-plot based on the (new) control state.
        self.refresh_plots()


    def save_preset(self):

        with open(rpath("models",self.sim_model,"data","params.yml"), "r") as f:
            presets = yaml.safe_load(f)["presets"]

        params_dict = to_plain(self.params)
        dialog = SaveDialog(presets.keys(), self)
        try:
            shortname, name, desc = dialog.bootstrap()
        except TypeError:
            return
        presets[shortname] = {"name": name, "desc": desc, "params": params_dict}

        _dump_to_yaml(presets, self.sim_model)
        
        preset_options_submenu = self.presets_submenu.addMenu(name)
        preset_options_submenu.setProperty("preset_id", shortname)
        load_action = qg.QAction("Load preset", self)
        delete_action = qg.QAction("Delete preset", self)
        rename_action = qg.QAction("Rename preset", self)
        desc_action = qg.QAction("View description", self)
        preset_options_submenu.addAction(load_action)
        preset_options_submenu.addAction(delete_action)
        preset_options_submenu.addAction(rename_action)
        preset_options_submenu.addAction(desc_action)
        load_action.triggered.connect(lambda _checked= False, name= shortname: self.load_preset(name))
        delete_action.triggered.connect(lambda _checked= False, name= shortname: self.delete_preset(name))
        rename_action.triggered.connect(lambda _checked= False, name= shortname: self.rename_preset(name))
        desc_action.triggered.connect(lambda _checked= False, name= shortname: self.view_desc(name))

        self.presets = presets

    def delete_preset(self, preset):
        del self.presets[preset]

        _dump_to_yaml(self.presets, self.sim_model)
        for action in self.presets_submenu.actions():
            submenu = action.menu()
            if submenu.property("preset_id") == preset:
                self.presets_submenu.removeAction(action)
                submenu.deleteLater()
                break

    def rename_preset(self, old_shortname):
        with open(rpath("models",self.sim_model,"data","params.yml"), "r") as f:
            presets = yaml.safe_load(f)["presets"]
        dialog = SaveDialog(presets.keys(), self, name_text= "New Name: ", desc_text= "(Optional) New Description")
        try:
            shortname, new_name, new_desc = dialog.bootstrap()
        except TypeError:
            return
        preset = presets[old_shortname]
        preset["name"] = new_name
        preset["desc"] = new_desc
        self.delete_preset(old_shortname)
        presets[shortname] = preset
        del presets[old_shortname]

        _dump_to_yaml(presets, self.sim_model)
        preset_options_submenu = self.presets_submenu.addMenu(new_name)
        preset_options_submenu.setProperty("preset_id", shortname)
        load_action = qg.QAction("Load preset", self)
        delete_action = qg.QAction("Delete preset", self)
        rename_action = qg.QAction("Rename preset", self)
        desc_action = qg.QAction("View description", self)
        preset_options_submenu.addAction(load_action)
        preset_options_submenu.addAction(delete_action)
        preset_options_submenu.addAction(rename_action)
        preset_options_submenu.addAction(desc_action)
        load_action.triggered.connect(lambda _checked= False, name= shortname: self.load_preset(name))
        delete_action.triggered.connect(lambda _checked= False, name= shortname: self.delete_preset(name))
        rename_action.triggered.connect(lambda _checked= False, name= shortname: self.rename_preset(name))
        desc_action.triggered.connect(lambda _checked= False, name= shortname: self.view_desc(name))

        self.presets = presets

    def view_desc(self, name):
        desc = self.presets[name]["desc"]
        dialog = DescDialog(self, desc)
        dialog.bootstrap()

    def view_demo_desc(self, demo):
        desc = self.demos[demo]["desc"]
        dialog = DescDialog(self, desc)
        dialog.bootstrap()
