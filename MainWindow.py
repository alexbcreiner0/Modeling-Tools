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

# from simulation.parameters import params_from_mapping, to_plain
from widgets.Dialogs import SaveDialog, DescDialog

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

# background worker for plot updating (this was the only way to get the computing message to properly display)
class SimWorker(qc.QObject):

    finished = qc.pyqtSignal(object, object, object) # traj, t, e

    def __init__(self, params, get_trajectories):
        super().__init__()
        self.params = params
        self.get_trajectories = get_trajectories

    @qc.pyqtSlot()
    def run(self):
        traj, t, e = self.get_trajectories(self.params)
        self.finished.emit(traj, t, e)

class MainWindow(qw.QMainWindow):

    def __init__(self, config):
        super().__init__()
        self.setWindowTitle("Dynamic Equilibrium Model")
        self.config = config
        self.settings = config["global_settings"]
        self.demos = config["demos"]
        self.thread = qc.QThread()
        self.thread.finished.connect(self.on_thread_finished)

        demo = self._find_default(self.demos)
        self.sim_model = demo["details"]["simulation_model"]
        self.params, self.get_trajectories, self.presets, panel_data, plotting_data, self.functions, self.default_dir = self._get_data(self.settings, demo)

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
        self.status_bar = self.statusBar()
        self.model_label = qw.QLabel(f"Model: {demo["name"]}")
        self.status_bar.addPermanentWidget(self.model_label)

        # make matplotlib stuff, need toolbar for below
        self.figure, self.axis = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = CustomNavigationToolbar(self.canvas, parent= self, default_dir= self.default_dir)
        self.removeToolBar(self.toolbar)
        
        # make top toolbar
        self.nav_toolbar = self._make_nav_toolbar()
        self.addToolBar(qc.Qt.ToolBarArea.TopToolBarArea, self.nav_toolbar)
        
        # Load and perform initial simulation, get trajectories
        self.traj, self.t, e = self.get_trajectories(self.params)

        self.graph_panel, self.control_panel, self.dropdown_choices = self._make_panels(plotting_data, panel_data, demo)
        self._set_graph_lims(demo, plotting_data)

        self.graph_panel.toolbar.pan()


        num_slots = len(self.graph_panel.axes)
        for slot_index in range(num_slots):
            cfg = self.control_panel.get_slot_config(slot_index)
            if cfg is None:
                continue
            dropdown_index, options, slot_cfg = cfg
            self.graph_panel.plot_slot(slot_index, dropdown_index, options, slot_cfg)

        self.main_layout = qw.QHBoxLayout()
        self.main_layout.addWidget(self.control_panel, stretch=3)
        self.main_layout.addWidget(self.graph_panel, stretch=5)


        self.update_figure_background(use_window_background=True)

        qc.QCoreApplication.instance().installEventFilter(self)

        main_container = qw.QWidget()
        main_container.setLayout(self.main_layout)

        self.setCentralWidget(main_container)

        self._install_focus_clear_filter()

    def _install_focus_clear_filter(self) -> None:
        # Put it on the window and also on the central widget / scroll areas if needed
        self.installEventFilter(self)
        cw = self.centralWidget()
        if cw:
            cw.installEventFilter(self)

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

                    # # fully transparent background
                    # fc = frame.get_facecolor()
                    # frame.set_facecolor((fc[0], fc[1], fc[2], 0.0))  # keep rgb, set alpha=0

                    # # fully opaque border
                    # ec = frame.get_edgecolor()
                    # frame.set_edgecolor((ec[0], ec[1], ec[2], 1.0))
                    # frame.set_linewidth(1.0)

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

    # def eventFilter(self, a0, a1):
    #     # Intercept 'P' key globally to toggle pan mode
    #     if a1.type() == qc.QEvent.Type.KeyPress and a1.key() == 80:
    #         if hasattr(self, "graph_panel") and self.graph_panel is not None:
    #             self.graph_panel.toolbar.pan()
    #         return True  # event handled; don't pass to any widget

    #         # Everywhere else: toggle pan on the Matplotlib toolbar
    #         if hasattr(self, "graph_panel") and self.graph_panel is not None:
    #             self.graph_panel.toolbar.pan()
    #         return True  # event handled; don't pass to other widgets

    #     return super().eventFilter(a0, a1)

    def eventFilter(self, a0, a1):
        if a1.type() == qc.QEvent.Type.KeyPress:
            # Only handle 'P' / 'p'
            if a1.key() == qc.Qt.Key.Key_P:
                fw = qw.QApplication.focusWidget()

                # If the user is typing anywhere, do NOT hijack 'p'
                if isinstance(fw, (qw.QLineEdit, qw.QTextEdit, qw.QPlainTextEdit, qw.QSpinBox, qw.QDoubleSpinBox)):
                    return False

                # Also: editable combo box has an internal line edit
                if isinstance(fw, qw.QComboBox) and fw.isEditable():
                    return False

                # Otherwise toggle pan
                if hasattr(self, "graph_panel") and self.graph_panel is not None:
                    self.graph_panel.toolbar.pan()
                    return True

        if a1.type() == qc.QEvent.Type.MouseButtonPress:
            w = qw.QApplication.widgetAt(a1.globalPosition().toPoint())
            if not _is_text_input_widget(w):
                fw = qw.QApplication.focusWidget()
                if fw is not None:
                    fw.clearFocus()

                # Give focus to something sane so keyboard shortcuts work
                self.setFocus(qc.Qt.FocusReason.MouseFocusReason)

        return super().eventFilter(a0, a1)

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
            self.traj, self.t, dropdown_choices, 
            self.params.T, plotting_data, self.canvas, 
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
        with open(rpath(f"config.yml"), "r") as f:
            fresh_config = yaml.safe_load(f)
        # extremely lazy and bad way of doing this
        _, _, _, _, plotting_data, _, _ = self._get_data(fresh_config["global_settings"], self._find_default(fresh_config["demos"]))
        self.graph_panel.data = format_plot_config(plotting_data, names)

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

    def _find_default(self, demos):

        for demo in demos:
            if "default" in demos[demo]:
                if demos[demo]["default"]: return demos[demo]
        return demos[next(iter(demos))]

    def _get_dropdown_choices(self, plotting_data):

        dropdown_choices = [plotting_data[dropdown_choice]["name"] for dropdown_choice in plotting_data]
        dropdown_tooltips = {plotting_data[choice]["name"]: plotting_data[choice]["tooltip"] for choice in plotting_data}
        for choice in dropdown_tooltips:
            if not dropdown_tooltips[choice]: dropdown_tooltips[choice] = "No notes"

        return dropdown_choices, dropdown_tooltips

    def on_thread_finished(self):

        self.thread.deleteLater()
        self.thread = None
        self.worker = None

    def _make_nav_toolbar(self):

        nav_toolbar = qw.QToolBar("Navigation")

        # xlower_entry, xupper_entry = qw.QLineEdit(), qw.QLineEdit()
        # ylower_entry, yupper_entry = qw.QLineEdit(), qw.QLineEdit()
        # entries = [xlower_entry, ylower_entry, xupper_entry, yupper_entry]
        # for entry in entries:
        #     entry.setMaximumWidth(80)

        # save_button = qw.QPushButton("Save Current Axes")
        # load_button = qw.QPushButton("Load Saved Axes")
        # buttons = [save_button, load_button]

        for i, action in enumerate(self.toolbar.actions()):
            if i == 10: continue
            nav_toolbar.addAction(action)

        nav_toolbar.addSeparator()
        style = self.style()
        bug_icon = style.standardIcon(qw.QStyle.StandardPixmap.SP_LineEditClearButton)

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

        # tight_layout_button = qw.QPushButton("Tight Layout")
        # tight_layout_button.clicked.connect(self.tight_layout)
        # nav_toolbar.addWidget(tight_layout_button)

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

        catch_icon = style.standardIcon(qw.QStyle.StandardPixmap.SP_DialogHelpButton)
        tight_layout_action = qg.QAction(catch_icon, "Make the plots adapt to their space better", self)
        tight_layout_action.triggered.connect(self.tight_layout)
        nav_toolbar.addAction(tight_layout_action)

        # xrange_label_from = qw.QLabel("X-axis: from")
        # xrange_label_to = qw.QLabel(" to ")
        # yrange_label_from = qw.QLabel("Y-axis: from")
        # yrange_label_to = qw.QLabel(" to ")

        nav_toolbar.addSeparator()
        # anything you want coming from the right of the toolbar should be put here

        # nav_toolbar.addWidget(buttons[0])
        # nav_toolbar.addWidget(buttons[1])
        # nav_toolbar.addSeparator()

        # nav_toolbar.addWidget(xrange_label_from)
        # nav_toolbar.addWidget(entries[0])
        # nav_toolbar.addWidget(xrange_label_to)
        # nav_toolbar.addWidget(entries[2])
        # nav_toolbar.addSeparator()
        # nav_toolbar.addWidget(yrange_label_from)
        # nav_toolbar.addWidget(entries[1])
        # nav_toolbar.addWidget(yrange_label_to)
        # nav_toolbar.addWidget(entries[3])

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

        if self.thread is not None:
            if self.thread.isRunning():
                self.thread.requestInterruption()
                self.status_bar.showMessage("Computation interrupted", msecs= 2000)

    def _make_menu(self, presets, demos, functions):

        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        params_menu = menu.addMenu("Parameters")
        demo_menu = menu.addMenu("Demos")
        view_menu = menu.addMenu("View")
        functions_menu = menu.addMenu("Sims")
        self.sim_choice = qg.QActionGroup(self)
        self.sim_choice.setExclusive(True)

        save_preset_action = qg.QAction("Save parameter settings", self)
        save_preset_action.triggered.connect(self.save_preset)
        file_menu.addAction(save_preset_action)
        presets_submenu = params_menu.addMenu("Parameter presets")
        self._create_presets_submenus(presets, presets_submenu)
    
        rerun_button = qg.QAction("Rerun Simulation", self)
        title_change = qg.QAction("Change plot titles", self)
        
        file_menu.addAction(rerun_button)
        rerun_button.triggered.connect(self.update_plot)

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

        return presets_submenu

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

    def update_plot(self, name= None, new_val= None):

        if name != None and name != False:
            setattr(self.params, name, new_val)

        if self.thread is not None and self.thread.isRunning():
            return

        self.status_bar.showMessage("Computing trajectories...")

        self.thread = qc.QThread(self)
        self.worker = SimWorker(self.params, self.get_trajectories)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.show_results)
        self.worker.finished.connect(self.thread.quit)
        # self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def show_results(self, traj, t, e):

        self.traj, self.t = traj, t
        if e != None:
            self.status_bar.showMessage(f"Simulation failed to complete. Exception caught: {str(e)}")
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

        # old stuff
        # options = self.control_panel.dropdown_widget.get_current_checked_boxes()
        # self.graph_panel.make_plot(self.traj, self.t, self.current_dropdown_choice, options)

    # retired
    # def new_check_update(self):

    #     options = self.control_panel.dropdown_widget.get_current_checked_boxes()
    #     self.graph_panel.make_plot(self.traj, self.t, self.current_dropdown_choice, options)

    # def switch_plot(self, index):

    #     self.current_dropdown_choice = index
    #     options = self.control_panel.dropdown_widget.get_current_checked_boxes()
    #     self.graph_panel.make_plot(self.traj, self.t, self.current_dropdown_choice, options)

    def keyPressEvent(self, a0):
        if a0 is not None:
            if a0.key() == 16777268: # F5
                self.update_plot()
            if a0.key() == 16777269: # F6
                self.tight_layout()
            # if a0.key() == 80:
            #     self.graph_panel.toolbar.pan()
            if a0.key() == 16777216: # ESC
                qw.QApplication.quit()

    def load_preset(self, preset):

        self.params = params_from_mapping(self.presets[preset]["params"], f"models/{self.sim_model}/simulation/parameters.py")
        # self.traj, self.t, e = self.get_trajectories(self.params)
        # options = self.control_panel.dropdown_widget.get_current_checked_boxes()
        # self.graph_panel.make_plot(self.traj, self.t, self.current_dropdown_choice, options)
        self.control_panel.load_new_params(self.params)

    def _get_data(self, settings, demo):

        sim_model = demo["details"]["simulation_model"]
        sim_function_name = demo["details"]["simulation_function"]
        default_preset = demo["details"]["default_preset"]
        default_dir = settings["default_save_dir"]

        presets = load_presets(sim_model)
        trajectories_module = importlib.import_module(f"models.{sim_model}.simulation.simulation")

        functions = {}
        for name, obj in inspect.getmembers(trajectories_module, inspect.isfunction):
            if obj.__module__ == trajectories_module.__name__:
                functions[name] = obj

        sim_function = getattr(trajectories_module, sim_function_name)

        if len(sys.argv) == 1:
            try:
                params_dict = presets[default_preset]
            except StopIteration:
                with open(rpath(f'models/{sim_model}/data/extra_data.yml'), 'r') as f:
                    default_presets = yaml.safe_load(f)
                _dump_to_yaml(default_presets, sim_model)
                params_dict = default_presets[next(iter(default_presets))]
        else:
            try:
                params_dict = presets[sys.argv[1]]
            except KeyError:
                print(f"Preset {sys.argv[1]} not found, loading the first thing in params.yaml.")
                params_dict = presets[next(iter(presets))]

        params = params_from_mapping(params_dict["params"], f"models/{sim_model}/simulation/parameters.py")

        with open(rpath(f"models/{sim_model}/data/plotting_data.yml")) as f:
            plotting_data = yaml.safe_load(f)

        with open(rpath(f"models/{sim_model}/data/control_panel_data.yml")) as f:
            panel_data = yaml.safe_load(f)

        return params, sim_function, presets, panel_data, plotting_data, functions, default_dir

    def load_demo(self, demo_name):

        demo = self.demos[demo_name]
        self.sim_model = demo["details"]["simulation_model"]
        self.params, self.get_trajectories, self.presets, panel_data, plotting_data, functions, self.default_dir  = self._get_data(self.settings, demo)

        model_settings = self.config["model_specific_settings"][self.sim_model]
        
        if model_settings is not None:
            if "commodity_names" in model_settings:
                com_names = model_settings["commodity_names"]
                plotting_data = format_plot_config(plotting_data, com_names)

        self.traj, self.t, e = self.get_trajectories(self.params)

        self.presets_submenu.clear()
        self._create_presets_submenus(self.presets, self.presets_submenu)

        self.main_layout.removeWidget(self.graph_panel)
        self.graph_panel.deleteLater()
        self.main_layout.removeWidget(self.control_panel)
        self.control_panel.deleteLater()

        # --- Tear down old panels ---
        self.main_layout.removeWidget(self.graph_panel)
        self.graph_panel.deleteLater()
        self.main_layout.removeWidget(self.control_panel)
        self.control_panel.deleteLater()

        # --- Reset the figure to a clean single-plot layout ---
        self.figure.clear()
        self.axis = self.figure.add_subplot(1, 1, 1)

        self.graph_panel, self.control_panel, self.dropdown_choices = self._make_panels(plotting_data, panel_data, demo)

        self._set_graph_lims(demo, plotting_data)
        self.main_layout.addWidget(self.control_panel, stretch=3)
        self.main_layout.addWidget(self.graph_panel, stretch=5)
        self.model_label.setText(f"Model: {demo["name"]}")
        self.update_plot()

    def load_sim(self, func, name):

        print(f"Setting sim function = {func}")
        self.get_trajectories = func
        action = self.sim_actions[name]
        action.setChecked(True)
        self.update_plot()

    def save_preset(self):

        with open(f"models/{self.sim_model}/data/params.yml", "r") as f:
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
        with open(f"models/{self.sim_model}/data/params.yml", "r") as f:
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
