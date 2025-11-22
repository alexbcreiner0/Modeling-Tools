from PyQt6 import (
    QtWidgets as qw,
    QtGui as qg,
    QtCore as qc
)
from qt_tools import recolor_icon
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib import pyplot as plt
import sys, importlib, yaml, math, inspect
from ControlPanel import ControlPanel
from GraphPanel import GraphPanel
from loader import load_presets, _dump_to_yaml, to_plain, params_from_mapping

# from simulation.parameters import params_from_mapping, to_plain
from widgets.Dialogs import SaveDialog, DescDialog

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

    def __init__(self, demos):
        super().__init__()
        self.setWindowTitle("Dynamic Equilibrium Model")
        self.demos = demos
        self.thread = qc.QThread()
        self.thread.finished.connect(self.on_thread_finished)

        demo = self._find_default(demos)
        self.sim_model = demo["details"]["simulation_model"]
        self.params, self.get_trajectories, self.presets, panel_data, plotting_data, self.functions = self._get_data(demo)

        # Create top bar menu
        self.presets_submenu = self._make_menu(self.presets, self.demos, self.functions)
        # print(self.sim_submenus)
        # print(self.get_trajectories.__name__)
        self.sim_actions[self.get_trajectories.__name__].setChecked(True)

        # make status bar
        saved_infos = [qw.QLabel("Saved x-axis: "), qw.QLabel("Saved y-axis: ")]
        self.saved_labels = [qw.QLabel(), qw.QLabel()]
        self.status_bar = self.statusBar()
        self.status_bar.addPermanentWidget(saved_infos[0])
        self.status_bar.addPermanentWidget(self.saved_labels[0])
        self.status_bar.addPermanentWidget(saved_infos[1])
        self.status_bar.addPermanentWidget(self.saved_labels[1])

        # make matplotlib stuff, need toolbar for below
        self.figure, self.axis = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.removeToolBar(self.toolbar)
        
        # make top toolbar
        self.nav_toolbar, self.toolbar_entries, self.toolbar_buttons = self._make_nav_toolbar()
        self.addToolBar(qc.Qt.ToolBarArea.TopToolBarArea, self.nav_toolbar)
        
        # Load and perform initial simulation, get trajectories
        self.traj, self.t, e = self.get_trajectories(self.params)

        self.graph_panel, self.control_panel, self.dropdown_choices = self._make_panels(plotting_data, panel_data)
        self._set_graph_lims(demo, plotting_data)
        self.graph_panel.toolbar.pan()

        self.main_layout = qw.QHBoxLayout()
        self.main_layout.addWidget(self.control_panel, stretch=3)
        self.main_layout.addWidget(self.graph_panel, stretch=5)

        main_container = qw.QWidget()
        main_container.setLayout(self.main_layout)
        # self.update_plot()

        self.setCentralWidget(main_container)

    def _set_graph_lims(self, demo, plotting_data):

        if "starting_xlim" in demo["details"]:
            x0, x1 = demo["details"]["starting_xlim"]
            self.graph_panel.xlower_entry.setText(f"{x0:g}")
            self.graph_panel.xupper_entry.setText(f"{x1:g}")
        if "starting_ylim" in demo["details"]:
            y0, y1 = demo["details"]["starting_ylim"]
            self.graph_panel.ylower_entry.setText(f"{y0:g}")
            self.graph_panel.yupper_entry.setText(f"{y1:g}")
        if "starting_plots" in demo["details"]:
            choice = demo["details"]["starting_plots"]
            if choice in plotting_data:
                name = plotting_data[choice]["name"]
                self.control_panel.dropdown_widget.dropdown_choices.setCurrentIndex(self.dropdown_choices.index(name))

    def _make_panels(self, plotting_data, panel_data):
        self.current_dropdown_choice = 0
        dropdown_choices, dropdown_tooltips = self._get_dropdown_choices(plotting_data)

        graph_panel = GraphPanel(
            self.traj, self.t, dropdown_choices, 
            self.params.T, plotting_data, self.canvas, 
            self.figure, self.axis, self.toolbar,
            self.toolbar_entries, self.toolbar_buttons, self.status_bar
        )
        graph_panel.saved_lims_changed.connect(self.update_saved_lims)
        xlim, ylim = graph_panel.xlim, graph_panel.ylim
        self.update_saved_lims(xlim, ylim)

        control_panel = ControlPanel(
            self.params, dropdown_choices, 
            dropdown_tooltips, panel_data, 
            plotting_data, self.sim_model
        )
        control_panel.paramChanged.connect(self.update_plot)
        control_panel.checkStateChanged.connect(self.new_check_update)
        control_panel.plotChoiceChanged.connect(self.switch_plot)

        return graph_panel, control_panel, dropdown_choices

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

        xlower_entry, xupper_entry = qw.QLineEdit(), qw.QLineEdit()
        ylower_entry, yupper_entry = qw.QLineEdit(), qw.QLineEdit()
        entries = [xlower_entry, ylower_entry, xupper_entry, yupper_entry]
        for entry in entries:
            entry.setMaximumWidth(80)

        save_button = qw.QPushButton("Save Current Axes")
        load_button = qw.QPushButton("Load Saved Axes")
        buttons = [save_button, load_button]

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
        catch_icon = style.standardIcon(qw.QStyle.StandardPixmap.SP_DialogHelpButton)
        grab_as_initial = qg.QAction(catch_icon, "Grab time as new initial conditions", self)
        grab_as_initial.triggered.connect(self.grab_as_initial)
        nav_toolbar.addWidget(self.grab_entry)
        nav_toolbar.addAction(grab_as_initial)

        spacer = qw.QWidget()
        spacer.setSizePolicy(
            qw.QSizePolicy.Policy.Expanding,
            qw.QSizePolicy.Policy.Preferred,
        )
        nav_toolbar.addWidget(spacer)

        xrange_label_from = qw.QLabel("X-axis: from")
        xrange_label_to = qw.QLabel(" to ")
        yrange_label_from = qw.QLabel("Y-axis: from")
        yrange_label_to = qw.QLabel(" to ")

        nav_toolbar.addSeparator()
        nav_toolbar.addWidget(buttons[0])
        nav_toolbar.addWidget(buttons[1])
        nav_toolbar.addSeparator()

        nav_toolbar.addWidget(xrange_label_from)
        nav_toolbar.addWidget(entries[0])
        nav_toolbar.addWidget(xrange_label_to)
        nav_toolbar.addWidget(entries[2])
        nav_toolbar.addSeparator()
        nav_toolbar.addWidget(yrange_label_from)
        nav_toolbar.addWidget(entries[1])
        nav_toolbar.addWidget(yrange_label_to)
        nav_toolbar.addWidget(entries[3])

        return nav_toolbar, entries, buttons

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
        functions_menu = menu.addMenu("Sims")
        self.sim_choice = qg.QActionGroup(self)
        self.sim_choice.setExclusive(True)

        save_preset_action = qg.QAction("Save parameter settings", self)
        save_preset_action.triggered.connect(self.save_preset)
        file_menu.addAction(save_preset_action)
        presets_submenu = params_menu.addMenu("Parameter presets")
        self._create_presets_submenus(presets, presets_submenu)
    
        rerun_button = qg.QAction("Rerun Simulation", self)
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

    def update_saved_lims(self, xlim, ylim):

        xlim_str = f"({xlim[0]:g}, {xlim[1]:g})"
        ylim_str = f"({ylim[0]:g}, {ylim[1]:g})"
        self.saved_labels[0].setText(xlim_str)
        self.saved_labels[1].setText(ylim_str)

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

        options = self.control_panel.dropdown_widget.get_current_checked_boxes()
        self.graph_panel.make_plot(self.traj, self.t, self.current_dropdown_choice, options)

    def new_check_update(self):

        options = self.control_panel.dropdown_widget.get_current_checked_boxes()
        self.graph_panel.make_plot(self.traj, self.t, self.current_dropdown_choice, options)

    def switch_plot(self, index):

        self.current_dropdown_choice = index
        options = self.control_panel.dropdown_widget.get_current_checked_boxes()
        self.graph_panel.make_plot(self.traj, self.t, self.current_dropdown_choice, options)

    def keyPressEvent(self, e):
        if e.key() == 16777268:
            self.update_plot()
        if e.key() == 80:
            self.graph_panel.toolbar.pan()
        if e.key() == 16777216:
            qw.QApplication.quit()

    def load_preset(self, preset):

        self.params = params_from_mapping(self.presets[preset]["params"], f"{self.sim_model}/simulation/parameters.py")
        # self.traj, self.t, e = self.get_trajectories(self.params)
        # options = self.control_panel.dropdown_widget.get_current_checked_boxes()
        # self.graph_panel.make_plot(self.traj, self.t, self.current_dropdown_choice, options)
        self.control_panel.load_new_params(self.params)

    def _get_data(self, demo):

        sim_model = demo["details"]["simulation_model"]
        sim_function_name = demo["details"]["simulation_function"]
        default_preset = demo["details"]["default_preset"]

        print(f"Loaded sim function: {sim_function_name}")
        print(f"Loaded preset: {default_preset}")

        presets = load_presets(sim_model)
        trajectories_module = importlib.import_module(f"{sim_model}.simulation.simulation")

        functions = {}
        for name, obj in inspect.getmembers(trajectories_module, inspect.isfunction):
            if obj.__module__ == trajectories_module.__name__:
                functions[name] = obj

        sim_function = getattr(trajectories_module, sim_function_name)

        if len(sys.argv) == 1:
            try:
                params_dict = presets[default_preset]
            except StopIteration:
                with open(f'{sim_model}/data/extra_data.yml', 'r') as f:
                    default_presets = yaml.safe_load(f)
                _dump_to_yaml(default_presets, sim_model)
                params_dict = default_presets[next(iter(default_presets))]
        else:
            try:
                params_dict = presets[sys.argv[1]]
            except KeyError:
                print(f"Preset {sys.argv[1]} not found, loading the first thing in params.yaml.")
                params_dict = presets[next(iter(presets))]

        params = params_from_mapping(params_dict["params"], f"{sim_model}/simulation/parameters.py")

        with open(f"{sim_model}/data/plotting_data.yml") as f:
            plotting_data = yaml.safe_load(f)

        with open(f"{sim_model}/data/control_panel_data.yml") as f:
            panel_data = yaml.safe_load(f)

        return params, sim_function, presets, panel_data, plotting_data, functions

    def load_demo(self, demo_name):

        demo = self.demos[demo_name]
        self.sim_model = demo["details"]["simulation_model"]
        self.params, self.get_trajectories, self.presets, panel_data, plotting_data, functions = self._get_data(demo)

        self.traj, self.t, e = self.get_trajectories(self.params)

        self.presets_submenu.clear()
        self._create_presets_submenus(self.presets, self.presets_submenu)

        self.main_layout.removeWidget(self.graph_panel)
        self.graph_panel.deleteLater()
        self.main_layout.removeWidget(self.control_panel)
        self.control_panel.deleteLater()

        self.graph_panel, self.control_panel, self.dropdown_choices = self._make_panels(plotting_data, panel_data)

        self._set_graph_lims(demo, plotting_data)
        self.main_layout.addWidget(self.control_panel, stretch=3)
        self.main_layout.addWidget(self.graph_panel, stretch=5)

    def load_sim(self, func, name):

        print(f"Setting sim function = {func}")
        self.get_trajectories = func
        action = self.sim_actions[name]
        action.setChecked(True)
        self.update_plot()

    def save_preset(self):

        with open(f"{self.sim_model}/data/params.yml", "r") as f:
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
        with open(f"{self.sim_model}/data/params.yml", "r") as f:
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
