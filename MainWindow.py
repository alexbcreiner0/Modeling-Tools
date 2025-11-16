from PyQt6 import (
    QtWidgets as qw,
    QtGui as qg,
    QtCore as qc
)
from pprint import pprint
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib import pyplot as plt
import yaml
import sys, importlib
from ControlPanel import ControlPanel
from GraphPanel import GraphPanel
from loader import load_presets, _dump_to_yaml, to_plain, params_from_mapping
from matplotlib.backend_bases import cursors

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
    def __init__(self, init_params, trajectory_function, presets, panel_data, plotting_data, current_path, demos):
        super().__init__()
        self.setWindowTitle("Dynamic Equilibrium Model")
        self.presets = presets
        self.demos = demos
        self.current_path = current_path

        # Extract dropdown info from data
        # dropdown_choices = [plotting_data[dropdown_choice]["name"] for dropdown_choice in plotting_data]
        # dropdown_tooltips = {plotting_data[choice]["name"]: plotting_data[choice]["tooltip"] for choice in plotting_data}
        # for choice in dropdown_tooltips:
        #     if not dropdown_tooltips[choice]: dropdown_tooltips[choice] = "No notes"
        dropdown_choices, dropdown_tooltips = self._get_dropdown_choices(plotting_data)

        self.figure, self.axis = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        self.nav_toolbar, entries, buttons = self._make_nav_toolbar()

        self.addToolBar(qc.Qt.ToolBarArea.TopToolBarArea, self.nav_toolbar)

        # Create top bar menu
        self.presets_submenu = self._make_menu(presets, demos)

        saved_infos = [qw.QLabel("Saved x-axis: "), qw.QLabel("Saved y-axis: ")]
        self.saved_labels = [qw.QLabel(), qw.QLabel()]
        self.status_bar = self.statusBar()
        self.status_bar.addPermanentWidget(saved_infos[0])
        self.status_bar.addPermanentWidget(self.saved_labels[0])
        self.status_bar.addPermanentWidget(saved_infos[1])
        self.status_bar.addPermanentWidget(self.saved_labels[1])

        # Load and perform initial simulation, get trajectories
        self.params = init_params
        self.get_trajectories = trajectory_function
        self.traj, self.t, e = self.get_trajectories(self.params)

        self.current_dropdown_choice = 0

        self.graph_panel = GraphPanel(
            self.traj, self.t, dropdown_choices, 
            self.params.T, plotting_data, self.canvas, 
            self.figure, self.axis, self.toolbar,
            entries, buttons, self.status_bar
        )

        self.graph_panel.saved_lims_changed.connect(self.update_saved_lims)
        xlim, ylim = self.graph_panel.xlim, self.graph_panel.ylim
        self.update_saved_lims(xlim, ylim)

        self.control_panel = ControlPanel(
            self.params, dropdown_choices, 
            dropdown_tooltips, panel_data, 
            plotting_data
        )
        self.control_panel.paramChanged.connect(self.update_plot)
        self.control_panel.checkStateChanged.connect(self.new_check_update)
        self.control_panel.plotChoiceChanged.connect(self.switch_plot)

        self.main_layout = qw.QHBoxLayout()
        self.main_layout.addWidget(self.control_panel, stretch=3)
        self.main_layout.addWidget(self.graph_panel, stretch=5)

        self.graph_panel.toolbar.pan()

        main_container = qw.QWidget()
        main_container.setLayout(self.main_layout)
        self.update_plot()

        self.setCentralWidget(main_container)

    def _get_dropdown_choices(self, plotting_data):
        dropdown_choices = [plotting_data[dropdown_choice]["name"] for dropdown_choice in plotting_data]
        dropdown_tooltips = {plotting_data[choice]["name"]: plotting_data[choice]["tooltip"] for choice in plotting_data}
        for choice in dropdown_tooltips:
            if not dropdown_tooltips[choice]: dropdown_tooltips[choice] = "No notes"

        return dropdown_choices, dropdown_tooltips

    def _make_nav_toolbar(self):

        nav_toolbar = qw.QToolBar("Navigation")

        xlower_entry, xupper_entry = qw.QLineEdit(), qw.QLineEdit()
        ylower_entry, yupper_entry = qw.QLineEdit(), qw.QLineEdit()
        entries = [xlower_entry, ylower_entry, xupper_entry, yupper_entry]

        save_button = qw.QPushButton("Save Current Axes")
        load_button = qw.QPushButton("Load Saved Axes")
        buttons = [save_button, load_button]

        for i, action in enumerate(self.toolbar.actions()):
            if i == 10: continue
            nav_toolbar.addAction(action)

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

    def _make_menu(self, presets, demos):

        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        demo_menu = menu.addMenu("Demos")

        save_preset_action = qg.QAction("Save parameter settings", self)
        save_preset_action.triggered.connect(self.save_preset)
        file_menu.addAction(save_preset_action)
        presets_submenu = file_menu.addMenu("Parameter presets")
        self._create_presets_submenus(presets, presets_submenu)
    
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
        if name != None:
            setattr(self.params, name, new_val)
        self.status_bar.showMessage("Computing trajectories...")

        self.thread = qc.QThread(self)
        self.worker = SimWorker(self.params, self.get_trajectories)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.show_results)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def show_results(self, traj, t, e):
        self.traj, self.t = traj, t
        if e != None:
            self.status_bar.showMessage(f"Simulation failed to complete. Exception caught: {str(e)}")
        # elif (len(self.t) - 1) // self.params.res < self.params.T:
        #     self.status_bar.showMessage("Simulation failed to complete. Likely a number got too big or small for the program to handle.", msecs= 3000)
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
        if e.key() == 80:
            self.graph_panel.toolbar.pan()
        if e.key() == 16777216:
            qw.QApplication.quit()

    def load_preset(self, preset):
        self.params = params_from_mapping(self.presets[preset]["params"], f"{self.current_path}/simulation/parameters.py")
        self.traj, self.t, e = self.get_trajectories(self.params)
        options = self.control_panel.dropdown_widget.get_current_checked_boxes()
        self.graph_panel.make_plot(self.traj, self.t, self.current_dropdown_choice, options)
        self.control_panel.load_new_params(self.params)

    def load_demo(self, demo):
        sim_model = self.demos[demo]["details"]["simulation_model"]
        sim_function = self.demos[demo]["details"]["simulation_function"]
        default_preset = self.demos[demo]["details"]["default_preset"]

        presets = load_presets(sim_model)
        trajectories_module = importlib.import_module(f"{sim_model}.simulation.simulation")
        self.get_trajectories = getattr(trajectories_module, sim_function)

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

        self.params = params_from_mapping(params_dict["params"], f"{sim_model}/simulation/parameters.py")
        self.traj, self.t, e = self.get_trajectories(self.params)

        self.presets_submenu.clear()
        self._create_presets_submenus(presets, self.presets_submenu)
        self.current_path = sim_model

        with open(f"{sim_model}/data/plotting_data.yml") as f:
            plotting_data = yaml.safe_load(f)

        with open(f"{sim_model}/data/control_panel_data.yml") as f:
            panel_data = yaml.safe_load(f)

        dropdown_choices, dropdown_tooltips = self._get_dropdown_choices(plotting_data)
        entries = self.graph_panel.entries
        buttons = self.graph_panel.buttons

        self.main_layout.removeWidget(self.graph_panel)
        self.graph_panel.deleteLater()
        self.main_layout.removeWidget(self.control_panel)
        self.control_panel.deleteLater()

        self.graph_panel = GraphPanel(
            self.traj, self.t, dropdown_choices, 
            self.params.T, plotting_data, self.canvas, 
            self.figure, self.axis, self.toolbar,
            entries, buttons, self.status_bar
        )

        self.control_panel = ControlPanel(
            self.params, dropdown_choices, 
            dropdown_tooltips, panel_data, 
            plotting_data
        )
        self.control_panel.paramChanged.connect(self.update_plot)
        self.control_panel.checkStateChanged.connect(self.new_check_update)
        self.control_panel.plotChoiceChanged.connect(self.switch_plot)

        if "starting_xlim" in self.demos[demo]["details"]:
            x0, x1 = self.demos[demo]["details"]["starting_xlim"]
            self.graph_panel.xlower_entry.setText(f"{x0:g}")
            self.graph_panel.xupper_entry.setText(f"{x1:g}")
        if "starting_ylim" in self.demos[demo]["details"]:
            y0, y1 = self.demos[demo]["details"]["starting_ylim"]
            self.graph_panel.ylower_entry.setText(f"{y0:g}")
            self.graph_panel.yupper_entry.setText(f"{y1:g}")
        if "starting_plots" in self.demos[demo]["details"]:
            choice = self.demos[demo]["details"]["starting_plots"]
            if choice in plotting_data:
                name = plotting_data[choice]["name"]
                self.control_panel.dropdown_widget.dropdown_choices.setCurrentIndex(dropdown_choices.index(name))
                # self.switch_plot(dropdown_choices.index(name))

        self.main_layout.addWidget(self.control_panel, stretch=3)
        self.main_layout.addWidget(self.graph_panel, stretch=5)

    def save_preset(self):
        with open(f"{self.current_path}/data/params.yml", "r") as f:
            presets = yaml.safe_load(f)["presets"]

        params_dict = to_plain(self.params)
        dialog = SaveDialog(presets.keys(), self)
        try:
            shortname, name, desc = dialog.bootstrap()
        except TypeError:
            return
        presets[shortname] = {"name": name, "desc": desc, "params": params_dict}

        _dump_to_yaml(presets, self.current_path)
        
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

        _dump_to_yaml(self.presets, self.current_path)
        for action in self.presets_submenu.actions():
            submenu = action.menu()
            if submenu.property("preset_id") == preset:
                self.presets_submenu.removeAction(action)
                submenu.deleteLater()
                break

    def rename_preset(self, old_shortname):
        with open(f"{self.current_path}/data/params.yml", "r") as f:
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

        _dump_to_yaml(presets, self.current_path)
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
