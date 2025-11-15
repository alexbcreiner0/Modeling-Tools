from PyQt6 import (
    QtWidgets as qw,
    QtGui as qg,
    QtCore as qc
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib import pyplot as plt
import yaml
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
    def __init__(self, init_params, trajectory_function, presets, panel_data, plotting_data, current_path):
        super().__init__()
        self.setWindowTitle("Dynamic Equilibrium Model")
        self.presets = presets
        self.current_path = current_path

        # Extract dropdown info from data
        dropdown_choices = [plotting_data[dropdown_choice]["name"] for dropdown_choice in plotting_data]
        dropdown_tooltips = {plotting_data[choice]["name"]: plotting_data[choice]["tooltip"] for choice in plotting_data}
        for choice in dropdown_tooltips:
            if not dropdown_tooltips[choice]: dropdown_tooltips[choice] = "No notes"

        self.figure, self.axis = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        self.nav_toolbar = qw.QToolBar("Navigation")

        for i, action in enumerate(self.toolbar.actions()):
            if i == 10: continue
            self.nav_toolbar.addAction(action)


        spacer = qw.QWidget()
        spacer.setSizePolicy(
            qw.QSizePolicy.Policy.Expanding,
            qw.QSizePolicy.Policy.Preferred,
        )
        self.nav_toolbar.addWidget(spacer)

        saved_infos = [qw.QLabel("Saved x-axis: "), qw.QLabel("Saved y-axis: ")]
        self.saved_labels = [qw.QLabel(), qw.QLabel()]

        xrange_label_from = qw.QLabel("X-axis: from")
        xrange_label_to = qw.QLabel(" to ")
        yrange_label_from = qw.QLabel("Y-axis: from")
        yrange_label_to = qw.QLabel(" to ")
        xlower_entry, xupper_entry = qw.QLineEdit(), qw.QLineEdit()
        ylower_entry, yupper_entry = qw.QLineEdit(), qw.QLineEdit()

        entries = [xlower_entry, ylower_entry, xupper_entry, yupper_entry]

        save_button = qw.QPushButton("Save Current Axes")
        load_button = qw.QPushButton("Load Saved Axes")

        self.nav_toolbar.addSeparator()
        self.nav_toolbar.addWidget(save_button)
        self.nav_toolbar.addWidget(load_button)
        self.nav_toolbar.addSeparator()

        self.nav_toolbar.addWidget(xrange_label_from)
        self.nav_toolbar.addWidget(xlower_entry)
        self.nav_toolbar.addWidget(xrange_label_to)
        self.nav_toolbar.addWidget(xupper_entry)
        self.nav_toolbar.addSeparator()
        self.nav_toolbar.addWidget(yrange_label_from)
        self.nav_toolbar.addWidget(ylower_entry)
        self.nav_toolbar.addWidget(yrange_label_to)
        self.nav_toolbar.addWidget(yupper_entry)

        self.addToolBar(qc.Qt.ToolBarArea.TopToolBarArea, self.nav_toolbar)

        # Create top bar menu
        menu = self.menuBar()
        file_menu = menu.addMenu("File")

        save_preset_action = qg.QAction("Save parameter settings", self)
        save_preset_action.triggered.connect(self.save_preset)
        file_menu.addAction(save_preset_action)
        self.presets_submenu = file_menu.addMenu("Parameter presets")
        for preset in presets:
            name = presets[preset]["name"]
            preset_options_submenu = self.presets_submenu.addMenu(name)
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
        quit_button = qg.QAction("Quit", self)
        file_menu.addAction(quit_button)
        quit_button.triggered.connect(qw.QApplication.quit)

        self.status_bar = self.statusBar()
        self.status_bar.addPermanentWidget(saved_infos[0])
        self.status_bar.addPermanentWidget(self.saved_labels[0])
        self.status_bar.addPermanentWidget(saved_infos[1])
        self.status_bar.addPermanentWidget(self.saved_labels[1])

        # Load and perform initial simulation, get trajectories
        self.params = init_params
        self.get_trajectories = trajectory_function
        self.traj, self.t, e = self.get_trajectories(self.params)

        # Create the control panel and the graph panel
        self.current_dropdown_choice = 0
        self.graph_panel = GraphPanel(self.traj, self.t, dropdown_choices, 
                                      self.params.T, plotting_data, self.canvas, 
                                      self.figure, self.axis, self.toolbar,
                                      entries, save_button, load_button, self.status_bar
                                      )

        self.graph_panel.saved_lims_changed.connect(self.update_saved_lims)
        xlim, ylim = self.graph_panel.xlim, self.graph_panel.ylim
        passxlim = (float(f"{float(xlim[0]):.3f}"), float(f"{float(xlim[1]):.3f}"))
        passylim = (float(f"{float(ylim[0]):.3f}"), float(f"{float(ylim[1]):.3f}"))
        self.update_saved_lims(xlim, ylim)
        self.control_panel = ControlPanel(self.params, dropdown_choices, dropdown_tooltips, panel_data, plotting_data)
        self.control_panel.paramChanged.connect(self.update_plot)
        self.control_panel.checkStateChanged.connect(self.new_check_update)
        self.control_panel.plotChoiceChanged.connect(self.switch_plot)

        main_layout = qw.QHBoxLayout()
        main_layout.addWidget(self.control_panel, stretch=3)
        main_layout.addWidget(self.graph_panel, stretch=5)

        main_container = qw.QWidget()
        main_container.setLayout(main_layout)
        self.update_plot()

        self.setCentralWidget(main_container)

    def update_saved_lims(self, xlim, ylim):
        self.saved_labels[0].setText(str(xlim))
        self.saved_labels[1].setText(str(ylim))

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
        elif (len(self.t) - 1) // self.params.res < self.params.T:
            self.status_bar.showMessage("Simulation failed to complete. Likely a number got too big or small for the program to handle.", msecs= 3000)
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
        if e.key() == 16777216:
            qw.QApplication.quit()

    def load_preset(self, preset):
        self.params = params_from_mapping(self.presets[preset]["params"], f"{self.current_path}/simulation/parameters.py")
        self.traj, self.t, e = self.get_trajectories(self.params)
        options = self.control_panel.dropdown_widget.get_current_checked_boxes()
        self.graph_panel.make_plot(self.traj, self.t, self.current_dropdown_choice, options)
        self.control_panel.load_new_params(self.params)

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

