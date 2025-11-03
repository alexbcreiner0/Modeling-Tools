from PyQt6 import (
    QtWidgets as qw,
    QtGui as qg
) 
from ControlPanel import ControlPanel
from GraphPanel import GraphPanel
from mathstuff import get_trajectories
from mathstuff import *
from parameters import Params, params_from_mapping
from dataclasses import asdict, is_dataclass
import yaml
from Dialogs import SaveDialog, DescDialog
# from init_data import params as init_params
import sys

def load_presets(path):
    try:
        with open(path, 'r') as f:
            doc = yaml.safe_load(f)
        return doc["presets"]
    except (FileNotFoundError, KeyError, TypeError):
        default_presets = {"baseline": {'name': 'Default', 'desc': '', 'params': {'A': [[0.2, 0.0, 0.4], [0.2, 0.8, 0.0], [0.0, 0.1, 0.1]], 'l':
 [0.7, 0.6, 0.3], 'b_bar': [0.6, 0.0, 0.2], 'c_bar': [0.2, 0.0, 0.4], 'alpha_w': 0.8, 'alpha_c': 0.7, 'alpha_L': 0.0, 'kappa': [1, 1, 1], 'eta': [2, 2, 2], 'eta_w': 0.25, 'L': 1, 'eta_r': 2, 'q0': [0.01, 0.1, 0.1], 'p0': [1.0, 0.8, 0.5], 's0': [0.01, 0.1, 0.25], 'm_w0': 0.5, 'w0': 0.5, 'r0': 0.0, 'T': 100}}}
        _dump_to_yaml(default_presets)
        return default_presets

def _dump_to_yaml(presets):
    class FlowDumper(yaml.SafeDumper):
        pass

    def _repr_list(dumper, data):
        # always use flow style for lists
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    FlowDumper.add_representer(list, _repr_list)
    FlowDumper.ignore_aliases = lambda *a, **k: True

    text = yaml.dump(
        {"presets": presets},
        Dumper=FlowDumper,
        sort_keys=False,
        indent=2,
        width=88
    )

    with open("params.yml", "w") as f:
        f.write(text)

class NoAliasDumper(yaml.SafeDumper): # ????? i thought yaml was supposed to make this easier wtf is this
    def ignore_aliases(self, data):
        return True

class MainWindow(qw.QMainWindow):
    def __init__(self, init_params, trajectory_function, entry_boxes, dropdown_choices, dropdown_tooltips, presets):
        super().__init__()
        self.setWindowTitle("Dynamic Equilibrium Model")
        self.presets = presets
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

        self.params = init_params
        self.get_trajectories = trajectory_function
        self.traj, self.t = self.get_trajectories(self.params)

        self.dropdown_choices = dropdown_choices
        self.current_dropdown_choice = 0
        self.graph_panel = GraphPanel(self.traj, self.t, self.dropdown_choices, self.params.T)
        self.control_panel = ControlPanel(
            entry_boxes, self.params,
            dropdown_choices, dropdown_tooltips)
        self.control_panel.paramChanged.connect(self.update_plot)
        self.control_panel.checkStateChanged.connect(self.new_check_update)
        self.graph_panel.TChanged.connect(self.update_plot)
        self.control_panel.plotChoiceChanged.connect(self.switch_plot)

        self.main_layout = qw.QHBoxLayout()
        self.main_layout.addWidget(self.control_panel, stretch=2)
        self.main_layout.addWidget(self.graph_panel, stretch=5)

        self.main_container = qw.QWidget()
        self.main_container.setLayout(self.main_layout)

        self.setCentralWidget(self.main_container)

    def update_plot(self, name, new_val):
        setattr(self.params, name, new_val)
        self.traj, self.t = self.get_trajectories(self.params)
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
        self.params = params_from_mapping(self.presets[preset]["params"])
        self.traj, self.t = self.get_trajectories(self.params)
        options = self.control_panel.dropdown_widget.get_current_checked_boxes()
        self.graph_panel.make_plot(self.traj, self.t, self.current_dropdown_choice, options)
        self.control_panel.load_new_params(self.params)

    def save_preset(self):
        with open("./params.yml", "r") as f:
            presets = yaml.safe_load(f)["presets"]

        params_dict = self.to_plain(self.params)
        dialog = SaveDialog(presets.keys(), self)
        try:
            shortname, name, desc = dialog.bootstrap()
        except TypeError:
            return
        presets[shortname] = {"name": name, "desc": desc, "params": params_dict}

        _dump_to_yaml(presets)
        
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

        _dump_to_yaml(presets)
        for action in self.presets_submenu.actions():
            submenu = action.menu()
            if submenu.property("preset_id") == preset:
                self.presets_submenu.removeAction(action)
                submenu.deleteLater()
                break

    def rename_preset(self, old_shortname):
        with open("params.yml", "r") as f:
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

        _dump_to_yaml(presets)
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
        dialog.exec()

    def to_plain(self, obj): # opaque as fuck chatgpt code for converting the parameters dataclass to a yaml-friendly dictionary
        """Recursively convert dataclass / numpy types to YAML-friendly Python types."""
        if is_dataclass(obj):
            obj = asdict(obj)
        if isinstance(obj, dict):
            return {k: self.to_plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.to_plain(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

if __name__ == "__main__":
    app = qw.QApplication([])

    presets = load_presets('./params.yml')

    if len(sys.argv) == 1:
        try:
            params_dict = presets[next(iter(presets))]
        except StopIteration:
            default_presets = {"baseline": {'name': 'Default', 'desc': '', 'params': {'A': [[0.2, 0.0, 0.4], [0.2, 0.8, 0.0], [0.0, 0.1, 0.1]], 'l':
     [0.7, 0.6, 0.3], 'b_bar': [0.6, 0.0, 0.2], 'c_bar': [0.2, 0.0, 0.4], 'alpha_w': 0.8, 'alpha_c': 0.7, 'alpha_L': 0.0, 'kappa': [1, 1, 1], 'eta': [2, 2, 2], 'eta_w': 0.25, 'L': 1, 'eta_r': 2, 'q0': [0.01, 0.1, 0.1], 'p0': [1.0, 0.8, 0.5], 's0': [0.01, 0.1, 0.25], 'm_w0': 0.5, 'w0': 0.5, 'r0': 0.0, 'T': 100}}}
            _dump_to_yaml(default_presets)
            params_dict = default_presets[next(iter(default_presets))]
    else:
        try:
            params_dict = presets[sys.argv[1]]
        except KeyError:
            print(f"Preset {sys.argv[1]} not found, loading the first thing in params.yaml.")
            params_dict = presets[next(iter(presets))]

    init_params = params_from_mapping(params_dict["params"])

    entry_boxes = {
        "r0": {"label": r"$r(0)= $", "range": (0,1), "init_val": init_params.r0, "num_type": "float", "tooltip": "Initial interest rate. Capitalists finance all production through borrowing at this rate."},
        "m_w0": {"label": r"$m_w(0)= $", "range": (0,1), "init_val": init_params.m_w0, "num_type": "float", "tooltip": "Initial worker savings. Total circulating money in the economy is fixed at M=1. Understand that all money in the simulation is possessed by either workers or capitalists. Thus the more money workers have, the less capitalists have, driving up interest rates."},
        "L": {"label": r"$L= $", "range": (0,100), "init_val": init_params.L, "num_type": "float", "tooltip": "Initial total amount of labor available for employment (can be broken into arbitrarily small pieces). Note this stands in for the amount of workers available."},
        "w0": {"label": r"$w(0)= $", "range": (0,1), "init_val": init_params.w0, "num_type": "float", "tooltip": "Initial hourly wage rate."},
        "alpha_w": {"label": r"$\alpha_w= $", "range": (0,1), "init_val": init_params.alpha_w, "num_type": "float", "tooltip": "Worker propensity to consume. Workers spend this proportion of their savings each period on means of subsistence."},
        "alpha_c": {"label": r"$\alpha_c= $", "range": (0,1), "init_val": init_params.alpha_c, "num_type": "float", "tooltip": "Capitalist propensity to consume. Capitalists spend this proportion of their savings each period on consumption."},
        "eta_w": {"label": r"$\eta_w= $", "range": (0,2), "init_val": init_params.eta_w, "num_type": "float", "tooltip": "Constant of integration representing the elasticity of the hourly wage with respect to unemployment. Essentially, the higher this number is, the more dramatically wages will change with unemployment."},
        "eta_r": {"label": r"$\eta_r= $", "range": (0,2), "init_val": init_params.eta_r, "num_type": "float", "tooltip": "Constant of integration representing the elasticity of the interest rate with respect to capitalist savings. Essentially, the higher this number is, the more dramatically the interest rate will fall with respect to changes in the total money posessed by capitalists."},
        "alpha_L": {"label": r"$\alpha_L= $", "range": (0,1), "init_val": init_params.alpha_L, "num_type": "float", "tooltip": "Rate of population growth of the working class."},
        "T": {"label": r"$T= $", "range": (10,1000), "init_val": init_params.T, "num_type": "int", "tooltip": "Number of periods simulated. Keep in mind larger values of this will slow down simulation speed."}
    }

    dropdown_choices = [
        "Prices vs Values",
        "Prices vs Equilibrium Prices",
        "Output and Supply",
        "Wages and Employment",
        "Money Rates of Profit",
        "Distribution of Labor Time",
        "Distribution of Income"
    ]

    dropdown_tooltips = {choice: "" for choice in dropdown_choices}
    dropdown_tooltips["Prices vs Values"] = "Labor values here are converted into amounts of money via multiplication by the hourly wage (hours * dollars/hour = dollars) for the sake of comparison. If you see values changing despite unchanging technology, this is the sole reason."

    window = MainWindow(init_params, get_trajectories, entry_boxes, dropdown_choices, dropdown_tooltips, presets)

    window.show()
    app.exec()

    with open("dimensions.txt", "w") as f:
        print(window.graph_panel.xlim, file= f)
        print(window.graph_panel.ylim, file= f)
