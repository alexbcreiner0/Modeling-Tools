from PyQt6 import (
    QtWidgets as qw,
    QtGui as qg
) 
import yaml
from loader import load_presets, _dump_to_yaml, params_from_mapping
import sys, importlib
from pathlib import Path
from MainWindow import MainWindow
import yaml

if __name__ == "__main__":
    app = qw.QApplication([])

    PROJECT_ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(PROJECT_ROOT))

    with open("default_settings.yml", "r") as f:
        default_settings = yaml.safe_load(f)

    sim_model = default_settings["simulation_model"]
    sim_function = default_settings["simulation_function"]
    default_preset = default_settings["default_preset"]

    presets = load_presets(sim_model)
    trajectories_module = importlib.import_module(f"{sim_model}.simulation.simulation")
    get_trajectories = getattr(trajectories_module, sim_function)

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

    init_params = params_from_mapping(params_dict["params"], f"{sim_model}/simulation/parameters.py")
    print(f"init_params loaded = {init_params}")

    with open(f"{sim_model}/data/plotting_data.yml") as f:
        plotting_data = yaml.safe_load(f)

    with open(f"{sim_model}/data/control_panel_data.yml") as f:
        panel_data = yaml.safe_load(f)

    window = MainWindow(init_params, get_trajectories, presets, panel_data, plotting_data)

    window.show()
    app.exec()

    with open("dimensions.txt", "w") as f:
        print(window.graph_panel.xlim, file= f)
        print(window.graph_panel.ylim, file= f)
