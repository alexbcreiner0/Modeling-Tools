import yaml
import os
import pprint
import subprocess
import json
import sys
from pathlib import Path

# This mostly contains wizards for creating new simulations.

def get_dict_from_cpp(executable_path: str) -> dict:

    """ Runs a C++ binary from the given path, takes its standard output, assumes it is in yaml format and turns it into a dictionary """
    completed = subprocess.run(
        [executable_path],
        check=True,
        capture_output=True,
        text=True,
    )

    output = completed.stdout.strip()
    data = yaml.safe_load(output)
    return data

def create_new_model_dir():

    name = input("Model folder name: ")

    if os.path.isdir(name):
        print("Folder already exists. Either rename it or delete it and try again.")
        return

    os.mkdir(f"models/{name}")
    os.mkdir(f"models/{name}/data")
    os.mkdir(f"models/{name}/simulation")
    with open(f"models/{name}/__init__.py", "w"): pass
    with open(f"models/{name}/data/control_panel_data.yml", "w"): pass
    with (
        open(f"models/{name}/data/params.yml", "w") as fout,
        open(f"templates/params_dot_yml.txt") as fin
    ):
        print(fin.read(), file= fout)
    with open(f"models/{name}/simulation/__init__.py", "w"): pass
    with (
        open(f"models/{name}/simulation/parameters.py", "w") as fout,
        open(f"templates/parameters_dot_py.txt", "r") as fin
    ):
        print(fin.read(), file= fout)
    with (
        open(f"models/{name}/simulation/simulation.py", "w") as fout,
        open(f"templates/simulation_dot_py.txt", "r") as fin
    ):
        print(fin.read(), file= fout)

    print("Done! Next steps are: \n 1. Fill in your simulation function in simulation/simulation.py",  
        "\n 2. Fill in your Params dataclass in simulation/parameters.py \n 3. Fill in your data/params.yml",
        "\n 4. Create and fill in your data/control_panel_data.yml file \n 5. Create and fill in your data/plotting_data.yml")
    print("A wizard currently exist for creating your plotting_data.yml file. Consider calling the new_plot_file() function next for that!")

def _add_plot_category(sim_name, yaml_name, name, title= None, x_label= None, y_label= None, tooltip= None, plots= None):

    with open(f"models/{sim_name}/data/plotting_data.yml", "r") as f:
        data = yaml.safe_load(f)

    if data is None: data = {}

    with open(f"models/{sim_name}/data/plotting_data_bak.yml", "w") as f:
        yaml.safe_dump(data, f)

    data[yaml_name] = {
        "name": name,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "tooltip": tooltip,
        "plots": plots
    }

    with open(f"models/{sim_name}/data/plotting_data.yml", "w") as f:
        yaml.safe_dump(data, f)

    os.remove(f"models/{sim_name}/data/plotting_data_bak.yml")
    print(f"Done adding category {name} to {sim_name}")

def _add_plot(sim_name, name, labels, category, traj_key, toggled= False, checkbox_name= None, linestyle= "solid", colors= ["red", "green", "blue"]):

    with open(f"models/{sim_name}/data/plotting_data.yml", "r") as f:
        data = yaml.safe_load(f)

    with open(f"models/{sim_name}/data/plotting_data_bak.yml", "w") as f:
        yaml.safe_dump(data, f)

    n = len(labels)
    colors = colors[0:n]
        
    plot_dict = {
        "labels": labels,
        "traj_key": traj_key,
        "toggled": toggled,
        "colors": colors,
        "linestyle": linestyle
    }

    if checkbox_name is not None:
        plot_dict["checkbox_name"] = checkbox_name
        plot_dict["toggled"] = toggled

    if data[category]["plots"] is None: data[category]["plots"] = {}

    data[category]["plots"][name] = plot_dict

    with open(f"models/{sim_name}/data/plotting_data.yml", "w") as f:
        yaml.safe_dump(data, f)

    os.remove(f"models/{sim_name}/data/plotting_data_bak.yml")
    print(f"Done adding plot {name} to {sim_name}.")

def _get_new_category_info() -> dict:

    output = {}

    correct = False
    while not correct:
        output["yaml_name"] = input("Internal name for category (no whitespace): ")
        output["name"] = input("Actual name for category (what the user will see): ")
        x_label_ans = input("x-axis title (leave blank for the default 'Time [t]'): ")
        output["x_label"] = "Time [t]" if not x_label_ans else x_label_ans
        y_label_ans = input("y-axis title (leave blank for a blank y-axis (e.g. rates or heterogeneous units)): ")
        output["y_label"] = None if not y_label_ans else y_label_ans
        tooltip_ans = input("Tooltip (any info you want the user to be able to see when hovering over the ? button). Leaving blank is fine: )")
        output["tooltip"] = None if not tooltip_ans else tooltip_ans

        print(f"Here is what you entered: ")
        pprint.pprint(output)
        feedback = input("Does this all look correct? (y/n) ")
        if feedback == "y": correct = True
            
    return output

def _get_new_plot_info(category= None) -> dict:
    output = {}

    correct = False
    while not correct:
        output["name"] = input("Plot name (internal, not for user): ")
        labels = []
        ans = input("Are you plotting a vector trajectory? (y/n) ")
        if ans.lower() == "y":
            done = False
            print("Type label names one at a time, or 'done' to finish.")
            i = 1
            while not done:
                string = input(f"Label {i}: ")
                if string.lower() == "done":
                    done = True
                    continue
                labels.append(string)
                i += 1
        else:
            labels.append(input("Name of plot (appears on legend): "))
        output["labels"] = labels
        if category is None:
            output["category"] = input("Category name (dropdown group) (case sensitive): ")
        else:
            output["category"] = category
        toggleable_answer = input("Toggleable? (y for yes, anything else to skip): ")
        toggleable = True if toggleable_answer == "y" else False
        if toggleable:
            output["checkbox_name"] = input("Checkbox name (press enter to skip): ")
            toggled_answer = input("Pre-toggled? y for yes, anything else for no: ")
            output["toggled"] = True if toggled_answer == "y" else False

        output["traj_key"] = input("Trajectory key: ")
        linestyle_answer = input("Linestyle (leave blank for solid, otherwise da for dashed and do for dotted: ")
        if linestyle_answer == "da":
            output["linestyle"] = "dashed"
        elif linestyle_answer == "do":
            output["linestyle"] = "dotted"
        else:
            output["linestyle"] = "solid"

        colors = []
        if len(labels) == 1:
            color = input("Color of plot (Basic names and hex codes are both accepted. Hex codes should begin with #): ")
            colors.append(color)
        else:
            print("Enter colors for your plots. Basic names and hex codes are both accepted. Hex codes should begin with #")
            for i in range(len(labels)):
                color = input(f"Color of plot {labels[i]}: ")
                colors.append(color)
        output["colors"] = colors

        print(f"You entered:")
        pprint.pprint(output)
        feedback = input("Does this all look correct? (y/n) ")
        if feedback.lower() == "y": correct = True

    return output

def _get_info_and_add_plot(sim_name, category= None):
    output = _get_new_plot_info(category)
    _add_plot(sim_name, **output)

def _get_info_and_add_category(sim_name):
    output = _get_new_category_info()
    _add_plot_category(sim_name, **output)

    return output["yaml_name"]

def new_plots(sim_name= None, category= None):
    """ Wizard for adding new plots to an existing plotting_data.yml file """
    if sim_name is None:
        sim_name = input("Enter folder name: ")
    if category is None:
        category = input("Enter category of plots to add to (internal name): ")

    first_run = True
    confirm_plot = "y"
    while True:
        if not first_run:
            confirm_plot = input("Create another plot for this category? (y/n) ")
        if not confirm_plot.lower() == "y":
            print("Done adding plots.")
            break

        _get_info_and_add_plot(sim_name, category)
        first_run = False

    print("Done")

def new_plot_file():
    """ Wizard for creating a new plotting_data.yml file """
    print("Before starting, you should have already created a folder for your simulation, and it should already contain a data subfolder.")
    confirm = input("If there is already a plotting_data.yml file, this process will DELETE it. Continue? (y/n) ")
    if not confirm.lower() == "y":
        print("Aborting.")
        return
    
    sim_name = input("Sim name (the name of the folder): ")
    try:
        with open(f"models/{sim_name}/data/plotting_data.yml", "w") as f:
            pass
    except OSError:
        print("Error! Did you create the necessary folders and spell the name right?")
        print("Aborting")
        return

    print("File created. Entering category and plot creation phase.")
    while True:
        confirm_category = input("Create a plot category? (y/n) ")
        if not confirm_category.lower() == "y":
            print("Done adding categories.")
            break

        category = _get_info_and_add_category(sim_name)

        while True:
            confirm_plot = input("Create a new plot for this category? (y/n) ")
            if not confirm_plot.lower() == "y":
                print("Done adding plots.")
                break

            _get_info_and_add_plot(sim_name, category)

if __name__ == "__main__":
    # Call your functions here
    
    # Both arguments optional for both of these
    # new_plots(sim_name, category) <- For adding plots to an existing plotting_data.yaml file
    # new_plot_file(sim_name, category) <- For creating a brand new file and initial population

    # new_plots()
    # create_new_model_dir()

