from PyQt6 import QtWidgets as qw
from ControlPanel import ControlPanel
from GraphPanel import GraphPanel
from mathstuff import get_trajectories
from mathstuff import *
from init_data import params as init_params
import sys

class MainWindow(qw.QMainWindow):
    def __init__(self, init_params, trajectory_function, entry_boxes, dropdown_choices, dropdown_tooltips):
        super().__init__()
        self.setWindowTitle("Dynamic Equilibrium Model")
        # self.resize(800,600)

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

if __name__ == "__main__":
    app = qw.QApplication([])

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

    window = MainWindow(init_params, get_trajectories, entry_boxes, dropdown_choices, dropdown_tooltips)

    window.show()
    app.exec()

    with open("dimensions.txt", "w") as f:
        print(window.graph_panel.xlim, file= f)
        print(window.graph_panel.ylim, file= f)
