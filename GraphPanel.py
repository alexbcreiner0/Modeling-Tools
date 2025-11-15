import sys
from PyQt6 import (
    QtCore as qc,
    QtWidgets as qw,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib import pyplot as plt
import numpy as np
import scienceplots
plt.style.use(["grid", "notebook"])

class GraphPanel(qw.QWidget):
    saved_lims_changed = qc.pyqtSignal(tuple, tuple)

    def __init__(self, init_traj, init_t, dropdown_choices, T,
                 plotting_data, canvas, figure, axis, toolbar,
                 entries, save_button, load_button):
        super().__init__()
        self.start_up = True
        self.data = plotting_data
        layout = qw.QVBoxLayout()
        self.dropdown_choices = dropdown_choices
        self.canvas = canvas
        self.figure, self.axis = figure, axis
        self.toolbar = toolbar

        try:
            with open("dimensions.txt", "r") as f:
                xlim_str = f.readline().strip().strip('()').split(',')
                ylim_str = f.readline().strip().strip('()').split(',')
                self.xlim = tuple([float(i) for i in xlim_str])
                self.ylim = tuple([float(i) for i in ylim_str])
        except FileNotFoundError:
            with open("dimensions.txt", "w") as f:
                print(str((0,50)),"\n",str((0,50)), file= f)
            self.xlim, self.ylim = (0,50), (0,50)
        self.saved_xlim, self.saved_ylim = self.xlim, self.ylim
        self.xlower_entry, self.xupper_entry = entries[0], entries[2]
        self.ylower_entry, self.yupper_entry = entries[1], entries[3]
        self.xlower_entry.setText(str(self.xlim[0]))
        self.xupper_entry.setText(str(self.xlim[1]))
        self.ylower_entry.setText(str(self.ylim[0]))
        self.yupper_entry.setText(str(self.ylim[1]))
        for entry in entries: 
            entry.setSizePolicy(qw.QSizePolicy.Policy.Fixed,qw.QSizePolicy.Policy.Fixed)
            entry.setFixedWidth(70)
            entry.textChanged.connect(self.edit_axes) # KEEP THIS
        self.save_button, self.load_button = save_button, load_button
        self.save_button.clicked.connect(self.save_axes)
        self.load_button.clicked.connect(self.load_axes)
        # self.saved_x_label, self.saved_y_label = saved_labels[0], saved_labels[1]
        # self.saved_lims_changed.emit(self.saved_xlim, self.saved_ylim)
        # self.saved_x_label.setText(str(self.saved_xlim))
        # self.saved_y_label.setText(str(self.saved_ylim))
      
        self.camera_controls = qw.QWidget()
        self.toolbar.pan()

        layout.addWidget(self.canvas, stretch=5)
        self.setLayout(layout)
        self.T = T
        self.edit_axes()
        self.make_plot(init_traj, init_t, 0, {})

        self.start_up = False

        self.axis.callbacks.connect("xlim_changed", self._on_axis_limits_changed)
        self.axis.callbacks.connect("xlim_changed", self._on_axis_limits_changed)
        self._block_axis_callback = False

    def _on_axis_limits_changed(self, ax):
        if self._block_axis_callback:
            return

        # Read current limits from the axes
        self.xlim = ax.get_xlim()
        self.ylim = ax.get_ylim()

        # Update the QLineEdits without re-triggering edit_axes
        widgets_and_values = [
            (self.xlower_entry, self.xlim[0]),
            (self.xupper_entry, self.xlim[1]),
            (self.ylower_entry, self.ylim[0]),
            (self.yupper_entry, self.ylim[1]),
        ]

        for w, v in widgets_and_values:
            w.blockSignals(True)
            text = f"{v:.3f}"
            w.setText(text)
            w.blockSignals(False)

    def edit_axes(self):
        try:
            new_xlim = (float(self.xlower_entry.text()), float(self.xupper_entry.text()))
            new_ylim = (float(self.ylower_entry.text()), float(self.yupper_entry.text()))

            self._block_axis_callback = True
            self.axis.set_xlim(new_xlim)
            self.axis.set_ylim(new_ylim)
            self.canvas.draw_idle()
            self._block_axis_callback = False

            self.xlim, self.ylim = new_xlim, new_ylim

        except ValueError:
            self._block_axis_callback = False

    def save_axes(self):
        self.saved_xlim, self.saved_ylim = (float(f"{float(self.xlim[0]):.3f}"), float(f"{float(self.xlim[1]):.3f}")), (float(f"{float(self.ylim[0]):.3f}"), float(f"{float(self.ylim[1]):.3f}"))
        self.saved_lims_changed.emit(self.saved_xlim, self.saved_ylim)

    def load_axes(self):
        self.xlim, self.ylim = self.saved_xlim, self.saved_ylim

        
        widgets_and_values = [
            (self.xlower_entry, self.xlim[0]),
            (self.xupper_entry, self.xlim[1]),
            (self.ylower_entry, self.ylim[0]),
            (self.yupper_entry, self.ylim[1]),
        ]

        for w, v in widgets_and_values:
            w.blockSignals(True)
            w.setText(f"{v:.3f}")   # or str(v)
            w.blockSignals(False)

        self.edit_axes()

    def make_plot(self, traj, t, dropdown_choice, options):
        current_xlim = self.axis.get_xlim()
        current_ylim = self.axis.get_ylim()
        self.axis.clear()
        self.axis.set_xlabel("Time [t]")

        dropdown_list = list(self.data.keys())
        plots = self.data[dropdown_list[dropdown_choice]]["plots"]

        for plot in plots:
            plot_dict = plots[plot]
            n = len(plot_dict["labels"])
            if "checkbox_name" in plot_dict:
                if plot_dict["checkbox_name"] in options or (self.start_up and "on_startup" in plot_dict):
                    if len(plot_dict["labels"]) == 1:
                        if "linestyle" in plot_dict: 
                            linestyle = plot_dict["linestyle"] 
                        else: 
                            linestyle = "solid"
                        self.axis.plot(t, traj[plot_dict["traj_key"]], color= plot_dict["colors"][0], linestyle= linestyle, label= plot_dict["labels"][0])
                    else:
                        n = len(plot_dict["labels"])
                        if "linestyle" in plot_dict:
                             linestyle = plot_dict["linestyle"] 
                        else: 
                            linestyle = "solid"
                        for i in range(n):
                            self.axis.plot(t, traj[plot_dict["traj_key"]][:,i], color= plot_dict["colors"][i], linestyle= linestyle, label= plot_dict["labels"][i])
            else:
                if len(plot_dict["labels"]) == 1:
                    if "linestyle" in plot_dict: 
                        linestyle = plot_dict["linestyle"] 
                    else: 
                        linestyle = "solid"
                    self.axis.plot(t, traj[plot_dict["traj_key"]], color= plot_dict["colors"][0], linestyle= linestyle, label= plot_dict["labels"][0])
                else:
                    n = len(plot_dict["labels"])
                    if "linestyle" in plot_dict:
                         linestyle = plot_dict["linestyle"] 
                    else: 
                        linestyle = "solid"
                    for i in range(n):
                        self.axis.plot(t, traj[plot_dict["traj_key"]][:,i], color= plot_dict["colors"][i], linestyle= linestyle, label= plot_dict["labels"][i])

        self.axis.set_xlim(current_xlim)
        self.axis.set_ylim(current_ylim)
        self.axis.legend()
        self.canvas.draw_idle()
        plt.tight_layout()

    # def make_plot2(self, traj, t, dropdown_choice, options):
    #     current_xlim = self.axis.get_xlim()
    #     current_ylim = self.axis.get_ylim()
    #     self.axis.clear()
    #     self.axis.set_xlabel("Time [t]")
    #     if dropdown_choice == 0: # prices

    #         if "Toggle prices" in options or self.start_up:
    #             self.axis.plot(t, traj["corn_prices"], color="red", label="Price of Corn")
    #             self.axis.plot(t, traj["iron_prices"], color="green", label="Price of Iron")
    #             self.axis.plot(t, traj["sugar_prices"], color="blue", label="Price of Sugar")

    #         if "Toggle values" in options or self.start_up:
    #             self.axis.plot(t, traj["corn_values"], color="red", linestyle="dashed", label="Value of Corn")
    #             self.axis.plot(t, traj["iron_values"], color="green", linestyle="dashed", label="Value of Iron")
    #             self.axis.plot(t, traj["sugar_values"], color="blue", linestyle="dashed", label="Value of Sugar")

    #         if "Toggle equilibrium prices" in options:
    #             self.axis.plot(t, traj["epr_corn_prices"], color="red", linestyle="dotted", label="Equilibrium Price of Corn")
    #             self.axis.plot(t, traj["epr_iron_prices"], color="green", linestyle="dotted", label="Equilibrium Price of Iron")
    #             self.axis.plot(t, traj["epr_sugar_prices"], color="blue", linestyle="dotted", label="Equilibrium Price of Sugar")

    #     elif dropdown_choice == 1: # prices vs epr prices

    #         if "Toggle prices" in options or self.start_up:
    #             self.axis.plot(t, traj["corn_prices"], color="red", label="Price of Corn")
    #             self.axis.plot(t, traj["iron_prices"], color="green", label="Price of Iron")
    #             self.axis.plot(t, traj["sugar_prices"], color="blue", label="Price of Sugar")

    #         if "Toggle values" in options or self.start_up:
    #             self.axis.plot(t, traj["corn_values"], color="red", linestyle="dotted", label="Value of Corn")
    #             self.axis.plot(t, traj["iron_values"], color="green", linestyle="dotted", label="Value of Iron")
    #             self.axis.plot(t, traj["sugar_values"], color="blue", linestyle="dotted", label="Value of Sugar")

    #         if "Toggle equilibrium prices" in options:
    #             self.axis.plot(t, traj["epr_corn_prices"], color="red", linestyle="dashed", label="Equilibrium Price of Corn")
    #             self.axis.plot(t, traj["epr_iron_prices"], color="green", linestyle="dashed", label="Equilibrium Price of Iron")
    #             self.axis.plot(t, traj["epr_sugar_prices"], color="blue", linestyle="dashed", label="Equilibrium Price of Sugar")

    #     elif dropdown_choice == 2: # outputs
    #         self.axis.plot(t, traj["s"][:,0], 'r-', label="Corn Supply")
    #         self.axis.plot(t, traj["s"][:,1], 'g-', label="Iron Supply")
    #         self.axis.plot(t, traj["s"][:,2], 'b-', label="Sugar Supply")

    #         self.axis.plot(t, traj["q"][:,0], 'r--', label="Corn Output")
    #         self.axis.plot(t, traj["q"][:,1], 'g--', label="Iron Output")
    #         self.axis.plot(t, traj["q"][:,2], 'b--', label="Sugar Output")

    #     elif dropdown_choice == 3: # wages and employment
    #         wages, employment = traj["w"], traj["total_labor_employed"]
    #         L = traj["L"]
    #         self.axis.plot(t, wages, 'k-', label="Hourly Wage")
    #         self.axis.plot(t, employment, 'r-', label="Employment")
    #         self.axis.plot(t, L-employment, 'b', label= "Size of Reserve Army")
    #         self.axis.plot(t, traj["labor_demand"], label= "Demand for Labor")

    #     elif dropdown_choice == 4: # rates
    #         interest = traj["r"]
    #         epr_profit_rates = traj["epr_profit_rates"]
    #         e = traj["e"]
    #         profit_rates = traj["profit_rates"]
    #         # self.axis.set_ylim(0,5)
    #         # self.axis.plot(t, interest, 'k--', label="Interest Rate")

    #         if "Toggle equilibrium RoP" in options:
    #             self.axis.plot(t, epr_profit_rates, color="orange", linestyle='--', label="Equilibrium Profit Rate")

    #         if "Toggle sectoral RoPs" in options:
    #             self.axis.plot(t, profit_rates[:,0], color="red", linestyle='-', label="Corn Sector Profit Rate")
    #             self.axis.plot(t, profit_rates[:,1], color="green", linestyle='-', label="Iron Sector Profit Rate")
    #             self.axis.plot(t, profit_rates[:,2], color="blue", linestyle='-', label="Sugar Sector Profit Rate")

    #         if "Toggle value RoP" in options:
    #             self.axis.plot(t, traj["value_rops"], label="Value Rate of Profit")

    #         if "Toggle rate of exploitation" in options:
    #             self.axis.plot(t, traj["e"], color="purple", linestyle="-", label="Rate of Exploitation")

    #         if "Toggle interest rate" in options:
    #             self.axis.plot(t, traj["r"], color="black", linestyle="--", label="Interest rate")

    #     elif dropdown_choice == 5: # value distn
    #         self.axis.plot(t, traj["surplus_vals"], 'g-', label="Surplus Value Produced")
    #         self.axis.plot(t, traj["values_ms"], 'r--', label="Value of Means of Subsistence")
    #         self.axis.plot(t, traj["cc_vals"], 'b-', label="Value of Constant Capital")

    #         # self.axis.set_ylabel("Hours [t]")

    #     elif dropdown_choice == 6:
    #         m_w = traj["m_w"]
    #         m_c = [1-m_wi for m_wi in m_w]
    #         wages = traj["w"]
    #         e = traj["e"]
    #         self.axis.plot(t, e, 'y--', label="Rate of Exploitation")
    #         self.axis.plot(t, wages, 'b-', label="Hourly Wage")
    #         self.axis.plot(t, m_w, 'k--', label="Worker savings")
    #         self.axis.plot(t, m_c, 'r--', label="Capitalist savings")

    #     self.axis.set_xlim(current_xlim)
    #     self.axis.set_ylim(current_ylim)
    #     self.axis.legend()
    #     self.canvas.draw_idle()
    #     plt.tight_layout()


