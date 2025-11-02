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
    TChanged = qc.pyqtSignal(str, int)

    def __init__(self, init_traj, init_t, dropdown_choices, T):
        super().__init__()
        self.start_up = True
        layout = qw.QVBoxLayout()
        self.dropdown_choices = dropdown_choices
        self.figure, self.axis = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.figure)
        bottom_layout = qw.QHBoxLayout()
        matplotlib_layout = qw.QHBoxLayout()
        sub_layout1 = qw.QGridLayout()
        xrange_label_from = qw.QLabel("X-axis: from")
        xrange_label_to = qw.QLabel(" to ")
        yrange_label_from = qw.QLabel("Y-axis: from")
        yrange_label_to = qw.QLabel(" to ")
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
        self.xlower_entry, self.xupper_entry = qw.QLineEdit(str(self.xlim[0])), qw.QLineEdit(str(self.xlim[1]))
        self.ylower_entry, self.yupper_entry = qw.QLineEdit(str(self.ylim[0])), qw.QLineEdit(str(self.ylim[1]))
        entries = [self.xlower_entry, self.ylower_entry, self.xupper_entry, self.yupper_entry]
        for entry in entries: 
            entry.setSizePolicy(qw.QSizePolicy.Policy.Fixed,qw.QSizePolicy.Policy.Fixed)
            entry.setFixedWidth(70)
            entry.textChanged.connect(self.edit_axes)
        self.save_button = qw.QPushButton("Save Current Axes")
        self.save_button.clicked.connect(self.save_axes)
        self.load_button = qw.QPushButton("Load Saved Axes")
        self.load_button.clicked.connect(self.load_axes)
        sub_layout1.addWidget(xrange_label_from, 0, 0)
        sub_layout1.addWidget(self.xlower_entry, 0, 1)
        sub_layout1.addWidget(xrange_label_to,0,2)
        sub_layout1.addWidget(self.xupper_entry,0,3)
        sub_layout1.addWidget(yrange_label_from,1,0)
        sub_layout1.addWidget(self.ylower_entry, 1, 1)
        sub_layout1.addWidget(yrange_label_to,1,2)
        sub_layout1.addWidget(self.yupper_entry,1,3)
        sub_layout1.addWidget(self.save_button,2,0,1,2)
        sub_layout1.addWidget(self.load_button,2,2,1,2)

        sub_layout2 = qw.QVBoxLayout()

        self.saved_x_info = qw.QLabel("Saved X-axis: ")
        self.saved_y_info = qw.QLabel("Saved y-axis: ")
        self.saved_x_label = qw.QLabel()
        self.saved_x_label.setText(str(self.saved_xlim))
        self.saved_y_label = qw.QLabel()
        self.saved_y_label.setText(str(self.saved_ylim))
        self.recompute_button = qw.QPushButton("Recompute")
        self.recompute_button.clicked.connect(self.recompute_T)

        sub_layout2.addWidget(self.saved_x_info)
        sub_layout2.addWidget(self.saved_x_label)
        sub_layout2.addWidget(self.saved_y_info)
        sub_layout2.addWidget(self.saved_y_label)
        sub_layout2.addWidget(self.recompute_button)

        sub_widget1 = qw.QWidget()
        sub_widget1.setLayout(sub_layout1)
        sub_widget2 = qw.QWidget()
        sub_widget2.setLayout(sub_layout2)

        matplotlib_layout.addWidget(sub_widget1)
        matplotlib_layout.addWidget(sub_widget2)
       
        self.camera_controls = qw.QWidget()
        self.camera_controls.setLayout(matplotlib_layout)

        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        layout.addWidget(self.canvas, stretch=5)
        bottom_layout.addWidget(self.toolbar, stretch=2)
        bottom_layout.addWidget(self.camera_controls, stretch=2)
        bottom_bar = qw.QWidget()
        bottom_bar.setLayout(bottom_layout)
        layout.addWidget(bottom_bar)

        self.setLayout(layout)
        self.T = T
        self.edit_axes()
        self.make_plot(init_traj, init_t, 0, {})
        self.start_up = False

    def recompute_T(self):
        print("Recomputing T")
        new_T = int(self.xlim[1])
        self.T = new_T
        self.TChanged.emit("T", new_T)

    def edit_axes(self):
        try:
            new_xlim = (float(self.xlower_entry.text()), float(self.xupper_entry.text()))
            new_ylim = (float(self.ylower_entry.text()), float(self.yupper_entry.text()))
            self.xlim, self.ylim = new_xlim, new_ylim
            self.axis.set_xlim(self.xlim)
            self.axis.set_ylim(self.ylim)
            self.canvas.draw_idle()
        except ValueError:
            pass

    def save_axes(self):
        self.saved_xlim, self.saved_ylim = self.xlim, self.ylim
        self.saved_x_label.setText(str(self.xlim))
        self.saved_y_label.setText(str(self.ylim))

    def load_axes(self):
        self.xlim, self.ylim = self.saved_xlim, self.saved_ylim
        self.xlower_entry.setText(str(self.xlim[0]))
        self.xupper_entry.setText(str(self.xlim[1]))
        self.ylower_entry.setText(str(self.ylim[0]))
        self.yupper_entry.setText(str(self.ylim[1]))
        self.edit_axes()

    def make_plot(self, traj, t, dropdown_choice, options):
        # if not xlim:
        current_xlim = self.axis.get_xlim()
        # else:
        #     current_xlim = xlim
        # if not ylim:
        current_ylim = self.axis.get_ylim()
        # else:
        #     current_ylim = ylim
        self.axis.clear()
        self.axis.set_xlabel("Time [t]")
        if dropdown_choice == 0: # prices
            corn_prices = traj["p"][:,0]
            iron_prices = traj["p"][:,1]
            sugar_prices = traj["p"][:,2]
            values = traj["values"]
            epr_prices = traj["epr_prices"]
            epr_corn_prices = [epr_prices_i[0] for epr_prices_i in epr_prices]
            epr_iron_prices = [epr_prices_i[1] for epr_prices_i in epr_prices]
            epr_sugar_prices = [epr_prices_i[2] for epr_prices_i in epr_prices]


            wages = traj["w"]

            wage_values = np.array([np.array(values) for i in range(self.T)])
            for i,w in enumerate(wages):
                wage_values[i] *= w
            corn_values, iron_values, sugar_values = wage_values[:,0], wage_values[:,1], wage_values[:,2]

            if "Toggle prices" in options or self.start_up:
                self.axis.plot(t, corn_prices, color="red", label="Price of Corn")
                self.axis.plot(t, iron_prices, color="green", label="Price of Iron")
                self.axis.plot(t, sugar_prices, color="blue", label="Price of Sugar")

            if "Toggle values" in options or self.start_up:
                self.axis.plot(t, corn_values, color="red", linestyle="dashed", label="Value of Corn")
                self.axis.plot(t, iron_values, color="green", linestyle="dashed", label="Value of Iron")
                self.axis.plot(t, sugar_values, color="blue", linestyle="dashed", label="Value of Sugar")

            if "Toggle equilibrium prices" in options:
                self.axis.plot(t, epr_corn_prices, color="red", linestyle="dotted", label="Equilibrium Price of Corn")
                self.axis.plot(t, epr_iron_prices, color="green", linestyle="dotted", label="Equilibrium Price of Iron")
                self.axis.plot(t, epr_sugar_prices, color="blue", linestyle="dotted", label="Equilibrium Price of Sugar")

            # self.axis.set_ylabel("Dollars [$]")

        elif dropdown_choice == 1: # prices vs epr prices
            corn_prices = traj["p"][:,0]
            iron_prices = traj["p"][:,1]
            sugar_prices = traj["p"][:,2]
            values = traj["values"]
            epr_prices = traj["epr_prices"]
            epr_corn_prices = [epr_prices_i[0] for epr_prices_i in epr_prices]
            epr_iron_prices = [epr_prices_i[1] for epr_prices_i in epr_prices]
            epr_sugar_prices = [epr_prices_i[2] for epr_prices_i in epr_prices]


            wages = traj["w"]

            wage_values = np.array([np.array(values) for i in range(self.T)])
            for i,w in enumerate(wages):
                wage_values[i] *= w
            corn_values, iron_values, sugar_values = wage_values[:,0], wage_values[:,1], wage_values[:,2]

            if "Toggle prices" in options or self.start_up:
                self.axis.plot(t, corn_prices, color="red", label="Price of Corn")
                self.axis.plot(t, iron_prices, color="green", label="Price of Iron")
                self.axis.plot(t, sugar_prices, color="blue", label="Price of Sugar")

            if "Toggle values" in options:
                self.axis.plot(t, corn_values, color="red", linestyle="dotted", label="Value of Corn")
                self.axis.plot(t, iron_values, color="green", linestyle="dotted", label="Value of Iron")
                self.axis.plot(t, sugar_values, color="blue", linestyle="dotted", label="Value of Sugar")

            if "Toggle equilibrium prices" in options or self.start_up:
                self.axis.plot(t, epr_corn_prices, color="red", linestyle="dashed", label="Equilibrium Price of Corn")
                self.axis.plot(t, epr_iron_prices, color="green", linestyle="dashed", label="Equilibrium Price of Iron")
                self.axis.plot(t, epr_sugar_prices, color="blue", linestyle="dashed", label="Equilibrium Price of Sugar")

            # self.axis.set_ylabel("Dollars [$]")

            # corn_prices = traj["p"][:,0]
            # iron_prices = traj["p"][:,1]
            # sugar_prices = traj["p"][:,2]
            # epr_prices = traj["epr_prices"]

            # epr_corn_prices = [epr_prices_i[0] for epr_prices_i in epr_prices]
            # epr_iron_prices = [epr_prices_i[1] for epr_prices_i in epr_prices]
            # epr_sugar_prices = [epr_prices_i[2] for epr_prices_i in epr_prices]

            # self.axis.plot(t, corn_prices, 'r', label="Price of Corn")
            # self.axis.plot(t, epr_corn_prices, 'r--', label="EPR Price of Corn")
            # self.axis.plot(t, iron_prices, 'g', label="Price of Iron")
            # self.axis.plot(t, epr_iron_prices, 'g--', label="EPR Price of Iron")
            # self.axis.plot(t, sugar_prices, 'b', label="Price of Sugar")
            # self.axis.plot(t, epr_sugar_prices, 'b--', label="EPR Price of Sugar")

        elif dropdown_choice == 2: # outputs
            self.axis.plot(t, traj["s"][:,0], 'r-', label="Corn Supply")
            self.axis.plot(t, traj["s"][:,1], 'g-', label="Iron Supply")
            self.axis.plot(t, traj["s"][:,2], 'b-', label="Sugar Supply")

            self.axis.plot(t, traj["q"][:,0], 'r--', label="Corn Output")
            self.axis.plot(t, traj["q"][:,1], 'g--', label="Iron Output")
            self.axis.plot(t, traj["q"][:,2], 'b--', label="Sugar Output")

            # self.axis.set_label("Units")

        elif dropdown_choice == 3: # wages and employment
            wages, employment = traj["w"], traj["total_labor_employed"]
            L = traj["L"]
            self.axis.plot(t, wages, 'k-', label="Hourly Wage")
            self.axis.plot(t, employment, 'r-', label="Employment")
            self.axis.plot(t, L-employment, 'b', label= "Size of Reserve Army")
            self.axis.plot(t, traj["labor_demand"], label= "Demand for Labor")

        elif dropdown_choice == 4: # rates
            interest = traj["r"]
            epr_profit_rates = traj["epr_profit_rates"]
            e = traj["e"]
            profit_rates = traj["profit_rates"]
            # self.axis.set_ylim(0,5)
            # self.axis.plot(t, interest, 'k--', label="Interest Rate")

            if "Toggle equilibrium RoP" in options:
                self.axis.plot(t, epr_profit_rates, color="orange", linestyle='--', label="EPR Profit Rate")

            if "Toggle sectoral RoPs" in options:
                self.axis.plot(t, profit_rates[:,0], color="red", linestyle='-', label="Corn Sector Profit Rate")
                self.axis.plot(t, profit_rates[:,1], color="green", linestyle='-', label="Iron Sector Profit Rate")
                self.axis.plot(t, profit_rates[:,2], color="blue", linestyle='-', label="Sugar Sector Profit Rate")

            if "Toggle value RoP" in options:
                self.axis.plot(t, traj["value_rops"], label="Value Rate of Profit")

            if "Toggle rate of exploitation" in options:
                self.axis.plot(t, traj["e"], color="purple", linestyle="-", label="Rate of Exploitation")

            if "Toggle interest rate" in options:
                self.axis.plot(t, traj["r"], color="black", linestyle="--", label="Interest rate")

        elif dropdown_choice == 5: # value distn
            self.axis.plot(t, traj["surplus_vals"], 'g-', label="Surplus Value Produced")
            self.axis.plot(t, traj["values_ms"], 'r--', label="Value of Means of Subsistence")
            self.axis.plot(t, traj["cc_vals"], 'b-', label="Value of Constant Capital")

            # self.axis.set_ylabel("Hours [t]")

        elif dropdown_choice == 6:
            m_w = traj["m_w"]
            m_c = [1-m_wi for m_wi in m_w]
            wages = traj["w"]
            e = traj["e"]
            self.axis.plot(t, e, 'y--', label="Rate of Exploitation")
            self.axis.plot(t, wages, 'b-', label="Hourly Wage")
            self.axis.plot(t, m_w, 'k--', label="Worker savings")
            self.axis.plot(t, m_c, 'r--', label="Capitalist savings")

        self.axis.set_xlim(current_xlim)
        self.axis.set_ylim(current_ylim)
        self.axis.legend()
        self.canvas.draw_idle()
        plt.tight_layout()


