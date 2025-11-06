<img width="1908" height="1026" alt="image" src="https://github.com/user-attachments/assets/016e9b36-8a78-442c-a310-73d21f475b60" />

Visualization suite initially created for exploring Ian Wright's dynamic equilibrium model. I've modified his model slightly by allowing for a growing labor force. With this change, we can witness a convergence towards uniform profit rates and the classical Sraffian price system (despite supply demand *disequilibrium*!). Assumptions of the model:

- All the classical Sraffian stuff (fixed unique production techniques for each commodity, uniform wages, etcetera - if you're here you don't need to hear me recite these). 
- A fixed amount of money circulating in the economy
- Population size changing at a fixed rate (e.g. dL/dt = kL for some k, where L is the available labor. k can be negative or zero)
- Workers accumulate money in the form of wages, capitalists accumulate it in the form of profit. Both workers and capitalists spend a fixed proportion of their savings each period on a fixed commodity bundle scaled according to how much that money can buy.
- Capitalists choose to produce more or less of something depending on whether the production process look profitable. E.g. if revenue looks to exceed cost at current prices then more is produced, revenue precedes cost means less is produced, break-even means no change in how much is produced.
- Wages change proportionally to changes in employment, and inversely proportionally to the size of the reserve army.
- Prices change proportionally to the change in supply, and approach infinity as supply approaches zero (supply changes according to the aforementioned capitalist and worker consumption as well as means of production usage)
- Capitalists borrow *all* money used to produce from creditors (or seen differently loan it to each other) at an interest rate which changes inversely proportionally to the change in the capitalist class's total savings, and which goes to infinity as those savings go to zero. This amount of money may exceed the total money circulating in the economy (e.g. fictitious capital), however debt does not accumulate. Rather, they simply surrender a portion of their revenue according to the interest rate. (Note, they are surrendering it to themselves. Hmm...)

The model is extremely robust and has a ton of possible applications. It's basically a (classical, e.g. non-probabilistic) capitalist economy in a box. Plug it into whatever you want. See [Ian Wright's PhD thesis](http://pinguet.free.fr/wrightthesis.pdf/) for more information on the model. There is a lot I am planning on adding and doing to this, and by the time you download it it will probably already look different from the screenshot. Have fun!

## To run
- Install [https://www.python.org/](Python) if you don't have it, make sure you check the 'add to system path' checkbox in the process if you are a Windows user.
- Clone the repo onto your computer (either by opening up a terminal and typing `git clone https://github.com/alexbcreiner0/Classical-Dynamic-Equilibrium-Model-Visualization-Tool.git` (must have git installed) or by downloading and extract the zip folder (found by clicking the green code button))
- Open up a terminal, navigate inside the folder to the folder:
```
cd Classical-Dynamic-Equilibrium-Model-Visualization-Tool
```
- (Optional but recommended) Create and enter a virtual environment:
```
python -m venv venv
source venv/bin/activate
```
- Install the necessary packages:
```
pip install -r requirements.txt
```
- Then run the `main.py` file:
```
python main.py
```

## Usage
This code is meant to be easily repurposed to quickly create similar applications for any model. As an example, I've added in a dynamic and constrained version of Morishima's model from Marx's Economics. To switch models, open up the `default_settings.yml` file and alter the simulation model to point at the folder you want. Try setting the `simulation_model` equal to `constrained_morishima` and see what happens!

How this works, and how to make your own models, is as follows. First, create a folder named something descriptive, in the outer directory. The directory structure must look like the following:

```
folder_name/
├── __init__.py
├── data/
|   |-- control_panel_data.yml
|   |-- extra_data.yml
|   |-- params.yml
|   |__ plotting_data.yml
└── simulation/
    ├── __init__.py
    ├── parameters.py
    └── simulation.py
```

To explain all of this, let's start inside of the simulation folder. 
### parameters.py
This file is only expected to have a single class defined, called `Params`, which is given the `@dataclass` decorator. This dataclass will contain all of the stuff which matters to the simulation. See one of the existing `parameters.py` files for guidance on how to define it - it is quite simple and self-explanatory.

### simulation.py
The file is expected to contain a function, whose name matches the `simulation_function` paremeter in the `default_settings.yml` file (it should default to `get_trajectories`). This function takes a single positional argument, which is expected to be an instance of the dataclass you defined in the above file. In here, you should do all of the relevant computations, and then return two things:
    1. A dictionary called `traj` containing *all* relevant trajectories that you will want to plot. Keys are expected to be strings and values are expected to be numpy arrays. These arrays may be multidimensional. For example, you could have `traj["p"]` be the evolving price vectors for any number of commodities, and would thereby be a matrix. 
    2. A numpy array of times for the x-axis, (e.g. `t = np.linspace(a,b,N)`)

### plotting_data.yml
This yaml file specifies the plots which are available to see through the dropdown widget of the control panel and the checkboxes. Sometimes, you will be plotting vector trajectories (e.g. prices for multiple commodities). It is expected that there be a single dictionary entry for all types of prices, e.g. that the dictionary value is a matrix whose columns are the individual trajectories. The number of labels and colors MUST match the number of columns of this matrix. Checkbox name is optional. If there is one then there MUST also be an entry called toggled, set to true or false, which specifies whether the plot will be showing at launch or not. Colors are necessary and expected, but linestyle is not and will default to solid. on_startup is a bandaid setting meant to address a bug. It's only really needed for plots on the first dropdown window that displays (dropdown options display in the order listed) when the app is launched, which have checkboxes but which are meant to be pre-checked on. Without it, they will not show up until the user changes something.

### control_panel_data.yml
This yaml file contains a per-row specification of the widgets which are to appear in the control panel. A control widget at present is assumed to always associate with a specific parameter of the model. The yaml name is not important, but the param_name MUST perfectly match the name of the associated parameter. The program will crash if there is not a label or tooltip field, but you can leave it blank. 

There are three types: matrix, vector, and scalar. Different widgets will be created depending on the type. If your control widget is a matrix or a vector, then the dimension of the matrix dim MUST be specified. For a matrix, the dim needs to be a list, e.g. [3,3] for a 3x3 matrix. For a vector, it just needs to be an integer. If your control widget is a scalar, then you MUST specify a slider range ([0,1] for a slider from 0 to 1) and a scalar_type (either "float" or "int")


### params.yml
Contains initial settings for your Params dataclass to instantiate to. The application can save and load new settings dynamically. The default preset is specified in the `default_settings.yml` file. If no presets are available or the file is missing, the program will try to create a new one, and populate it initially with a preset that it will look for in the `extra_data.yml` file

### extra_data.yml
At present, all that this contains is a default preset to fall back on if something goes wrong when loading one from the `params.yml` file. Not strictly necessary.

The overall idea is this: 
- Create a simulation function and a Params dataclass
- Create a plotting_data.yml file containing the info for what should be plotted
- Create a control_panel_data.yml file specifying what controls should be available.
- Create a params.yml file specifying default initial conditions
- Point the app to these things by altering the default_settings.yml file
- Run the application, and you have a nice model to mess around with!
