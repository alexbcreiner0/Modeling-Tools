![Demo](assets/demo.gif)

# Tex's Modelling Tools

Model visualization and exploration toolkit. Create Desmos-style interfaces quickly and easily for models of all kinds. Simulation PDE's? Running agent-based simulation? No problem! All you need to do is declare your system parameters, make a single Python function which executes the simulation and returns a dictionary of the plots you want to display, and write up a small yaml file declaring the controls you want to have. The tools will handle the rest and create a display of all of your plots so that you can edit your parameters and view the results in real time. 

# Installation
This software was primarily developed for the visualization of a few specific models which I am currently writing papers for. If you are just trying to use the accompanying software to those papers, binary releases are available for you which can be simply downloaded and ran to display the relevant model. Just click the appropriate link in the section directly below this one. If you are interested in interacting directly with the tools yourself and making alterations or building your own models, see the instructions below that.

## Binary Releases and Accompanying Papers

- [For my upcoming paper titled 'Empirical Redemption of Marx's Law of the TEndential Fall in the Rate of Profit Within Dynamic Cross-Dual Disequilibrium Models, click here](https://github.com/alexbcreiner0/Modeling-Tools/releases/tag/v1.0.0)
   - [The paper (currently in pre-publishing](https://www.alexcreiner.com/documents/rate-of-profit-paper.pdf)

## Download and Run the Code Yourself
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
