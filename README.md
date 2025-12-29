![Demo](assets/demo.gif)

Model visualization and exploration toolkit. Made primarily for the exploration of dynamic classical gravitation models, in particular that of Ian Wright

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
