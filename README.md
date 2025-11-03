<img width="1908" height="1026" alt="image" src="https://github.com/user-attachments/assets/016e9b36-8a78-442c-a310-73d21f475b60" />

Visualization tool for exploring Ian Wright's dynamic equilibrium model. I've modified his model slightly by allowing for a growing labor force. With this change, we can witness a convergence towards equilibrium profit rates and the classical Sraffian price system. Assumptions of the model:

- All the classical Sraffian stuff (fixed unique production techniques for each commodity, uniform wages, etcetera - if you're here you don't need to hear me recite these). 
- A fixed amount of money circulating in the economy
- Population size changing at a fixed rate (e.g. dL/dt = kL for some k, where L is the available labor. k can be negative or zero)
- Workers accumulate money in the form of wages, capitalists accumulate it in the form of profit. Both workers and capitalists spend a fixed proportion of their savings each period on a fixed commodity bundle scaled according to how much that money can buy.
- Capitalists choose to produce more or less of something depending on whether the production process look profitable. E.g. if revenue looks to exceed cost at current prices then more is produced, revenue precedes cost means less is produced, break-even means no change in how much is produced.
- Wages change proportionally to changes in employment, and inversely proportionally to the size of the reserve army.
- Prices change proportionally to the change in supply, and approach infinity as supply approaches zero (supply changes according to the aforementioned capitalist and worker consumption as well as means of production usage)
- Capitalists borrow *all* money used to produce from creditors (or seen differently loan it to each other) at an interest rate which changes inversely proportionally to the change in the capitalist class's total savings, and which goes to infinity as those savings go to zero. This amount of money may exceed the total money circulating in the economy (e.g. fictitious capital), however debt does not accumulate. Rather, they simply surrender a portion of their revenue according to the interest rate. (Note, they are surrendering it to themselves. Hmm...)

The model is extremely robust and has a ton of possible applications. It's basically a (classical, e.g. non-probabilistic) capitalist economy in a box. Plug it into whatever you want. See [http://pinguet.free.fr/wrightthesis.pdf/](Ian Wright's PhD thesis) for more information on the model. There is a lot I am planning on adding and doing to this, and by the time you download it it will probably already look different from the screenshot. Have fun!

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
