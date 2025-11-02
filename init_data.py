from numpy import random, array
from mathstuff import Params

random.seed(0)
n = 3
A = array([[0.2, 0, 0.4],
              [0.2, 0.8, 0],
              [0, 0.1, 0.1]])
l = array([0.7, 0.6, 0.3])
b_bar = array([0.6, 0, 0.2])
c_bar = array([0.2, 0, 0.4])
alpha_w = 0.8
alpha_c = 0.7
alpha_L = 0.101
kappa = array([1,1,1])
eta = array([2,2,2])
eta_w = 0.25 # no
L = 1
eta_r = 2 # no
q0 = array([0.01, 0.1, 0.1])
p0 = array([1, 0.8, 0.5])
s0 = array([0.01, 0.1, 0.25])
mw0 = 0.5
w0 = 0.5 # no
r0 = 0
T = 100 # no

params = Params(
  A=A, L=L, l=l, b_bar=b_bar, c_bar=c_bar, 
  alpha_w=alpha_w, alpha_c=alpha_c, alpha_L = alpha_L,
  kappa=kappa, eta=eta, eta_w=eta_w, eta_r=eta_r,
  w0=w0, r0=r0, q0=q0, p0=p0, s0=s0, m_w0=mw0, T=T
)

if __name__ == "__main__":
    # costs = A.T@p0+w0*l
    # profits = p0 - costs
    # profit_rates = profits
    # for i in range(len(p0)):
    #     profit_rates[i] = profits[i]/costs[i]
    # print(profit_rates)

    # final_prices = array([5.55118273e-06, 1.26817048e-05, 3.51861921e-06])
    # final_wages = 1.8162946988667283e-06
    # costs = A.T@final_prices + final_wages*l
    # profits = final_prices - costs
    # profit_rates = profits
    # for i in range(len(p0)):
    #     profit_rates[i] = profits[i]/costs[i]
    # print(profit_rates)
    print(params.alpha_L)
   
