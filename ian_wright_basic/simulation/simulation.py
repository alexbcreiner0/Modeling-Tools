import numpy as np
import copy
from scipy.integrate import solve_ivp
from typing import Callable, Dict, Any
from .parameters import Params
from .CapitalistEconomy import *

# This is where the app will look for a get_trajectories function. 
def basic_unrestricted_economy(params):
    """ Unchanged dynamics, nothing special happening """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomy(sim_params)

    e = None
    for i in range(params.T):
        try:
            economy.step()
        except Exception as error:
            e = error
            break

    traj, t = economy.traj, economy.t
    return traj, t, e

def basic_economy_fixed_real_wage(params):
    """ Unchanged dynamics, nothing special happening """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomyFixedRealWage(sim_params)

    e = None
    for i in range(params.T):
        try:
            economy.step()
        except Exception as error:
            e = error
            break

    traj, t = economy.traj, economy.t
    return traj, t, e

def periodic_culs_perturbation_basic_unrestricted(params):
    """ Unchanged dynamics, but with random capital using labor saving cost reducing technological changes introduced every 20 cycles """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomy(sim_params)

    e = None
    for i in range(params.T):
        # try:
        if i % 20 == 0 and i != 0:
            economy.implement_culs_shock(0.1)
        economy.step()
        # except Exception as error:
        #     e = error
        #     break

    traj, t = economy.traj, economy.t
    return traj, t, e

def periodic_ls_perturbation_basic_unrestricted(params):
    """ Unchanged dynamics, but with random capital using labor saving cost reducing technological changes introduced every 20 cycles """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomy(sim_params)

    e = None
    for i in range(params.T):
        try:
            if i % 20 == 0 and i != 0:
                economy.implement_culs_shock(0.1, cu= False)
            economy.step()
        except Exception as error:
            e = error
            break

    traj, t = economy.traj, economy.t
    return traj, t, e

def periodic_cslu_perturbation_basic_unrestricted(params):
    """ Unchanged dynamics, but with random capital using labor saving cost reducing technological changes introduced every 20 cycles """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomy(sim_params)

    e = None
    for i in range(params.T):
        try:
            if i % 20 == 0 and i != 0:
                economy.implement_cslu_shock(0.1)
            economy.step()
        except Exception as error:
            e = error
            break

    traj, t = economy.traj, economy.t
    return traj, t, e

def periodic_cs_perturbation_basic_unrestricted(params):
    """ Unchanged dynamics, but with random capital using labor saving cost reducing technological changes introduced every 20 cycles """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomy(sim_params)

    e = None
    for i in range(params.T):
        try:
            if i % 20 == 0 and i != 0:
                economy.implement_cslu_shock(0.1, lu= False)
            economy.step()
        except Exception as error:
            e = error
            break

    traj, t = economy.traj, economy.t
    return traj, t, e

def periodic_culs_perturbation_fixed_real_wage(params):
    """ Unchanged dynamics, but with random capital using labor saving cost reducing technological changes introduced every 20 cycles """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomyFixedRealWage(sim_params)

    e = None
    for i in range(params.T):
        try:
            if i % 20 == 0 and i != 0:
                economy.implement_culs_shock(0.2)
            economy.step()
        except Exception as error:
            e = error
            break

    traj, t = economy.traj, economy.t
    return traj, t, e

def periodic_ls_perturbation_fixed_real_wage(params):
    """ Unchanged dynamics, but with random capital using labor saving cost reducing technological changes introduced every 20 cycles """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomyFixedRealWage(sim_params)

    e = None
    for i in range(params.T):
        try:
            if i % 20 == 0 and i != 0:
                economy.implement_culs_shock(0.1, cu= False)
            economy.step()
        except Exception as error:
            e = error
            break

    traj, t = economy.traj, economy.t
    return traj, t, e

def periodic_cslu_perturbation_fixed_real_wage(params):
    """ Unchanged dynamics, but with random capital using labor saving cost reducing technological changes introduced every 20 cycles """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomyFixedRealWage(sim_params)

    e = None
    for i in range(params.T):
        try:
            if i % 20 == 0 and i != 0:
                economy.implement_cslu_shock(0.1)
            economy.step()
        except Exception as error:
            e = error
            break

    traj, t = economy.traj, economy.t
    return traj, t, e

def periodic_cs_perturbation_fixed_real_wage(params):
    """ Unchanged dynamics, but with random capital using labor saving cost reducing technological changes introduced every 20 cycles """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomyFixedRealWage(sim_params)

    e = None
    for i in range(params.T):
        try:
            if i % 20 == 0 and i != 0:
                economy.implement_cslu_shock(0.1, lu= False)
            economy.step()
        except Exception as error:
            e = error
            break

    traj, t = economy.traj, economy.t
    return traj, t, e


def single_supply_shock_perturbation_basic_unrestricted(params):
    """ Unchanged dynamics, but with a supply shock implemented at time t = 50 """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomy(sim_params)

    e = None
    for i in range(params.T):
        try:
            economy.step()
            if i == 50:
                s = economy.check_supply()
                deduction = 0.5*s
                economy.exo_supply_shock(deduction)
        except Exception as error:
            e = error
            break

    traj, t = economy.traj, economy.t
    return traj, t, e

def single_uniform_labor_productivity_perturbation_basic_unrestricted(params):
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomy(sim_params)

    e = None
    for i in range(params.T):
        try:
            economy.step()
            if i == 50:
                q, p, s, l, m_w, L = economy._split_state(economy.y)
                l *= 0.5
                y = np.concatenate([q, p, s, l, np.array([m_w]), np.array([L])])
                economy.y = y
        except Exception as error:
            e = error
            break

    traj, t = economy.traj, economy.t
    return traj, t, e

def single_labor_productivity_perturbation_basic_unrestricted(params):
    """ Dynamics are altered so that the real wage is fixed at b. At t = 50, the living labor vector is suddenly halved """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomy(sim_params)

    e = None
    for i in range(params.T):
        try:
            economy.step()
            if i == 50:
                q, p, s, l, m_w, L = economy._split_state(economy.y)
                l *= 0.5
                y = np.concatenate([q, p, s, l, np.array([m_w]), np.array([L])])
                economy.y = y
        except Exception as error:
            e = error
            break

    traj, t = economy.traj, economy.t
    return traj, t, e


def single_labor_productivity_perturbation_fixed_real_wage(params):
    """ Dynamics are altered so that the real wage is fixed at b. At t = 50, the living labor vector is suddenly halved """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomyFixedRealWage(sim_params)

    e = None
    for i in range(params.T):
        try:
            economy.step()
            if i == 50:
                q, p, s, l, m_w, L = economy._split_state(economy.y)
                l *= 0.5
                y = np.concatenate([q, p, s, l, np.array([m_w]), np.array([L])])
                economy.y = y
        except Exception as error:
            e = error
            break

    traj, t = economy.traj, economy.t
    return traj, t, e

def single_culs_perturbation_fixed_real_wage(params):
    """ Dynamics are altered so that the real wage is fixed at b. At t = 50, the living labor vector is suddenly halved """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomyFixedRealWage(sim_params)

    e = None
    for i in range(params.T):
        try:
            economy.step()
            if i == 50:
                q, p, s, l, m_w, L = economy._split_state(economy.y)
                l *= 0.5
                y = np.concatenate([q, p, s, l, np.array([m_w]), np.array([L])])
                economy.y = y
        except Exception as error:
            e = error
            break

    traj, t = economy.traj, economy.t
    return traj, t, e



def single_labor_productivity_perturbation_fixed_money_wage(params):
    """ Dynamics are altered so that the real wage is fixed at b. At t = 50, the living labor vector is suddenly halved """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomyFixedMoneyWage(sim_params)

    e = None
    for i in range(params.T):
        try:
            economy.step()
            if i == 50:
                q, p, s, l, m_w, L = economy._split_state(economy.y)
                l *= 0.5
                y = np.concatenate([q, p, s, l, np.array([m_w]), np.array([L])])
                economy.y = y
        except Exception as error:
            e = error
            break

    traj, t = economy.traj, economy.t
    return traj, t, e

def single_labor_productivity_perturbation_fixed_employment(params):
    """ Dynamics are altered such that employment is held non-decreasing. Every 20 cycles, a random capital using labor saving cost reducing technical change is introduced somewhere in the economy. """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomyFixedEmployment(sim_params)

    e = None
    for i in range(params.T):
        try:
            economy.step()
            if i == 50:
                q, p, s, l, m_w, L = economy._split_state(economy.y)
                q_old = q.copy()
                l_old = l.copy()
                l *= 0.5

                delta_E = l_old.dot(q_old) - l.dot(q)
                q_new = q + delta_E*(l/(l.dot(l))) # works out such that new_l.dot(new_q) = old_l.dot(old_q)

                y = np.concatenate([q_new, p, s, l, np.array([m_w]), np.array([L])])
                economy.traj["q"][-1] = q_new

                economy.y = y
        except Exception as error:
            e = error
            print(e)
            break

    traj, t = economy.traj, economy.t
    return traj, t, e

def single_labor_productivity_perturbation_fixed_money_wage_fixed_employment(params):
    """ Dynamics are altered such that the money wage is fixed and employment is held non-decreasing.  """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomyFixedMoneyWageFixedEmployment(sim_params)

    e = None
    for i in range(params.T):
        # print(f"i = {i}")
        try:
            economy.step()
            # print(f"Step complete: y = {economy.y}")
            if i == 50:
                q, p, s, l, m_w, L = economy._split_state(economy.y)
                q_old = q.copy()
                l_old = l.copy()
                l *= 0.5

                delta_E = l_old.dot(q_old) - l.dot(q)
                q_new = q + delta_E*(l/(l.dot(l))) # works out such that new_l.dot(new_q) = old_l.dot(old_q)

                y = np.concatenate([q_new, p, s, l, np.array([m_w]), np.array([L])])
                economy.traj["q"][-1] = q_new

                economy.y = y
        except Exception as error:
            e = error
            print(e)
            break

    traj, t = economy.traj, economy.t
    return traj, t, e

def single_labor_productivity_perturbation_fixed_real_wage_fixed_employment(params):
    """ Dynamics are altered such that the money wage is fixed and employment is held non-decreasing. Every 20 cycles, a random capital using labor saving cost reducing technical change is introduced somewhere in the economy. """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomyFixedRealWageFixedEmployment(sim_params)

    e = None
    for i in range(params.T):
        # print(f"i = {i}")
        try:
            economy.step()
            # print(f"Step complete: y = {economy.y}")
            if i == 50:
                q, p, s, l, m_w, L = economy._split_state(economy.y)
                q_old = q.copy()
                l_old = l.copy()
                l *= 0.5

                delta_E = l_old.dot(q_old) - l.dot(q)
                q_new = q + delta_E*(l/(l.dot(l))) # works out such that new_l.dot(new_q) = old_l.dot(old_q)

                y = np.concatenate([q_new, p, s, l, np.array([m_w]), np.array([L])])
                economy.traj["q"][-1] = q_new

                economy.y = y
        except Exception as error:
            e = error
            print(e)
            break

    traj, t = economy.traj, economy.t
    return traj, t, e

def periodic_culs_perturbation_fixed_money_wage_fixed_employment(params):
    """ Dynamics are altered such that the money wage is fixed and employment is held non-decreasing. Every 20 cycles, a random capital using labor saving cost reducing technical change is introduced somewhere in the economy. """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomyFixedMoneyWageFixedEmployment(sim_params)

    e = None
    for i in range(params.T):
        # print(f"i = {i}")
        try:
            economy.step()
            # print(f"Step complete: y = {economy.y}")
            if i != 0 and i % 20 == 0:
                q, p, s, l, m_w, L = economy._split_state(economy.y)
                q_old = q.copy()
                l_old = l.copy()
                q, p, s, l, m_w, L = economy.implement_culs_shock(0.15)

                delta_E = l_old.dot(q_old) - l.dot(q)
                q_new = q + delta_E*(l/(l.dot(l))) # works out such that new_l.dot(new_q) = old_l.dot(old_q)

                y = np.concatenate([q_new, p, s, l, np.array([m_w]), np.array([L])])
                economy.traj["q"][-1] = q_new

                economy.y = y
        except Exception as error:
            e = error
            print(e)
            break

    traj, t = economy.traj, economy.t
    return traj, t, e

def periodic_culs_perturbation_fixed_exploitation_fixed_employment(params):
    """ Dynamics are altered such that the rate of exploitation is fixed and employment is held non-decreasing. Every 20 cycles, a random capital using labor saving cost reducing technical change is introduced somewhere in the economy. """
    sim_params = copy.deepcopy(params)
    economy = CapitalistEconomyFixedExploitationFixedEmployment(sim_params)

    e = None
    for i in range(params.T):
        try:
            economy.step()
            if i != 0 and i % 20 == 0:
                q, p, s, l, m_w, L = economy._split_state(economy.y)
                l_old = l.copy()
                q_old = q.copy()

                q, p, s, l, m_w, L = economy.implement_culs_shock(0.15)

                delta_E = l_old.dot(q_old) - l.dot(q)
                q_new = q + delta_E*(l/(l.dot(l))) # works out such that new_l.dot(new_q) = old_l.dot(old_q)

                current_e = economy.traj["e"][-1]
                economy.target_e = current_e

                y = np.concatenate([q_new, p, s, l, np.array([m_w]), np.array([L])])
                economy.traj["q"][-1] = q_new

                economy.y = y
        except Exception as error:
            e = error
            break

    traj, t = economy.traj, economy.t
    return traj, t, e
