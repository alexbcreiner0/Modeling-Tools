import numpy as np
import sympy as smp
import copy
from .Economy import Economy
import sys

def get_trajectories(params):
    sim_params = copy.deepcopy(params)
    e = None

    try:
        economy = Economy(sim_params)
    except Exception as error:
        e = error
        return {}, np.array([]), e

    for i in range(params.T):
        try:
            economy.step()
        except Exception as error:
            e = error
            with open("log.txt", "w") as f:
                print(economy.traj, file= f)
            break

    try:
        economy.get_analytic_curves()
    except Exception as error:
        e = error
        with open("log.txt", "w") as f:
            print(economy.traj, file= f)

    traj, t = economy.traj, economy.t
    return traj, t, e

def get_trajectories_bak(params):

    traj = {}

    include_curves= []

    init_y1, init_y2 = params.y1i, params.y2i
    exploit = params.e
    accum = params.a
    comp1, comp2 = params.k1, params.k2
    T = params.T
    res = params.res

    y1i, y2i, e, a, k1, k2 = smp.symbols('y_{1i}, y_{2i}, e, a, k_1, k_2')

    c1 = k1/(1+e+k1)
    v1 = (1-c1)/(1+e)
    s1 = e*v1
    c2 = k2 / (1+e+k2)
    v2 = (1-c2)/(1+e)
    s2 = e*v2
    l1 = (1+e)*v1
    l2 = (1+e)*v2
    
    b = 1-a
    M11, M12 = c1, c2
    M21 = (b*s1*c1+v1) / (1-b*s2)
    M22 = (b*s1*c2+v2) / (1-b*s2)
    
    theta1 = 1/2*(M11+M22+smp.sqrt((M11-M22)**2+4*M12*M21))
    theta2 = 1/2*(M11+M22-smp.sqrt((M11-M22)**2+4*M12*M21))
    m11, m21 = 1,1
    
    m12 = (theta1 - M11) / M12 * m11
    m22 = -1*(M11 - theta2) / M12 * m21
    
    r1 = (m22*y1i - m21*y2i) / (m11*m22 - m21*m12)
    r2 = (-1*m12*y1i + m11*y2i) / (m11*m22 - m21*m12)
    
    t = smp.symbols('t')

    if comp1 == comp2:
        y1 = (y1i*M11+y2i*M21)/(M11**2+M21**2)*(1/theta1)**t*M11
        y2 = (y1i*M11+y2i*M21)/(M11**2+M21**2)*(1/theta1)**t*M21
    else:    
        y1 = r1*m11*(1/theta1)**t + r2*m21*(1/theta2)**t
        y2 = r1*m12*(1/theta1)**t + r2*m22*(1/theta2)**t

    by1 = m11 / m12 * y2i
    by2 = m12 / m11 * y1i
    rg = (m22*y1i - m21*by1) / (m11*m22 - m12*m21)

    pg = 1 / (0.5*(c1 + v2 + smp.sqrt((c1-v2)**2 + 4*v1*c2))) - 1
    pg = pg.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]).evalf()
    
    #j1 = (1+pg) / (1+(c1/(v1+s1))) + ((1+pg)**2*(c1/(v1+s1))) / ((1-(pg*c1)/(v1+s1))*(1+c1/(v1+s1)))
    #j2 = (1+pg) / (1+(c2/(v2+s2))) + ((1+pg)**2*(c2/(v2+s2))) / ((1-(pg*c1)/(v1+s1))*(1+c2/(v2+s2)))

    p1 = 1/(1+pg) - v2
    p2 = c2
    p1_real = p1.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]).evalf()
    p2_real = p2.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]).evalf()

    w_real = p2_real / (1+exploit)

    j1 = p1/p2*(1+e)
    j1 = j1.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]).evalf()
    j2 = e+1
    j2 = j2.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]).evalf()
    
    N11, N12 = c1, c2
    N21 = (b*pg*(j1/j2*c1 + v1)*c1 + v1) / (1-b*pg*(j1/j2*c2+v2))
    N22 = (b*pg*(j1/j2*c1+v1)*c2+v2) / (1-b*pg*(j1/j2*c2+v2))

    theta1p = 0.5*(N11+N22+smp.sqrt((N11-N22)**2+4*N12*N21))
    theta2p = 0.5*(N11+N22-smp.sqrt((N11-N22)**2+4*N12*N21))   

    theta1p = theta1p.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
    theta2p = theta2p.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
    
    n11, n21 = 1,1
    n12 = (theta1p - N11) / N12 * n11
    n12 = n12.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
    n22 = -1*(N11 - theta2p) / N12 * n21
    n22 = n22.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
    r1p = (n22*y1i - n21*y2i) / (n11*n22 - n21*n12)
    r2p = (-1*n12*y1i + n11*y2i) / (n11*n22 - n21*n12)
    bgp1 = n11/n12*y2i
    bgp2 = n12/n11*y1i
    be1 = M11/M21*y2i
    be2 = M21/M11*y1i
    bep1 = N11/N21*y2i
    bep2 = N21/N11*y1i
    rgp = (n22*y1i - n21*bgp1) / (n11*n22 - n12*n21) #worthless?
    rlp = (n22*y1i - n21*y2i) / (n11*n22 - n12*n21)

    if comp1 == comp2:
        y1p = (y1i*N11+y2i*N21)/(N11**2+N21**2)*(1/theta1p)**t*N11
        y2p = (y1i*N11+y2i*N21)/(N11**2+N21**2)*(1/theta1p)**t*N21
    else:
        y1p = r1p*n11*(1/theta1p)**t + r2p*n21*(1/theta2p)**t
        y2p = r1p*n12*(1/theta1p)**t + r2p*n22*(1/theta2p)**t
    
    if params.balanced and params.balanced_indep == "y1":
        if "money_curves" in include_curves:
            if comp1 == comp2:
                bep1 = N11/N21*y2i
                init_y1 = bep1.subs([(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
            else:
                bgp1 = n11/n12*y2i
                init_y1 = bgp1.subs([(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
        else:
            if comp1 == comp2:
                init_y1 = be1.subs([(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
            else:
                init_y1 = by1.subs([(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
    if params.balanced and params.balanced_indep == "y2":
        if "money_curves" in include_curves:
            if comp1 == comp2:
                init_y2 = bep2.subs([(y1i,init_y1),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
            else:
                init_y2 = bgp2.subs([(y1i,init_y1),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
        else:
            if comp1 == comp2:
                init_y2 = be2.subs([(y2i,init_y1),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
            else:
                init_y2 = by2.subs([(y1i,init_y1),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])

    y1_func = smp.lambdify(t, y1.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]))
    y2_func = smp.lambdify(t, y2.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]))

    y1p_func = smp.lambdify(t, y1p.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]))
    y2p_func = smp.lambdify(t, y2p.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]))
    
    c1_real = c1.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
    c2_real = c2.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])     
    v1_real = v1.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])       
    v2_real = v2.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])    
    s1_real = s1.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
    s2_real = s2.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]) 
    j1_real = j1.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
    j2_real = j2.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
    l1_real = l1.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
    l2_real = l2.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
    pg_real = pg.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)])
#    return y1_func, y2_func, c1_real, v1_real, s1_real, c2_real, v2_real, s2_real
    
    y1_balanced = r1*m11*(1/theta1)**t
    y2_balanced = r1*m12*(1/theta1)**t
    y1_bal_func = smp.lambdify(t, y1_balanced.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]))
    y2_bal_func = smp.lambdify(t, y2_balanced.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]))

    y1p_bal = rlp*n11*(1/theta1p)**t
    y2p_bal = rlp*n12*(1/theta1p)**t
    y1p_bal_func = smp.lambdify(t, y1p_bal.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]))
    y2p_bal_func = smp.lambdify(t, y2p_bal.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]))

    d = (l1*(y1.subs(t,t+1) - y1) + l2*(y2.subs(t,t+1) - y2)) / (l1*y1 + l2*y2)
        # d_func = smp.lambdify(t, d.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]))

    K = (c1*y1 + c2*y2) / (v1*y1 + v2*y2)
        # K_func = smp.lambdify(t, K.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]))

    Pb = (s1*rg*m11*(1/theta1)**t + s2*rg*m12*(1/theta1)**t) / ((c1+v1)*rg*m11*(1/theta1)**t + (c2+v2)*rg*m12*(1/theta1)**t)

    roav = ((c1 + v1)*(y1.subs(t,t+1)-y1) + (c2+v2)*(y2.subs(t,t+1)-y2)) / ((c1+v1)*y1 + (c2+v2)*y2)
    # roav_func = smp.lambdify(t, roav.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]))
    
    roamp = ((p1_real*c1_real+p2_real*v1_real)*(y1p.subs(t,t+1) - y1p) + (p1_real*c2_real+p2_real*v2_real)*(y2p.subs(t,t+1) - y2p)) / ((p1_real*c1_real+p2_real*v1_real)*y1p + (p1*c2+p2*v2)*y2p)
    roamp_func = smp.lambdify(t, roamp.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]))
        
    xlims = (0,T)
    t_nums = np.linspace(xlims[0], xlims[1], res, dtype=complex)
    y1_out, y2_out = y1_func(t_nums).real.astype(float), y2_func(t_nums).real.astype(float)
    y1p_out, y2p_out = y1p_func(t_nums).real.astype(float), y2p_func(t_nums).real.astype(float)

    yp_out = np.column_stack((y1p_out, y2p_out))
    y_out = np.column_stack((y1_out.real, y2_out.real))

    if params.money_reinvestment:
        traj["y"] = yp_out
    else:
        traj["y"] = y_out
    
    change_y1 = y1.subs(t,t+1)-y1
    change_y1_func = smp.lambdify(t, change_y1.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]))
    change_y1_out = change_y1_func(t_nums).real.astype(float)
    change_y2 = y2.subs(t,t+1)-y2
    change_y2_func = smp.lambdify(t, change_y2.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]))
    change_y2_out = change_y2_func(t_nums).real.astype(float)
    
    change_y1p = y1p.subs(t,t+1)-y1p
    change_y1p_func = smp.lambdify(t, change_y1p.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]))
    change_y1p_out = change_y1p_func(t_nums).real.astype(float)
    change_y2p = y2p.subs(t,t+1)-y2p
    change_y2p_func = smp.lambdify(t, change_y2p.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]))
    change_y2p_out = change_y2p_func(t_nums).real.astype(float)

    net_y_val = y2_out + (y1_out - c1_real*y1_out - c2_real*y2_out)
    net_y_price = p2_real*y2_out + p1_real*(y1_out - c1_real*y1_out - c2_real*y2_out)

    MELT = net_y_price / net_y_val

    traj["MELT"] = MELT
    e_n = (1 - w_real / MELT)/(w_real/MELT)
    money_profit = net_y_price - w_real*(l1_real*y1_out + l2_real*y2_out)
    S_n = (l1_real*y1_out + l2_real*y2_out) - (w_real / MELT)*l1_real*y1_out - (w_real / MELT)*l2_real*y2_out

    traj["NI_total_surplus"] = S_n
    traj["NI_e"] = e_n

    # C matrix
    c11, c12 = 0, 0
    c21 = (1-a)*(s1*y1_out+s2*y2_out)*(c1*p1+v1*p2) / (p1*(c1*y1_out+c2*y2_out) + p2*(v1*y1_out+v2*y2_out))
    c22 = (1-a)*(s1*y1_out+s2*y2_out)*(c2*p1+v2*p2) / (p1*(c1*y1_out+c2*y2_out) + p2*(v1*y1_out+v2*y2_out))

    if params.money_reinvestment:
        y1p_bal_out, y2p_bal_out = y1p_bal_func(t_nums), y2p_bal_func(t_nums)
        yp_bal_out = np.column_stack((y1p_bal_out.real, y2p_bal_out.real))
        traj["y_bal"] = yp_bal_out
    else:
        y1_bal_out, y2_bal_out = y1_bal_func(t_nums), y2_bal_func(t_nums)
        y_bal_out = np.column_stack((y1_bal_out.real, y2_bal_out.real))
        traj["y_bal"] = y_bal_out.real

    d_out = ((v1_real+s1_real)*change_y1_out + (v2_real+s2_real)*change_y2_out) / ((v1_real+s1_real)*y1_out + (v2_real+s2_real)*y2_out)
    traj["labor_demand"] = d_out.real 
#        d_out[np.abs(np.diff(d_out, prepend=d_out[0])) > 5] = np.nan

    K_out = (c1_real*y1_out + c2_real*y2_out) / (v1_real*y1_out + v2_real*y2_out)
    traj["overall_composition"] = K_out.real

    k1_out = c1_real / v1_real * np.ones(res)
    k2_out = c2_real / v2_real * np.ones(res)
    k_sect_out = np.column_stack((k1_out.real, k2_out.real))
    traj["k_sects"] = k_sect_out

    p1_out = s1_real / (c1_real + v1_real) * np.ones(res)
    p2_out = s2_real / (c2_real + v2_real) * np.ones(res)

    traj["sectoral_value_profit_rates"] = np.column_stack((p1_out.real, p2_out.real))

    P_out = (s1_real*y1_out + s2_real*y2_out) / ((c1_real+v1_real)*y1_out + (c2_real+v2_real)*(y2_out))
    traj["val_profit"] = P_out
    Pb_out = Pb.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]) * np.ones(res)

    traj["val_profit_bal"] = Pb_out

    traj["pg_out"] = pg.subs([(y1i,init_y1),(y2i,init_y2),(a,accum),(e,exploit),(k1,comp1),(k2,comp2)]) * np.ones(res)
    roam_out = ((p1_real*c1_real+p2_real*v2_real)*change_y1_out + (p1_real*c2_real+p2_real*v2_real)*change_y2_out) / ((p1_real*c1_real+p2_real*v2_real)*y1_out + (p1_real*c2_real+p2_real*v2_real)*y2_out)

    if params.money_reinvestment:
        roavp_out = ((c1_real + v1_real)*change_y1p_out + (c2_real+v2_real)*change_y2p_out) / ((c1_real+v1_real)*y1p_out + (c2_real+v2_real)*y2p_out)
        traj["roav"] = roavp_out
        roamp_out = roamp_func(t_nums)
        traj["roam"] = roamp_out
    else:
        traj["roav"] = ((c1_real + v1_real)*change_y1_out + (c2_real+v2_real)*change_y2_out) / ((c1_real+v1_real)*y1_out + (c2_real+v2_real)*y2_out)
        traj["roam"] = roam_out

    gdp_out = ((v1_real+s1_real)*change_y1_out + (v2_real+s2_real)*change_y2_out)/ ((v1_real+s1_real)*y1_out + (v2_real+s2_real)*y2_out)
    traj["gdp"] = gdp_out

    gdpp_out = ((j2_real*v1_real+j1_real*s1_real)*change_y1p_out 
                    + (j2_real*v2_real+j2_real*s2_real)*change_y2p_out) / ((j2_real*v1_real+j1_real*s1_real)*y1p_out + (j2_real*v2_real+j2_real*s2_real)*y2p_out)
    traj["gdpp"] = gdpp_out

       
    p1_out = (pg_real - v2_real) * y1_out
    p2_out = c2_real * y2_out
    p_vec_out = np.column_stack((p1_out.real, p2_out.real))
    traj["p_vec"] = p_vec_out

    p_out = p1_out + p2_out
    traj["p_out"] = p_out

    e_out = exploit*np.ones(len(t_nums))
    traj["e"] = e_out

    # dont remember what this is
    # if "specific_prices" in include_curves:
    #     p1_out = j1*y1_out
    #     p2_out = j2*y2_out
    #     p_out = p1_out + p2_out

    p1s_out = j1*y1_out
    p2s_out = j2*y2_out
    ps_vec_out = np.column_stack((p1s_out.real, p2s_out.real))
    traj["ps_vec"] = ps_vec_out

    ps_out = p1_out + p2_out
    traj["ps_out"] = ps_out

    total_out = y1_out + y2_out
    traj["total_val"] = total_out

    total_profit_out = (j1_real-(j1_real*c1_real+j2_real*v1_real))*y1_out + (j2_real-(j1_real*c2_real+j2_real*v2_real))*y2_out
    traj["total_money_profit"] = total_profit_out

    total_surplus = s1_real*y1_out + s2_real*y2_out
    traj["total_surplus"] = total_surplus

    # if return_yip:
    #     return y1_out, y2_out, y1p_out, y2p_out, P_out[0], pg, j1, j2, c1_real, v1_real, s1_real, c2_real, v2_real, s2_real

    labels = {
        "y1": "Department I",
        "y2": "Department II",
        "y1b": "Dept. I Balanced",
        "y2b": "Dept. II Balanced",
        "y1p": "Department I",
        "y2p": "Department II",
        "labor_demand": "Demand for Labor",
        "overall_composition": "Composition of Capital",
        "technical_compositions": ("Dept. I Compositon", "Dept. II Composition"),
        "departmental_rops": ("Dept. I Rate of Profit", "Dept. II Rate of Profit"),
        "overall_rop": "Overall Value Rate of Profit",
        "balanced_rop": "Balanced Growth Rate of Profit",
        "golden_rop": "Actual Equilibrium Rate of Profit",
        "roav": "Rate of Accumulation (Measured in Value)",
        "roam": "Rate of Accumulation (Measured in Money)",
        "roavp": "Rate of Accumulation (Measured in Value)",
        "roamp": "Rate of Accumulation (Measured in Money)",
        "gdp": "Gross National Product (Measured in Value)",
        "gdpp": "Gross National Product (Measured in Money)",
        "money_curves": ("Deptartment I", "Department II"),
        "balanced_money": ("Dept. I Balanced", "Dept. II Balanced"),
        "prices": "Total dollar output",
        "total_output": "Total value output",
        "specific_prices": "Total output price",
        "total_profit": "Total profit",
        "total_surplus": "Total surplus value",
        "MELT": "MELT",
        "NE_surplus": "NE Total Surplus Value"
    }

    if "gdp" in include_curves:
        curve, = ax.plot(t_nums, gdp_out, color="red")
        curve.set_label(labels["gdp"])

    if "gdpp" in include_curves:
        curve, = ax.plot(t_nums, gdpp_out, '--', color="red")
        curve.set_label(labels["gdpp"])

    if "money_curves" in include_curves:
        y1p_curve, = ax.plot(t_nums, y1p_out, color="red")
        y2p_curve, = ax.plot(t_nums, y2p_out, color="blue")
        y1p_curve.set_label(labels["money_curves"][0])
        y2p_curve.set_label(labels["money_curves"][1])

    if "balanced_money" in include_curves:
        y1pb_curve, = ax.plot(t_nums, y1p_bal_out, '--', color='red')
        y2pb_curve, = ax.plot(t_nums, y2p_bal_out, '--', color='blue')
        y1pb_curve.set_label(labels["money_curves"][0])
        y2pb_curve.set_label(labels["money_curves"][1])

    if "prices" in include_curves:
        p_curve, = ax.plot(t_nums, p_out, color='green')
        p_curve.set_label(labels["prices"])

    if "total_output" in include_curves:
        total_curve, = ax.plot(t_nums, total_out, '--', color='green')
        total_curve.set_label(labels["total_output"])

    if "specific_prices" in include_curves:
        p_curve, = ax.plot(t_nums, p_out, color='green')
        p_curve.set_label(labels["specific_prices"])

    if "total_surplus" in include_curves:
        total_surplus_curve, = ax.plot(t_nums, total_surplus, '--', color="teal")
        total_surplus_curve.set_label(labels["total_surplus"])
   
    if "total_profit" in include_curves:
        total_profit_curve, = ax.plot(t_nums, profit_out, color='teal')
        total_profit_curve.set_label(labels["total_profit"])

    if "MELT" in include_curves:
        MELT_curve, = ax.plot(t_nums, MELT, color='yellow')
        MELT_curve.set_label(labels["MELT"])

    if "exploitations" in include_curves:
        e_n_curve, = ax.plot(t_nums, e_n, color="purple")
        e_n_curve.set_label("New Interpretation $e$")
        e_out = exploit*np.ones(len(t_nums))
        e_curve, = ax.plot(t_nums, e_out, color="pink")
        e_curve.set_label("Classical $e$")

    if "NE_surplus" in include_curves:
        ne_surplus_curve, = ax.plot(t_nums, S_n, color="red")
        ne_surplus_curve.set_label("New Interpretation Surplus Value")
        ne_profit_out = MELT*S_n
        ne_profit_curve, = ax.plot(t_nums, ne_profit_out, color="green")
        ne_profit_curve.set_label("NE Surplus Times MELT")
        
    # if not omit_legend:
    #     ax.legend(loc=loc, fancybox=False, edgecolor="black", fontsize=legend_font_size)
    # if title:
    #     if title_font_size:
    #         ax.set_title(title, fontsize=title_font_size)
    #     else:
    #         ax.set_title(title)
    # if not omit_xlabel: ax.set_xlabel("Time [t]")
    # if not omit_ylabel:
    #     if not ylabel:
    #         ax.set_ylabel("Total Output Value (Hours)")
    #     else:
    #         ax.set_ylabel(ylabel)
    # if display_params:
        # summary = fr'''$y_{{1i}} = {init_y1:.3f}$
# $y_{{2i}} = {init_y2:.3f}$
# $e = {exploit}$
# $a = {accum}$
# $k_1 = {comp1}$
# $k_2 = {comp2}$'''
        # if specific_params: summary = specific_params
        # ax.text(params_loc[0], params_loc[1], summary,
        #         transform=ax.transAxes,
        #         verticalalignment='top',
        #         horizontalalignment='left',
        #         multialignment='left',
        #         linespacing=1.2,
        #         bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black'))
    # if filename: plt.savefig(filename)
    # if return_axis: 
        # return ax
    # else: 
        # plt.show()

    return traj, t_nums.real, None


