import pybamm
import numpy as np
from scipy.interpolate import interp1d
from utils import *


def compute_sensitivity(model, parameter_values,
                        current_fn, t_eval,
                        target_param, soc=0.5, perturbation=0.05,
                        soc2theta_n=None, soc2theta_p=None):

    parameter_values = parameter_values.copy()

    c_n_init, c_p_init = soc2conc(soc, parameter_values, soc2theta_n, soc2theta_p)

    parameter_values.update({
        "Initial concentration in negative electrode [mol.m-3]": c_n_init,
        "Initial concentration in positive electrode [mol.m-3]": c_p_init,
        "Current function [A]": current_fn,
    })

    # nominal simulation
    solver = pybamm.CasadiSolver(atol=1e-9, rtol=1e-9, mode="safe")
    sol_nominal = pybamm.Simulation(model, parameter_values=parameter_values, solver=solver).solve(t_eval=t_eval)
    t_ref     = sol_nominal["Time [s]"].data
    V_nominal = sol_nominal["Terminal voltage [V]"].data

    _theta = parameter_values[target_param]

    def get_voltage(sol):
        t = sol["Time [s]"].data
        V = sol["Terminal voltage [V]"].data
        if len(V) != len(V_nominal):
            V = interp1d(t, V, bounds_error=False, fill_value="extrapolate")(t_ref)
        return V

    if callable(_theta):
        parameter_values.update({target_param: lambda *args, s=(1+perturbation), f=_theta: f(*args) * s})
        V_upper = get_voltage(pybamm.Simulation(model, parameter_values=parameter_values, solver=solver).solve(t_eval=t_eval))
        parameter_values.update({target_param: lambda *args, s=(1-perturbation), f=_theta: f(*args) * s})
        V_lower = get_voltage(pybamm.Simulation(model, parameter_values=parameter_values, solver=solver).solve(t_eval=t_eval))
        parameter_values.update({target_param: _theta})
    else:
        parameter_values.update({target_param: _theta * (1 + perturbation)})
        V_upper = get_voltage(pybamm.Simulation(model, parameter_values=parameter_values, solver=solver).solve(t_eval=t_eval))
        parameter_values.update({target_param: _theta * (1 - perturbation)})
        V_lower = get_voltage(pybamm.Simulation(model, parameter_values=parameter_values, solver=solver).solve(t_eval=t_eval))
        parameter_values.update({target_param: _theta})

    sensitivity = (V_upper - V_lower) / (2 * perturbation * V_nominal)

    return sensitivity, V_nominal