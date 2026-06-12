import pybamm
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from scipy.interpolate import interp1d


def make_sinusoidal_input(freq_hz=1.0, n_periods=10, dc_offset_a=0.0, ac_mag_a=10/1000, pts_per_period=20):
    T_period = 1.0 / freq_hz
    T_total  = n_periods * T_period
    t_eval   = np.linspace(0, T_total, pts_per_period * n_periods)

    omega = 2 * np.pi * freq_hz

    def current_fn(t):
        return dc_offset_a + ac_mag_a * pybamm.sin(omega * t)

    return current_fn, t_eval


def build_soc2theta(param):
    model = pybamm.lithium_ion.SPMe()
    experiment = pybamm.Experiment([
        "Discharge at C/3 until 2.5V",
        "Hold at 2.5V until C/50",
    ])
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
    sol = sim.solve()

    theta_n = sol["Negative electrode stoichiometry"].data
    theta_p = sol["Positive electrode stoichiometry"].data
    capacity = sol["Discharge capacity [A.h]"].data
    soc = (1 - (capacity - capacity[0]) / (capacity[-1] - capacity[0]))  # 0~1

    soc2theta_n = interp1d(soc, theta_n, kind='linear', bounds_error=False, fill_value='extrapolate')
    soc2theta_p = interp1d(soc, theta_p, kind='linear', bounds_error=False, fill_value='extrapolate')

    return soc2theta_n, soc2theta_p


def soc2conc(soc, param, soc2theta_n, soc2theta_p):
    c_max_n = param["Maximum concentration in negative electrode [mol.m-3]"]
    c_max_p = param["Maximum concentration in positive electrode [mol.m-3]"]
    return float(soc2theta_n(soc)) * c_max_n, float(soc2theta_p(soc)) * c_max_p