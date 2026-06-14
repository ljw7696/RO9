# %% Import necessary libraries
import pybamm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
import random
import functools
import matplotlib as mpl
import plotly.graph_objects as go
import pickle
from plotly.subplots import make_subplots
from joblib import Parallel, delayed


from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze


import gc
gc.collect()

import logging
# Set the logging level to ERROR to suppress warnings
pybamm.set_logging_level("ERROR")



## Read supplementary files
from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent))



%load_ext autoreload
%autoreload 2

from utils import *
# from simplify import *

# %% Base Model Run
initial_soc = 0.6


######## Default model run ########
model = make_model("SPMe")

base_param = make_base_params("Chen2020", soc=initial_soc)

experiment = make_experiment(profile_id=4)

sol = run_model(model, base_param, experiment)


t_query = get_t_query(sol)

VIT = get_VIT(sol, t_query)

state_names = [
    "Average negative particle concentration [mol.m-3]",
    "Average positive particle concentration [mol.m-3]",
]
States = get_states(
    sol,
    state_names,
    t_query,
)

plot_results(VIT, kind="VIT")
plot_results(States, kind="states")



# %% Single Theta Sensitivity
initial_soc = 0.5


MODEL_OPTIONS = {
    "surface form": "differential",
    "contact resistance": "true",
}

model = make_model("SPMe", options=MODEL_OPTIONS)


base_param = make_base_params("Chen2020", soc=initial_soc,
                              sensitivity_ready=True)

experiment = make_experiment(profile_id=4)
experiment = make_experiment(profile_id="hppc")


theta_values_nominal = {
    "Negative electrode exchange-current density [A.m-2]": 6.48e-7, ##
    "Positive electrode exchange-current density [A.m-2]": 3.42e-6, ##

    "Negative electrode double-layer capacity [F.m-2]": 0.2,
    "Positive electrode double-layer capacity [F.m-2]": 0.2,

    "Negative electrode active material volume fraction": 0.75,
    "Positive electrode active material volume fraction": 0.665,

    "Cation transference number": 0.2594,

    "Electrolyte conductivity [S.m-1]": 0.9487, ##
    "Electrolyte diffusivity [m2.s-1]": 1.7694e-10, ##

    "Positive electrode porosity": 0.335,
    "Negative electrode porosity": 0.25,
    "Separator porosity": 0.47,

    "Negative particle diffusivity [m2.s-1]": 3.3e-14,
    "Positive particle diffusivity [m2.s-1]": 4.0e-15,

    "Negative electrode conductivity [S.m-1]": 215.0,
    "Positive electrode conductivity [S.m-1]": 0.18,

    # "Negative electrode thickness [m]",
    # "Positive electrode thickness [m]",
    # "Separator thickness [m]",
    
    # "Negative particle radius [m]",
    # "Positive particle radius [m]",
}


sensitivity_targets = list(theta_values_nominal.keys())


target_params, target_values = prepare_sensitivity_inputs(
    base_param,
    sensitivity_targets,
    values=theta_values_nominal,
)


sol = run_model(
    model,
    target_params,
    experiment,
    inputs=target_values,
    calculate_sensitivities=True,
)


t_query = get_t_query(sol)

VIT = get_VIT(sol, t_query)

state_names = [
    "Average negative particle concentration [mol.m-3]",
    "Average positive particle concentration [mol.m-3]",
]
States = get_states(
    sol,
    state_names,
    t_query,
)

Sens_V = get_sensitivities(
    sol,
    "Voltage [V]",
    sensitivity_targets,
    t_query,
    theta_values=target_values,
    normalize=True,
)

Sens_V_scaled = scale_voltage_sens_by_ocv(
    Sens_V,
    base_param,
)

Sens_csp = get_sensitivities(
    sol,
    "Average positive particle concentration [mol.m-3]",
    sensitivity_targets,
    t_query,
    theta_values=target_values,
    normalize=True,
)

Sens_csn = get_sensitivities(
    sol,
    "Average negative particle concentration [mol.m-3]",
    sensitivity_targets,
    t_query,
    theta_values=target_values,
    normalize=True,
)

plot_results(VIT, kind="VIT")
plot_results(States, kind="states")
plot_results(
    Sens_V_scaled,
    kind="sensitivities",
    max_cols=4,
    figsize_per_plot=(4, 3),
)


plot_sensitivity_comparison_yy(
    Sens_V_scaled,
    Sens_csp,
    label_left="Voltage sensitivity",
    label_right="csp sensitivity",
)

plot_sensitivity_comparison_yy(
    Sens_V_scaled,
    Sens_csn,
    label_left="Voltage sensitivity",
    label_right="csn sensitivity",
)


plot_sensitivity_comparison_yy(
    Sens_csn,
    Sens_csp,
    label_left="csn sensitivity",
    label_right="csp sensitivity",
)




# %% Normalized sensitivities plot
endpoints = get_soc_endpoints(base_param)


Sens_soc_n, Sens_soc_p = concentrations_to_soc_sensitivities(
    Sens_csn,
    Sens_csp,
    endpoints,
)


plot_sensitivity_comparison_yy(
    Sens_V_scaled,
    Sens_soc_p,
    label_left="V sensitivity",
    label_right="soc_p sensitivity",
    suptitle="Voltage-SOC_p Sensitivity Comparison",
)

plot_sensitivity_comparison_yy(
    Sens_V_scaled,
    Sens_soc_n,
    label_left="V sensitivity",
    label_right="soc_n sensitivity",
    suptitle="Voltage-SOC_n Sensitivity Comparison",
)

plot_sensitivity_comparison_yy(
    Sens_soc_p,
    Sens_soc_n,
    label_left="soc_p sensitivity",
    label_right="soc_n sensitivity",
    suptitle="SOC Sensitivity Comparison",
)



# %% Normalized FIM computation
V_span = 4.2 - 2.5

cases = {
    "No noise": {
        "sigma_V": None,
        "sigma_SOC": None,
    },
    "1mV + 1%SOC": {
        "sigma_V": 1/1000 / V_span,
        "sigma_SOC": 0.3/100,
    },
    "2mV + 2%SOC": {
        "sigma_V": 2/1000 / V_span,
        "sigma_SOC": 0.6/100,
    },
    "3mV + 3%SOC": {
        "sigma_V": 3/1000 / V_span,
        "sigma_SOC": 0.9/100,
    },
}

for case_name, noise in cases.items():

    print("\n" + "="*80)
    print(case_name)
    print("="*80)

    F_V = compute_fim(
        Sens_V_scaled,
        params=sensitivity_targets,
        sigma=noise["sigma_V"],
    )

    F_soc_n = compute_fim(
        Sens_soc_n,
        params=sensitivity_targets,
        sigma=noise["sigma_SOC"],
    )

    F_soc_p = compute_fim(
        Sens_soc_p,
        params=sensitivity_targets,
        sigma=noise["sigma_SOC"],
    )

    F_V_soc_n = F_V + F_soc_n
    F_V_soc_p = F_V + F_soc_p
    F_V_soc_np = F_V + F_soc_n + F_soc_p

    print("Condition numbers")
    print("V       :", np.linalg.cond(F_V))
    print("V+SOC_n :", np.linalg.cond(F_V_soc_n))
    print("V+SOC_p :", np.linalg.cond(F_V_soc_p))
    print("V+Both  :", np.linalg.cond(F_V_soc_np))

    print()

    base = np.linalg.slogdet(F_V)[1]

    print("Δlogdet")
    print("SOC_n :", np.linalg.slogdet(F_V_soc_n)[1] - base)
    print("SOC_p :", np.linalg.slogdet(F_V_soc_p)[1] - base)
    print("Both  :", np.linalg.slogdet(F_V_soc_np)[1] - base)

    print()

    C_V = np.linalg.pinv(F_V)
    C_np = np.linalg.pinv(F_V_soc_np)

    improve = np.diag(C_V) / np.diag(C_np)

    print("CRLB improvement")
    for p, r in zip(sensitivity_targets, improve):
        print(
            f"{PARAM_LABELS.get(p,p):25s}"
            f"{r:10.2f}"
        )
# %%
