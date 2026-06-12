import pybamm
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time

from utils import *


def update_parameters(param, target_params={}):
    for key, value in target_params.items():
        param[key] = value
    return param


def get_symbol(param):
    param_labels = {
        "Negative electrode exchange-current density [A.m-2]": r"$k^-$",
        "Positive electrode exchange-current density [A.m-2]": r"$k^+$",
        "Negative electrode double-layer capacity [F.m-2]": r"$C_{dl}^-$",
        "Positive electrode double-layer capacity [F.m-2]": r"$C_{dl}^+$",
        "Negative electrode active material volume fraction": r"$\varepsilon_s^-$",
        "Positive electrode active material volume fraction": r"$\varepsilon_s^+$",
        "Cation transference number": r"$t^+$",
        "Electrolyte conductivity [S.m-1]": r"$\kappa$",
        "Electrolyte diffusivity [m2.s-1]": r"$D_e$",
        "Positive electrode porosity": r"$\varepsilon_e^+$",
        "Negative electrode porosity": r"$\varepsilon_e^-$",
        "Separator porosity": r"$\varepsilon_e^{sep}$",
        "Negative electrode thickness [m]": r"$L^-$",
        "Positive electrode thickness [m]": r"$L^+$",
        "Separator thickness [m]": r"$L^{sep}$",
        "Negative particle diffusivity [m2.s-1]": r"$D_s^-$",
        "Negative particle radius [m]": r"$R_s^-$",
        "Positive particle diffusivity [m2.s-1]": r"$D_s^+$",
        "Positive particle radius [m]": r"$R_s^+$",
    }
    return param_labels.get(param, param)