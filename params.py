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

