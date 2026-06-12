import pybamm
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
import pandas as pd                    # ← 추가
from pathlib import Path               # ← 추가
from scipy.interpolate import interp1d


# Profile ID → CSV filename 매핑
PROFILE_CSV_MAP = {
    'FUDS': "FUDS_scaled.csv",
    'US06': "US06_scaled.csv",
    'WLTP': "WLTP_scaled.csv",
}
DATA_DIR = Path("data")


def make_experiment(profile_id, soc: float) -> pybamm.Experiment:
    """
    Parameters
    ----------
    profile_id : int (1~5), str ('hppc', 'FUDS', 'US06', 'WLTP')
    soc        : float, 0.0~1.0  (profile 5 에서만 사용)

    Returns
    -------
    pybamm.Experiment
    """

    if profile_id == 1:
        # 0.5C pulse: discharge 1s / rest 1s / charge 1s -> 50 cycles
        steps = [
            ("Discharge at 0.5C for 1 second",
             "Rest for 1 second",
             "Charge at 0.5C for 1 second")
        ] * 50

    elif profile_id == 2:
        steps = [
            ("Discharge at 1C for 1 second",
             "Rest for 1 second",
             "Charge at 1C for 1 second",
             "Rest for 1 second")
        ] * 50

    elif profile_id == 3:
        steps = [
            ("Discharge at 2C for 1 second",
             "Rest for 1 second",
             "Charge at 2C for 1 second",
             "Rest for 1 second")
        ] * 50

    elif profile_id == 4:
        # 0.5C discharge 5 min, rest 20 min, 0.5C charge 5 min
        steps = [(
            "Rest for 10 second",
            "Discharge at 0.5C for 5 minutes",
            "Rest for 20 minutes",
            "Charge at 0.5C for 5 minutes",
            "Rest for 20 minutes",
        )]

    elif profile_id == 5:
        # 1C discharge by 10% SOC per step
        n_reps = int(round(soc / 0.1))
        steps = [
            ("Discharge at 1C for 6 minutes",
             "Rest for 20 minutes")
        ] * n_reps

    elif profile_id == 'hppc':
        steps = [(
            "Rest for 10 seconds",
            "Discharge at 1C for 30 seconds",
            "Rest for 300 seconds",
            "Charge at 1C for 30 seconds",
            "Rest for 300 seconds",
            "Discharge at C/3 for 540 seconds",
            "Rest for 600 seconds",
            "Charge at C/3 for 540 seconds",
            "Rest for 600 seconds",
        )]

    elif profile_id in PROFILE_CSV_MAP:
        csv_path = DATA_DIR / PROFILE_CSV_MAP[profile_id]
        df = pd.read_csv(csv_path)
        
        t_data = df["TestTime_s_"].values.astype(float)
        I_data = df["Current_A_"].values.astype(float)
        
        # t=0 부터 시작하도록 shift (defensive)
        t_data = t_data - t_data[0]
        duration_s = float(t_data[-1])
        
        current_interp = pybamm.Interpolant(
            t_data, I_data, pybamm.t, name=f"{profile_id}_current"
        )
        
        step = pybamm.step.current(
            current_interp,
            duration=f"{duration_s} seconds",
            termination=["2.5 V", "4.2 V"],
        )
        return pybamm.Experiment([step, "Rest for 300 seconds"])   # ← rest 추가

    # elif profile_id in PROFILE_CSV_MAP:
    #     # CSV 기반 drive-cycle: FUDS / US06 / WLTP
    #     csv_path = DATA_DIR / PROFILE_CSV_MAP[profile_id]
    #     df = pd.read_csv(csv_path)
        
    #     t_data = df["TestTime_s_"].values.astype(float)
    #     I_data = df["Current_A_"].values.astype(float)
        
    #     # PyBaMM Interpolant: t → I(t)
    #     current_interp = pybamm.Interpolant(
    #         t_data, I_data, pybamm.t, name=f"{profile_id}_current"
    #     )
        
    #     duration_s = float(t_data[-1] - t_data[0])
    #     step = pybamm.step.current(
    #         current_interp,
    #         duration=f"{duration_s} seconds",
    #     )
    #     return pybamm.Experiment([step])

    else:
        raise ValueError(
            f"profile_id must be 1~5, 'hppc', or one of {list(PROFILE_CSV_MAP.keys())}, got {profile_id}"
        )

    return pybamm.Experiment(steps)




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
    soc = (1 - (capacity - capacity[0]) / (capacity[-1] - capacity[0]))

    soc2theta_n = interp1d(soc, theta_n, kind='linear', bounds_error=False, fill_value='extrapolate')
    soc2theta_p = interp1d(soc, theta_p, kind='linear', bounds_error=False, fill_value='extrapolate')

    return soc2theta_n, soc2theta_p



def soc2conc(soc, param, soc2theta_n, soc2theta_p):
    c_max_n = param["Maximum concentration in negative electrode [mol.m-3]"]
    c_max_p = param["Maximum concentration in positive electrode [mol.m-3]"]
    return float(soc2theta_n(soc)) * c_max_n, float(soc2theta_p(soc)) * c_max_p


def set_init_concentration(param, soc):
    """SOC 기반 초기 농도 설정 (in-place). build_soc2theta 내부 호출."""
    soc2theta_n, soc2theta_p = build_soc2theta(param)
    c_n_init, c_p_init = soc2conc(soc, param, soc2theta_n, soc2theta_p)
    param.update({
        "Initial concentration in negative electrode [mol.m-3]": c_n_init,
        "Initial concentration in positive electrode [mol.m-3]": c_p_init,
    })
    return param

# How to use:
# parameter_values = parameter_values.copy()

# c_n_init, c_p_init = soc2conc(soc, parameter_values, soc2theta_n, soc2theta_p)

# parameter_values.update({
#     "Initial concentration in negative electrode [mol.m-3]": c_n_init,
#     "Initial concentration in positive electrode [mol.m-3]": c_p_init,
#     "Current function [A]": current_fn,
# })