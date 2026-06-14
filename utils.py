import pybamm
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
import pandas as pd                    # ← 추가
from scipy.interpolate import interp1d

from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent / "data"



######################################################################
###################### SOC Conversion Utilities ######################
######################################################################
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


def get_soc_endpoints(param):

    soc2theta_n, soc2theta_p = build_soc2theta(param)

    c_n_max = param["Maximum concentration in negative electrode [mol.m-3]"]
    c_p_max = param["Maximum concentration in positive electrode [mol.m-3]"]

    return {
        "csn_at_soc100": float(soc2theta_n(1.0) * c_n_max),
        "csp_at_soc100": float(soc2theta_p(1.0) * c_p_max),

        "csn_at_soc0": float(soc2theta_n(0.0) * c_n_max),
        "csp_at_soc0": float(soc2theta_p(0.0) * c_p_max),
    }

# How to use:
# parameter_values = parameter_values.copy()

# c_n_init, c_p_init = soc2conc(soc, parameter_values, soc2theta_n, soc2theta_p)

# parameter_values.update({
#     "Initial concentration in negative electrode [mol.m-3]": c_n_init,
#     "Initial concentration in positive electrode [mol.m-3]": c_p_init,
#     "Current function [A]": current_fn,
# })



######################################################################
###################### electrochem Model Setup #######################
######################################################################
def make_model(model_name="SPMe", options=None):
    """
    Create a PyBaMM lithium-ion model.
    model_name: "DFN", "SPMe", or "SPM"
    """
    if options is None:
        options = {}

    model_name = model_name.lower()

    if model_name == "dfn":
        return pybamm.lithium_ion.DFN(options=options)
    elif model_name == "spme":
        return pybamm.lithium_ion.SPMe(options=options)
    elif model_name == "spm":
        return pybamm.lithium_ion.SPM(options=options)
    else:
        raise ValueError("model_name must be one of: 'DFN', 'SPMe', 'SPM'")
    


def make_base_params(param_set="Chen2020", soc=None, sensitivity_ready=False):
    """
    Create PyBaMM parameter values.

    If sensitivity_ready=True, convert selected callable parameters
    into scalar/nominal forms for sensitivity/FIM/parameter ID.
    """
    params = pybamm.ParameterValues(param_set)

    if sensitivity_ready:
        c_e_ref = 1000.0
        T_ref_K = 298.15

        kappa_nominal = float(
            params["Electrolyte conductivity [S.m-1]"](c_e_ref, T_ref_K)
        )
        De_nominal = float(
            params["Electrolyte diffusivity [m2.s-1]"](c_e_ref, T_ref_K)
        )

        params["Electrolyte conductivity [S.m-1]"] = kappa_nominal
        params["Electrolyte diffusivity [m2.s-1]"] = De_nominal

        params["Negative electrode exchange-current density [A.m-2]"] = j0_neg_nominal
        params["Positive electrode exchange-current density [A.m-2]"] = j0_pos_nominal

        params["Contact resistance [Ohm]"] = 0.005

    if soc is not None:
        set_init_concentration(params, soc)

    return params



# Profile ID → CSV filename 매핑
PROFILE_CSV_MAP = {
    'FUDS': "FUDS_scaled.csv",
    'US06': "US06_scaled.csv",
    'WLTP': "WLTP_scaled.csv",
}

def make_experiment(profile_id) -> pybamm.Experiment:
    """
    Parameters
    ----------
    profile_id : int (1~5), str ('hppc', 'FUDS', 'US06', 'WLTP')
    soc        : float, 0.0~1.0  (profile 5 에서만 사용)

    Returns
    -------
    pybamm.Experiment
    """

    if profile_id == 'sin_01Hz':
        f, pts = 0.1, 100
        n_cycles = 10
        dt = 1.0 / (f * pts)
        t_data = np.arange(n_cycles * pts + 1) * dt
        I_data = 5.0 * np.sin(2.0 * np.pi * f * t_data)
        current_interp = pybamm.Interpolant(t_data, I_data, pybamm.t, name="sin_01Hz_current")
        step = pybamm.step.current(current_interp, duration=f"{t_data[-1]} seconds")
        return pybamm.Experiment([step])

    elif profile_id == 'sin_1Hz':
        f, pts = 1.0, 100
        n_cycles = 10
        dt = 1.0 / (f * pts)
        t_data = np.arange(n_cycles * pts + 1) * dt
        I_data = 5.0 * np.sin(2.0 * np.pi * f * t_data)
        current_interp = pybamm.Interpolant(t_data, I_data, pybamm.t, name="sin_1Hz_current")
        step = pybamm.step.current(current_interp, duration=f"{t_data[-1]} seconds")
        return pybamm.Experiment([step])

    elif profile_id == 'sin_10Hz':
        f, pts = 10.0, 100
        n_cycles = 10
        dt = 1.0 / (f * pts)
        t_data = np.arange(n_cycles * pts + 1) * dt
        I_data = 5.0 * np.sin(2.0 * np.pi * f * t_data)
        current_interp = pybamm.Interpolant(t_data, I_data, pybamm.t, name="sin_10Hz_current")
        step = pybamm.step.current(current_interp, duration=f"{t_data[-1]} seconds")
        return pybamm.Experiment([step])

    elif profile_id == 'sin_100Hz':
        f, pts = 100.0, 50
        n_cycles = 10
        dt = 1.0 / (f * pts)
        t_data = np.arange(n_cycles * pts + 1) * dt
        I_data = 5.0 * np.sin(2.0 * np.pi * f * t_data)
        current_interp = pybamm.Interpolant(t_data, I_data, pybamm.t, name="sin_100Hz_current")
        step = pybamm.step.current(current_interp, duration=f"{t_data[-1]} seconds")
        return pybamm.Experiment([step])

    elif profile_id == 4:
        # 0.5C discharge 5 min, rest 20 min, 0.5C charge 5 min
        steps = [(
            "Rest for 10 second",
            "Discharge at 0.5C for 5 minutes",
            "Rest for 20 minutes",
            "Charge at 0.5C for 5 minutes",
            "Rest for 20 minutes",
        )]

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

    elif profile_id == "C20_dchg":
        steps = [(
            "Discharge at C/20 until 2.5 V",
            "Hold at 2.5 V until C/50",
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

    else:
        raise ValueError(
            f"profile_id must be 1~5, 'hppc', or one of {list(PROFILE_CSV_MAP.keys())}, got {profile_id}"
        )

    return pybamm.Experiment(steps)



def run_model(
    model,
    params,
    experiment,
    inputs=None,
    calculate_sensitivities=False,
    solver=None,
):
    """
    Run a PyBaMM simulation and return the solution.
    """
    if solver is None:
        solver = pybamm.IDAKLUSolver()

    sim = pybamm.Simulation(
        model,
        experiment=experiment,
        parameter_values=params,
        solver=solver,
    )

    sol = sim.solve(
        inputs=inputs,
        calculate_sensitivities=calculate_sensitivities,
    )

    return sol




######################################################################
###################### Sensitivity Model Setup #######################
######################################################################
# ============================================================
# Exchange-current nominal values
# ============================================================
m_ref_neg_nominal = 6.48e-7
m_ref_pos_nominal = 3.42e-6
def j0_neg_nominal(c_e, c_s_surf, c_s_max, T):
    arrh = pybamm.exp(
        35000 / pybamm.constants.R * (1 / 298.15 - 1 / T)
    )

    return (
        m_ref_neg_nominal
        * arrh
        * c_e**0.5
        * c_s_surf**0.5
        * (c_s_max - c_s_surf) ** 0.5
    )


def j0_pos_nominal(c_e, c_s_surf, c_s_max, T):
    arrh = pybamm.exp(
        17800 / pybamm.constants.R * (1 / 298.15 - 1 / T)
    )

    return (
        m_ref_pos_nominal
        * arrh
        * c_e**0.5
        * c_s_surf**0.5
        * (c_s_max - c_s_surf) ** 0.5
    )



# ============================================================
# Modified Parameter Functions for Sensitivity
# ============================================================
def j0_neg_input(c_e, c_s_surf, c_s_max, T):
    m_ref = pybamm.InputParameter(
        "Negative electrode exchange-current density [A.m-2]"
    )

    arrh = pybamm.exp(
        35000 / pybamm.constants.R * (1 / 298.15 - 1 / T)
    )

    return (
        m_ref
        * arrh
        * c_e**0.5
        * c_s_surf**0.5
        * (c_s_max - c_s_surf) ** 0.5
    )


def j0_pos_input(c_e, c_s_surf, c_s_max, T):
    m_ref = pybamm.InputParameter(
        "Positive electrode exchange-current density [A.m-2]"
    )

    arrh = pybamm.exp(
        17800 / pybamm.constants.R * (1 / 298.15 - 1 / T)
    )

    return (
        m_ref
        * arrh
        * c_e**0.5
        * c_s_surf**0.5
        * (c_s_max - c_s_surf) ** 0.5
    )



def prepare_sensitivity_inputs(params, sensitivity_targets, values=None):
    sensitivity_params = params.copy()
    theta_values = {}

    special_params = {
        "Negative electrode exchange-current density [A.m-2]": (
            j0_neg_input,
            m_ref_neg_nominal,
        ),
        "Positive electrode exchange-current density [A.m-2]": (
            j0_pos_input,
            m_ref_pos_nominal,
        ),
    }

    for name in sensitivity_targets:
        if name in special_params:
            input_function, nominal_val = special_params[name]
            sensitivity_params[name] = input_function
        else:
            val = sensitivity_params[name]

            if callable(val):
                raise ValueError(
                    f"{name} is callable. "
                    "Use sensitivity_ready=True or add special handling."
                )

            nominal_val = float(val)
            sensitivity_params.update({name: "[input]"})

        if values is None:
            theta_values[name] = nominal_val
        else:
            theta_values[name] = values[name]

    return sensitivity_params, theta_values




def compute_fim(
    Sens,
    params=None,
    sigma=None,
):
    """
    Compute Fisher Information Matrix

    sigma=None:
        F = J.T @ J

    sigma=float:
        F = J.T @ J / sigma**2
    """

    if params is None:
        params = [k for k in Sens.keys() if k != "t"]

    J = np.column_stack([
        np.asarray(Sens[p]).squeeze()
        for p in params
    ])

    if sigma is None:
        F = J.T @ J
    else:
        F = (J.T @ J) / sigma**2

    return F



def fim_diagonal_comparison(F_base, F_new, params):
    base_diag = np.diag(F_base)
    new_diag = np.diag(F_new)

    ratio = new_diag / base_diag
    diff = new_diag - base_diag

    for p, b, n, r, d in zip(params, base_diag, new_diag, ratio, diff):
        label = PARAM_LABELS.get(p, p)
        print(f"{label:20s}  base={b:.3e}  new={n:.3e}  ratio={r:.3f}  diff={d:.3e}")





######################################################################
###################### Results Visualization #######################
######################################################################
def get_t_query(sol, dt=1.0):
    """
    Create uniform query time vector from solution.
    """

    t_raw = sol["Time [s]"].entries

    return np.arange(
        t_raw[0],
        t_raw[-1] + dt,
        dt,
    )



def interp_to_query(t_raw, y_raw, t_query=None):
    """
    Interpolate y_raw(t_raw) onto t_query.

    If t_query is None, return original t_raw and y_raw.
    """
    t_raw = np.asarray(t_raw).flatten()
    y_raw = np.asarray(y_raw)

    if t_query is None:
        return t_raw, y_raw

    t_query = np.asarray(t_query).flatten()

    if y_raw.ndim == 1:
        y_query = interp1d(
            t_raw,
            y_raw,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )(t_query)

    else:
        y_query = np.vstack([
            interp1d(
                t_raw,
                y_raw[:, i],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )(t_query)
            for i in range(y_raw.shape[1])
        ]).T

    return t_query, y_query



def get_VIT(sol, t_query=None):
    t_raw = sol["Time [s]"].entries

    _, V = interp_to_query(t_raw, sol["Voltage [V]"].entries, t_query)
    t, I = interp_to_query(t_raw, sol["Current [A]"].entries, t_query)

    out = {
        "t": t,
        "V": V,
        "I": I,
    }

    try:
        _, T = interp_to_query(t_raw, sol["X-averaged cell temperature [K]"].entries, t_query)
        out["T"] = T
    except KeyError:
        pass

    return out



def get_states(sol, state_names, t_query=None):
    t_raw = sol["Time [s]"].entries

    states = {}
    t_out = t_raw if t_query is None else np.asarray(t_query).flatten()
    states["t"] = t_out

    for name in state_names:
        _, y = interp_to_query(t_raw, sol[name].entries, t_query)
        states[name] = y

    return states



def get_sensitivities(
    sol,
    output_name,
    sensitivity_targets,
    t_query=None,
    theta_values=None,
    normalize=False,
):
    t_raw = sol["Time [s]"].entries

    sensitivities = {}
    t_out = t_raw if t_query is None else np.asarray(t_query).flatten()
    sensitivities["t"] = t_out

    for target in sensitivity_targets:
        sens = sol[output_name].sensitivities[target]
        sens = np.asarray(sens).squeeze()

        if normalize:
            if theta_values is None:
                raise ValueError("theta_values must be provided when normalize=True.")
            sens = theta_values[target] * sens

        _, sens_q = interp_to_query(t_raw, sens, t_query)
        sensitivities[target] = sens_q

    return sensitivities



def scale_voltage_sens_by_ocv(
    sensitivities,
    parameter_values,
):

    ocv_min = parameter_values["Open-circuit voltage at 0% SOC [V]"]
    ocv_max = parameter_values["Open-circuit voltage at 100% SOC [V]"]

    ocv_range = ocv_max - ocv_min

    scaled = {"t": sensitivities["t"]}

    for key, value in sensitivities.items():
        if key == "t":
            continue

        scaled[key] = np.asarray(value) / ocv_range

    return scaled



def get_soc_endpoints(param):

    soc2theta_n, soc2theta_p = build_soc2theta(param)

    c_n_max = param["Maximum concentration in negative electrode [mol.m-3]"]
    c_p_max = param["Maximum concentration in positive electrode [mol.m-3]"]

    return {
        "csn_100": float(soc2theta_n(1.0) * c_n_max), # SOC 100%
        "csp_0": float(soc2theta_p(1.0) * c_p_max),   # SOC 100%

        "csn_0": float(soc2theta_n(0.0) * c_n_max),   # SOC 0%
        "csp_100": float(soc2theta_p(0.0) * c_p_max), # SOC 0% 
    }



def concentrations_to_soc_sensitivities(
    Sens_csn,
    Sens_csp,
    endpoints,
):

    csn_span = endpoints["csn_100"] - endpoints["csn_0"]
    csp_span = endpoints["csp_0"] - endpoints["csp_100"]

    Sens_soc_n = Sens_csn.copy()
    Sens_soc_p = Sens_csp.copy()

    for key in Sens_csn:

        if key == "t":
            continue

        Sens_soc_n[key] = Sens_csn[key] / csn_span
        Sens_soc_p[key] = -Sens_csp[key] / csp_span

    return Sens_soc_n, Sens_soc_p



def plot_results(
    results,
    kind=None,
    max_cols=4,
    figsize_per_plot=(4, 3),
    suptitle=None,
):
    t = results["t"]
    keys = [k for k in results.keys() if k != "t"]

    if kind == "VIT" or set(keys).issubset({"V", "I", "T"}):
        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3))
        if len(keys) == 1:
            axes = [axes]

        labels = {
            "V": ("Voltage [V]", "Voltage"),
            "I": ("Current [A]", "Current"),
            "T": ("Temperature [K]", "Temperature"),
        }

        for ax, key in zip(axes, keys):
            ylabel, title = labels.get(key, (key, key))
            ax.plot(t, np.asarray(results[key]).squeeze())
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.4)

        if suptitle is not None:
            fig.suptitle(suptitle)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            plt.tight_layout()

        plt.show()
        return

    n = len(keys)
    ncols = min(max_cols, int(np.ceil(np.sqrt(n))))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )

    axes = np.array(axes).reshape(-1)

    for i, key in enumerate(keys):
        ax = axes[i]
        y = np.asarray(results[key]).squeeze()

        if kind == "sensitivities":
            title = PARAM_LABELS.get(key, key)
            ylabel = "Sensitivity"
        else:
            title = key
            ylabel = key

        ax.plot(t, y)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.4)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if suptitle is not None:
        fig.suptitle(suptitle)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    plt.show()



def plot_sensitivity_comparison_yy(
    sens_left,
    sens_right,
    label_left="Left",
    label_right="Right",
    max_cols=4,
    figsize_per_plot=(4, 3),
    suptitle=None,
):
    t = sens_left["t"]
    params = [k for k in sens_left.keys() if k != "t"]

    n = len(params)
    ncols = min(max_cols, int(np.ceil(np.sqrt(n))))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )
    axes = np.array(axes).reshape(-1)

    for i, param in enumerate(params):
        ax1 = axes[i]
        ax2 = ax1.twinx()

        y_left = np.asarray(sens_left[param]).squeeze()
        y_right = np.asarray(sens_right[param]).squeeze()

        line1, = ax1.plot(
            t,
            y_left,
            linewidth=1.5,
            label=label_left,
        )

        line2, = ax2.plot(
            t,
            y_right,
            color="red",
            linewidth=1.5,
            label=label_right,
        )

        ax1.set_ylabel(label_left)

        ax2.set_ylabel(
            label_right,
            color="red",
        )

        ax2.tick_params(
            axis="y",
            colors="red",
        )

        ax2.spines["right"].set_color("red")

        title = PARAM_LABELS.get(param, param)
        ax1.set_title(title)

        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel(label_left)
        ax2.set_ylabel(label_right)

        ax1.grid(True, alpha=0.4)

        if i == 0:
            ax1.legend([line1, line2], [label_left, label_right], loc="best")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if suptitle is not None:
        fig.suptitle(suptitle)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    plt.show()



# ============================================================
# Parameter display labels
# ============================================================
PARAM_LABELS = {
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
    "Positive particle diffusivity [m2.s-1]": r"$D_s^+$",

    "Negative electrode conductivity [S.m-1]": r"$\sigma^-$",
    "Positive electrode conductivity [S.m-1]": r"$\sigma^+$",

    "Negative particle radius [m]": r"$R_s^-$",
    "Positive particle radius [m]": r"$R_s^+$",
}