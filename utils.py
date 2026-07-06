import pybamm
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
import pandas as pd                    # ← 추가
from scipy.interpolate import interp1d

from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent / "data"





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
            "Rest for 150 seconds",
            "Charge at 1C for 30 seconds",
            "Rest for 150 seconds",
            "Discharge at C/3 for 540 seconds",
            "Rest for 600 seconds",
            "Charge at C/3 for 540 seconds",
            "Rest for 600 seconds",
        )]

    elif profile_id == "C3_dchg":
        steps = [(
            "Rest for 10 seconds",
            "Discharge at C/3 for 1080 seconds",
            "Rest for 600 seconds",
        )]

    elif profile_id == "1C_dchg":
        steps = [(
            "Rest for 10 seconds",
            "Discharge at 1C for 360 seconds",
            "Rest for 600 seconds",
        )]

    elif profile_id == "2C_dchg":
        steps = [(
            "Rest for 10 seconds",
            "Discharge at 2C for 180 seconds",
            "Rest for 600 seconds",
        )]

    elif profile_id == "5C_dchg":
        steps = [(
            "Rest for 10 seconds",
            "Discharge at 5C for 72 seconds",
            "Rest for 600 seconds",
        )]

    elif profile_id == "C50_dchg":
        steps = [(
            "Discharge at C/50 until 2.5 V",
        )]

    elif profile_id == "3C_dchg":
        steps = [(
            "Discharge at 3C until 2.5 V",
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
        # max_num_steps caps the integrator's internal steps per output point.
        # A healthy solve uses far fewer; a pathologically stiff parameter set
        # (the optimizer can wander into one) would otherwise take ever-smaller
        # steps and hang forever. With the cap it raises instead -> callers'
        # try/except turns it into a fit penalty rather than a stuck worker.
        solver = pybamm.IDAKLUSolver(options={"max_num_steps": 50000})

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




######################################################################
###################### Sensitivity Analysis #######################
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



# Standard Chen2020 graphite OCP, captured once for the shifted variant below.
_GRAPHITE_OCP_CHEN2020 = pybamm.ParameterValues("Chen2020")[
    "Negative electrode OCP [V]"
]


def ocp_n_discrepancy(sto):
    """Chen2020 graphite (anode) OCP_n with a small, SOC-localized POSITIVE bias.

    Adds a Gaussian bump on top of the standard Chen2020 graphite OCP:
        - `peak`  mV up at `center`
        - `floor` mV up at the edges
    NOTE: `center`/`width` are in graphite STOICHIOMETRY, not cell SOC.

    Use as the 'true' OCP_n for a reference model while your model keeps the
    unshifted Chen2020 OCP -> study OCP-mismatch (structural model error).
    Set via:  params["Negative electrode OCP [V]"] = ocp_n_discrepancy
    """
    U = _GRAPHITE_OCP_CHEN2020(sto)

    peak = 3e-3     # +15 mV at center
    floor = 2e-3     # +10 mV at edges
    center = 0.50    # graphite stoichiometry (NOT cell SOC)
    width = 0.18

    bump = pybamm.exp(-((sto - center) ** 2) / (2 * width ** 2))
    dU = floor + (peak - floor) * bump   # positive bias, added to anode OCP
    return U + dU


def Dsn_discrepancy(c_s, T):
    """Discrepancy: concentration-dependent NEGATIVE particle diffusivity.

    Physically realistic: D DECREASES as the particle fills (higher c_s), like
    the electrolyte. Truth uses this; the fit uses a constant Ds- and cannot
    reproduce the shape -> structural mismatch.
    D0 = nominal Ds-; with offset 0.9, D == D0 at sto=0.9. Ranges
    ~7.94*D0 (empty, sto=0) .. 0.79*D0 (full, sto=1): 10x swing.
    """
    D0 = 3.3e-14
    cmax = 33133.0          # Chen2020 negative max concentration [mol.m-3]
    sto = c_s / cmax
    return D0 * 10 ** (0.9 - sto)   # D0 at sto=0.9; 7.94*D0 (empty) .. 0.79*D0 (full): 10x swing


def Dsp_discrepancy(c_s, T):
    """Discrepancy: concentration-dependent POSITIVE particle diffusivity.

    D DECREASES as the particle fills (higher c_s). For the cathode, "full"
    is at cell SOC 0% (lithiated), so D is low there and high near SOC 100%.
    D0 = nominal Ds+; with offset 0.85, D == D0 at sto=0.85. Ranges
    ~7.08*D0 (empty, sto=0) .. 0.71*D0 (full, sto=1): 10x swing.
    """
    D0 = 4.0e-15
    cmax = 63104.0
    sto = c_s / cmax
    return D0 * 10 ** (1.72 * (0.85 - sto))   # window 0.27~0.85 안에서 10x


# Named RC-overpotential discrepancy branches (lumped dynamics SPMe lacks).
# Each branch is a saturating parallel-RC driven by current:
#   R     = V_max / I_REF   -> steady drop at the reference (1C) current
#   tau   = R * C           -> relaxation time constant
#   v_max -> HARD cap on |V_rc| so a high C-rate doesn't push the branch past
#            its intended magnitude (binds only when |I| > I_REF).
# Toggle each independently in DISC: "rc_short" and/or "rc_long".
RC_I_REF = 5.0   # reference current [A] (~1C for the LGM50 cell) used to size R
RC_SPECS = {
    # rc_short: 5 mV @ 5 A -> R=1 mOhm, tau=30 s, capped at 5 mV
    "rc_short": dict(R=5.0e-3 / RC_I_REF, tau=30.0,  v_max=5.0e-3),
    # rc_long: 20 mV @ 5 A -> R=4 mOhm, tau=300 s, capped at 20 mV
    "rc_long":  dict(R=20.0e-3 / RC_I_REF, tau=300.0, v_max=20.0e-3),
}


def Vrc_discrepancy(t, I, branches):
    """Lumped RC-overpotential discrepancy: extra relaxation dynamics SPMe lacks.

    Sums one or more (optionally saturating) parallel-RC branches driven by the
    cell current I(t):
        tau * dV_rc/dt + V_rc = R * I,     tau = R*C
    Exact zero-order-hold update (stable on PyBaMM's non-uniform t grid):
        a = exp(-dt_k / tau);   V[k] = a*V[k-1] + R*(1-a)*I[k]
    with V[0] = 0 (relaxed start), then clamped to |V| <= v_max if given.

    Usage (truth-side only; the fit stays the clean SPMe):
        sol  = run_model(...)                       # SPMe truth
        t, I = sol["Time [s]"].entries, sol["Current [A]"].entries
        V_truth = sol["Voltage [V]"].entries - Vrc_discrepancy(t, I, branches)
    NOTE the MINUS: PyBaMM current is +ve on discharge and an overpotential
    LOWERS terminal voltage, so V_rc is subtracted (it is >=0 while I>0).

    Parameters
    ----------
    t, I     : (N,) arrays   time [s] and current [A] from the SPMe solution
    branches : list of branch specs, each either
                 dict(R=.., tau=.., v_max=..)   (v_max optional), or
                 a tuple (R, tau) / (R, tau, v_max).
               Pass e.g. [RC_SPECS["rc_short"], RC_SPECS["rc_long"]].

    Returns
    -------
    (N,) array  V_rc(t) [V] = sum over branches
    """
    t = np.asarray(t, dtype=float)
    I = np.asarray(I, dtype=float)
    n = t.shape[0]
    dt = np.diff(t)
    V_rc = np.zeros(n)
    for br in branches:
        if isinstance(br, dict):
            R, tau, v_max = br["R"], br["tau"], br.get("v_max")
        elif len(br) == 3:
            R, tau, v_max = br
        else:
            (R, tau), v_max = br, None
        v = 0.0
        branch = np.zeros(n)
        for k in range(1, n):
            a = np.exp(-dt[k - 1] / tau)
            v = a * v + R * (1.0 - a) * I[k]
            if v_max is not None:
                v = max(-v_max, min(v_max, v))   # saturate |V_rc| <= v_max
            branch[k] = v
        V_rc += branch
    return V_rc


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



def residual_sensitivity_decomposition(
    r,
    S,
    sigma=None,
    t=None,
    labels=None,
    plot=False,
    rtol=1e-6,
):
    """Split a fit residual into the part inside the sensitivity span (which
    parameters COULD reduce) and the orthogonal part (which NO parameter can
    reduce -> the model-discrepancy fingerprint).

        r_parallel = U_k U_kᵀ r    (U_k = left singular vectors of S above rtol)
        r_perp     = r - r_parallel

    Uses an SVD with a rank tolerance so near-degenerate (collinear) sensitivity
    columns don't blow up the projection -- the QR version is unreliable when
    cond(S) is large.

    How to read the output (at a CONVERGED least-squares fit):
      * Optimality is Sᵀr = 0, so RMS(r_parallel) ~ 0 by construction.
        -> RMS(r_parallel) NOT ~0  => not converged, or a parameter hit a bound.
      * RMS(r_perp) vs the noise floor is the real signal:
        -> RMS(r_perp) >> noise     => structural lack of fit (discrepancy).
      * the SHAPE of r_perp(t) is the discrepancy a model #2 must represent.
      * rank < #params => some columns are collinear (not separately
        identifiable); those directions are dropped from the projection.

    All magnitudes are reported as RMS in mV (= L2 norm / sqrt(N)), matching RMSE.

    Parameters
    ----------
    r : (N,) array           residual, V_true - V_fit  [V]
    S : (N, p) array         sensitivity matrix dV/dtheta (one column per param)
    sigma : float/(N,) or None  measurement noise std; weights the projection so
                                orthogonality matches the least-squares objective
                                (uniform sigma -> identical to unweighted)
    t : (N,) or None         time vector (x-axis for the plot)
    labels : list or None    parameter names (only used in the printout)
    plot : bool              overlay r, r_parallel, r_perp [mV] if True
    rtol : float             singular values below rtol*max are treated as zero
                             (drops collinear directions); default 1e-6

    Returns
    -------
    dict: r_parallel, r_perp, rms_r, rms_parallel, rms_perp, frac_parallel,
          frac_perp, rank, n_params, cond_S, singular_values
    """
    r = np.asarray(r, dtype=float).reshape(-1)
    S = np.asarray(S, dtype=float)
    N = r.shape[0]
    if S.ndim != 2 or S.shape[0] != N:
        raise ValueError(f"S must be (N, p) matching r (N={N}); got {S.shape}")
    n_params = S.shape[1]

    # weight into the space where least-squares optimality (Sᵀr=0) holds
    if sigma is None:
        w = np.ones_like(r)
    else:
        w = 1.0 / np.broadcast_to(np.asarray(sigma, dtype=float), r.shape)

    rw = r * w
    Sw = S * w[:, None]

    # SVD-based projection, robust to near-degenerate columns
    U, svals, _ = np.linalg.svd(Sw, full_matrices=False)
    smax = svals[0] if svals.size else 0.0
    keep = svals > rtol * smax
    rank = int(keep.sum())
    Uk = U[:, keep]

    r_par_w = Uk @ (Uk.T @ rw) if rank else np.zeros_like(rw)
    r_perp_w = rw - r_par_w

    # back to volts (components still sum to r)
    r_parallel = r_par_w / w
    r_perp = r_perp_w / w

    def rms(v):
        return float(np.sqrt(np.mean(v ** 2)))

    rms_r = rms(r)
    cond_S = float(smax / svals[-1]) if svals[-1] > 0 else np.inf
    out = {
        "r_parallel": r_parallel,
        "r_perp": r_perp,
        "rms_r": rms_r,
        "rms_parallel": rms(r_parallel),
        "rms_perp": rms(r_perp),
        "frac_parallel": rms(r_parallel) / rms_r if rms_r else 0.0,
        "frac_perp": rms(r_perp) / rms_r if rms_r else 0.0,
        "rank": rank,
        "n_params": n_params,
        "cond_S": cond_S,
        "singular_values": svals,
    }

    print("residual sensitivity decomposition"
          + (f"  (params: {labels})" if labels else ""))
    print(f"  RMS r        = {out['rms_r'] * 1e3:.4f} mV")
    print(f"  RMS r_par    = {out['rms_parallel'] * 1e3:.4f} mV"
          f"   (fraction {out['frac_parallel']:.3f}) -> ~0 if converged")
    print(f"  RMS r_perp   = {out['rms_perp'] * 1e3:.4f} mV"
          f"   (fraction {out['frac_perp']:.3f}) -> discrepancy if >> noise")
    print(f"  rank(S)      = {rank}/{n_params}"
          f"   (rtol={rtol:g}; full cond(S)={cond_S:.2e})")
    if rank < n_params:
        print(f"  NOTE: {n_params - rank} collinear direction(s) dropped "
              f"-> those params are not separately identifiable")

    if plot:
        import matplotlib.pyplot as plt
        x = t if t is not None else np.arange(N)
        plt.figure(figsize=(10, 4))
        plt.plot(x, r * 1e3, label="residual", lw=1.0)
        plt.plot(x, r_parallel * 1e3, label="parallel (reducible)", lw=1.0)
        plt.plot(x, r_perp * 1e3, label="orthogonal (discrepancy)", lw=1.2)
        plt.axhline(0, color="k", lw=0.4)
        plt.xlabel("Time [s]" if t is not None else "index")
        plt.ylabel("mV")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return out



def print_relative_sensitivity(
    Sens,
    params=None,
    title=None,
):
    """
    Relative sensitivity based on RMS

    rel_i = RMS_i / sum(RMS)

    Sum(rel_i) = 1
    """

    if params is None:
        params = [k for k in Sens.keys() if k != "t"]

    rms = {}

    for p in params:
        s = np.asarray(Sens[p]).squeeze()
        rms[p] = np.sqrt(np.mean(s**2))

    total = sum(rms.values())

    rel = {
        p: rms[p] / total
        for p in params
    }

    if title is not None:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    print(
        "Parameter".ljust(25),
        "RMS".rjust(12),
        "Relative".rjust(12),
    )

    for p, val in sorted(
        rel.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(
            f"{PARAM_LABELS.get(p,p):25s}"
            f"{rms[p]:12.3e}"
            f"{100*val:11.2f}%"
        )

    return rel



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




def sensitivity_ranking(Sens, params):
    rows = []

    for p in params:
        s = np.asarray(Sens[p]).squeeze()
        rms = np.sqrt(np.mean(s**2))
        peak = np.max(np.abs(s))

        rows.append((p, rms, peak))

    rows.sort(key=lambda x: x[1], reverse=True)

    return rows


def print_sensitivity_ranking(profile_id, Sens, title, params, top_n=None):
    print("\n" + "-" * 80)
    print(f"{profile_id} | {title}")
    print("-" * 80)

    rows = sensitivity_ranking(Sens, params)

    if top_n is not None:
        rows = rows[:top_n]

    print("Parameter".ljust(25), "RMS".rjust(12), "Peak".rjust(12))

    for p, rms, peak in rows:
        print(
            f"{PARAM_LABELS.get(p,p):25s}"
            f"{rms:12.3e}"
            f"{peak:12.3e}"
        )




def run_perturbed_profiles_single_update(
    name,
    initial_soc,
    experiment,
    theta_values_nominal,
    model_options,
    frac=0.01,
):
    base_numeric = make_base_params(
        "Chen2020",
        soc=initial_soc,
        sensitivity_ready=False,
    )

    nominal_val = float(theta_values_nominal[name])

    def solve_with_value(val):
        p = base_numeric.copy()
        p.update({name: val})

        model_fd = make_model(
            "SPMe",
            options=model_options,
        )

        sol = run_model(
            model_fd,
            p,
            experiment,
        )

        return {
            "t": sol["Time [s]"].entries,
            "V": sol["Voltage [V]"].entries,
            "csn_avg": sol["Average negative particle concentration [mol.m-3]"].entries,
            "csp_avg": sol["Average positive particle concentration [mol.m-3]"].entries,
        }

    return {
        "nominal": solve_with_value(nominal_val),
        "lower": solve_with_value(nominal_val * (1.0 - frac)),
        "upper": solve_with_value(nominal_val * (1.0 + frac)),
    }



# %% FD sensitivities using single-parameter update

# def compute_fd_sensitivity_single_update(
#     sensitivity_targets,
#     initial_soc,
#     experiment,
#     theta_values_nominal,
#     model_options,
#     endpoints,
#     V_span,
#     frac=0.01,
# ):
#     Sens_V_fd = {"t": None}
#     Sens_soc_n_fd = {"t": None}
#     Sens_soc_p_fd = {"t": None}

#     for name in sensitivity_targets:
#         print(name)

#         profiles = run_perturbed_profiles_single_update(
#             name=name,
#             initial_soc=initial_soc,
#             experiment=experiment,
#             theta_values_nominal=theta_values_nominal,
#             model_options=model_options,
#             frac=frac,
#         )

#         t_low = profiles["lower"]["t"]
#         t_up = profiles["upper"]["t"]

#         t_end = min(t_low[-1], t_up[-1])
#         t_common = profiles["nominal"]["t"][profiles["nominal"]["t"] <= t_end]

#         if Sens_V_fd["t"] is None:
#             Sens_V_fd["t"] = t_common
#             Sens_soc_n_fd["t"] = t_common
#             Sens_soc_p_fd["t"] = t_common

#         def interp_case(case, key):
#             return interp1d(
#                 profiles[case]["t"],
#                 profiles[case][key],
#                 bounds_error=False,
#                 fill_value=np.nan,
#             )(t_common)

#         V_low = interp_case("lower", "V")
#         V_up = interp_case("upper", "V")

#         csn_low = interp_case("lower", "csn_avg")
#         csn_up = interp_case("upper", "csn_avg")

#         csp_low = interp_case("lower", "csp_avg")
#         csp_up = interp_case("upper", "csp_avg")

#         S_V = (V_up - V_low) / (2.0 * frac)
#         S_csn = (csn_up - csn_low) / (2.0 * frac)
#         S_csp = (csp_up - csp_low) / (2.0 * frac)

#         csn_span = endpoints["csn_100"] - endpoints["csn_0"]
#         csp_span = endpoints["csp_0"] - endpoints["csp_100"]

#         Sens_V_fd[name] = S_V / V_span
#         Sens_soc_n_fd[name] = S_csn / csn_span
#         Sens_soc_p_fd[name] = -S_csp / csp_span

#         print(
#             f"{PARAM_LABELS.get(name, name):25s}"
#             f" t_end={t_end:8.1f}"
#         )

#     return Sens_V_fd, Sens_soc_n_fd, Sens_soc_p_fd


def compute_fd_sensitivity_single_update_fast(
    sensitivity_targets,
    initial_soc,
    experiment,
    theta_values_nominal,
    model_options,
    endpoints,
    V_span,
    frac=0.01,
):
    base_numeric = make_base_params(
        "Chen2020",
        soc=initial_soc,
        sensitivity_ready=False,
    )

    model_nom = make_model("SPMe", options=model_options)
    sol_nom = run_model(model_nom, base_numeric, experiment)

    t_ref = sol_nom["Time [s]"].entries

    Sens_V_fd = {"t": t_ref}
    Sens_soc_n_fd = {"t": t_ref}
    Sens_soc_p_fd = {"t": t_ref}

    csn_span = endpoints["csn_100"] - endpoints["csn_0"]
    csp_span = endpoints["csp_0"] - endpoints["csp_100"]

    def solve_with_param(name, val):
        p = base_numeric.copy()
        p.update({name: val})

        model_fd = make_model("SPMe", options=model_options)
        sol = run_model(model_fd, p, experiment)

        return {
            "t": sol["Time [s]"].entries,
            "V": sol["Voltage [V]"].entries,
            "csn_avg": sol["Average negative particle concentration [mol.m-3]"].entries,
            "csp_avg": sol["Average positive particle concentration [mol.m-3]"].entries,
        }

    def interp_to_ref(profile, key):
        return interp1d(
            profile["t"],
            profile[key],
            bounds_error=False,
            fill_value="extrapolate",
        )(t_ref)

    for name in sensitivity_targets:
        print("FD:", PARAM_LABELS.get(name, name))

        nominal_val = float(theta_values_nominal[name])

        low = solve_with_param(name, nominal_val * (1.0 - frac))
        up = solve_with_param(name, nominal_val * (1.0 + frac))

        print(
            "  t_end low/up/ref:",
            low["t"][-1],
            up["t"][-1],
            t_ref[-1],
        )

        V_low = interp_to_ref(low, "V")
        V_up = interp_to_ref(up, "V")

        csn_low = interp_to_ref(low, "csn_avg")
        csn_up = interp_to_ref(up, "csn_avg")

        csp_low = interp_to_ref(low, "csp_avg")
        csp_up = interp_to_ref(up, "csp_avg")

        S_V = (V_up - V_low) / (2.0 * frac)
        S_csn = (csn_up - csn_low) / (2.0 * frac)
        S_csp = (csp_up - csp_low) / (2.0 * frac)

        Sens_V_fd[name] = S_V / V_span
        Sens_soc_n_fd[name] = S_csn / csn_span
        Sens_soc_p_fd[name] = -S_csp / csp_span

    return Sens_V_fd, Sens_soc_n_fd, Sens_soc_p_fd



def plot_perturbed_profiles(
    profiles,
    name,
):
    label = PARAM_LABELS.get(name, name)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4),
    )

    plot_info = [
        ("V", "Voltage [V]"),
        ("csn_avg", "Average csn [mol m-3]"),
        ("csp_avg", "Average csp [mol m-3]"),
    ]

    for ax, (key, ylabel) in zip(axes, plot_info):
        for case, style in [
            ("nominal", "-"),
            ("lower", "--"),
            ("upper", "--"),
        ]:
            ax.plot(
                profiles[case]["t"],
                profiles[case][key],
                style,
                label=case,
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(key)
        ax.grid(True, alpha=0.4)

    axes[0].legend()

    fig.suptitle(label)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


######################################################################
###################### Results Visualization #######################
######################################################################
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
    figsize_per_plot=(5, 3),
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

    line1_ref = None
    line2_ref = None

    for i, param in enumerate(params):
        ax1 = axes[i]
        ax2 = ax1.twinx()

        y_left = np.asarray(sens_left[param]).squeeze()
        y_right = np.asarray(sens_right[param]).squeeze()

        line1, = ax1.plot(t, y_left, linewidth=2.0)
        line2, = ax2.plot(t, y_right, color="red", linewidth=2.0)

        if i == 0:
            line1_ref, line2_ref = line1, line2

        ax1.set_ylabel(label_left, fontsize=13)
        ax2.set_ylabel(label_right, color="red", fontsize=13)
        ax2.tick_params(axis="y", colors="red", labelsize=12)
        ax1.tick_params(axis="both", labelsize=12)
        ax2.spines["right"].set_color("red")

        ax1.set_title(PARAM_LABELS.get(param, param), fontsize=15)
        ax1.set_xlabel("Time [s]", fontsize=13)
        ax1.grid(True, alpha=0.4)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # legend를 figure 왼쪽 위 (suptitle 라인 왼쪽)로
    fig.legend(
        [line1_ref, line2_ref],
        [label_left, label_right],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=14,
        frameon=True,
    )

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=17)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()













################################################################
