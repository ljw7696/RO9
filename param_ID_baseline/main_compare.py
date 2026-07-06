"""Compare two models on the SAME experiment / initial SOC:

  Case 1 - pure PyBaMM SPMe + Chen2020 (all functional forms intact,
           default options: no contact resistance, default surface form).
  Case 2 - current setup: sensitivity-ready params (electrolyte conductivity
           kappa and diffusivity De frozen to scalar nominal values),
           contact resistance 0.005, model options
           {surface form: differential, contact resistance: true}.

Shows how much the modeling choices (frozen functional params + contact
resistance + options) move the terminal voltage.
"""
# %% Imports / config
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pybamm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

pybamm.set_logging_level("ERROR")
from utils import (
    make_base_params,
    make_model,
    run_model,
    make_experiment,
    prepare_sensitivity_inputs,
)

PROFILE_ID = "WLTP"
initial_soc = 0.85
MODEL_OPTIONS = {"surface form": "differential", "contact resistance": "true"}

# the params your setup passes as inputs (their nominal values)
theta_values_nominal = {
    "Negative electrode exchange-current density [A.m-2]": 6.48e-7,
    "Positive electrode exchange-current density [A.m-2]": 3.42e-6,
    "Electrolyte conductivity [S.m-1]": 0.9487,
    "Electrolyte diffusivity [m2.s-1]": 1.7694e-10,
    "Negative particle diffusivity [m2.s-1]": 3.3e-14,
    "Positive particle diffusivity [m2.s-1]": 4.0e-15,
}
fit_params = list(theta_values_nominal.keys())

experiment = make_experiment(profile_id=PROFILE_ID)


# %% Case 2: current (sensitivity-ready) setup
base = make_base_params("Chen2020", soc=initial_soc, sensitivity_ready=True)
p2, iv2 = prepare_sensitivity_inputs(base, fit_params, values=theta_values_nominal)
model2 = make_model("SPMe", options=MODEL_OPTIONS)
sol2 = run_model(model2, p2, experiment, inputs=iv2)
t2 = sol2["Time [s]"].entries
V2 = sol2["Voltage [V]"].entries


# %% Case 1: standard PyBaMM SPMe + Chen2020 with FUNCTIONAL kappa/De intact.
# Same contact resistance (0.005) AND same model options (surface form +
# contact resistance) as Case 2, so the ONLY remaining difference is the frozen
# kappa/De vs functional kappa/De. Clean isolation of the electrolyte-transport effect.
param_pure = make_base_params("Chen2020", soc=initial_soc, sensitivity_ready=False)
param_pure["Contact resistance [Ohm]"] = 0.005
model_pure = make_model("SPMe", options=MODEL_OPTIONS)   # same options as Case 2
sol1 = run_model(model_pure, param_pure, experiment)
t1 = sol1["Time [s]"].entries
V1 = sol1["Voltage [V]"].entries


# %% Difference + plot
V2_on_t1 = interp1d(t2, V2, bounds_error=False, fill_value="extrapolate")(t1)
dV_mV = (V2_on_t1 - V1) * 1e3

print(f"Profile={PROFILE_ID}, initial_soc={initial_soc}")
print(f"t_end: pure={t1[-1]:.1f}s, current={t2[-1]:.1f}s")
print(f"V difference (current - pure):  "
      f"max={np.max(np.abs(dV_mV)):.2f} mV,  RMS={np.sqrt(np.mean(dV_mV**2)):.2f} mV")

fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(t1, V1, lw=1.5, label="pure PyBaMM SPMe (functional)")
ax[0].plot(t2, V2, "--", lw=1.5, label="current setup (frozen kappa/De + Rc)")
ax[0].set_ylabel("Voltage [V]")
ax[0].legend()
ax[0].grid(alpha=0.4)

ax[1].plot(t1, dV_mV, color="r", lw=1.0)
ax[1].axhline(0, color="k", lw=0.6)
ax[1].set_ylabel("ΔV [mV]\n(current − pure)")
ax[1].set_xlabel("Time [s]")
ax[1].grid(alpha=0.4)

plt.tight_layout()
plt.savefig(Path(__file__).parent / "model_compare.png", dpi=130)
plt.show()
print("saved: model_compare.png")


# %% Built-in PyBaMM voltage decomposition (quick view)
# Stacks: OCV + reaction/concentration overpotentials + ohmic losses = terminal V
pybamm.plot_voltage_components(sol1)   # pure PyBaMM
pybamm.plot_voltage_components(sol2)   # current setup


# %% Manual component extraction -> plot each component one by one
# Battery-level, X-averaged voltage components (all in volts).
# Terminal V = OCV + reaction + concentration + electrolyte ohmic + solid ohmic
#              - I*R_contact   (the contact-resistance term shows up in Residual)
COMPONENTS = {
    # Thermodynamic equilibrium voltage = U_pos(SOC) - U_neg(SOC). Depends ONLY
    # on how full each electrode is (state of charge). This is the baseline the
    # cell would sit at with zero current; everything below is a loss on top.
    "OCV": "Battery open-circuit voltage [V]",

    # Kinetic (activation) loss: extra voltage to drive the intercalation
    # reaction at the particle surfaces (Butler-Volmer). Larger when current is
    # high or exchange-current density j0 (k+/k-) is small.
    "Reaction overpotential (eta)": "X-averaged battery reaction overpotential [V]",

    # Loss from Li+ concentration gradients (in electrolyte + particle surface
    # vs average). Builds up as current flows; tied to De and transference no.
    "Concentration overpotential (De)": "X-averaged battery concentration overpotential [V]",

    # Ionic IR drop: resistance to Li+ moving THROUGH the electrolyte. Set by
    # electrolyte conductivity kappa and porosity. <- frozen kappa shows up here.
    "Electrolyte ohmic (kappa)": "X-averaged battery electrolyte ohmic losses [V]",

    # Electronic IR drop: resistance to electrons moving through the solid
    # electrode matrix. Set by electrode conductivity sigma+/sigma-.
    "Solid ohmic (sigma)": "X-averaged battery solid phase ohmic losses [V]",
}


def get_components(sol, label):
    """Pull available voltage components from a solution into {name: array}."""
    t = sol["Time [s]"].entries
    out = {"t": t}
    for name, var in COMPONENTS.items():
        try:
            out[name] = sol[var].entries
        except KeyError:
            print(f"[{label}] missing: {var}")
    # terminal voltage and the leftover (contact resistance + numerics)
    V = sol["Voltage [V]"].entries
    out["Terminal V"] = V
    summed = sum(out[n] for n in COMPONENTS if n in out)
    out["Residual (contact R etc.)"] = V - summed
    return out


comp1 = get_components(sol1, "pure")
comp2 = get_components(sol2, "current")

# one subplot per component; pure vs current on LEFT axis, difference (current -
# pure, in mV) on the RIGHT axis.
names = [n for n in COMPONENTS] + ["Residual (contact R etc.)", "Terminal V"]
fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.2 * len(names)), sharex=True)
for ax, name in zip(axes, names):
    has1 = name in comp1
    has2 = name in comp2

    if has1:
        ax.plot(comp1["t"], comp1[name], lw=1.2, color="blue",
                label="pure PyBaMM SPMe")
    if has2:
        ax.plot(comp2["t"], comp2[name], lw=1.2, color="red", ls="--",
                label="my SPMe")
    ax.set_ylabel(name, fontsize=8)
    ax.grid(alpha=0.4)
    ax.legend(fontsize=7, loc="upper left")

    # difference (my - pure) on the right axis, in mV: black dashed
    if has1 and has2:
        c2_on_t1 = interp1d(
            comp2["t"], comp2[name], bounds_error=False, fill_value="extrapolate"
        )(comp1["t"])
        d_mV = (c2_on_t1 - comp1[name]) * 1e3
        axr = ax.twinx()
        axr.plot(comp1["t"], d_mV, color="black", ls="--", lw=0.9,
                 label="difference")
        axr.axhline(0, color="black", lw=0.4, ls=":")
        axr.set_ylabel("Δ [mV]", color="black", fontsize=7)
        axr.tick_params(axis="y", labelcolor="black", labelsize=6)

axes[-1].set_xlabel("Time [s]")
fig.suptitle(f"Voltage components (left) + difference current-pure [mV] (right)\n"
             f"{PROFILE_ID}, SOC={initial_soc}")
plt.tight_layout(rect=[0, 0, 1, 0.985])
plt.savefig(Path(__file__).parent / "voltage_components.png", dpi=130)
plt.show()
print("saved: voltage_components.png")



# ============================================================
# DISCREPANCY PARAM-ID PIPELINE  (fit + bias + residual decomposition)
# moved from main_sim.py.  Self-contained: re-runs Imports/Setup below.
# ============================================================
# %% Imports
import time
import gc
import sys
from pathlib import Path

import pybamm
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

sys.path.append(str(Path.cwd().parent))
from utils import *

gc.collect()
pybamm.set_logging_level("ERROR")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# %% Setup
log("SCRIPT START")

initial_soc = 0.85
N_STARTS = 30          # 여기 바꾸면 됨
ADD_NOISE = True
NOISE_STD = 1e-3      # 1 mV
MAX_NFEV = 100

# Jacobian for the TRF optimizer. Pick ONE:
#   "fd"       -> finite differences with an enlarged step (DIFF_STEP) so the
#                 perturbation clears the ODE solver's ~1e-4 V noise floor.
#   "analytic" -> PyBaMM sODE (calculate_sensitivities); exact dV/dtheta,
#                 ~1 solve/iteration, no FD noise. (Validate vs FD if unsure.)
JAC_MODE = "fd"         # "fd" (enlarged finite-diff step) or "analytic" (sODE)
DIFF_STEP = 1e-2        # only used when JAC_MODE == "fd"

# Model discrepancy injected into the TRUTH data only (synthetic "real" cell).
# Truth = MY SPMe + discrepancy at theta_values_nominal; the fit uses MY SPMe
# WITHOUT it -> controlled structural mismatch. Toggle each independently:
#   "ocp" -> anode OCP_n bias (ocp_n_discrepancy)
#   "Dsn" -> anode  particle diffusivity, concentration-dependent
#   "Dsp" -> cathode particle diffusivity, concentration-dependent
#   set() / [] -> no discrepancy (inverse-crime sanity: fit recovers theta_true)
DISCREPANCIES = {"ocp", "Dsp", "Dsn"}    # any subset, e.g. {"Dsn"}, {"ocp","Dsp"}, {"ocp","Dsn","Dsp"}

MODEL_OPTIONS = {
    "surface form": "differential",
    "contact resistance": "true",
}

# Single source of truth for the drive profile. Change this ONE line to switch
# profiles; it is used for BOTH the nominal truth data and the optimizer worker.
PROFILE_ID = "hppc"
experiment = make_experiment(profile_id=PROFILE_ID)

theta_values_nominal = {
    "Negative electrode exchange-current density [A.m-2]": 6.48e-7,
    "Positive electrode exchange-current density [A.m-2]": 3.42e-6,
    "Negative electrode double-layer capacity [F.m-2]": 0.2,
    "Positive electrode double-layer capacity [F.m-2]": 0.2,
    "Negative electrode active material volume fraction": 0.75,
    "Positive electrode active material volume fraction": 0.665,
    "Cation transference number": 0.2594,
    "Electrolyte conductivity [S.m-1]": 0.9487,
    "Electrolyte diffusivity [m2.s-1]": 1.7694e-10,
    "Positive electrode porosity": 0.335,
    "Negative electrode porosity": 0.25,
    "Separator porosity": 0.47,
    "Negative particle diffusivity [m2.s-1]": 3.3e-14,
    "Positive particle diffusivity [m2.s-1]": 4.0e-15,
    "Negative electrode conductivity [S.m-1]": 215.0,
    "Positive electrode conductivity [S.m-1]": 0.18,
}

PARAM_LABELS_ASCII = {
    "Negative electrode exchange-current density [A.m-2]": "k-",
    "Positive electrode exchange-current density [A.m-2]": "k+",
    "Negative electrode double-layer capacity [F.m-2]": "Cdl-",
    "Positive electrode double-layer capacity [F.m-2]": "Cdl+",
    "Negative electrode active material volume fraction": "eps_s-",
    "Positive electrode active material volume fraction": "eps_s+",
    "Cation transference number": "t+",
    "Electrolyte conductivity [S.m-1]": "kappa",
    "Electrolyte diffusivity [m2.s-1]": "De",
    "Positive electrode porosity": "eps_e+",
    "Negative electrode porosity": "eps_e-",
    "Separator porosity": "eps_e_sep",
    "Negative particle diffusivity [m2.s-1]": "Ds-",
    "Positive particle diffusivity [m2.s-1]": "Ds+",
    "Negative electrode conductivity [S.m-1]": "sigma-",
    "Positive electrode conductivity [S.m-1]": "sigma+",
}

PARAM_BOUNDS = {
    "Negative electrode exchange-current density [A.m-2]": (1e-9, 1e-5),
    "Positive electrode exchange-current density [A.m-2]": (1e-8, 1e-4),

    # "Negative electrode double-layer capacity [F.m-2]": (0.05, 0.8),
    # "Positive electrode double-layer capacity [F.m-2]": (0.05, 0.8),

    # "Negative electrode active material volume fraction": (0.1, 0.9),
    # "Positive electrode active material volume fraction": (0.1, 0.9),

    # "Cation transference number": (0.1, 0.45),

    "Electrolyte conductivity [S.m-1]": (0.3, 3.0),
    "Electrolyte diffusivity [m2.s-1]": (1e-12, 1e-9),

    # "Positive electrode porosity": (0.1, 0.7),
    # "Negative electrode porosity": (0.1, 0.7),
    # "Separator porosity": (0.1, 0.7),

    "Negative particle diffusivity [m2.s-1]": (1e-17, 1e-13),
    "Positive particle diffusivity [m2.s-1]": (1e-17, 1e-13),

    # "Negative electrode conductivity [S.m-1]": (50.0, 500.0),
    # "Positive electrode conductivity [S.m-1]": (0.05, 5.0),
}

# Fit exactly the params that have a bound above. Comment out a bound in
# PARAM_BOUNDS -> that param is automatically dropped from the fit (held fixed).
fit_params = list(PARAM_BOUNDS.keys())

log_scaled_params = [
    "Negative electrode exchange-current density [A.m-2]",
    "Positive electrode exchange-current density [A.m-2]",
    "Electrolyte conductivity [S.m-1]",
    "Electrolyte diffusivity [m2.s-1]",
    "Negative particle diffusivity [m2.s-1]",
    "Positive particle diffusivity [m2.s-1]",
    "Negative electrode conductivity [S.m-1]",
    "Positive electrode conductivity [S.m-1]",
]

sigma_V_fit = 1e-3


# %% Transform helpers
def check_bounds(params):
    for p in params:
        if p not in PARAM_BOUNDS:
            raise KeyError(f"No bound defined for: {p}")

        lo, hi = PARAM_BOUNDS[p]
        val = theta_values_nominal[p]

        if not (lo <= val <= hi):
            raise ValueError(
                f"Nominal outside bounds for {p}: "
                f"lo={lo}, nominal={val}, hi={hi}"
            )


def theta_dim_to_bar(theta_dim, params):
    theta_bar = []

    for p in params:
        val = float(theta_dim[p])
        lo, hi = PARAM_BOUNDS[p]

        if p in log_scaled_params:
            z = (np.log(val) - np.log(lo)) / (np.log(hi) - np.log(lo))
        else:
            z = (val - lo) / (hi - lo)

        theta_bar.append(z)

    return np.asarray(theta_bar)


def theta_bar_to_dim(theta_bar, params):
    theta_dim = {}

    for z, p in zip(theta_bar, params):
        lo, hi = PARAM_BOUNDS[p]
        z = float(np.clip(z, 0.0, 1.0))

        if p in log_scaled_params:
            theta_dim[p] = np.exp(np.log(lo) + z * (np.log(hi) - np.log(lo)))
        else:
            theta_dim[p] = lo + z * (hi - lo)

    return theta_dim


def solve_voltage(theta_dim, tag, discrepancy=None):
    t_start = time.time()
    discrepancy = set(discrepancy or [])

    # Functional setup (same as main_soc_sensitivity.py):
    # j0 stays a function of concentration; the theta_dim values for the two
    # exchange-current-density params enter as the rate prefactor m_ref,
    # NOT as a flat j0 scalar. All fit_params are passed as PyBaMM inputs.
    log(f"{tag}: build parameters")
    base = make_base_params("Chen2020", soc=initial_soc, sensitivity_ready=True)

    # TRUTH ONLY: inject known structural discrepancies. The fit never sets these.
    # "ocp": biased anode OCP (set before inputs, since OCP is not a fit param).
    if "ocp" in discrepancy:
        base["Negative electrode OCP [V]"] = ocp_n_discrepancy

    p, input_values = prepare_sensitivity_inputs(base, fit_params, values=theta_dim)

    # Particle-diffusivity discrepancies (anode/cathode independent). Ds-/Ds+ are
    # fit params (constant inputs); replace the chosen one(s) with concentration-
    # dependent functions and drop them from the inputs dict.
    DSN = "Negative particle diffusivity [m2.s-1]"
    DSP = "Positive particle diffusivity [m2.s-1]"
    if "Dsn" in discrepancy:               # anode
        p[DSN] = Dsn_discrepancy
        input_values.pop(DSN, None)
    if "Dsp" in discrepancy:               # cathode
        p[DSP] = Dsp_discrepancy
        input_values.pop(DSP, None)

    log(f"{tag}: build model")
    model = make_model("SPMe", options=MODEL_OPTIONS)

    log(f"{tag}: solve start")
    # forward solve only (calculate_sensitivities defaults to False)
    sol = run_model(model, p, experiment, inputs=input_values)

    log(f"{tag}: solve done ({time.time() - t_start:.2f} sec total)")

    # current comes from the SAME solve (no extra run needed)
    t = sol["Time [s]"].entries
    V = sol["Voltage [V]"].entries
    I = sol["Current [A]"].entries

    return t, V, I


# %% Nominal synthetic data
log("Check parameter bounds")
check_bounds(fit_params)

log("Prepare nominal theta_bar")
theta_bar0 = theta_dim_to_bar(theta_values_nominal, fit_params)
theta_dim0 = theta_bar_to_dim(theta_bar0, fit_params)

log("Build TRUTH synthetic voltage data (with discrepancy if enabled)")
t_data, V_data_clean, I_data = solve_voltage(
    theta_dim0, tag="TRUTH", discrepancy=DISCREPANCIES
)
log(f"Truth data ready: len={len(t_data)}, t_end={t_data[-1]:.2f}, "
    f"discrepancies={DISCREPANCIES}")

V_data = V_data_clean.copy()

if ADD_NOISE:
    rng_noise = np.random.default_rng(0)
    V_data = V_data + rng_noise.normal(0.0, NOISE_STD, size=V_data.shape)
    log(f"Added Gaussian voltage noise: std={NOISE_STD * 1e3:.2f} mV")


# %% Plot truth (with discrepancy) vs my clean model (no discrepancy)
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# my own model at theta_true WITHOUT discrepancies = what the fit can represent
t_mine, V_mine, I_mine = solve_voltage(theta_dim0, tag="MY-MODEL", discrepancy=set())

# difference (the part the clean fit cannot reproduce)
V_mine_on_t = interp1d(
    t_mine, V_mine, bounds_error=False, fill_value="extrapolate"
)(t_data)
dV_mV = (V_data_clean - V_mine_on_t) * 1e3

fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(t_data, V_data_clean, "b", lw=1.5,
           label=f"truth (SPMe + {sorted(DISCREPANCIES)})")
ax[0].plot(t_mine, V_mine, "r--", lw=1.5, label="my model (no discrepancy)")
ax[0].set_ylabel("Voltage [V]")
ax[0].legend()
ax[0].grid(alpha=0.4)

ax[1].plot(t_data, dV_mV, "k", lw=1.0)
ax[1].axhline(0, color="gray", lw=0.5)
ax[1].set_ylabel("ΔV [mV]\n(truth − mine)")
ax[1].set_xlabel("Time [s]")
ax[1].grid(alpha=0.4)

fig.suptitle(f"Truth vs clean model at theta_true  |  discrepancies={sorted(DISCREPANCIES)}")
plt.tight_layout()
plt.show()


# %% Worker: one local-optimization start (self-contained for parallel use)
import os
from joblib import Parallel, delayed


def run_one_start(
    start_id,
    t_data,
    V_data,
    fit_params,
    param_bounds,
    log_scaled_params,
    initial_soc,
    model_options,
    sigma_V_fit,
    max_nfev,
    profile_id,           # always passed explicitly from the driver (PROFILE_ID)
    jac_mode="fd",        # "fd" (finite diff) or "analytic" (PyBaMM sODE)
    diff_step=1e-2,       # FD perturbation; only used when jac_mode == "fd"
):
    """Run TRF local optimization from one random start.

    Fully self-contained: builds its own experiment/model and imports utils
    functions internally, so it can be pickled and run in a separate process
    by joblib (no reliance on notebook/module globals).

    jac_mode:
      "fd"       -> scipy estimates the Jacobian by finite differences using
                    `diff_step` (enlarged to clear the solver noise floor).
      "analytic" -> Jacobian from PyBaMM sODE sensitivities (dV/dtheta), one
                    solve per iteration, no FD noise.
    """
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.interpolate import interp1d
    import pybamm

    pybamm.set_logging_level("ERROR")
    from utils import (
        make_base_params,
        prepare_sensitivity_inputs,
        make_model,
        run_model,
        make_experiment,
    )

    log_scaled = set(log_scaled_params)
    n_p = len(fit_params)
    experiment = make_experiment(profile_id=profile_id)
    want_sens = (jac_mode == "analytic")

    def theta_bar_to_dim(theta_bar):
        out = {}
        for z, p in zip(theta_bar, fit_params):
            lo, hi = param_bounds[p]
            z = float(np.clip(z, 0.0, 1.0))
            if p in log_scaled:
                out[p] = float(np.exp(np.log(lo) + z * (np.log(hi) - np.log(lo))))
            else:
                out[p] = float(lo + z * (hi - lo))
        return out

    def dtheta_ddz(theta_dim):
        # d(theta_dim)/d(theta_bar): chain-rule factor for the analytic Jacobian
        d = np.zeros(n_p)
        for k, p in enumerate(fit_params):
            lo, hi = param_bounds[p]
            if p in log_scaled:
                d[k] = theta_dim[p] * (np.log(hi) - np.log(lo))
            else:
                d[k] = (hi - lo)
        return d

    # single-slot cache so residual() and jac() at the same x share one solve
    cache = {"key": None, "Vq": None, "S": None, "ok": False, "theta_dim": None}

    def _solve(theta_bar):
        key = np.asarray(theta_bar).tobytes()
        if cache["key"] == key:
            return
        cache["key"] = key
        theta_dim = theta_bar_to_dim(theta_bar)
        cache["theta_dim"] = theta_dim
        try:
            base = make_base_params("Chen2020", soc=initial_soc, sensitivity_ready=True)
            p, iv = prepare_sensitivity_inputs(base, fit_params, values=theta_dim)
            model = make_model("SPMe", options=model_options)
            sol = run_model(
                model, p, experiment, inputs=iv,
                calculate_sensitivities=want_sens,
            )
            t = sol["Time [s]"].entries
            V = sol["Voltage [V]"].entries
            Vq = interp1d(t, V, bounds_error=False, fill_value="extrapolate")(t_data)

            S = None
            if want_sens:
                sens = sol["Voltage [V]"].sensitivities
                S = np.zeros((len(t_data), n_p))
                for k, name in enumerate(fit_params):
                    s_arr = np.asarray(sens[name]).reshape(-1)
                    S[:, k] = interp1d(
                        t, s_arr, bounds_error=False, fill_value="extrapolate"
                    )(t_data)

            cache.update(Vq=Vq, S=S, ok=bool(np.all(np.isfinite(Vq))))
        except Exception:
            cache.update(Vq=None, S=None, ok=False)

    def residual(theta_bar):
        _solve(theta_bar)
        if not cache["ok"]:
            return np.ones_like(V_data) * 1e6
        r = (cache["Vq"] - V_data) / sigma_V_fit
        if np.any(~np.isfinite(r)):
            return np.ones_like(V_data) * 1e6
        return r

    def jac(theta_bar):
        _solve(theta_bar)
        if not cache["ok"] or cache["S"] is None:
            return np.zeros((len(V_data), n_p))
        # dr/dz_k = (1/sigma) * dV/dtheta_dim_k * dtheta_dim_k/dz_k
        scale = dtheta_ddz(cache["theta_dim"])
        return (cache["S"] / sigma_V_fit) * scale[np.newaxis, :]

    # reproducible init: depends only on start_id, not on run order
    rng = np.random.default_rng(start_id)
    theta_bar_init = rng.uniform(0.0, 1.0, size=n_p)

    ls_kwargs = dict(
        bounds=(np.zeros(n_p), np.ones(n_p)),
        method="trf",
        max_nfev=max_nfev,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        verbose=0,
    )
    if jac_mode == "analytic":
        res = least_squares(residual, theta_bar_init, jac=jac, **ls_kwargs)
    else:
        res = least_squares(residual, theta_bar_init, diff_step=diff_step, **ls_kwargs)

    r_hat = residual(res.x)
    rmse_hat = float(np.sqrt(np.mean((r_hat * sigma_V_fit) ** 2)) * 1e3)

    return {
        "start_id": start_id,
        "success": bool(res.success),
        "cost": float(res.cost),
        "rmse_mV": rmse_hat,
        "jac_mode": jac_mode,
        "theta_init": theta_bar_to_dim(theta_bar_init),
        "theta_hat": theta_bar_to_dim(res.x),
        "theta_bar_init": np.asarray(theta_bar_init),
        "theta_bar_hat": np.asarray(res.x),
    }


# %% Initial residual check (serial sanity, at nominal)
log("Check initial residual at nominal")
t_fit0, V_fit0, _ = solve_voltage(theta_dim0, tag="NOMINAL-CHECK")
V_fit0_q = interp1d(
    t_fit0, V_fit0, bounds_error=False, fill_value="extrapolate"
)(t_data)
rmse0 = np.sqrt(np.mean((V_fit0_q - V_data) ** 2)) * 1e3
log(f"Initial nominal RMSE vs noisy data = {rmse0:.6f} mV (expect ~{NOISE_STD * 1e3:.2f})")


# %% Multi-start TRF (parallel)
##################################################
##################################################
##################################################
# n_jobs: number of parallel worker processes. Capped at #starts and #cores.
# Lower it if memory is tight (each worker holds its own PyBaMM model).
N_JOBS = min(N_STARTS, os.cpu_count() or 1)
N_JOBS = 12

log(f"Multistart: {N_STARTS} starts across {N_JOBS} parallel workers")
t_opt = time.time()

all_results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(run_one_start)(
        start_id,
        t_data,
        V_data,
        fit_params,
        PARAM_BOUNDS,
        log_scaled_params,
        initial_soc,
        MODEL_OPTIONS,
        sigma_V_fit,
        MAX_NFEV,
        PROFILE_ID,
        JAC_MODE,
        DIFF_STEP,
    )
    for start_id in range(N_STARTS)
)

# loky may return out of order -> sort by start_id for stable output
all_results.sort(key=lambda r: r["start_id"])
log(f"Multistart done: {N_STARTS} starts in {time.time() - t_opt:.1f} sec")

for result in all_results:
    log(
        f"START {result['start_id'] + 1} | success={result['success']} "
        f"| cost={result['cost']:.3e} | RMSE={result['rmse_mV']:.6f} mV"
    )


# %% Summary
print("\nInitial -> Final parameter ratios", flush=True)

for result in all_results:
    print("\n" + "-" * 80, flush=True)
    print(
        f"start={result['start_id']} "
        f"success={result['success']} "
        f"cost={result['cost']:.3e} "
        f"RMSE={result['rmse_mV']:.6f} mV",
        flush=True,
    )

    for p in fit_params:
        label = PARAM_LABELS_ASCII.get(p, p)
        init_ratio = result["theta_init"][p] / theta_values_nominal[p]
        final_ratio = result["theta_hat"][p] / theta_values_nominal[p]

        print(
            f"{label:10s} "
            f"init={init_ratio:9.3f} "
            f"final={final_ratio:9.3f}",
            flush=True,
        )

log("SCRIPT END")


# %% Init -> Final of survived (converged) starts
# "Survived" = fit succeeded (RMSE below threshold), i.e. not a failed/penalty start.
SURVIVE_RMSE = 10.0   # mV

survived = [r for r in all_results if r["rmse_mV"] < SURVIVE_RMSE]
print(f"\nSurvived starts: {len(survived)}/{len(all_results)} "
      f"(RMSE < {SURVIVE_RMSE} mV)", flush=True)

for r in survived:
    print("\n" + "-" * 70, flush=True)
    print(f"start={r['start_id']}  RMSE={r['rmse_mV']:.3f} mV", flush=True)
    for p in fit_params:
        label = PARAM_LABELS_ASCII.get(p, p)
        init_ratio = r["theta_init"][p] / theta_values_nominal[p]
        final_ratio = r["theta_hat"][p] / theta_values_nominal[p]
        print(
            f"    {label:10s} "
            f"init={init_ratio:9.3f}  final={final_ratio:9.3f}",
            flush=True,
        )


# %% Parameter bias readout: recovered theta_hat vs known theta_true
# theta_true = theta_values_nominal (the values used to GENERATE the truth data).
# ratio = theta_hat / theta_true.  ratio ~ 1.0  => recovered well.
# With USE_DISCREPANCY=True the clean (#1) fit cannot recover theta_true, so the
# deviation of the median ratio from 1.0 IS the discrepancy-induced bias, and a
# large spread across starts flags non-identifiability.
print(f"\nParameter bias over {len(survived)} survived starts "
      f"(discrepancies={DISCREPANCIES}):", flush=True)
print(f"{'param':10s}{'median ratio':>14s}{'spread(max-min)':>17s}"
      f"{'|median-1|':>12s}", flush=True)

bias_norm = 0.0
for p in fit_params:
    ratios = np.array(
        [r["theta_hat"][p] / theta_values_nominal[p] for r in survived]
    )
    label = PARAM_LABELS_ASCII.get(p, p)
    med = np.median(ratios)
    spread = ratios.max() - ratios.min()
    bias_norm += (med - 1.0) ** 2
    print(f"{label:10s}{med:14.3f}{spread:17.3f}{abs(med - 1.0):12.3f}",
          flush=True)

print(f"\noverall bias  ||median_ratio - 1||_2 = {np.sqrt(bias_norm):.3f}",
      flush=True)


# %%
import pickle
import time

fname = f"multistart_results_{time.strftime('%Y%m%d_%H%M%S')}.pkl"

with open(fname, "wb") as f:
    pickle.dump(all_results, f)

print("saved:", fname)




# %% Plot fitted voltage over nominal (valid starts only)
import matplotlib.pyplot as plt

# Starts whose solve failed land on the flat penalty -> astronomically high RMSE.
# Skip them so the plot stays on a sensible voltage scale.
RMSE_VALID_MAX = 10.0   # mV

plt.figure(figsize=(9, 4))

# nominal "truth" voltage
plt.plot(t_data, V_data_clean, "k--", lw=2, label="nominal (truth)", zorder=5)

for result in all_results:
    sid = result["start_id"]
    if result["rmse_mV"] > RMSE_VALID_MAX:
        log(f"skip start {sid} in plot (RMSE={result['rmse_mV']:.3g} mV)")
        continue

    t_fit, V_fit, _ = solve_voltage(result["theta_hat"], tag=f"PLOT start {sid}")
    plt.plot(
        t_fit, V_fit, alpha=0.7,
        label=f"start {sid} (RMSE={result['rmse_mV']:.3f} mV)",
    )

plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("Fitted voltage vs nominal (valid starts)")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()




# %% Plot fitted SOC_n and SOC_p over nominal (valid starts only)
import matplotlib.pyplot as plt

# concentration endpoints (0% / 100% SOC).
# SOC math is the same as main_soc_sensitivity.py:
#   SOC_n = (csn - csn_0)   / (csn_100 - csn_0)
#   SOC_p = (csp - csp_100) / (csp_0   - csp_100)
# IMPORTANT: build the endpoints from a FULL-SOC cell (no soc= override). Passing
# an already-partial-SOC param makes build_soc2theta discharge from 50%, so its
# "soc=1.0" anchors to the 50% state and SOC(0) wrongly reads 1.0.
base_for_endpoints = make_base_params("Chen2020", sensitivity_ready=True)
endpoints = get_soc_endpoints(base_for_endpoints)
csn_span = endpoints["csn_100"] - endpoints["csn_0"]
csp_span = endpoints["csp_0"] - endpoints["csp_100"]


def solve_soc(theta_dim, tag=""):
    """Functional-path solve returning t, SOC_n, SOC_p (avg-particle based)."""
    base = make_base_params("Chen2020", soc=initial_soc, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, fit_params, values=theta_dim)
    model = make_model("SPMe", options=MODEL_OPTIONS)
    sol = run_model(model, p, experiment, inputs=iv)

    t = sol["Time [s]"].entries
    csn = sol["Average negative particle concentration [mol.m-3]"].entries
    csp = sol["Average positive particle concentration [mol.m-3]"].entries

    soc_n = (csn - endpoints["csn_0"]) / csn_span
    soc_p = (csp - endpoints["csp_100"]) / csp_span
    return t, soc_n, soc_p


RMSE_VALID_MAX = 10.0   # mV; skip failed/penalty starts

# nominal trajectory
t_nom, socn_nom, socp_nom = solve_soc(theta_values_nominal, tag="nominal")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(t_nom, socn_nom, "k--", lw=2, label="nominal", zorder=5)
axes[1].plot(t_nom, socp_nom, "k--", lw=2, label="nominal", zorder=5)

for result in all_results:
    sid = result["start_id"]
    if result["rmse_mV"] > RMSE_VALID_MAX:
        log(f"skip start {sid} in SOC plot (RMSE={result['rmse_mV']:.3g} mV)")
        continue

    t_fit, socn, socp = solve_soc(result["theta_hat"], tag=f"start {sid}")
    axes[0].plot(t_fit, socn, alpha=0.7, label=f"start {sid}")
    axes[1].plot(t_fit, socp, alpha=0.7, label=f"start {sid}")

axes[0].set_title("Negative electrode SOC$_n$")
axes[1].set_title("Positive electrode SOC$_p$")
for ax, yl in zip(axes, ["SOC_n", "SOC_p"]):
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(yl)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()




# %%
print("initial_soc =", initial_soc)
print("SOC_n(0) =", socn_nom[0], "  SOC_p(0) =", socp_nom[0])



# %% BEFORE vs AFTER fitting: residual decomposition (parallel / orthogonal)
# Two residuals against the same data V_data (= discrepant truth + 1 mV noise,
# built in the TRUTH cell above):
#   r_before = V_data - V_spme(theta_nominal)   clean model BEFORE fitting
#   r_after  = V_data - V_spme(theta_hat)       clean model AFTER  fitting
# Each is split along its OWN local sensitivity directions (S at that theta):
#   r_parallel   = part parameters COULD remove   (in col(S))
#   r_orthogonal = part parameters CANNOT remove  (discrepancy)
# Expectation: fitting removes the parallel part -> r_after_parallel ~ 0, and
# r_after_orthogonal ~ the structural discrepancy that biasing could not fix.
import pickle, glob, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# auto-pick the most recent NON-EMPTY multistart .pkl (skips failed/0-byte saves).
# Override with an explicit filename if you want a specific run.
_pkls = [f for f in glob.glob("multistart_results_*.pkl") if os.path.getsize(f) > 0]
RESULTS_FILE = max(_pkls, key=os.path.getmtime)
print("using:", RESULTS_FILE)
with open(RESULTS_FILE, "rb") as f:
    all_results = pickle.load(f)
SURVIVE_RMSE = 10.0
survived = [r_ for r_ in all_results if r_["rmse_mV"] < SURVIVE_RMSE]
best = min(survived, key=lambda rr: rr["rmse_mV"])
theta_hat = best["theta_hat"]
print(f"loaded {len(all_results)} starts; best survived start={best['start_id']} "
      f"(RMSE={best['rmse_mV']:.3f} mV)")

labels = [PARAM_LABELS_ASCII.get(p_, p_) for p_ in fit_params]


def clean_V_and_S(theta_dim):
    """Clean SPMe at theta_dim -> (V on t_data, relative S = theta*dV/dtheta)."""
    base = make_base_params("Chen2020", soc=initial_soc, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, fit_params, values=theta_dim)
    model = make_model("SPMe", options=MODEL_OPTIONS)
    sol = run_model(model, p, experiment, inputs=iv, calculate_sensitivities=True)
    tt = sol["Time [s]"].entries
    Vq = interp1d(tt, sol["Voltage [V]"].entries,
                  bounds_error=False, fill_value="extrapolate")(t_data)
    S = np.column_stack([
        interp1d(tt, np.asarray(sol["Voltage [V]"].sensitivities[name]).reshape(-1),
                 bounds_error=False, fill_value="extrapolate")(t_data)
        for name in fit_params
    ])
    S = S * np.array([theta_dim[name] for name in fit_params])[None, :]  # relative
    return Vq, S


# BEFORE: clean model at theta_nominal (= theta_true).  S evaluated at theta_nominal.
V_before, S_before = clean_V_and_S(theta_values_nominal)
r_before = V_data - V_before

# AFTER: clean model at theta_hat.  S evaluated at theta_hat.
V_after, S_after = clean_V_and_S(theta_hat)
r_after = V_data - V_after

# Three decompositions:
#  1) BEFORE          : r_before  split along S(theta_nominal)
#  2) AFTER (own)     : r_after   split along S(theta_hat)   -> r_par ~ 0 (optimality)
#  3) AFTER (nominal) : r_after   split along S(theta_nominal) -> SAME frame as (1),
#     so r_before vs this is apples-to-apples (direction fixed); and comparing it
#     to (2) shows how much the sensitivity directions rotated nominal -> theta_hat.
print("\n--- BEFORE fitting (model & S at theta_nominal) ---")
dec_before = residual_sensitivity_decomposition(
    r_before, S_before, sigma=sigma_V_fit, t=t_data, labels=labels, plot=False)
print("\n--- AFTER fitting (model & S at theta_hat) ---")
dec_after = residual_sensitivity_decomposition(
    r_after, S_after, sigma=sigma_V_fit, t=t_data, labels=labels, plot=False)
print("\n--- AFTER fitting, S at theta_nominal (fixed/nominal frame) ---")
dec_after_nom = residual_sensitivity_decomposition(
    r_after, S_before, sigma=sigma_V_fit, t=t_data, labels=labels, plot=False)

# del_r = r_before - r_after = V(theta_hat) - V(theta_nominal): the voltage change
# the fit produced. Decomposed at S(theta_nominal): parallel = linear part of the
# move, orthogonal = nonlinear leftover (linearity check of the parameter move).
del_r = r_before - r_after
print("\n--- del_r = r_before - r_after, S at theta_nominal (linearity of the move) ---")
dec_delr = residual_sensitivity_decomposition(
    del_r, S_before, sigma=sigma_V_fit, t=t_data, labels=labels, plot=False)

print("\n===================== RMS [mV] =====================")
print(f"{'':26s}{'RMS r':>10s}{'RMS r_par':>12s}{'RMS r_perp':>12s}")
rows = [
    ("before (S_nom)", dec_before),
    ("after  (S_hat)", dec_after),
    ("after  (S_nom)", dec_after_nom),
    ("del_r  (S_nom)", dec_delr),
]
for nm, d in rows:
    print(f"{nm:26s}{d['rms_r']*1e3:10.3f}{d['rms_parallel']*1e3:12.3f}"
          f"{d['rms_perp']*1e3:12.3f}")

# plot: 4 panels, same form
fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
panels = [
    (axes[0], "BEFORE  (S @ theta_nominal)", r_before, dec_before),
    (axes[1], "AFTER   (S @ theta_hat)", r_after, dec_after),
    (axes[2], "AFTER   (S @ theta_nominal, fixed frame)", r_after, dec_after_nom),
    (axes[3], "del_r = before - after  (S @ theta_nominal)", del_r, dec_delr),
]
for ax, name, rr, d in panels:
    ax.plot(t_data, rr * 1e3, label="residual", lw=1.0)
    ax.plot(t_data, d["r_parallel"] * 1e3, label="parallel (reducible)", lw=1.0)
    ax.plot(t_data, d["r_perp"] * 1e3, label="orthogonal (discrepancy)", lw=1.2)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel("mV")
    ax.set_title(f"{name}:  RMS r_par={d['rms_parallel']*1e3:.2f} mV,  "
                 f"r_perp={d['rms_perp']*1e3:.2f} mV")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4)
axes[-1].set_xlabel("Time [s]")
fig.suptitle("Residual decomposition: before / after (S_hat) / after (S_nom) / del_r")
plt.tight_layout()
plt.show()


# %% Pick theta_effective: 10 best HPPC fits (parameter ratios + Vrmse)
# Inspect the 10 lowest-RMSE fits from the HPPC discrepancy run: which params
# converge to the SAME value (tight cluster -> identifiable, trustworthy
# theta_effective) vs which scatter. Use this to choose theta_effective for the
# WLTP generalization test below.
import pickle, glob, os
import numpy as np
import matplotlib.pyplot as plt

# Which fit to inspect. Pick ONE:
PKL = "multistart_results_20260623_214201_soc65.pkl"   # CLEAN fit (no discrepancy)
# PKL = "multistart_results_20260625_124741.pkl"       # DISCREPANCY fit (for the WLTP test)
# _p = [f for f in glob.glob("multistart_results_*.pkl") if os.path.getsize(f) > 0]
# PKL = max(_p, key=os.path.getmtime)                  # or: latest non-empty

with open(PKL, "rb") as f:
    _res = pickle.load(f)
print("using:", PKL)

top10 = sorted(_res, key=lambda r: r["rmse_mV"])[:10]
labels = [PARAM_LABELS_ASCII.get(p_, p_) for p_ in fit_params]

# table: ratio = theta_hat / theta_nominal
print("\n10 best theta_effective  (ratio = theta_hat / theta_nominal)")
print(f"{'rank':>4s}{'start':>6s}{'RMSE_mV':>10s}   "
      + "".join(f"{lab:>9s}" for lab in labels))
for rank, r in enumerate(top10):
    ratios = [r["theta_hat"][p_] / theta_values_nominal[p_] for p_ in fit_params]
    print(f"{rank:>4d}{r['start_id']:>6d}{r['rmse_mV']:>10.3f}   "
          + "".join(f"{x:>9.3f}" for x in ratios))

# plot: parameter ratios of the 10 best (cluster vs scatter)
plt.figure(figsize=(9, 5))
x = np.arange(len(fit_params))
for rank, r in enumerate(top10):
    ratios = [r["theta_hat"][p_] / theta_values_nominal[p_] for p_ in fit_params]
    is_best = rank == 0
    plt.plot(x, ratios, "o-",
             color="red" if is_best else "gray",
             alpha=0.95 if is_best else 0.4,
             lw=2.0 if is_best else 0.8,
             label="best (rank 0)" if is_best else None)
plt.axhline(1.0, color="k", ls="--", lw=0.8, label="nominal (= 1)")
plt.yscale("log")
plt.xticks(x, labels)
plt.ylabel("theta_hat / theta_nominal")
plt.title("10 best HPPC fits: parameter ratios (tight = identifiable)")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()


# %% Discrepancy SOC sweep: Dsn + Dsp ONLY (no OCP) -> theta_eff at each SOC
# Truth = clean SPMe + concentration-dependent Ds (anode + cathode) at theta_nominal,
# on HPPC, at each SOC. Fit the CLEAN model -> biased theta_eff. Save a pkl per SOC.
import pickle, time
import numpy as np

DISC = {"Dsn", "Dsp"}                    # diffusivity discrepancies only (NO ocp)
SOC_SWEEP = [0.85, 0.65, 0.50, 0.30]
HPPC = make_experiment(profile_id="hppc")
DSN = "Negative particle diffusivity [m2.s-1]"
DSP = "Positive particle diffusivity [m2.s-1]"
N_JOBS = 12


def discrepant_truth(soc):
    """Clean SPMe + Dsn/Dsp discrepancy at theta_nominal, given SOC -> t, V(+noise)."""
    base = make_base_params("Chen2020", soc=soc, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, fit_params, values=theta_values_nominal)
    if "Dsn" in DISC:
        p[DSN] = Dsn_discrepancy
        iv.pop(DSN, None)
    if "Dsp" in DISC:
        p[DSP] = Dsp_discrepancy
        iv.pop(DSP, None)
    sol = run_model(make_model("SPMe", options=MODEL_OPTIONS), p, HPPC, inputs=iv)
    t = sol["Time [s]"].entries
    V = sol["Voltage [V]"].entries.copy()
    if ADD_NOISE:
        V = V + np.random.default_rng(0).normal(0.0, NOISE_STD, size=V.shape)
    return t, V


disc_results = {}
for soc in SOC_SWEEP:
    log(f"======== discrepancy fit (Dsn+Dsp, no ocp) at SOC {soc} ========")
    t_soc, V_soc = discrepant_truth(soc)

    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(run_one_start)(
            sid, t_soc, V_soc, fit_params, PARAM_BOUNDS, log_scaled_params,
            soc, MODEL_OPTIONS, sigma_V_fit, MAX_NFEV, "hppc", JAC_MODE, DIFF_STEP)
        for sid in range(N_STARTS)
    )
    results.sort(key=lambda r: r["start_id"])
    disc_results[soc] = results

    fname = f"disc_dsnp_hppc_soc{int(soc * 100)}_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(results, f)
    log(f"  saved {fname}")

    # theta_eff = median over survived (biased away from 1.0 by the Ds discrepancy)
    survived = [r for r in results if r["rmse_mV"] < 10.0]
    print(f"  theta_eff median over {len(survived)} survived "
          f"(theta_hat/theta_nominal):")
    for p_ in fit_params:
        ratios = np.array([r["theta_hat"][p_] / theta_values_nominal[p_]
                           for r in survived])
        print(f"    {PARAM_LABELS_ASCII.get(p_, p_):8s} {np.median(ratios):8.3f}")


# %% theta_eff across SOC (Dsn+Dsp discrepancy): table + plot from the 4 pkls
# Loads the disc_dsnp_hppc_soc*.pkl files and compares the biased theta_eff vs SOC.
# A parameter whose ratio SWINGS with SOC is absorbing the (concentration-dependent)
# discrepancy -> the cross-condition inconsistency that flags model error.
import pickle, glob, re
import numpy as np
import matplotlib.pyplot as plt

_files = {}
for f in glob.glob("disc_dsnp_hppc_soc*.pkl"):
    m = re.search(r"soc(\d+)", f)
    if m:
        _files[int(m.group(1))] = f
socs = sorted(_files)                       # [30, 50, 65, 85]
labels = [PARAM_LABELS_ASCII.get(p_, p_) for p_ in fit_params]
SURVIVE_RMSE = 10.0

med = {}
for soc in socs:
    with open(_files[soc], "rb") as fp:
        res = pickle.load(fp)
    surv = [r for r in res if r["rmse_mV"] < SURVIVE_RMSE]
    med[soc] = np.array([
        np.median([r["theta_hat"][p_] / theta_values_nominal[p_] for r in surv])
        for p_ in fit_params
    ])

print("theta_eff median (theta_hat / theta_nominal), Dsn+Dsp discrepancy (no OCP)")
print("param   " + "".join(f"{s:>9d}%" for s in socs))
for i, lab in enumerate(labels):
    print(f"{lab:8s}" + "".join(f"{med[s][i]:>9.3f} " for s in socs))

plt.figure(figsize=(8, 5))
for i, lab in enumerate(labels):
    plt.plot(socs, [med[s][i] for s in socs], "o-", label=lab)
plt.axhline(1.0, color="k", ls="--", lw=0.8, label="nominal (=1)")
plt.yscale("log")
plt.xlabel("SOC [%]")
plt.ylabel("theta_eff / theta_nominal")
plt.title("theta_eff bias vs SOC  (Dsn+Dsp discrepancy, no OCP)")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()


# %% Voltage profiles per SOC (Dsn+Dsp): discrepant truth vs fitted theta_eff
# Per SOC: solve the discrepant truth (SPMe + Dsn/Dsp at theta_nominal, dashed)
# and the CLEAN model at every survived theta_eff (gray) on HPPC.
import pickle, glob, re
import matplotlib.pyplot as plt

_files = {}
for f in glob.glob("disc_dsnp_hppc_soc*.pkl"):
    m = re.search(r"soc(\d+)", f)
    if m:
        _files[int(m.group(1))] = f

HPPC = make_experiment(profile_id="hppc")
DSN = "Negative particle diffusivity [m2.s-1]"
DSP = "Positive particle diffusivity [m2.s-1]"
SOC_LAYOUT = [[85, 65], [50, 30]]
SURVIVE_RMSE = 10.0


def _solve(theta, soc, discrepant):
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, fit_params, values=theta)
    if discrepant:
        p[DSN] = Dsn_discrepancy; iv.pop(DSN, None)
        p[DSP] = Dsp_discrepancy; iv.pop(DSP, None)
    sol = run_model(make_model("SPMe", options=MODEL_OPTIONS), p, HPPC, inputs=iv)
    return sol["Time [s]"].entries, sol["Voltage [V]"].entries


fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for i in range(2):
    for j in range(2):
        soc = SOC_LAYOUT[i][j]
        ax = axes[i, j]
        with open(_files[soc], "rb") as fp:
            res = pickle.load(fp)
        surv = [r for r in res if r["rmse_mV"] < SURVIVE_RMSE]

        for r in surv:                                   # fitted theta_eff (clean)
            tf, Vf = _solve(r["theta_hat"], soc, discrepant=False)
            ax.plot(tf, Vf, color="gray", alpha=0.4, lw=0.8)

        tt, Vt = _solve(theta_values_nominal, soc, discrepant=True)   # truth
        ax.plot(tt, Vt, "k--", lw=2, label="truth (SPMe+Dsn+Dsp)", zorder=5)

        ax.set_title(f"SOC {soc}%  ({len(surv)} fits)")
        ax.grid(alpha=0.4)
        ax.legend(fontsize=8)
        if i == 1:
            ax.set_xlabel("Time [s]")
        if j == 0:
            ax.set_ylabel("Voltage [V]")

fig.suptitle("Voltage: discrepant truth vs fitted theta_eff (Dsn+Dsp, HPPC)")
plt.tight_layout()
plt.show()


# %% Dsn+Dsp: top-10 params + V RMSE, and residual decomposition per SOC
# Per SOC: print the 10 lowest-RMSE fits (ratios + RMSE), then at the BEST theta_eff
# split the residual r = truth - SPMe(theta_eff) into parallel (parameter-reducible)
# and orthogonal (discrepancy) parts. r_perp >> noise = structural error parameters
# cannot remove.
import pickle, glob, re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

_files = {}
for f in glob.glob("disc_dsnp_hppc_soc*.pkl"):
    m = re.search(r"soc(\d+)", f)
    if m:
        _files[int(m.group(1))] = f

HPPC = make_experiment(profile_id="hppc")
DSN = "Negative particle diffusivity [m2.s-1]"
DSP = "Positive particle diffusivity [m2.s-1]"
SOC_LAYOUT = [[85, 65], [50, 30]]
labels = [PARAM_LABELS_ASCII.get(p_, p_) for p_ in fit_params]


def _truth_V(soc):
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, fit_params, values=theta_values_nominal)
    p[DSN] = Dsn_discrepancy; iv.pop(DSN, None)
    p[DSP] = Dsp_discrepancy; iv.pop(DSP, None)
    sol = run_model(make_model("SPMe", options=MODEL_OPTIONS), p, HPPC, inputs=iv)
    t = sol["Time [s]"].entries
    V = sol["Voltage [V]"].entries.copy()
    V = V + np.random.default_rng(0).normal(0.0, NOISE_STD, size=V.shape)
    return t, V


def _clean_V_and_S(theta, soc, t_data):
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, fit_params, values=theta)
    sol = run_model(make_model("SPMe", options=MODEL_OPTIONS), p, HPPC,
                    inputs=iv, calculate_sensitivities=True)
    tt = sol["Time [s]"].entries
    Vq = interp1d(tt, sol["Voltage [V]"].entries,
                  bounds_error=False, fill_value="extrapolate")(t_data)
    S = np.column_stack([
        interp1d(tt, np.asarray(sol["Voltage [V]"].sensitivities[n]).reshape(-1),
                 bounds_error=False, fill_value="extrapolate")(t_data)
        for n in fit_params
    ])
    S = S * np.array([theta[n] for n in fit_params])[None, :]   # relative
    return Vq, S


fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for i in range(2):
    for j in range(2):
        soc = SOC_LAYOUT[i][j]
        ax = axes[i, j]
        with open(_files[soc], "rb") as fp:
            res = pickle.load(fp)
        top10 = sorted(res, key=lambda r: r["rmse_mV"])[:10]

        print(f"\n================ SOC {soc}% : top 10 (ratio theta_hat/theta_nom) "
              f"================")
        print(f"{'rank':>4}{'RMSE_mV':>9}  " + "".join(f"{l:>8}" for l in labels))
        for rank, r in enumerate(top10):
            rr = [r["theta_hat"][p_] / theta_values_nominal[p_] for p_ in fit_params]
            print(f"{rank:>4}{r['rmse_mV']:>9.3f}  " + "".join(f"{x:>8.3f}" for x in rr))

        # residual decomposition at the best theta_eff
        theta_eff = top10[0]["theta_hat"]
        t_d, V_d = _truth_V(soc)
        Vq, S = _clean_V_and_S(theta_eff, soc, t_d)
        r = V_d - Vq
        d = residual_sensitivity_decomposition(
            r, S, sigma=sigma_V_fit, t=t_d, labels=labels, plot=False)
        print(f"  >> decomposition at best: RMS r={d['rms_r']*1e3:.2f}, "
              f"r_par={d['rms_parallel']*1e3:.2f}, r_perp={d['rms_perp']*1e3:.2f} mV")

        ax.plot(t_d, r * 1e3, label="residual", lw=1.0)
        ax.plot(t_d, d["r_parallel"] * 1e3, label="parallel (param)", lw=1.0)
        ax.plot(t_d, d["r_perp"] * 1e3, label="orthogonal (discrepancy)", lw=1.2)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"SOC {soc}%:  r_par={d['rms_parallel']*1e3:.1f}, "
                     f"r_perp={d['rms_perp']*1e3:.1f} mV")
        ax.grid(alpha=0.4)
        ax.legend(fontsize=7)
        if i == 1:
            ax.set_xlabel("Time [s]")
        if j == 0:
            ax.set_ylabel("mV")

fig.suptitle("Dsn+Dsp: residual decomposition at best theta_eff, per SOC")
plt.tight_layout()
plt.show()

# %%
