# %% Imports
import os
# Cap math-library threads to 1 PER PROCESS *before* numpy/pybamm import.
# Each joblib worker runs its own PyBaMM solve; without this, every worker
# spawns a full BLAS/OpenMP thread pool -> 12 workers x many threads badly
# oversubscribe the CPU and a 1s solve crawls to 10-20s. loky workers inherit
# this environment, so the cap reaches them too.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

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
N_STARTS = 40          # 여기 바꾸면 됨
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

# main_sim.py = CLEAN REFERENCE (no model discrepancy): inverse-crime sanity
# (the fit should recover theta_true, ratios ~ 1) + identifiability of the fit
# params (sensitivity S spectrum / rank). The model-DISCREPANCY study lives in
# main_compare.py.

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

    "Electrolyte conductivity [S.m-1]": (0.1, 3.0),
    "Electrolyte diffusivity [m2.s-1]": (1e-12, 1e-9),

    # "Positive electrode porosity": (0.1, 0.7),
    # "Negative electrode porosity": (0.1, 0.7),
    # "Separator porosity": (0.1, 0.7),

    "Negative particle diffusivity [m2.s-1]": (5e-16, 1e-12),
    "Positive particle diffusivity [m2.s-1]": (5e-16, 1e-12),

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


def solve_voltage(theta_dim, tag):
    t_start = time.time()

    # Functional setup (same as main_soc_sensitivity.py):
    # j0 stays a function of concentration; the theta_dim values for the two
    # exchange-current-density params enter as the rate prefactor m_ref,
    # NOT as a flat j0 scalar. All fit_params are passed as PyBaMM inputs.
    log(f"{tag}: build parameters")
    base = make_base_params("Chen2020", soc=initial_soc, sensitivity_ready=True)
    p, input_values = prepare_sensitivity_inputs(base, fit_params, values=theta_dim)

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

log("Build nominal synthetic voltage data (clean model at theta_true)")
t_data, V_data_clean, I_data = solve_voltage(theta_dim0, tag="NOMINAL")
log(f"Nominal data ready: len={len(t_data)}, t_end={t_data[-1]:.2f}")

V_data = V_data_clean.copy()

if ADD_NOISE:
    rng_noise = np.random.default_rng(0)
    V_data = V_data + rng_noise.normal(0.0, NOISE_STD, size=V_data.shape)
    log(f"Added Gaussian voltage noise: std={NOISE_STD * 1e3:.2f} mV")


# %% Plot nominal voltage and current
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plot_results(
    {"t": t_data, "V": V_data_clean, "I": I_data},
    kind="VIT",
    suptitle=f"Nominal {PROFILE_ID} (clean model at theta_true)",
)


# %% Worker: one local-optimization start (self-contained for parallel use)
import os
from joblib import Parallel, delayed

# The rate sweep runs each start in its OWN subprocess with a per-start timeout
# (so a stalled IDAKLU solve can't hang the sweep) via run_multistart_subprocess.
# That pool executes the CANONICAL copy of run_one_start in rate_fit_worker.py.
# The inline run_one_start below is kept only for the legacy in-kernel joblib
# cell ("Multi-start TRF (parallel)"). Edit rate_fit_worker.py for sweep changes.
from pathlib import Path as _Path
if str(_Path.cwd()) not in sys.path:          # rate_fit_worker.py sits next to this file
    sys.path.append(str(_Path.cwd()))
from rate_fit_worker import run_multistart_subprocess


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

    # Build model/params/Simulation ONCE; only the InputParameter VALUES change
    # per solve. The OLD code rebuilt make_model + Simulation on every residual
    # (hundreds per start), leaking ~18 MB/solve of C-level memory -> workers
    # grew unbounded, the OS killed one, and joblib then deadlocked (the "stuck"
    # hang). The param STRUCTURE from prepare_sensitivity_inputs is
    # theta-independent, so reusing it is exact. ~100x fewer allocations + flat RSS.
    _base = make_base_params("Chen2020", soc=initial_soc, sensitivity_ready=True)
    _p_struct, _ = prepare_sensitivity_inputs(
        _base, fit_params, values=theta_bar_to_dim(np.full(n_p, 0.5))
    )
    _sim = pybamm.Simulation(
        make_model("SPMe", options=model_options),
        experiment=experiment,
        parameter_values=_p_struct,
        solver=pybamm.IDAKLUSolver(options={"max_num_steps": 50000}),
    )

    def _solve(theta_bar):
        key = np.asarray(theta_bar).tobytes()
        if cache["key"] == key:
            return
        cache["key"] = key
        theta_dim = theta_bar_to_dim(theta_bar)
        cache["theta_dim"] = theta_dim
        try:
            iv = {name: theta_dim[name] for name in fit_params}
            sol = _sim.solve(inputs=iv, calculate_sensitivities=want_sens)
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


# %% Recovery check: recovered theta_hat vs known theta_true
# theta_true = theta_values_nominal (the values used to GENERATE the data).
# ratio = theta_hat / theta_true.  Since this is the CLEAN model (no discrepancy,
# inverse crime), every identifiable parameter should recover ratio ~ 1.0.
# A median far from 1.0 or a large spread flags a NON-identifiable parameter.
print(f"\nParameter recovery over {len(survived)} survived starts "
      f"(clean model, ratios should be ~1.0):", flush=True)
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





# %% Load 4 SOC runs and plot truth vs identified-parameter voltages (2x2)
# Layout:  top row 85, 65 ;  bottom row 50, 30  (percent SOC).
# For each SOC: solve the clean model at theta_true (truth, dashed) and at every
# survived theta_hat from that run (gray) on the CURRENT experiment.
##########################################################
###################### Plot Results ######################
##########################################################
import pickle
import matplotlib.pyplot as plt

SOC_FILES = {
    85: "multistart_results_20260623_215819_soc85.pkl",
    65: "multistart_results_20260623_214201_soc65.pkl",
    50: "multistart_results_20260623_071649_soc50.pkl",
    30: "multistart_results_20260623_163927_soc30.pkl",
}
SOC_LAYOUT = [[85, 65], [50, 30]]   # subplot grid
SURVIVE_RMSE = 6                 # mV; skip failed/penalty starts


def solve_V_at_soc(theta_dim, soc_frac):
    """Clean SPMe voltage at a given initial SOC and parameter set."""
    base = make_base_params("Chen2020", soc=soc_frac, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, fit_params, values=theta_dim)
    model = make_model("SPMe", options=MODEL_OPTIONS)
    sol = run_model(model, p, experiment, inputs=iv)
    return sol["Time [s]"].entries, sol["Voltage [V]"].entries


fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for i in range(2):
    for j in range(2):
        soc = SOC_LAYOUT[i][j]
        ax = axes[i, j]

        with open(SOC_FILES[soc], "rb") as fp:
            results = pickle.load(fp)
        survived = [r for r in results if r["rmse_mV"] < SURVIVE_RMSE]

        # identified-parameter voltages (one gray curve per survived start)
        for r in survived:
            t_f, V_f = solve_V_at_soc(r["theta_hat"], soc / 100.0)
            ax.plot(t_f, V_f, color="gray", alpha=0.4, lw=0.8)

        # truth (clean model at theta_true), dashed black on top
        t_truth, V_truth = solve_V_at_soc(theta_values_nominal, soc / 100.0)
        ax.plot(t_truth, V_truth, "k--", lw=2, label="truth", zorder=5)

        ax.set_title(f"SOC {soc}%  ({len(survived)} fits)")
        ax.grid(alpha=0.4)
        ax.legend(fontsize=8)
        if i == 1:
            ax.set_xlabel("Time [s]")
        if j == 0:
            ax.set_ylabel("Voltage [V]")

fig.suptitle(f"Truth vs identified-parameter voltages across SOC  ({PROFILE_ID})")
plt.tight_layout()
plt.show()


# %% Identifiability test of the 6 fit params (FIM / CRLB at theta_true)
# Uses existing utils tools: get_sensitivities (relative dV/dlntheta) + compute_fim.
#   eigenvalues of FIM  -> how many parameter directions are observable
#   cond(FIM)           -> overall ill-conditioning (= cond(S)^2)
#   CRLB = sqrt(diag(F^-1)) -> best achievable RELATIVE std per parameter
import numpy as np
import matplotlib.pyplot as plt

# clean solve at theta_true WITH analytic sensitivities
base = make_base_params("Chen2020", soc=initial_soc, sensitivity_ready=True)
p_id, iv_id = prepare_sensitivity_inputs(base, fit_params, values=theta_values_nominal)
model_id = make_model("SPMe", options=MODEL_OPTIONS)
sol_id = run_model(model_id, p_id, experiment, inputs=iv_id, calculate_sensitivities=True)

t_q = get_t_query(sol_id)
# relative sensitivities: theta * dV/dtheta  (dimensionless, comparable columns)
Sens = get_sensitivities(
    sol_id, "Voltage [V]", fit_params, t_q,
    theta_values=theta_values_nominal, normalize=True,
)

# Fisher Information Matrix (sigma = measurement noise std)
F = compute_fim(Sens, fit_params, sigma=sigma_V_fit)

labels = [PARAM_LABELS_ASCII.get(p_, p_) for p_ in fit_params]
evals = np.sort(np.linalg.eigvalsh(F))[::-1]          # descending
cond = evals[0] / evals[-1]

print(f"\nIdentifiability of {len(fit_params)} params  "
      f"(profile={PROFILE_ID}, SOC={initial_soc})")
print("FIM eigenvalues (normalized to max):")
print(np.array2string(evals / evals[0], precision=2))
print(f"cond(FIM) = {cond:.2e}   (= cond(S)^2; large => some directions weak)")

# CRLB: relative std per parameter (lower = better identified)
crlb = np.sqrt(np.diag(np.linalg.pinv(F)))
print("\nCRLB relative std per parameter (lower = better identified):")
for lab, c in sorted(zip(labels, crlb), key=lambda x: x[1]):
    print(f"  {lab:8s} {c:10.3e}")

# eigen-spectrum plot
plt.figure(figsize=(7, 4))
plt.semilogy(range(1, len(evals) + 1), evals / evals[0], "o-")
plt.xlabel("eigenvalue index")
plt.ylabel(r"$\lambda_i / \lambda_{max}$")
plt.title(f"FIM eigen-spectrum — identifiability of {len(fit_params)} params")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()


# %% Identifiability across the 4 SOCs (FIM / CRLB at theta_true, each SOC)
# Same metric as the cell above, swept over SOC -> see which params are better
# identified at which SOC (e.g. does Ds- improve away from 85%?).
import numpy as np
import matplotlib.pyplot as plt

SOC_LIST = [0.30, 0.50, 0.65, 0.85]
id_labels = [PARAM_LABELS_ASCII.get(p_, p_) for p_ in fit_params]


def identifiability_at_soc(soc):
    base = make_base_params("Chen2020", soc=soc, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, fit_params, values=theta_values_nominal)
    model = make_model("SPMe", options=MODEL_OPTIONS)
    sol = run_model(model, p, experiment, inputs=iv, calculate_sensitivities=True)
    t_q = get_t_query(sol)
    Sens = get_sensitivities(
        sol, "Voltage [V]", fit_params, t_q,
        theta_values=theta_values_nominal, normalize=True,
    )
    F = compute_fim(Sens, fit_params, sigma=sigma_V_fit)
    crlb = np.sqrt(np.diag(np.linalg.pinv(F))) * 100.0   # relative std [%]
    return crlb, float(np.linalg.cond(F))


crlb_by_soc = {}
cond_by_soc = {}
for soc in SOC_LIST:
    crlb_by_soc[soc], cond_by_soc[soc] = identifiability_at_soc(soc)

# table: rows = params, cols = SOC
print(f"\nCRLB relative std [%] per parameter vs SOC  (profile={PROFILE_ID})")
print("param   " + "".join(f"{int(s * 100):>9d}%" for s in SOC_LIST))
for i, lab in enumerate(id_labels):
    print(f"{lab:8s}" + "".join(f"{crlb_by_soc[s][i]:>9.2f} " for s in SOC_LIST))
print("cond(FIM)" + "".join(f"{cond_by_soc[s]:>9.1e} " for s in SOC_LIST))

# plot CRLB vs SOC per parameter
plt.figure(figsize=(8, 5))
for i, lab in enumerate(id_labels):
    plt.plot([s * 100 for s in SOC_LIST],
             [crlb_by_soc[s][i] for s in SOC_LIST], "o-", label=lab)
plt.yscale("log")
plt.xlabel("SOC [%]")
plt.ylabel("CRLB relative std [%]  (lower = better identified)")
plt.title(f"Parameter identifiability vs SOC ({PROFILE_ID})")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# %% Init -> Final parameter ratios per SOC (from the saved .pkl runs)
# ratio = theta / theta_nominal.  init = random start, final = recovered fit.
# Printed for the survived starts of each SOC run.
import pickle

SOC_FILES = {
    30: "multistart_results_20260623_163927_soc30.pkl",
    50: "multistart_results_20260623_071649_soc50.pkl",
    65: "multistart_results_20260623_214201_soc65.pkl",
    85: "multistart_results_20260623_215819_soc85.pkl",
}
SURVIVE_RMSE = 10.0   # mV

for soc in [30, 50, 65, 85]:
    with open(SOC_FILES[soc], "rb") as f:
        results = pickle.load(f)
    survived = [r for r in results if r["rmse_mV"] < SURVIVE_RMSE]

    print("\n" + "#" * 72)
    print(f"#  SOC {soc}%   ({len(survived)}/{len(results)} survived, "
          f"RMSE < {SURVIVE_RMSE} mV)")
    print("#" * 72)

    for r in survived:
        print(f"\nstart={r['start_id']}  RMSE={r['rmse_mV']:.3f} mV")
        for p_ in r["theta_hat"]:           # use the run's own params (robust)
            label = PARAM_LABELS_ASCII.get(p_, p_)
            init_ratio = r["theta_init"][p_] / theta_values_nominal[p_]
            final_ratio = r["theta_hat"][p_] / theta_values_nominal[p_]
            print(f"    {label:8s} init={init_ratio:9.3f}  final={final_ratio:9.3f}")


# %% Rate x SOC sweep: fit on each discharge profile at each SOC, save pkl
# For each (profile, SOC): build truth at theta_nominal + noise (+ discrepancy if
# DISC set), run a 30-start fit, save a pkl, print median recovery.
# -> identifiability vs C-rate and SOC. 16 fits total (short profiles).
import pickle, os

RATE_PROFILES = ["hppc", "C3_dchg", "1C_dchg", "2C_dchg", "5C_dchg", "sin_1Hz"]
# Per-profile starting SOCs. HPPC walks across SOC via its pulse train, so it
# uses the classic HPPC levels; the partial discharges use their own window.
# sin_1Hz is a tiny (10 s) symmetric ripple -> negligible SOC drift, so it uses
# the HPPC SOC levels (same operating points as HPPC).
SOC_SWEEP = [0.90, 0.70, 0.55, 0.40]          # default (discharge profiles)
SOC_BY_PROFILE = {"hppc": [0.85, 0.65, 0.50, 0.30],
                  "sin_1Hz": [0.85, 0.65, 0.50, 0.30]}   # overrides per profile_id
# 6 workers (= physical P-cores on the i7-13800H). With per-worker threads
# capped to 1 (see Imports), this avoids the CPU/memory oversubscription that
# made one case take 2+ hours. Each worker holds ~1 GB, so 6 is also RAM-safe.
N_JOBS = 12

# Per-start wall-clock timeout. IDAKLU can stall for minutes on isolated
# interior parameter combinations (resonance bands no bound can exclude). Each
# start runs in its own subprocess; one exceeding this is killed and recorded as
# a penalty (filtered out by the survivor threshold), so the sweep never hangs.
# Per-profile because HPPC is ~3x longer than the discharges (single solve ~1.2s
# vs ~0.4s) so a legit full-budget HPPC start can take minutes. These bound true
# (infinite) stalls only; legit starts finish well under them.
START_TIMEOUT_S = 240                         # default (discharge profiles)
PROFILE_TIMEOUT = {"hppc": 600}               # overrides per profile_id

# Discrepancy injected into the TRUTH only (the fit stays the clean model):
#   set()              -> clean (pure SPMe + white noise)
#   {"Dsn","Dsp"}      -> concentration-dependent particle diffusivity (in the PDE)
#   {"ocp"}            -> OCP discrepancy (shifted OCP_n)
#   {"rc_short"}       -> fast lumped RC overpotential (R=1mOhm, tau=30s, cap 5mV)
#   {"rc_long"}        -> slow lumped RC overpotential (R=4mOhm, tau=300s, cap 20mV)
# All compose, e.g. {"Dsn","Dsp","rc_short","rc_long"}. RC branch params live in
# utils.RC_SPECS (edit there to retune R/tau/v_max).
DISC = {"rc_short", "rc_long"}     # V_truth = SPMe + rc_short + rc_long (-> tag "rc_long_rc_short")
TAG = "_".join(sorted(DISC)) if DISC else "clean"     # filename tag
OCP = "Negative electrode OCP [V]"
DSN = "Negative particle diffusivity [m2.s-1]"
DSP = "Positive particle diffusivity [m2.s-1]"

# save all pkls into a dedicated folder (created if missing)
# RUN_LABEL separates runs that share the same DISC tag but differ otherwise
# (e.g. bound choices): files go in rate_sweep_pkl/<RUN_LABEL>/. Set to "wide"
# for this run, "narrow" for the previous one.
RUN_LABEL = "wide"
RESULTS_DIR = os.path.join("rate_sweep_pkl", RUN_LABEL)
os.makedirs(RESULTS_DIR, exist_ok=True)

rate_results = {}
for profile in RATE_PROFILES:
    rate_exp = make_experiment(profile_id=profile)
    socs = SOC_BY_PROFILE.get(profile, SOC_SWEEP)        # per-profile SOC list
    timeout_s = PROFILE_TIMEOUT.get(profile, START_TIMEOUT_S)
    for soc in socs:
        log(f"=========== fit ({TAG}): {profile} at SOC {soc} ===========")

        # truth at theta_nominal + noise (+ discrepancies if DISC set). OCP and
        # Ds are param-level injections (before the solve); rc_* is post-process.
        base = make_base_params("Chen2020", soc=soc, sensitivity_ready=True)
        p_t, iv_t = prepare_sensitivity_inputs(base, fit_params, values=theta_values_nominal)
        if "ocp" in DISC:
            p_t[OCP] = ocp_n_discrepancy
        if "Dsn" in DISC:
            p_t[DSN] = Dsn_discrepancy; iv_t.pop(DSN, None)
        if "Dsp" in DISC:
            p_t[DSP] = Dsp_discrepancy; iv_t.pop(DSP, None)
        sol_t = run_model(make_model("SPMe", options=MODEL_OPTIONS), p_t, rate_exp, inputs=iv_t)
        t_soc = sol_t["Time [s]"].entries
        V_soc = sol_t["Voltage [V]"].entries.copy()
        # lumped RC overpotential (post-process on V using the truth current).
        # MINUS: overpotential lowers terminal voltage on discharge. Each rc_*
        # toggle in DISC adds its branch (params from utils.RC_SPECS).
        rc_branches = [RC_SPECS[k] for k in DISC if k in RC_SPECS]
        if rc_branches:
            I_soc = sol_t["Current [A]"].entries
            V_soc = V_soc - Vrc_discrepancy(t_soc, I_soc, rc_branches)
        if ADD_NOISE:
            V_soc = V_soc + np.random.default_rng(0).normal(0.0, NOISE_STD, size=V_soc.shape)
        log(f"  truth ready: len={len(t_soc)}, t_end={t_soc[-1]:.0f}")

        # multistart fit (CLEAN model). Each start runs in its OWN subprocess
        # with START_TIMEOUT_S; a stalled IDAKLU solve is killed -> penalty, so
        # the sweep can't hang. Results come back ordered by start_id.
        common_args = dict(
            t_data=t_soc, V_data=V_soc, fit_params=fit_params,
            param_bounds=PARAM_BOUNDS, log_scaled_params=log_scaled_params,
            initial_soc=soc, model_options=MODEL_OPTIONS, sigma_V_fit=sigma_V_fit,
            max_nfev=MAX_NFEV, profile_id=profile, jac_mode=JAC_MODE, diff_step=DIFF_STEP,
        )
        results = run_multistart_subprocess(
            common_args, N_STARTS, N_JOBS, timeout_s,
            workdir=os.path.join(RESULTS_DIR, "_tmp"),
        )
        rate_results[(profile, soc)] = results

        # fixed name per (profile, SOC) -> reruns OVERWRITE the previous pkl
        fname = os.path.join(
            RESULTS_DIR,
            f"rate_{TAG}_{profile}_soc{int(soc * 100)}.pkl",
        )
        with open(fname, "wb") as f:
            pickle.dump(results, f)
        log(f"  saved {fname}")

        # recovery check: median theta_hat/theta_nominal over survived (~1 = retrieved)
        survived = [r for r in results if r["rmse_mV"] < 10.0]
        print(f"  recovery over {len(survived)}/{len(results)} survived "
              f"(median theta_hat/theta_nominal):")
        for p_ in fit_params:
            ratios = np.array([r["theta_hat"][p_] / theta_values_nominal[p_] for r in survived])
            print(f"    {PARAM_LABELS_ASCII.get(p_, p_):8s} {np.median(ratios):8.3f}")


# %% Rate sweep analysis: locate the saved pkls (profile x SOC)
# Set RATE_TAG to match the sweep you want to analyze:
#   "Dsn_Dsp"  -> the Ds-discrepancy sweep
#   "clean"    -> the clean sweep
import glob, re, pickle, os
import numpy as np
import matplotlib.pyplot as plt

RATE_TAG = "clean"   # "clean" | "Dsn_Dsp_rc_long_rc_short" | "Dsn_Dsp_ocp_rc_long_rc_short"
RUN_LABEL = "wide"                      # which subfolder to read: "wide" or "narrow"
RATE_DIR = os.path.join("rate_sweep_pkl", RUN_LABEL)
PROFILES = ["hppc", "C3_dchg", "1C_dchg", "2C_dchg", "5C_dchg"]
# per-profile SOCs (must match the sweep): HPPC uses 85/65/50/30, discharges 90/70/55/40
SOCS_DEFAULT = [90, 70, 55, 40]
SOCS_BY_PROFILE = {"hppc": [85, 65, 50, 30]}
def socs_of(profile):
    return SOCS_BY_PROFILE.get(profile, SOCS_DEFAULT)
labels = [PARAM_LABELS_ASCII.get(p_, p_) for p_ in fit_params]

rate_files = {}
for f in glob.glob(os.path.join(RATE_DIR, f"rate_{RATE_TAG}_*.pkl")):
    # matches both fixed names (..._soc90.pkl) and old timestamped ones (..._soc90_20260629_083740.pkl)
    m = re.search(rf"rate_{RATE_TAG}_(.+?)_soc(\d+)(?:_\d+_\d+)?\.pkl", os.path.basename(f))
    if m:
        rate_files[(m.group(1), int(m.group(2)))] = f
print(f"found {len(rate_files)} rate-sweep pkls for tag '{RATE_TAG}'")


# %% Print best fits per case (profile x SOC), within +RMSE_TOL of the best
# Only prints fits whose RMSE is within RMSE_TOL of the case's best (e.g. 0.05 =
# up to best*1.05). These are the ~equally-good fits; big theta spread among them
# = a flat/non-identifiable direction. Also capped at MAX_SHOW rows.
RMSE_TOL = 0.05      # +5% of best RMSE
MAX_SHOW = 40
for profile in PROFILES:
    for soc in socs_of(profile):
        if (profile, soc) not in rate_files:
            print(f"\n[missing] {profile} SOC{soc}%")
            continue
        with open(rate_files[(profile, soc)], "rb") as fp:
            res = pickle.load(fp)
        ranked = sorted(res, key=lambda r: r["rmse_mV"])
        best = ranked[0]["rmse_mV"]
        keep = [r for r in ranked if r["rmse_mV"] <= best * (1 + RMSE_TOL)][:MAX_SHOW]
        print(f"\n===== {profile}  SOC {soc}% : {len(keep)} fits within +{RMSE_TOL*100:.0f}% "
              f"of best ({best:.3f} mV)   (ratio theta_hat/theta_nom) =====")
        print(f"{'rank':>4}{'RMSE_mV':>9}  " + "".join(f"{l:>8}" for l in labels))
        for rank, r in enumerate(keep):
            rr = [r["theta_hat"][p_] / theta_values_nominal[p_] for p_ in fit_params]
            print(f"{rank:>4}{r['rmse_mV']:>9.3f}  " + "".join(f"{x:>8.3f}" for x in rr))


# %% Voltage profiles of the 10 best per case (5x4 grid: rows=profile, cols=SOC)
# Set DISC_ANALYSIS to the SAME discrepancies the sweep used, so the dashed
# "truth" line is the ACTUAL discrepant V the fits were chasing (not clean).
DISC_ANALYSIS = {"Dsn", "Dsp", "rc_short", "rc_long"}

def _solve_V(theta, profile, soc, discrepant=False):
    exp = make_experiment(profile_id=profile)
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, fit_params, values=theta)
    if discrepant and "ocp" in DISC_ANALYSIS:
        p["Negative electrode OCP [V]"] = ocp_n_discrepancy
    if discrepant and "Dsn" in DISC_ANALYSIS:
        p[DSN] = Dsn_discrepancy; iv.pop(DSN, None)
    if discrepant and "Dsp" in DISC_ANALYSIS:
        p[DSP] = Dsp_discrepancy; iv.pop(DSP, None)
    sol = run_model(make_model("SPMe", options=MODEL_OPTIONS), p, exp, inputs=iv)
    t, V = sol["Time [s]"].entries, sol["Voltage [V]"].entries.copy()
    if discrepant:
        rc = [RC_SPECS[k] for k in DISC_ANALYSIS if k in RC_SPECS]
        if rc:
            V = V - Vrc_discrepancy(t, sol["Current [A]"].entries, rc)
    return t, V


n_rows, n_cols = len(PROFILES), max(len(socs_of(p)) for p in PROFILES)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 14))
for i, profile in enumerate(PROFILES):
    for j, soc in enumerate(socs_of(profile)):
        ax = axes[i, j]
        if (profile, soc) not in rate_files:
            ax.set_visible(False)
            continue
        with open(rate_files[(profile, soc)], "rb") as fp:
            res = pickle.load(fp)
        top10 = sorted(res, key=lambda r: r["rmse_mV"])[:10]

        for r in top10:                                  # 10 best fits (gray, clean model)
            tf, Vf = _solve_V(r["theta_hat"], profile, soc)
            ax.plot(tf, Vf, color="gray", alpha=0.4, lw=0.8)
        tt, Vt = _solve_V(theta_values_nominal, profile, soc, discrepant=True)  # actual truth
        ax.plot(tt, Vt, "r--", lw=1.5, label="discrepant truth", zorder=5)

        ax.set_title(f"{profile}  SOC{soc}%", fontsize=9)
        ax.grid(alpha=0.3)
        if i == n_rows - 1:
            ax.set_xlabel("Time [s]")
        if j == 0:
            ax.set_ylabel("V")

fig.suptitle("Rate sweep: 10 best fits (gray) vs discrepant truth (red dashed)")
plt.tight_layout()
plt.show()

# %% Print SOC endpoints (particle concentration at 0% / 100% SOC)
_bp = make_base_params("Chen2020", sensitivity_ready=True)
ep = get_soc_endpoints(_bp)
csn_max = _bp["Maximum concentration in negative electrode [mol.m-3]"]
csp_max = _bp["Maximum concentration in positive electrode [mol.m-3]"]
print(f"cs_max:  neg = {csn_max:.1f}   pos = {csp_max:.1f}  [mol/m3]")
print("get_soc_endpoints  [mol/m3]  ->  stoichiometry (cs/cs_max):")
for k, v in ep.items():
    cmax = csn_max if k.startswith("csn") else csp_max
    print(f"  {k:14s} = {v:9.2f}   ->  {v/cmax:.4f}")


# %% Clean vs discrepant V(theta_nominal) [1:1 overlay per profile x SOC]
# Overlays, at the SAME theta_nominal:  V_clean  vs  V_clean + discrepancies.
# The gap between them IS the injected discrepancy (what the fit has to absorb).
# Needs only the Setup-cell globals; does NOT need the fit pkls. Set DISC_COMPARE
# to the discrepancies you want to show (default = the current run's, no OCP).
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

DISC_COMPARE = {"Dsn", "Dsp", "rc_short", "rc_long"}   # add "ocp" to include the OCP shift
_OCP = "Negative electrode OCP [V]"
_DSN = "Negative particle diffusivity [m2.s-1]"
_DSP = "Positive particle diffusivity [m2.s-1]"
CMP_PROFILES = ["hppc", "C3_dchg", "1C_dchg", "2C_dchg", "5C_dchg"]
CMP_SOCS = {"hppc": [85, 65, 50, 30]}          # others default below
CMP_SOCS_DEFAULT = [90, 70, 55, 40]

def _Vnom(profile, soc, disc):
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, fit_params, values=theta_values_nominal)
    if "ocp" in disc:
        p[_OCP] = ocp_n_discrepancy
    if "Dsn" in disc:
        p[_DSN] = Dsn_discrepancy; iv.pop(_DSN, None)
    if "Dsp" in disc:
        p[_DSP] = Dsp_discrepancy; iv.pop(_DSP, None)
    sol = run_model(make_model("SPMe", options=MODEL_OPTIONS), p,
                    make_experiment(profile_id=profile), inputs=iv)
    t, V = sol["Time [s]"].entries, sol["Voltage [V]"].entries.copy()
    rc = [RC_SPECS[k] for k in disc if k in RC_SPECS]
    if rc:
        V = V - Vrc_discrepancy(t, sol["Current [A]"].entries, rc)
    return t, V

_nr = len(CMP_PROFILES)
_nc = max(len(CMP_SOCS.get(p, CMP_SOCS_DEFAULT)) for p in CMP_PROFILES)
fig, axes = plt.subplots(_nr, _nc, figsize=(16, 14))
for i, profile in enumerate(CMP_PROFILES):
    for j, soc in enumerate(CMP_SOCS.get(profile, CMP_SOCS_DEFAULT)):
        ax = axes[i, j]
        tc, Vc = _Vnom(profile, soc, set())            # clean V(theta_nom)
        td, Vd = _Vnom(profile, soc, DISC_COMPARE)     # + discrepancies
        ax.plot(tc, Vc, "k-",  lw=1.6, label="V(theta_nom)")
        ax.plot(td, Vd, "r--", lw=1.6, label="+ discrepancies")
        Vd_on_c = interp1d(td, Vd, bounds_error=False, fill_value="extrapolate")(tc)
        axt = ax.twinx()                               # dV (green) for the 1:1 gap
        axt.plot(tc, (Vd_on_c - Vc) * 1e3, color="tab:green", lw=0.8, alpha=0.6)
        axt.set_ylabel("dV [mV]", color="tab:green", fontsize=7)
        axt.tick_params(axis="y", labelcolor="tab:green", labelsize=6)
        ax.set_title(f"SOC{soc}%", fontsize=9)
        ax.grid(alpha=0.3)
        if i == 0 and j == 0:
            ax.legend(fontsize=7, loc="lower left")
        if i == _nr - 1:
            ax.set_xlabel("Time [s]")
        if j == 0:
            ax.set_ylabel("V")
fig.suptitle(f"Clean vs discrepant V(theta_nominal)   [DISC={sorted(DISC_COMPARE)}]")
plt.tight_layout(rect=[0.05, 0, 1, 0.97])
# profile name as a row label on the far left, outside the panels
for i, profile in enumerate(CMP_PROFILES):
    pos = axes[i, 0].get_position()
    fig.text(0.015, (pos.y0 + pos.y1) / 2, profile, rotation=90,
             ha="center", va="center", fontsize=12, fontweight="bold")
plt.show()
# %%
