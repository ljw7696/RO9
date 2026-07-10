# %% Driver: per-SOC-region JOINT fit (minimize mean voltage RMSE over the 5 profiles).
# Same config + multistart machinery as main_rate_sweep.py; the worker (joint_fit_worker.py)
# is rate_fit_worker.py with only the residual changed (5 profiles stacked, equal weight).
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, pickle, time
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import numpy as np
from joint_fit_worker import run_multistart_subprocess

# ==== settings (edit these) ====
N_STARTS = 50            # random uniform initial points (multistart)
N_JOBS = 12             # concurrent subprocesses
# per-region hard timeout = sum of each profile's original timeout (they run sequentially)
PROFILE_TIMEOUT = {"hppc": 600}          # from main_rate_sweep.py
DEFAULT_TIMEOUT = 240                     # discharge profiles
# ================================

# ---- same fit config as main_rate_sweep.py ----
PARAM_BOUNDS = {
    "Negative electrode exchange-current density [A.m-2]": (1e-9, 1e-5),
    "Positive electrode exchange-current density [A.m-2]": (1e-8, 1e-4),
    "Electrolyte conductivity [S.m-1]": (0.1, 3.0),
    "Electrolyte diffusivity [m2.s-1]": (1e-12, 1e-9),
    "Negative particle diffusivity [m2.s-1]": (5e-16, 1e-12),
    "Positive particle diffusivity [m2.s-1]": (5e-16, 1e-12),
}
fit_params = list(PARAM_BOUNDS.keys())
log_scaled_params = list(PARAM_BOUNDS.keys())     # all fitted params are log-scaled
sigma_V_fit = 1e-3
MODEL_OPTIONS = {"surface form": "differential", "contact resistance": "true"}
MAX_NFEV = 100
DIFF_STEP = 1e-2
JAC_MODE = "fd"

TRUTH = pickle.load(open("meta_genmatrix_rc_long_rc_short_wide_both.pkl", "rb"))["truth"]


def region(soc):
    return 1 if soc >= 80 else 2 if soc >= 60 else 3 if soc >= 48 else 4


OUT = {}
for rg in [1, 2, 3, 4]:
    conds = sorted([k for k in TRUTH if region(k[1]) == rg], key=lambda k: k[0])
    profiles = [(prof, soc / 100.0, TRUTH[(prof, soc)][0], TRUTH[(prof, soc)][1])
                for (prof, soc) in conds]
    timeout_s = sum(PROFILE_TIMEOUT.get(prof, DEFAULT_TIMEOUT) for (prof, soc) in conds)
    common_args = dict(
        profiles=profiles, fit_params=fit_params, param_bounds=PARAM_BOUNDS,
        log_scaled_params=log_scaled_params, model_options=MODEL_OPTIONS,
        sigma_V_fit=sigma_V_fit, max_nfev=MAX_NFEV, jac_mode=JAC_MODE, diff_step=DIFF_STEP,
    )
    print(f"\n=== region {rg}: {[f'{c[0]}_s{c[1]}' for c in conds]}  "
          f"(timeout {timeout_s}s/start, {N_STARTS} starts) ===", flush=True)
    t0 = time.time()
    R = run_multistart_subprocess(common_args, N_STARTS, N_JOBS, timeout_s,
                                  workdir=f"_joint_tmp/region{rg}", py_exe=sys.executable)
    R.sort(key=lambda r: r["rmse_mV"])
    OUT[rg] = {"conds": conds, "all": R, "best": R[0]}
    n_stall = sum(r.get("stalled") for r in R)
    print(f"  done {time.time()-t0:.0f}s | {n_stall}/{N_STARTS} stalled | "
          f"best mean_rmse={R[0]['rmse_mV']:.3f} mV", flush=True)

pickle.dump(OUT, open("joint_fit_results.pkl", "wb"))
print("\nsaved joint_fit_results.pkl  ->  run `python joint_fit_report.py`", flush=True)
