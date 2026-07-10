# %% Region-1 test: theta_comb with outliers EXCLUDED (cutoff) vs INCLUDED (all 5), DL.
# Then held-out RMSE on WLTP/FUDS/US06 at SOC 85% for: comb(excl), comb(incl), theta_joint.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, pickle
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import numpy as np
import pybamm
pybamm.set_logging_level("ERROR")
from utils import (make_base_params, make_model, run_model, make_experiment,
                   prepare_sensitivity_inputs, Vrc_discrepancy, RC_SPECS)

DSN = "Negative particle diffusivity [m2.s-1]"; DSP = "Positive particle diffusivity [m2.s-1]"
FP = ["Negative electrode exchange-current density [A.m-2]",
      "Positive electrode exchange-current density [A.m-2]",
      "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]", DSN, DSP]
LAB = ["k-", "k+", "kappa", "De", "Ds-", "Ds+"]
NOM = np.array([6.48e-7, 3.42e-6, 0.9487, 1.7694e-10, 3.3e-14, 4.0e-15])
NOM_D = {FP[j]: NOM[j] for j in range(6)}
CRLB_FLOOR, CRLB_MAX, LN10 = 0.01, 0.50, np.log(10.0)
OPTS = {"surface form": "differential", "contact resistance": "true"}
RC = [RC_SPECS["rc_short"], RC_SPECS["rc_long"]]

eff = pickle.load(open("meta_cache_rc_long_rc_short_wide.pkl", "rb"))
OUT = pickle.load(open("joint_fit_results.pkl", "rb"))
OK = [k for k in sorted(eff) if eff[k].get("ok")]
conds1 = [k for k in OK if k[1] >= 80]     # region 1


def fit_DL(y, sy):
    k = len(y); w0 = 1.0 / sy**2
    yb = np.sum(w0 * y) / np.sum(w0)
    Q = np.sum(w0 * (y - yb)**2)
    c = np.sum(w0) - np.sum(w0**2) / np.sum(w0)
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    w = 1.0 / (sy**2 + tau2); mu = np.sum(w * y) / np.sum(w)
    return mu


def combine(include_outliers):
    theta = {}
    for j in range(6):
        ks = conds1 if include_outliers else [k for k in conds1 if eff[k]["crlb"][j] <= CRLB_MAX]
        y = np.array([np.log10(eff[k]["theta_hat"][FP[j]] / NOM[j]) for k in ks])
        sy = np.clip([eff[k]["crlb"][j] for k in ks], CRLB_FLOOR, None) / LN10
        theta[FP[j]] = NOM[j] * 10**fit_DL(y, sy)
    return theta


theta_excl = combine(False)      # cutoff (current default)
theta_incl = combine(True)       # include railed outliers
theta_joint = OUT[1]["best"]["theta_hat"]

print("Region 1 theta (%dev from nominal):")
print(f"{'param':6}{'comb(excl)':>12}{'comb(incl)':>12}{'joint':>10}")
for j, l in enumerate(LAB):
    print(f"{l:6}{(theta_excl[FP[j]]/NOM[j]-1)*100:>+11.0f}%"
          f"{(theta_incl[FP[j]]/NOM[j]-1)*100:>+11.0f}%{(theta_joint[FP[j]]/NOM[j]-1)*100:>+9.0f}%")


def make_truth(profile, soc):
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, FP, values=NOM_D)
    sol = run_model(make_model("SPMe", options=OPTS), p, make_experiment(profile_id=profile),
                    inputs=iv, solver=pybamm.IDAKLUSolver(options={"max_num_steps": 50000}))
    t = sol["Time [s]"].entries; V = sol["Voltage [V]"].entries.copy()
    V = V - Vrc_discrepancy(t, sol["Current [A]"].entries, RC)
    return t, V + np.random.default_rng(0).normal(0.0, 1e-3, size=V.shape)


def sim_V(theta, profile, soc, t_ref):
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, FP, values=theta)
    sol = run_model(make_model("SPMe", options=OPTS), p, make_experiment(profile_id=profile),
                    inputs=iv, solver=pybamm.IDAKLUSolver(options={"max_num_steps": 50000}))
    return np.interp(t_ref, sol["Time [s]"].entries, sol["Voltage [V]"].entries)


print("\nHeld-out RMSE (mV) at SOC 85% (region 1):")
print(f"{'cycle':8}{'comb(excl)':>12}{'comb(incl)':>12}{'joint':>10}")
for dc in ["WLTP", "FUDS", "US06"]:
    t_ref, V_data = make_truth(dc, 85)
    re = np.sqrt(np.mean((sim_V(theta_excl, dc, 85, t_ref) - V_data) ** 2)) * 1e3
    ri = np.sqrt(np.mean((sim_V(theta_incl, dc, 85, t_ref) - V_data) ** 2)) * 1e3
    rj = np.sqrt(np.mean((sim_V(theta_joint, dc, 85, t_ref) - V_data) ** 2)) * 1e3
    print(f"{dc:8}{re:>12.2f}{ri:>12.2f}{rj:>10.2f}")
