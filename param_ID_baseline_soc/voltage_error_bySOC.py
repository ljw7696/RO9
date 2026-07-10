# %% Per-SOC-region voltage-error grid (same 5x4 layout; gray = region's theta_eff,i only)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, pickle, threading
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import numpy as np
import pybamm
pybamm.set_logging_level("ERROR")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import (make_base_params, make_model, run_model, make_experiment,
                   prepare_sensitivity_inputs)

OPTS = {"surface form": "differential", "contact resistance": "true"}
SIM_TIMEOUT = 30.0
DSN = "Negative particle diffusivity [m2.s-1]"; DSP = "Positive particle diffusivity [m2.s-1]"
FP = ["Negative electrode exchange-current density [A.m-2]",
      "Positive electrode exchange-current density [A.m-2]",
      "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]", DSN, DSP]
LAB = ["k-", "k+", "kappa", "De", "Ds-", "Ds+"]
NOM = np.array([6.48e-7, 3.42e-6, 0.9487, 1.7694e-10, 3.3e-14, 4.0e-15])

eff = pickle.load(open("meta_cache_rc_long_rc_short_wide.pkl", "rb"))
results = pickle.load(open("results_bySOC.pkl", "rb"))
GEN = pickle.load(open("meta_genmatrix_rc_long_rc_short_wide_both.pkl", "rb"))
CURVES, TRUTH = GEN["curves"], GEN["truth"]
OK = [k for k in sorted(eff) if eff[k].get("ok")]


def region(soc):
    return 1 if soc >= 80 else 2 if soc >= 60 else 3 if soc >= 48 else 4


def flabel(k):
    return f"{k[0]}_s{k[1]}"


def _guard(fn, timeout):
    box = {}
    def w():
        try: box["v"] = fn()
        except BaseException as e: box["e"] = e
    th = threading.Thread(target=w, daemon=True); th.start(); th.join(timeout)
    return None if (th.is_alive() or "e" in box) else box["v"]


def sim_V(theta, profile, soc, t_ref):
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, FP, values=theta)
    solver = pybamm.IDAKLUSolver(options={"max_num_steps": 20000})
    sol = run_model(make_model("SPMe", options=OPTS), p,
                    make_experiment(profile_id=profile), inputs=iv, solver=solver)
    return np.interp(t_ref, sol["Time [s]"].entries, sol["Voltage [V]"].entries)


# region -> list of its condition keys (the 5 theta_eff,i sources)
REG_CONDS = {rg: [k for k in OK if region(k[1]) == rg] for rg in [1, 2, 3, 4]}
THETA_COMB = {rg: {FP[j]: results[rg][LAB[j]]["theta_comb"]
                   if not np.isnan(results[rg][LAB[j]]["theta_comb"]) else NOM[j]
                   for j in range(len(FP))} for rg in [1, 2, 3, 4]}
# simulate theta_comb(region) on each of its own conditions
COMB_CURVE = {}
for rg in [1, 2, 3, 4]:
    for tk in REG_CONDS[rg]:
        t_ref, _ = TRUTH[tk]
        COMB_CURVE[tk] = _guard(lambda: sim_V(THETA_COMB[rg], tk[0], tk[1], t_ref), SIM_TIMEOUT)

PROFILES_ORDER = ["1C_dchg", "2C_dchg", "5C_dchg", "C3_dchg", "hppc"]
socs_of = {}
for k in OK:
    socs_of.setdefault(k[0], []).append(k[1])
for p in socs_of:
    socs_of[p] = sorted(socs_of[p])
nrow, ncol = len(PROFILES_ORDER), max(len(v) for v in socs_of.values())

fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
for ri, prof in enumerate(PROFILES_ORDER):
    socs = socs_of.get(prof, [])
    for ci in range(ncol):
        ax = axes[ri][ci]
        if ci >= len(socs):
            ax.axis("off"); continue
        soc = socs[ci]; tk = (prof, soc); rg = region(soc)
        t_ref, V_truth = TRUTH[tk]
        hppc_k = next(k for k in REG_CONDS[rg] if k[0] == "hppc")   # region's hppc theta -> blue
        for k in REG_CONDS[rg]:                       # this region's theta_eff,i
            Vs = CURVES.get((flabel(k), tk))
            if Vs is None:
                continue
            if k == hppc_k:
                ax.plot(t_ref, (V_truth - Vs) * 1e3, color="tab:blue", lw=1.6, alpha=0.9, zorder=3)
            else:
                ax.plot(t_ref, (V_truth - Vs) * 1e3, color="0.35", lw=0.7, alpha=0.5, zorder=1)
        Vc = COMB_CURVE.get(tk)                        # region theta_combined (red, on top)
        if Vc is not None:
            ax.plot(t_ref, (V_truth - Vc) * 1e3, color="red", lw=1.7, zorder=4)
        ax.axhline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
        if Vc is not None:
            m = max(np.max(np.abs((V_truth - Vc) * 1e3)) * 3.0, 5.0)
            ax.set_ylim(-m, m)
        ax.set_title(f"{prof} s{soc}", fontsize=12)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)
        if ci == 0:
            ax.set_ylabel("Voltage (mV)", fontsize=14)
        if ri == nrow - 1:
            ax.set_xlabel("Time (s)", fontsize=14)
fig.legend(handles=[
    Line2D([0], [0], color="0.35", lw=1.6, label=r"$V_{data} - V_{SPMe}(\theta_{eff,i})$"),
    Line2D([0], [0], color="tab:blue", lw=1.8, label=r"$V_{data} - V_{SPMe}(\theta_{hppc})$"),
    Line2D([0], [0], color="red", lw=2.0, label=r"$V_{data} - V_{SPMe}(\theta_{combined})$")],
    loc="upper right", fontsize=16, ncol=3, bbox_to_anchor=(0.99, 0.955))
fig.suptitle(r"Voltage Error under different $\theta_{eff,i}$  (per SOC region)", y=1.0, fontsize=17)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("meta_voltage_error_bySOC.png", dpi=120, bbox_inches="tight")
print("saved meta_voltage_error_bySOC.png")
