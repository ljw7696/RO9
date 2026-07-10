# %% Per-SOC-region WLTP held-out plots: overlay + error (2x2, one panel per SOC=region)
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
WLTP_TIMEOUT = 60.0
DSN = "Negative particle diffusivity [m2.s-1]"; DSP = "Positive particle diffusivity [m2.s-1]"
FP = ["Negative electrode exchange-current density [A.m-2]",
      "Positive electrode exchange-current density [A.m-2]",
      "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]", DSN, DSP]
LAB = ["k-", "k+", "kappa", "De", "Ds-", "Ds+"]
NOM = np.array([6.48e-7, 3.42e-6, 0.9487, 1.7694e-10, 3.3e-14, 4.0e-15])
WLTP_SOCS = [85, 65, 50, 30]

eff = pickle.load(open("meta_cache_rc_long_rc_short_wide.pkl", "rb"))
results = pickle.load(open("results_bySOC.pkl", "rb"))
WALL = pickle.load(open("meta_wltp_rc_long_rc_short_wide_both.pkl", "rb"))
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


REG_CONDS = {rg: [k for k in OK if region(k[1]) == rg] for rg in [1, 2, 3, 4]}
THETA_COMB = {rg: {FP[j]: results[rg][LAB[j]]["theta_comb"]
                   if not np.isnan(results[rg][LAB[j]]["theta_comb"]) else NOM[j]
                   for j in range(len(FP))} for rg in [1, 2, 3, 4]}
# theta_comb(region) on WLTP at that SOC
COMB_W = {}
for soc in WLTP_SOCS:
    rg = region(soc); tW = WALL[soc]["t"]
    COMB_W[soc] = _guard(lambda: sim_V(THETA_COMB[rg], "WLTP", soc, tW), WLTP_TIMEOUT)
    r = results[rg]
    print(f"WLTP soc{soc} (region {rg}) theta_comb simulated")

# ---------- Fig 1: overlay ----------
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for ax, soc in zip(axes.ravel(), WLTP_SOCS):
    rg = region(soc); D = WALL[soc]; tW, VW = D["t"], D["truth"]
    for k in REG_CONDS[rg]:                       # region theta_eff,i (gray)
        Vs = D["curves"].get(flabel(k))
        if Vs is not None:
            ax.plot(tW, Vs, color="0.35", lw=0.9, alpha=0.6, zorder=1)
    ax.plot(tW, VW, color="tab:blue", lw=1.8, zorder=3)               # V_data
    if COMB_W[soc] is not None:
        ax.plot(tW, COMB_W[soc], color="red", ls="--", lw=1.8, zorder=4)   # combined
    ax.set_title(f"WLTP SOC {soc}%", fontsize=12)
    ax.grid(alpha=0.3); ax.tick_params(axis="both", labelsize=11)
    ax.set_ylabel("Voltage (V)", fontsize=14); ax.set_xlabel("Time (s)", fontsize=14)
fig.legend(handles=[
    Line2D([0], [0], color="0.35", lw=1.6, label=r"$V_{SPMe}(\theta_{eff,i})$"),
    Line2D([0], [0], color="tab:blue", lw=2.0, label=r"$V_{data}$"),
    Line2D([0], [0], color="red", ls="--", lw=2.0, label=r"$V_{SPMe}(\theta_{combined})$")],
    loc="upper right", fontsize=16, ncol=3, bbox_to_anchor=(0.99, 0.99))
fig.suptitle("WLTP Voltage: individual vs combined  (per SOC region)", y=1.0, fontsize=17)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("meta_wltp_overlays_bySOC.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved meta_wltp_overlays_bySOC.png")

# ---------- Fig 2: error ----------
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for ax, soc in zip(axes.ravel(), WLTP_SOCS):
    rg = region(soc); D = WALL[soc]; tW, VW = D["t"], D["truth"]
    fivec_lbl = flabel(next(k for k in REG_CONDS[rg] if k[0] == "5C_dchg"))   # region's 5C theta -> blue
    for k in REG_CONDS[rg]:
        lbl = flabel(k); Vs = D["curves"].get(lbl)
        if Vs is None:
            continue
        if lbl == fivec_lbl:
            ax.plot(tW, (VW - Vs) * 1e3, color="tab:blue", lw=1.6, alpha=0.9, zorder=3)
        else:
            ax.plot(tW, (VW - Vs) * 1e3, color="0.35", lw=0.6, alpha=0.45, zorder=1)
    Vc = COMB_W[soc]
    if Vc is not None:
        ax.plot(tW, (VW - Vc) * 1e3, color="red", lw=1.7, zorder=4)
    ax.axhline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
    if Vc is not None:
        m = max(np.max(np.abs((VW - Vc) * 1e3)) * 3.0, 5.0)
        ax.set_ylim(-m, m)
    ax.set_title(f"WLTP SOC {soc}%", fontsize=12)
    ax.grid(alpha=0.3); ax.tick_params(axis="both", labelsize=11)
    ax.set_ylabel("Voltage (mV)", fontsize=14); ax.set_xlabel("Time (s)", fontsize=14)
fig.legend(handles=[
    Line2D([0], [0], color="0.35", lw=1.6, label=r"$V_{data} - V_{SPMe}(\theta_{eff,i})$"),
    Line2D([0], [0], color="tab:blue", lw=1.8, label=r"$V_{data} - V_{SPMe}(\theta_{5C})$"),
    Line2D([0], [0], color="red", lw=2.0, label=r"$V_{data} - V_{SPMe}(\theta_{combined})$")],
    loc="upper right", fontsize=16, ncol=3, bbox_to_anchor=(0.99, 0.96))
fig.suptitle(r"WLTP: Voltage Error under different $\theta_{eff,i}$  (per SOC region)", y=1.0, fontsize=17)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("meta_wltp_error_bySOC.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved meta_wltp_error_bySOC.png")
