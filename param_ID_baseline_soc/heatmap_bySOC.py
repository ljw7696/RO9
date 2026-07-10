# %% Per-SOC-region voltage-RMSE rank heatmaps (one PNG per region)
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
from matplotlib.colors import LogNorm

CMIN, CMAX = 1.0, 100.0   # log color range (mV); blow-ups saturate, values still annotated


def _fmt(v):
    return "x" if not np.isfinite(v) else (f"{v:.0f}" if v >= 10 else f"{v:.1f}")


def tlabel(r):
    return r"$\theta_{combined}$" if r == "COMBINED" else r"$\theta_{" + r.replace("_", r"\_") + "}$"
from utils import (make_base_params, make_model, run_model, make_experiment,
                   prepare_sensitivity_inputs)

OPTS = {"surface form": "differential", "contact resistance": "true"}
RATE_TAG = "rc_long_rc_short"
SIM_TIMEOUT = 30.0
DSN = "Negative particle diffusivity [m2.s-1]"; DSP = "Positive particle diffusivity [m2.s-1]"
FP = ["Negative electrode exchange-current density [A.m-2]",
      "Positive electrode exchange-current density [A.m-2]",
      "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]", DSN, DSP]
LAB = ["k-", "k+", "kappa", "De", "Ds-", "Ds+"]
NOM = np.array([6.48e-7, 3.42e-6, 0.9487, 1.7694e-10, 3.3e-14, 4.0e-15])

eff = pickle.load(open("meta_cache_rc_long_rc_short_wide.pkl", "rb"))
ceff = pickle.load(open("meta_ceff_wide.pkl", "rb"))
results = pickle.load(open("results_bySOC.pkl", "rb"))
GEN = pickle.load(open("meta_genmatrix_rc_long_rc_short_wide_both.pkl", "rb"))
RMSE, TRUTH = GEN["rmse"], GEN["truth"]
OK = [k for k in sorted(eff) if eff[k].get("ok")]


def region(soc):
    return 1 if soc >= 80 else 2 if soc >= 60 else 3 if soc >= 48 else 4


RLAB = {1: "SOC ~85-90", 2: "SOC ~65-70", 3: "SOC ~50-55", 4: "SOC ~30-40"}


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


def flabel(k):                       # cache source label (keeps _dchg)
    return f"{k[0]}_s{k[1]}"


def short(k):                        # display label
    return f"{k[0].replace('_dchg','')}_s{k[1]}"


DATA = {}
for rg in [1, 2, 3, 4]:
    conds = sorted([k for k in OK if region(k[1]) == rg], key=lambda k: ceff[k[0]])
    theta_comb = {FP[j]: results[rg][LAB[j]]["theta_comb"]
                  if not np.isnan(results[rg][LAB[j]]["theta_comb"]) else NOM[j]
                  for j in range(len(FP))}
    rows = [short(k) for k in conds] + ["COMBINED"]
    # RMSE matrix: rows(candidates) x cols(conditions)
    RM = np.full((len(rows), len(conds)), np.nan)
    for ci, tk in enumerate(conds):
        t_ref, V_truth = TRUTH[tk]
        for ri, k in enumerate(conds):                      # individual theta_eff,i from cache
            RM[ri, ci] = RMSE[(flabel(k), tk)]
        Vs = _guard(lambda: sim_V(theta_comb, tk[0], tk[1], t_ref), SIM_TIMEOUT)
        RM[-1, ci] = np.inf if Vs is None else float(np.sqrt(np.mean((Vs - V_truth) ** 2)) * 1e3)
    # rank within each column (1 = best/lowest RMSE)
    RANK = np.zeros_like(RM, dtype=int)
    for ci in range(len(conds)):
        order = np.argsort(RM[:, ci])
        for r, ri in enumerate(order):
            RANK[ri, ci] = r + 1
    # plot actual RMSE (mV), log color
    n = len(rows)
    Mc = np.clip(RM, CMIN, CMAX)
    fig, ax = plt.subplots(figsize=(max(6, len(conds) * 1.2), max(5, n * 0.75)))
    im = ax.imshow(Mc, aspect="auto", cmap="RdYlGn_r", norm=LogNorm(vmin=CMIN, vmax=CMAX))
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([f"{tk[0]}\ns{tk[1]}" for tk in conds], fontsize=10, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels([tlabel(r) for r in rows], fontsize=12)
    for yi, cl in enumerate(rows):
        if cl == "COMBINED":
            lab = ax.get_yticklabels()[yi]; lab.set_fontweight("bold"); lab.set_color("blue")
            ax.axhline(yi - 0.5, color="blue", lw=1.5); ax.axhline(yi + 0.5, color="blue", lw=1.5)
        for xi in range(len(conds)):
            ax.text(xi, yi, _fmt(RM[yi, xi]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="voltage RMSE (mV)")
    ax.set_xlabel("Profile Name")
    ax.set_ylabel(r"$\theta_{eff}$")
    ax.set_title(f"Voltage RMSE (mV)  ({RLAB[rg]})")
    plt.tight_layout()
    fname = f"meta_rmse_heatmap_region{rg}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    DATA[rg] = (rows, conds, RM)
    print(f"saved {fname}  (COMBINED RMSE mV: {[round(float(v),1) for v in RM[-1]]})")

pickle.dump({rg: (DATA[rg][0], [list(c) for c in DATA[rg][1]], DATA[rg][2])
             for rg in DATA}, open("rmse_matrices_bySOC.pkl", "wb"))
print("saved rmse_matrices_bySOC.pkl")

# ---- combined 2x2 ----
fig, axes = plt.subplots(2, 2, figsize=(17, 13))
for rg, ax in zip([1, 2, 3, 4], axes.flat):
    rows, conds, RM = DATA[rg]
    n = len(rows)
    Mc = np.clip(RM, CMIN, CMAX)
    im = ax.imshow(Mc, aspect="auto", cmap="RdYlGn_r", norm=LogNorm(vmin=CMIN, vmax=CMAX))
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([f"{tk[0]}\ns{tk[1]}" for tk in conds], fontsize=12, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels([tlabel(r) for r in rows], fontsize=13)
    for yi, cl in enumerate(rows):
        if cl == "COMBINED":
            lab = ax.get_yticklabels()[yi]; lab.set_fontweight("bold"); lab.set_color("blue")
            ax.axhline(yi - 0.5, color="blue", lw=1.8); ax.axhline(yi + 0.5, color="blue", lw=1.8)
        for xi in range(len(conds)):
            ax.text(xi, yi, _fmt(RM[yi, xi]), ha="center", va="center", fontsize=12)
    ax.set_xlabel("Profile Name", fontsize=14)
    ax.set_ylabel(r"$\theta_{eff}$", fontsize=16)
    ax.set_title(RLAB[rg], fontsize=16)
fig.suptitle("Voltage RMSE (mV)  (per SOC region)", fontsize=19)
fig.subplots_adjust(hspace=0.40, wspace=0.42, right=0.87, top=0.93)
cax = fig.add_axes([0.89, 0.15, 0.03, 0.7])
cb = fig.colorbar(im, cax=cax)
cb.set_label("voltage RMSE (mV)", fontsize=15)
cb.ax.tick_params(labelsize=13)
plt.savefig("meta_rmse_heatmap_bySOC_2x2.png", dpi=120)
plt.close()
print("saved meta_rmse_heatmap_bySOC_2x2.png")
