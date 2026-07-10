# %% Show r = theta / theta_nom per start (sorted by mean voltage RMSE) + plot best theta_joint fits
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, pickle
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import numpy as np

FP = ["Negative electrode exchange-current density [A.m-2]",
      "Positive electrode exchange-current density [A.m-2]",
      "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]",
      "Negative particle diffusivity [m2.s-1]", "Positive particle diffusivity [m2.s-1]"]
LAB = ["k-", "k+", "kappa", "De", "Ds-", "Ds+"]
NOM = np.array([6.48e-7, 3.42e-6, 0.9487, 1.7694e-10, 3.3e-14, 4.0e-15])
RLAB = {1: "SOC~85-90", 2: "SOC~65-70", 3: "SOC~50-55", 4: "SOC~30-40"}

OUT = pickle.load(open("joint_fit_results.pkl", "rb"))
comb = pickle.load(open("results_bySOC_DLo.pkl", "rb"))

# ================= 1) ratio table (r = theta/theta_nom, sorted by mean RMSE) =================
for rg in sorted(OUT):
    R = sorted(OUT[rg]["all"], key=lambda r: r["rmse_mV"])   # ascending by mean RMSE
    print("=" * 90)
    print(f"REGION {rg} ({RLAB[rg]})   r = theta / theta_nom   (sorted by mean voltage RMSE)")
    print("=" * 90)
    print(f"{'rank':>4}{'start':>6}{'meanRMSE':>10}   " + "".join(f"{l:>9}" for l in LAB))
    rc = [comb[rg][LAB[j]]["theta_comb"] / NOM[j] for j in range(6)]
    print(f"{'comb':>4}{'--':>6}{'--':>10}   " + "".join(f"{v:>9.3f}" for v in rc))
    print("-" * 90)
    for i, r in enumerate(R, 1):
        if r.get("stalled"):
            print(f"{i:>4}{r['start_id']:>6}{'STALLED':>10}")
            continue
        ratios = [r["theta_hat"][FP[j]] / NOM[j] for j in range(6)]
        print(f"{i:>4}{r['start_id']:>6}{r['rmse_mV']:>10.3f}   "
              + "".join(f"{v:>9.3f}" for v in ratios))
    print()


# ================= 2) plot: best theta_joint fit to the region's training profiles =================
import pybamm
pybamm.set_logging_level("ERROR")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import (make_base_params, make_model, run_model, make_experiment,
                   prepare_sensitivity_inputs, Vrc_discrepancy, RC_SPECS)

OPTS = {"surface form": "differential", "contact resistance": "true"}
TRUTH = pickle.load(open("meta_genmatrix_rc_long_rc_short_wide_both.pkl", "rb"))["truth"]


def sim_V(theta, profile, soc, t_ref):
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, FP, values=theta)
    sol = run_model(make_model("SPMe", options=OPTS), p, make_experiment(profile_id=profile),
                    inputs=iv, solver=pybamm.IDAKLUSolver(options={"max_num_steps": 50000}))
    return np.interp(t_ref, sol["Time [s]"].entries, sol["Voltage [V]"].entries)


_NOM_D = {FP[j]: NOM[j] for j in range(6)}
_RC = [RC_SPECS["rc_short"], RC_SPECS["rc_long"]]


def make_truth(profile, soc):                # SPMe(theta_nom) + rc discrepancy + noise
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, FP, values=_NOM_D)
    sol = run_model(make_model("SPMe", options=OPTS), p, make_experiment(profile_id=profile),
                    inputs=iv, solver=pybamm.IDAKLUSolver(options={"max_num_steps": 50000}))
    t = sol["Time [s]"].entries; V = sol["Voltage [V]"].entries.copy()
    V = V - Vrc_discrepancy(t, sol["Current [A]"].entries, _RC)
    V = V + np.random.default_rng(0).normal(0.0, 1e-3, size=V.shape)
    return t, V


OV_HANDLES = [Line2D([0], [0], color="tab:blue", lw=2, label=r"$V_{data}$"),
              Line2D([0], [0], color="red", ls="--", lw=2, label=r"$V_{SPMe}(\theta_{joint})$"),
              Line2D([0], [0], color="tab:green", ls="--", lw=2, label=r"$V_{SPMe}(\theta_{comb})$")]
ERR_HANDLES = [Line2D([0], [0], color="red", lw=2, label=r"$V_{data}-V_{SPMe}(\theta_{joint})$"),
               Line2D([0], [0], color="tab:green", lw=2, label=r"$V_{data}-V_{SPMe}(\theta_{comb})$")]

TR = {}                                # training RMSEs: TR[rg] = [(cond, rj, rc), ...]
WL = {}                                # WLTP RMSEs:     WL[soc] = (rj, rc)
regions = sorted(OUT)
ncol = max(len(OUT[rg]["conds"]) for rg in regions)
fig, axes = plt.subplots(len(regions), ncol, figsize=(4 * ncol, 3 * len(regions)))       # overlay
fige, axese = plt.subplots(len(regions), ncol, figsize=(4 * ncol, 3 * len(regions)))     # error
for ri, rg in enumerate(regions):
    conds = OUT[rg]["conds"]
    theta_j = OUT[rg]["best"]["theta_hat"]
    theta_c = {FP[j]: comb[rg][LAB[j]]["theta_comb"] for j in range(6)}
    for ci in range(ncol):
        ax, axe = axes[ri][ci], axese[ri][ci]
        if ci >= len(conds):
            ax.axis("off"); axe.axis("off"); continue
        prof, soc = conds[ci]
        t_ref, V_data = TRUTH[(prof, soc)]
        Vj = sim_V(theta_j, prof, soc, t_ref)
        Vc = sim_V(theta_c, prof, soc, t_ref)
        rj = np.sqrt(np.mean((Vj - V_data) ** 2)) * 1e3
        rc = np.sqrt(np.mean((Vc - V_data) ** 2)) * 1e3
        TR.setdefault(rg, []).append((f"{prof}_s{soc}", rj, rc))
        ttl = f"{prof}_s{soc}  (joint {rj:.1f} / comb {rc:.1f} mV)"
        # --- overlay ---
        ax.plot(t_ref, V_data, color="tab:blue", lw=1.8, zorder=2)
        ax.plot(t_ref, Vj, color="red", ls="--", lw=1.6, zorder=3)
        ax.plot(t_ref, Vc, color="tab:green", ls="--", lw=1.6, zorder=4)
        ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3); ax.tick_params(labelsize=10)
        # --- error ---
        axe.axhline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
        axe.plot(t_ref, (V_data - Vj) * 1e3, color="red", lw=1.2, zorder=3)
        axe.plot(t_ref, (V_data - Vc) * 1e3, color="tab:green", lw=1.2, zorder=4)
        axe.set_title(ttl, fontsize=10); axe.grid(alpha=0.3); axe.tick_params(labelsize=10)
        if ci == 0:
            ax.set_ylabel(f"{RLAB[rg]}\nVoltage (V)", fontsize=12)
            axe.set_ylabel(f"{RLAB[rg]}\nV error (mV)", fontsize=12)
        if ri == len(regions) - 1:
            ax.set_xlabel("Time (s)", fontsize=12); axe.set_xlabel("Time (s)", fontsize=12)
fig.legend(handles=OV_HANDLES, loc="upper right", fontsize=13, ncol=3, bbox_to_anchor=(0.99, 1.0))
fig.suptitle(r"$\theta_{joint}$ vs $\theta_{comb}$ fit to training profiles, per SOC region", fontsize=15, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("joint_fit_overlays_DLo.png", dpi=120, bbox_inches="tight")
fige.legend(handles=ERR_HANDLES, loc="upper right", fontsize=13, ncol=2, bbox_to_anchor=(0.99, 1.0))
fige.suptitle(r"Voltage error $V_{data}-V_{SPMe}(\theta)$: training profiles, per SOC region", fontsize=15, y=1.0)
fige.tight_layout(rect=[0, 0, 1, 0.97])
fige.savefig("joint_fit_errors_DLo.png", dpi=120, bbox_inches="tight")
print("saved joint_fit_overlays.png, joint_fit_errors.png")


# ================= 3) WLTP held-out: theta_joint vs theta_comb (never fit to WLTP) =================
def _region(soc):
    return 1 if soc >= 80 else 2 if soc >= 60 else 3 if soc >= 48 else 4


WALL = pickle.load(open("meta_wltp_rc_long_rc_short_wide_both.pkl", "rb"))
WLTP_SOCS = [85, 65, 50, 30]        # = the 4 region SOC levels
fig, axes = plt.subplots(2, 2, figsize=(16, 9))       # overlay
fige, axese = plt.subplots(2, 2, figsize=(16, 9))     # error
print("\nWLTP held-out RMSE (mV):")
for ax, axe, soc in zip(axes.ravel(), axese.ravel(), WLTP_SOCS):
    rg = _region(soc)
    tW, VW = WALL[soc]["t"], WALL[soc]["truth"]
    theta_j = OUT[rg]["best"]["theta_hat"]
    theta_c = {FP[j]: comb[rg][LAB[j]]["theta_comb"] for j in range(6)}
    Vj = sim_V(theta_j, "WLTP", soc, tW)
    Vc = sim_V(theta_c, "WLTP", soc, tW)
    rj = np.sqrt(np.mean((Vj - VW) ** 2)) * 1e3
    rc = np.sqrt(np.mean((Vc - VW) ** 2)) * 1e3
    WL[soc] = (rj, rc)
    ttl = f"WLTP SOC {soc}%  (joint {rj:.1f} / comb {rc:.1f} mV)"
    ax.plot(tW, VW, color="tab:blue", lw=1.6, zorder=2)
    ax.plot(tW, Vj, color="red", ls="--", lw=1.4, zorder=3)
    ax.plot(tW, Vc, color="tab:green", ls="--", lw=1.4, zorder=4)
    ax.set_title(ttl, fontsize=12); ax.grid(alpha=0.3); ax.tick_params(labelsize=10)
    ax.set_ylabel("Voltage (V)", fontsize=12); ax.set_xlabel("Time (s)", fontsize=12)
    axe.axhline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
    axe.plot(tW, (VW - Vj) * 1e3, color="red", lw=1.0, zorder=3)
    axe.plot(tW, (VW - Vc) * 1e3, color="tab:green", lw=1.0, zorder=4)
    axe.set_title(ttl, fontsize=12); axe.grid(alpha=0.3); axe.tick_params(labelsize=10)
    axe.set_ylabel("V error (mV)", fontsize=12); axe.set_xlabel("Time (s)", fontsize=12)
fig.legend(handles=OV_HANDLES, loc="upper right", fontsize=13, ncol=3, bbox_to_anchor=(0.99, 1.0))
fig.suptitle(r"WLTP held-out: $\theta_{joint}$ vs $\theta_{comb}$", fontsize=15, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("joint_wltp_overlays_DLo.png", dpi=120, bbox_inches="tight")
fige.legend(handles=ERR_HANDLES, loc="upper right", fontsize=13, ncol=2, bbox_to_anchor=(0.99, 1.0))
fige.suptitle(r"WLTP held-out voltage error $V_{data}-V_{SPMe}(\theta)$", fontsize=15, y=1.0)
fige.tight_layout(rect=[0, 0, 1, 0.96])
fige.savefig("joint_wltp_errors_DLo.png", dpi=120, bbox_inches="tight")
print("saved joint_wltp_overlays.png, joint_wltp_errors.png")


# ================= 4) voltage RMSE summary (mV) =================
print("\n" + "=" * 60)
print("VOLTAGE RMSE (mV):  theta_joint  vs  theta_comb")
print("=" * 60)
print("\n--- TRAINING profiles ---")
print(f"{'condition':16}{'joint':>9}{'comb':>9}")
allj, allc = [], []
for rg in regions:
    for cond, rj, rc in TR[rg]:
        print(f"{cond:16}{rj:>9.2f}{rc:>9.2f}")
        allj.append(rj); allc.append(rc)
    mj = np.mean([x[1] for x in TR[rg]]); mc = np.mean([x[2] for x in TR[rg]])
    print(f"  -> region {rg} MEAN : {mj:>7.2f}{mc:>9.2f}")
print(f"{'TRAIN overall mean':16}{np.mean(allj):>9.2f}{np.mean(allc):>9.2f}")

print("\n--- WLTP held-out ---")
print(f"{'condition':16}{'joint':>9}{'comb':>9}")
for soc in WLTP_SOCS:
    rj, rc = WL[soc]
    print(f"{'WLTP_s'+str(soc):16}{rj:>9.2f}{rc:>9.2f}")
print(f"{'WLTP mean':16}{np.mean([WL[s][0] for s in WLTP_SOCS]):>9.2f}"
      f"{np.mean([WL[s][1] for s in WLTP_SOCS]):>9.2f}")

# ----- other held-out drive cycles: FUDS, US06 (truth generated fresh) -----
for dc in ["FUDS", "US06"]:
    print(f"\n--- {dc} held-out ---")
    print(f"{'condition':16}{'joint':>9}{'comb':>9}")
    rjs, rcs = [], []
    for soc in WLTP_SOCS:
        rg = _region(soc)
        t_ref, V_data = make_truth(dc, soc)
        theta_j = OUT[rg]["best"]["theta_hat"]
        theta_c = {FP[j]: comb[rg][LAB[j]]["theta_comb"] for j in range(6)}
        rj = np.sqrt(np.mean((sim_V(theta_j, dc, soc, t_ref) - V_data) ** 2)) * 1e3
        rc = np.sqrt(np.mean((sim_V(theta_c, dc, soc, t_ref) - V_data) ** 2)) * 1e3
        rjs.append(rj); rcs.append(rc)
        print(f"{dc+'_s'+str(soc):16}{rj:>9.2f}{rc:>9.2f}")
    print(f"{dc+' mean':16}{np.mean(rjs):>9.2f}{np.mean(rcs):>9.2f}")
