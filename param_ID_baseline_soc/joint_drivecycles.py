# %% Held-out drive cycles (WLTP, FUDS, US06): theta_joint vs theta_comb, per SOC region.
# Truth = SPMe(theta_nom) + rc_short + rc_long discrepancy + noise (same as the sweep).
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, pickle
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
                   prepare_sensitivity_inputs, Vrc_discrepancy, RC_SPECS)

FP = ["Negative electrode exchange-current density [A.m-2]",
      "Positive electrode exchange-current density [A.m-2]",
      "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]",
      "Negative particle diffusivity [m2.s-1]", "Positive particle diffusivity [m2.s-1]"]
LAB = ["k-", "k+", "kappa", "De", "Ds-", "Ds+"]
NOM = np.array([6.48e-7, 3.42e-6, 0.9487, 1.7694e-10, 3.3e-14, 4.0e-15])
NOM_D = {FP[j]: NOM[j] for j in range(6)}
OPTS = {"surface form": "differential", "contact resistance": "true"}
NOISE_STD = 1e-3
RC = [RC_SPECS["rc_short"], RC_SPECS["rc_long"]]     # rc_long_rc_short discrepancy
DRIVE_CYCLES = ["WLTP", "FUDS", "US06"]
SOCS = [85, 65, 50, 30]                              # = the 4 region SOC levels

OUT = pickle.load(open("joint_fit_results.pkl", "rb"))
comb = pickle.load(open("results_bySOC.pkl", "rb"))


def region(soc):
    return 1 if soc >= 80 else 2 if soc >= 60 else 3 if soc >= 48 else 4


def make_truth(profile, soc):
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, FP, values=NOM_D)
    sol = run_model(make_model("SPMe", options=OPTS), p, make_experiment(profile_id=profile),
                    inputs=iv, solver=pybamm.IDAKLUSolver(options={"max_num_steps": 50000}))
    t = sol["Time [s]"].entries; V = sol["Voltage [V]"].entries.copy()
    V = V - Vrc_discrepancy(t, sol["Current [A]"].entries, RC)
    V = V + np.random.default_rng(0).normal(0.0, NOISE_STD, size=V.shape)
    return t, V


def sim_V(theta, profile, soc, t_ref):
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, FP, values=theta)
    sol = run_model(make_model("SPMe", options=OPTS), p, make_experiment(profile_id=profile),
                    inputs=iv, solver=pybamm.IDAKLUSolver(options={"max_num_steps": 50000}))
    return np.interp(t_ref, sol["Time [s]"].entries, sol["Voltage [V]"].entries)


OV = [Line2D([0], [0], color="tab:blue", lw=2, label=r"$V_{data}$"),
      Line2D([0], [0], color="red", ls="--", lw=2, label=r"$V_{SPMe}(\theta_{joint})$"),
      Line2D([0], [0], color="tab:green", ls="--", lw=2, label=r"$V_{SPMe}(\theta_{comb})$")]
ER = [Line2D([0], [0], color="red", lw=2, label=r"$V_{data}-V_{SPMe}(\theta_{joint})$"),
      Line2D([0], [0], color="tab:green", lw=2, label=r"$V_{data}-V_{SPMe}(\theta_{comb})$")]

for dc in DRIVE_CYCLES:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))       # overlay
    fige, axese = plt.subplots(2, 2, figsize=(16, 9))     # error
    print(f"\n{dc} held-out RMSE (mV):  {'joint':>8}{'comb':>8}")
    for ax, axe, soc in zip(axes.ravel(), axese.ravel(), SOCS):
        rg = region(soc)
        t_ref, V_data = make_truth(dc, soc)
        theta_j = OUT[rg]["best"]["theta_hat"]
        theta_c = {FP[j]: comb[rg][LAB[j]]["theta_comb"] for j in range(6)}
        Vj = sim_V(theta_j, dc, soc, t_ref)
        Vc = sim_V(theta_c, dc, soc, t_ref)
        rj = np.sqrt(np.mean((Vj - V_data) ** 2)) * 1e3
        rc = np.sqrt(np.mean((Vc - V_data) ** 2)) * 1e3
        print(f"  {dc} SOC {soc}% (region {rg}): {rj:>8.2f}{rc:>8.2f}")
        ttl = f"{dc} SOC {soc}%  (joint {rj:.1f} / comb {rc:.1f} mV)"
        ax.plot(t_ref, V_data, color="tab:blue", lw=1.6, zorder=2)
        ax.plot(t_ref, Vj, color="red", ls="--", lw=1.4, zorder=3)
        ax.plot(t_ref, Vc, color="tab:green", ls="--", lw=1.4, zorder=4)
        ax.set_title(ttl, fontsize=12); ax.grid(alpha=0.3); ax.tick_params(labelsize=10)
        ax.set_ylabel("Voltage (V)", fontsize=12); ax.set_xlabel("Time (s)", fontsize=12)
        axe.axhline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
        axe.plot(t_ref, (V_data - Vj) * 1e3, color="red", lw=1.0, zorder=3)
        axe.plot(t_ref, (V_data - Vc) * 1e3, color="tab:green", lw=1.0, zorder=4)
        axe.set_title(ttl, fontsize=12); axe.grid(alpha=0.3); axe.tick_params(labelsize=10)
        axe.set_ylabel("V error (mV)", fontsize=12); axe.set_xlabel("Time (s)", fontsize=12)
    fig.legend(handles=OV, loc="upper right", fontsize=13, ncol=3, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(rf"{dc} held-out: $\theta_{{joint}}$ vs $\theta_{{comb}}$", fontsize=15, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"joint_{dc.lower()}_overlays.png", dpi=120, bbox_inches="tight")
    fige.legend(handles=ER, loc="upper right", fontsize=13, ncol=2, bbox_to_anchor=(0.99, 1.0))
    fige.suptitle(rf"{dc} held-out voltage error $V_{{data}}-V_{{SPMe}}(\theta)$", fontsize=15, y=1.0)
    fige.tight_layout(rect=[0, 0, 1, 0.96])
    fige.savefig(f"joint_{dc.lower()}_errors.png", dpi=120, bbox_inches="tight")
    plt.close(fig); plt.close(fige)
    print(f"  saved joint_{dc.lower()}_overlays.png, joint_{dc.lower()}_errors.png")
