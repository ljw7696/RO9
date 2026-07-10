# %% WLTP held-out RMSE table: theta_combined(region) vs theta_hppc(region)
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

results = pickle.load(open("results_bySOC.pkl", "rb"))
WALL = pickle.load(open("meta_wltp_rc_long_rc_short_wide_both.pkl", "rb"))


def region(soc):
    return 1 if soc >= 80 else 2 if soc >= 60 else 3 if soc >= 48 else 4


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


rows = []
for soc in WLTP_SOCS:
    rg = region(soc)
    tW, VW = WALL[soc]["t"], WALL[soc]["truth"]
    theta_comb = {FP[j]: results[rg][LAB[j]]["theta_comb"]
                  if not np.isnan(results[rg][LAB[j]]["theta_comb"]) else NOM[j]
                  for j in range(len(FP))}
    Vc = _guard(lambda: sim_V(theta_comb, "WLTP", soc, tW), WLTP_TIMEOUT)
    rc = float(np.sqrt(np.mean((Vc - VW) ** 2)) * 1e3)
    fivec_lbl = f"5C_dchg_s{ {85: 90, 65: 70, 50: 55, 30: 40}[soc] }"
    rh = WALL[soc]["rmse"][fivec_lbl]
    win = r"$\theta_{combined}$" if rc < rh else r"$\theta_{5C}$"
    diff = abs(rc - rh)
    rows.append((f"WLTP SOC {soc}%", rc, rh, f"{win} (by {diff:.3f} mV)"))
    print(f"WLTP SOC {soc}%:  theta_comb={rc:.3f} mV   theta_5C={rh:.3f} mV   -> {win.strip('$')} by {diff:.3f}")

# ---- render styled table ----
fig, ax = plt.subplots(figsize=(9, 2.6))
ax.axis("off")
col_labels = ["Profile", r"$\theta_{combined}$", r"$\theta_{eff,5C}$", "Winner"]
cell_text = [[r[0], f"{r[1]:.3f} mV", f"{r[2]:.3f} mV", r[3]] for r in rows]
tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(13); tbl.scale(1, 2.0)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("0.3")
    if r == 0:
        cell.set_facecolor("0.15"); cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("0.22" if r % 2 else "0.28")
        cell.set_text_props(color="white")
fig.patch.set_facecolor("0.22")
plt.savefig("meta_wltp_table_bySOC.png", dpi=150, bbox_inches="tight", facecolor="0.22")
print("saved meta_wltp_table_bySOC.png")
