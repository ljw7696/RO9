# %% Explore model discrepancies: clean SPMe vs SPMe+discrepancy across profiles
"""Pick a discrepancy set (DISC) and overlay, at theta_nominal:
     V_clean(SPMe)   vs   V_clean + discrepancy
for every (profile, SOC). The gap (green dV) shows what the discrepancy does and
how it varies across profile & SOC -> use it to decide which discrepancies to keep.
Start with just {"Dsp"}; add "ocp"/"Dsn"/"rc_short"/"rc_long" later."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import numpy as np
import pybamm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
pybamm.set_logging_level("ERROR")
from utils import (make_base_params, make_model, run_model, make_experiment,
                   prepare_sensitivity_inputs, ocp_n_discrepancy, Dsn_discrepancy,
                   Dsp_discrepancy, Vrc_discrepancy, RC_SPECS)

MODEL_OPTIONS = {"surface form": "differential", "contact resistance": "true"}
OCP = "Negative electrode OCP [V]"
DSN = "Negative particle diffusivity [m2.s-1]"; DSP = "Positive particle diffusivity [m2.s-1]"
FP = ["Negative electrode exchange-current density [A.m-2]",
      "Positive electrode exchange-current density [A.m-2]",
      "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]", DSN, DSP]
NOM = {FP[0]: 6.48e-7, FP[1]: 3.42e-6, FP[2]: 0.9487, FP[3]: 1.7694e-10, DSN: 3.3e-14, DSP: 4.0e-15}

# ---- pick the discrepancy to study ----
DISC = {"rc_long", "rc_short"}            # try Dsp ALONE first; later e.g. {"ocp"}, {"Dsn"}, {"rc_short","rc_long"}
SHOW_DFN = False           # also overlay plain DFN at theta_nom (SPMe's real reduction error). SLOW.

PROFILES = ["hppc", "C3_dchg", "1C_dchg", "2C_dchg", "5C_dchg"]
SOCS_BY_PROFILE = {"hppc": [85, 65, 50, 30]}
SOCS_DEFAULT = [90, 70, 55, 40]


def V_at_nom(profile, soc, disc, model="SPMe"):
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, FP, values=NOM)
    if "ocp" in disc:
        p[OCP] = ocp_n_discrepancy
    if "Dsn" in disc:
        p[DSN] = Dsn_discrepancy; iv.pop(DSN, None)
    if "Dsp" in disc:
        p[DSP] = Dsp_discrepancy; iv.pop(DSP, None)
    sol = run_model(make_model(model, options=MODEL_OPTIONS), p,
                    make_experiment(profile_id=profile), inputs=iv)
    t, V = sol["Time [s]"].entries, sol["Voltage [V]"].entries.copy()
    rc = [RC_SPECS[k] for k in disc if k in RC_SPECS]
    if rc:
        V = V - Vrc_discrepancy(t, sol["Current [A]"].entries, rc)
    return t, V


nr = len(PROFILES)
nc = max(len(SOCS_BY_PROFILE.get(p, SOCS_DEFAULT)) for p in PROFILES)
fig, axes = plt.subplots(nr, nc, figsize=(16, 14))
tag = "+".join(sorted(DISC))
for i, profile in enumerate(PROFILES):
    for j, soc in enumerate(SOCS_BY_PROFILE.get(profile, SOCS_DEFAULT)):
        ax = axes[i, j]
        tc, Vc = V_at_nom(profile, soc, set())        # clean SPMe
        td, Vd = V_at_nom(profile, soc, DISC)         # SPMe + discrepancy
        ax.plot(tc, Vc, "k-", lw=1.6, label="V_SPMe")
        ax.plot(td, Vd, "r--", lw=1.6, label=f"V_SPMe+{tag}")
        if SHOW_DFN:
            try:
                tf, Vf = V_at_nom(profile, soc, set(), model="DFN")   # clean DFN (no injected disc)
                ax.plot(tf, Vf, color="tab:blue", ls=":", lw=1.6, label="V_DFN")
            except Exception:
                pass
        dV = interp1d(td, Vd, bounds_error=False, fill_value="extrapolate")(tc) - Vc
        axt = ax.twinx()
        axt.plot(tc, dV * 1e3, color="tab:green", lw=0.8, alpha=0.6)
        axt.set_ylabel(r"$V_{data} - V_{SPMe}$ [mV]", color="tab:green", fontsize=11)
        axt.tick_params(axis="y", labelcolor="tab:green", labelsize=10)
        ax.set_title(f"SOC{soc}%   (max|dV|={np.max(np.abs(dV))*1e3:.0f}mV)", fontsize=12)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)
        if i == nr - 1:
            ax.set_xlabel("Time (s)", fontsize=14)
        if j == 0:
            ax.set_ylabel("Voltage (V)", fontsize=14)
_handles = [
    Line2D([0], [0], color="r", ls="--", lw=1.6, label=r"$V_{data}$"),
    Line2D([0], [0], color="k", lw=1.6, label=r"$V_{SPMe}$"),
]
if SHOW_DFN:
    _handles.append(Line2D([0], [0], color="tab:blue", ls=":", lw=1.6, label="V_DFN"))
_handles.append(Line2D([0], [0], color="tab:green", lw=1.2, label=r"$V_{data} - V_{SPMe}$ (right axis)"))
fig.legend(handles=_handles, loc="upper right", ncol=len(_handles), fontsize=14,
           bbox_to_anchor=(0.995, 0.998), framealpha=0.9)
fig.suptitle("Injected Discrepancies", x=0.35, y=1.0, fontsize=18)
plt.tight_layout(rect=[0.05, 0, 1, 0.97])
for i, profile in enumerate(PROFILES):     # profile as left row-label
    pos = axes[i, 0].get_position()
    fig.text(0.015, (pos.y0 + pos.y1) / 2, profile, rotation=90,
             ha="center", va="center", fontsize=16, fontweight="bold")
plt.savefig("discrepancy_explore.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"saved discrepancy_explore.png  (DISC={sorted(DISC)})")
