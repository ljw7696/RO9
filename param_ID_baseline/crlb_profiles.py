# %% CRLB per parameter, per (profile, SOC), at theta_nominal
"""Which profile identifies which parameter? Computes the Cramer-Rao lower bound
(relative, = sigma_theta/theta) for each of the 6 parameters at theta_nominal on
the clean SPMe, for every (profile, SOC). Small CRLB = well identified.
Prints: (1) CRLB_rel table, (2) per-condition parameter ranking, (3) best
condition per parameter. Analytic sensitivities; ~1-3 s per condition."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import numpy as np
import pybamm
pybamm.set_logging_level("ERROR")
from utils import (make_base_params, make_model, run_model, make_experiment,
                   prepare_sensitivity_inputs, get_sensitivities, compute_fim)

OPTS = {"surface form": "differential", "contact resistance": "true"}
DSN = "Negative particle diffusivity [m2.s-1]"; DSP = "Positive particle diffusivity [m2.s-1]"
FP = ["Negative electrode exchange-current density [A.m-2]",
      "Positive electrode exchange-current density [A.m-2]",
      "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]", DSN, DSP]
LAB = ["k-", "k+", "kappa", "De", "Ds-", "Ds+"]
NOM = {FP[0]: 6.48e-7, FP[1]: 3.42e-6, FP[2]: 0.9487, FP[3]: 1.7694e-10, DSN: 3.3e-14, DSP: 4.0e-15}
SIGMA = 1e-3

# (profile, SOC%) grid -- matches your sweeps
CONDS = [("hppc", s) for s in (85, 65, 50, 30)]
for prof in ("C3_dchg", "1C_dchg", "2C_dchg", "5C_dchg"):
    CONDS += [(prof, s) for s in (90, 70, 55, 40)]


def crlb_rel(profile, soc):
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, FP, values=NOM)
    sol = run_model(make_model("SPMe", options=OPTS), p,
                    make_experiment(profile_id=profile), inputs=iv,
                    calculate_sensitivities=True)
    Sens = get_sensitivities(sol, "Voltage [V]", FP, theta_values=NOM, normalize=True)
    F = compute_fim(Sens, FP, sigma=SIGMA)          # FIM for ln(theta)
    crlb = np.sqrt(np.clip(np.diag(np.linalg.pinv(F)), 0, None))
    return crlb, float(np.linalg.cond(F)), F


rows = {}
print("computing CRLB per condition...", flush=True)
for (prof, soc) in CONDS:
    cr, cond, F = crlb_rel(prof, soc)
    rows[(prof, soc)] = (cr, cond, F)

# (1) table
print("\nCRLB_rel (%) at theta_nominal  [small = well identified]")
print(f"{'condition':16}" + "".join(f"{l:>8}" for l in LAB) + f"{'cond(F)':>10}{'cond(J)':>9}")
for k in CONDS:
    cr, cond, _F = rows[k]
    print(f"{k[0]+' '+str(k[1]):16}" + "".join(f"{100*x:>7.1f}%" for x in cr)
          + f"{cond:>10.1e}{np.sqrt(cond):>9.0f}")   # cond(J)=sqrt(cond(F)); compare to 1e3-1e4

# (2) per-condition ranking (best -> worst identified)
print("\nper-condition ranking (best -> worst identified):")
for k in CONDS:
    cr = rows[k][0]
    order = np.argsort(cr)
    print(f"  {k[0]+' '+str(k[1]):14}: " + "  <  ".join(f"{LAB[i]}({100*cr[i]:.1f}%)" for i in order))

# (3) best condition for each parameter
print("\nbest condition per parameter (lowest CRLB_rel):")
for j, l in enumerate(LAB):
    best = min(CONDS, key=lambda k: rows[k][0][j])
    worst = max(CONDS, key=lambda k: rows[k][0][j])
    print(f"  {l:6} best: {best[0]+' '+str(best[1]):14} ({100*rows[best][0][j]:5.1f}%)"
          f"   worst: {worst[0]+' '+str(worst[1]):14} ({100*rows[worst][0][j]:6.1f}%)")

# (4) FULL FIM eigen-spectrum per condition: all 6 axes of the uncertainty ellipsoid.
# For each of the 6 eigenvectors (directions), unc = 100/sqrt(eigenvalue) = the
# relative uncertainty along that direction (%). Sorted best (small unc) -> worst
# (large unc = flattest). The 6 numbers are NOT per-parameter; each is a direction.
print("\nFIM eigen-spectrum per condition: 6 direction-uncertainties 100/sqrt(lambda) (%)")
print("  (ellipsoid axes, best->worst; the biggest = flattest = worst-identified direction)")
print(f"{'condition':16}" + "".join(f"{'ax'+str(i+1):>8}" for i in range(len(LAB))))
for k in CONDS:
    F = rows[k][2]
    w, _ = np.linalg.eigh(F)                        # ascending eigenvalues
    unc = np.sort(100.0 / np.sqrt(np.clip(w, 1e-30, None)))   # ascending unc = best->worst
    print(f"{k[0]+' '+str(k[1]):16}" + "".join(f"{u:>8.2f}" for u in unc))

# and which parameters make up each condition's WORST (flattest) axis
print("\nworst (flattest) axis composition per condition  [eigvec weights, |w|>0.15]:")
print(f"{'condition':16}{'unc%':>8}   flat combination")
for k in CONDS:
    F = rows[k][2]
    w, V = np.linalg.eigh(F)
    lam = max(float(w[0]), 1e-30)
    vec = V[:, 0]
    if vec[np.argmax(np.abs(vec))] < 0:
        vec = -vec
    order = np.argsort(-np.abs(vec))
    combo = "  ".join(f"{vec[i]:+.2f}*{LAB[i]}" for i in order if abs(vec[i]) > 0.15)
    print(f"{k[0]+' '+str(k[1]):16}{100/np.sqrt(lam):>7.1f}%   {combo}")

# (5) WHY cond is large: cond(J) = worst-axis / best-axis.
# If the big cond comes from a tiny BEST axis (a super-identified param), it's
# benign; if from a huge WORST axis, it's a real identifiability failure.
def _dom(vec):
    return LAB[int(np.argmax(np.abs(vec)))]
print("\nwhy cond is large:  cond(J) = worst-axis unc / best-axis unc")
print(f"{'condition':16}{'best axis':>18}{'worst axis':>18}{'cond(J)':>9}")
for k in CONDS:
    w, V = np.linalg.eigh(rows[k][2])
    best_u = 100 / np.sqrt(max(float(w[-1]), 1e-30))   # largest eigenvalue = best axis
    worst_u = 100 / np.sqrt(max(float(w[0]), 1e-30))   # smallest eigenvalue = worst axis
    bp, wp = _dom(V[:, -1]), _dom(V[:, 0])
    print(f"{k[0]+' '+str(k[1]):16}{bp+' '+format(best_u,'.4f')+'%':>18}"
          f"{wp+' '+format(worst_u,'.1f')+'%':>18}{worst_u/best_u:>9.0f}")
