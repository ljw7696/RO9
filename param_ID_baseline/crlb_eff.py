"""CRLB at theta_eff (the best fit per condition) + intersection cloud.

For each (profile, SOC): take theta_eff = lowest-RMSE fit, put a CRLB ellipsoid
on it (Fisher at theta_eff, not at nominal), then:
  - test which conditions' ellipsoids are mutually COMPATIBLE (Mahalanobis<chi2)
  - plot the theta_eff/theta_nom cloud with CRLB bars.
Non-overlapping ellipsoids across conditions = no common theta = misspecification.
Set RATE_TAG / RUN_LABEL. Run cell-by-cell.
"""
# %% config + load theta_eff per condition
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, glob, re, pickle, threading
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import numpy as np
import pybamm
import matplotlib.pyplot as plt
from scipy.stats import chi2
pybamm.set_logging_level("ERROR")
from utils import (make_base_params, make_model, run_model, make_experiment,
                   prepare_sensitivity_inputs, get_sensitivities, compute_fim)

OPTS = {"surface form": "differential", "contact resistance": "true"}
DSN = "Negative particle diffusivity [m2.s-1]"; DSP = "Positive particle diffusivity [m2.s-1]"
FP = ["Negative electrode exchange-current density [A.m-2]",
      "Positive electrode exchange-current density [A.m-2]",
      "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]", DSN, DSP]
LAB = ["k-", "k+", "kappa", "De", "Ds-", "Ds+"]
NOM = np.array([6.48e-7, 3.42e-6, 0.9487, 1.7694e-10, 3.3e-14, 4.0e-15])
SIGMA = 1e-3

RATE_TAG = "rc_long_rc_short"   # "clean" | "rc_long_rc_short" | "Dsn_Dsp_rc_long_rc_short"
RUN_LABEL = "wide"
EXCLUDE_PROFILES = {"sin_1Hz"}  # single-tone: degenerate + can't see rc -> drop from analysis
RATE_DIR = os.path.join("rate_sweep_pkl", RUN_LABEL)
BOUNDS_BY_RUN = {
    "wide":   {FP[0]: (1e-9, 1e-5), FP[1]: (1e-8, 1e-4), FP[2]: (0.1, 3.0),
               FP[3]: (1e-12, 1e-9), DSN: (5e-16, 1e-12), DSP: (5e-16, 1e-12)},
    "narrow": {FP[0]: (1e-7, 1e-4), FP[1]: (1e-7, 1e-4), FP[2]: (0.1, 3.0),
               FP[3]: (1e-12, 1e-9), DSN: (5e-16, 1e-12), DSP: (5e-16, 1e-12)},
}
BOUNDS = BOUNDS_BY_RUN[RUN_LABEL]
LNRANGE = np.array([np.log(BOUNDS[p][1]) - np.log(BOUNDS[p][0]) for p in FP])
CHI2_CRIT = chi2.ppf(0.95, df=len(FP))
SOLVE_TIMEOUT = 90.0

# theta_eff = best (lowest RMSE) fit per condition
eff = {}
for f in glob.glob(os.path.join(RATE_DIR, f"rate_{RATE_TAG}_*.pkl")):
    m = re.search(rf"rate_{RATE_TAG}_(.+?)_soc(\d+)\.pkl", os.path.basename(f))
    if not m:
        continue
    if m.group(1) in EXCLUDE_PROFILES:
        continue
    res = pickle.load(open(f, "rb"))
    good = [r for r in res if not r.get("stalled")]
    if not good:
        continue
    best = min(good, key=lambda r: r["rmse_mV"])
    eff[(m.group(1), int(m.group(2)))] = dict(
        theta_hat=best["theta_hat"], bar=np.asarray(best["theta_bar_hat"], float),
        rmse=best["rmse_mV"])
CONDS = sorted(eff)
print(f"theta_eff loaded for {len(CONDS)} conditions from {RATE_DIR} (tag {RATE_TAG})")


# %% CRLB at each theta_eff (thread-guarded against stalls)
def _core(theta_hat, soc, profile):
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, FP, values=theta_hat)
    solver = pybamm.IDAKLUSolver(options={"max_num_steps": 20000})
    sol = run_model(make_model("SPMe", options=OPTS), p,
                    make_experiment(profile_id=profile), inputs=iv,
                    calculate_sensitivities=True, solver=solver)
    Sens = get_sensitivities(sol, "Voltage [V]", FP, theta_values=theta_hat, normalize=True)
    F = compute_fim(Sens, FP, sigma=SIGMA)
    cov_norm = np.linalg.pinv(F)
    crlb_rel = np.sqrt(np.clip(np.diag(cov_norm), 0, None))
    cov_bar = cov_norm / np.outer(LNRANGE, LNRANGE)
    return crlb_rel, cov_bar, float(np.linalg.cond(F))

def crlb_at(theta_hat, soc, profile):
    box = {}
    def work():
        try:
            box["v"] = _core(theta_hat, soc, profile)
        except BaseException as e:
            box["e"] = e
    th = threading.Thread(target=work, daemon=True); th.start(); th.join(SOLVE_TIMEOUT)
    if th.is_alive():
        raise TimeoutError("stalled")
    if "e" in box:
        raise box["e"]
    return box["v"]

print(f"\n{'condition':16}{'RMSE':>7}   " + "".join(f"{l:>8}" for l in LAB) + f"{'cond(F)':>10}")
for k in CONDS:
    try:
        cr, cov, cond = crlb_at(eff[k]["theta_hat"], k[1], k[0])
        eff[k].update(crlb=cr, cov=cov, cond=cond, ok=True)
        ratio = np.array([eff[k]["theta_hat"][p] for p in FP]) / NOM
        print(f"{k[0]+' '+str(k[1]):16}{eff[k]['rmse']:>6.2f}m   "
              + "".join(f"{x:>8.2f}" for x in ratio) + f"{cond:>10.1e}")
    except Exception as e:
        eff[k]["ok"] = False
        print(f"{k[0]+' '+str(k[1]):16}  CRLB failed: {type(e).__name__}")
OK = [k for k in CONDS if eff[k].get("ok")]
print(f"\nCRLB ok for {len(OK)}/{len(CONDS)} conditions")


# %% intersection: which theta_eff ellipsoids are mutually compatible
n = len(OK)
compat = np.zeros((n, n), bool)
for a in range(n):
    for b in range(n):
        dmu = eff[OK[a]]["bar"] - eff[OK[b]]["bar"]
        S = eff[OK[a]]["cov"] + eff[OK[b]]["cov"]
        d2 = float(dmu @ np.linalg.pinv(S) @ dmu)
        compat[a, b] = d2 < CHI2_CRIT
deg = compat.sum(1) - 1
order = np.argsort(-deg)
print(f"\nchi2(95%, df={len(FP)}) = {CHI2_CRIT:.1f}.  degree = # other conditions each intersects")
for a in order:
    k = OK[a]
    print(f"  {k[0]+' soc'+str(k[1]):16} degree={deg[a]:>3} / {n-1}")
hub = int(order[0]); consensus = [hub]
for a in order[1:]:
    if all(compat[a, c] for c in consensus):
        consensus.append(int(a))
print(f"\nlargest mutually-compatible set: {len(consensus)}/{n} conditions share a common theta")


# %% cloud: theta_eff/theta_nom per parameter, with CRLB bars
# Blue band = across-condition spread of theta_eff (the empirical "joint
# distribution"), shown only for parameters that actually have a spread
# (default: Ds- only; others collapse onto 1 so a band is meaningless).
SHADE_PARAMS = ["Ds-"]                      # which params to shade; add more if wanted
BAND_PCT = (16, 84)                         # percentile band (16-84 ~ +/-1 sigma)

fig, ax = plt.subplots(figsize=(11, 6))
ratios = {l: np.array([eff[k]["theta_hat"][FP[j]] / NOM[j] for k in OK])
          for j, l in enumerate(LAB)}      # all theta_eff/theta_nom per param

# shaded joint-distribution band (Ds- only, by default)
for l in SHADE_PARAMS:
    j = LAB.index(l)
    lo, hi = np.percentile(ratios[l], BAND_PCT)
    med = np.median(ratios[l])
    ax.fill_between([j - 0.42, j + 0.42], lo, hi, color="tab:blue", alpha=0.15, zorder=0,
                    label=f"{l} spread ({BAND_PCT[0]}-{BAND_PCT[1]}%)")
    ax.plot([j - 0.42, j + 0.42], [med, med], color="tab:blue", lw=1.5, alpha=0.6, zorder=1)

# the theta_eff points + CRLB bars (single neutral color; consensus coloring dropped)
for a, k in enumerate(OK):
    r = np.array([eff[k]["theta_hat"][p] for p in FP]) / NOM
    x = np.arange(len(FP)) + np.random.default_rng(a).uniform(-0.28, 0.28)
    ax.errorbar(x, r, yerr=r * eff[k]["crlb"], fmt="o", ms=5, color="dimgray",
                alpha=0.6, elinewidth=0.8, capsize=2, zorder=3)
ax.axhline(1.0, color="k", ls="--", lw=1.2, label="theta_nom (ratio=1)")
ax.set_yscale("log"); ax.set_xticks(range(len(FP))); ax.set_xticklabels(LAB)
ax.set_ylabel("theta_eff / theta_nom")
ax.set_title(f"[{RATE_TAG} / {RUN_LABEL}] theta_eff cloud + CRLB "
             f"(blue band = across-condition spread)")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("crlb_eff_cloud.png", dpi=120, bbox_inches="tight")
plt.show()
print("saved crlb_eff_cloud.png")


# %% between-profile scatter (s) vs within-profile CRLB (w), per parameter
# s = std of ln(theta_eff/theta_nom) across conditions  (BETWEEN-profile spread)
# w = mean within-profile CRLB_rel (= sigma/theta = the plotted error bar)
# both are relative std -> directly comparable.  s/w:
#   ~1  -> CRLB COVERS the variation (consistent, well-calibrated)
#   >>1 -> variation EXCEEDS the CRLB (over-dispersed -> structural / CRLB too tight)
print(f"\nbetween-profile scatter vs within-profile CRLB  ({len(OK)} conditions)")
print(f"{'param':6}{'s(between)':>12}{'w(within)':>12}{'s/w':>8}   coverage")
for j, l in enumerate(LAB):
    logr = np.log(np.array([eff[k]["theta_hat"][FP[j]] / NOM[j] for k in OK]))
    s = float(np.std(logr))                                  # between-profile relative std
    w = float(np.mean([eff[k]["crlb"][j] for k in OK]))      # mean within-profile CRLB_rel
    ratio = s / w if w > 0 else float("inf")
    tag = ("CRLB covers" if ratio < 1.5 else
           "borderline" if ratio < 3 else "OVER-dispersed")
    print(f"{l:6}{100*s:>11.2f}%{100*w:>11.2f}%{ratio:>8.1f}   {tag}")


# %% per-parameter theta_eff vs C-rate (CC discharge profiles), colored by SOC
# 6 subplots, one per fit param: y = theta_eff/theta_nom (log), x = C-rate,
# color = SOC level, error bars = CRLB (sigma/theta). Only the CONSTANT-current
# discharges have a single C-rate, so HPPC and the sine are excluded here.
CRATE = {"C3_dchg": 1.0 / 3.0, "1C_dchg": 1.0, "2C_dchg": 2.0, "5C_dchg": 5.0}
CRATE_LABEL = {1.0 / 3.0: "C/3", 1.0: "1C", 2.0: "2C", 5.0: "5C"}
soc_levels = sorted({k[1] for k in OK if k[0] in CRATE})
cmap = plt.get_cmap("viridis")
scolor = {s: cmap(i / max(1, len(soc_levels) - 1)) for i, s in enumerate(soc_levels)}

fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
for j, l in enumerate(LAB):
    ax = axes[j // 3][j % 3]
    for s in soc_levels:
        xs, ys, es = [], [], []
        for prof, cr in CRATE.items():
            k = (prof, s)
            if k in OK:                      # only conditions with a successful CRLB
                r = eff[k]["theta_hat"][FP[j]] / NOM[j]
                xs.append(cr); ys.append(r); es.append(r * eff[k]["crlb"][j])
        if xs:
            o = np.argsort(xs)
            xs, ys, es = np.array(xs)[o], np.array(ys)[o], np.array(es)[o]
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=5, capsize=3, lw=1.3,
                        color=scolor[s], label=f"SOC {s}%")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.7)     # truth
    ax.set_title(l, fontweight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(list(CRATE.values()))
    ax.set_xticklabels([CRATE_LABEL[c] for c in CRATE.values()])
    ax.grid(alpha=0.3, which="both")
    if j % 3 == 0:
        ax.set_ylabel("theta_eff / theta_nom")
    if j // 3 == 1:
        ax.set_xlabel("C-rate")
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=len(soc_levels), fontsize=9,
           title="SOC")
fig.suptitle(f"[{RATE_TAG}/{RUN_LABEL}] theta_eff vs C-rate per parameter "
             f"(dashed = truth, color = SOC)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("crlb_eff_vs_crate.png", dpi=120, bbox_inches="tight")
plt.show()
print("saved crlb_eff_vs_crate.png")
