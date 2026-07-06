# %% Per-parameter modality among the near-best fits (within +RMSE_TOL of best)
"""For each (profile, SOC), keep fits within +RMSE_TOL of the best RMSE, then for
EACH parameter separately count its 1D modes (distinct value-clusters).
Symbols per cell:
   1     = unimodal & tight   (identified)
   flat  = unimodal but WIDE  (continuous ridge -> non-identifiable, not discrete)
   2,3.. = genuinely bi/multimodal (that many separated clusters)
Set RATE_TAG / RUN_LABEL / RMSE_TOL and run."""
import glob, os, re, pickle
import numpy as np

RATE_TAG = "rc_long_rc_short"   # "clean" | "rc_long_rc_short" | "Dsn_Dsp_rc_long_rc_short"
RUN_LABEL = "wide"        # "wide" | "narrow"
RMSE_TOL = 0.05           # keep fits with rmse <= best*(1+RMSE_TOL)
GAP = 0.15                # theta_bar gap that separates two modes
NARROW = 0.10             # theta_bar spread below which a unimodal param is "tight"
MINSIZE = 2               # a 1D cluster needs >= this many points
LAB = ["k-", "k+", "kappa", "De", "Ds-", "Ds+"]
RATE_DIR = os.path.join("rate_sweep_pkl", RUN_LABEL)


def modes_1d(vals):
    v = np.sort(np.asarray(vals, float))
    groups = np.split(v, np.where(np.diff(v) > GAP)[0] + 1)
    groups = [g for g in groups if len(g) >= MINSIZE]
    n = len(groups)
    width = float(v.max() - v.min()) if len(v) else 0.0
    return n, width


files = {}
for f in glob.glob(os.path.join(RATE_DIR, f"rate_{RATE_TAG}_*.pkl")):
    m = re.search(rf"rate_{RATE_TAG}_(.+?)_soc(\d+)\.pkl", os.path.basename(f))
    if m:
        files[(m.group(1), int(m.group(2)))] = f

# theta_nominal for the ratio (same order as LAB)
DSN = "Negative particle diffusivity [m2.s-1]"; DSP = "Positive particle diffusivity [m2.s-1]"
FP = ["Negative electrode exchange-current density [A.m-2]",
      "Positive electrode exchange-current density [A.m-2]",
      "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]", DSN, DSP]
NOM = np.array([6.48e-7, 3.42e-6, 0.9487, 1.7694e-10, 3.3e-14, 4.0e-15])

print(f"run: {RATE_TAG}/{RUN_LABEL}, fits within +{RMSE_TOL*100:.0f}% of best.")
print("cell = median(theta_eff/theta_nom);  '~'=flat (range in notes),  '*N'=N modes\n")
print(f"{'condition':16}{'nfit':>5}   " + "".join(f"{l:>10}" for l in LAB))
tally = {l: {"tight": 0, "flat": 0, "bi": 0, "multi": 0} for l in LAB}
notes = []
for k in sorted(files):
    res = pickle.load(open(files[k], "rb"))
    ranked = sorted(res, key=lambda r: r["rmse_mV"])
    best = ranked[0]["rmse_mV"]
    keep = [r for r in ranked if r["rmse_mV"] <= best * (1 + RMSE_TOL)]
    bars = np.array([np.asarray(r["theta_bar_hat"], float) for r in keep])
    ratios = np.array([[r["theta_hat"][p] / NOM[j] for j, p in enumerate(FP)] for r in keep])
    cells = []
    for j, l in enumerate(LAB):
        n, w = modes_1d(bars[:, j])
        rj = ratios[:, j]; med = np.median(rj)
        if n >= 2:
            cells.append(f"{med:.2f}*{n}"); tally[l]["multi" if n >= 3 else "bi"] += 1
            notes.append(f"  {k[0]} soc{k[1]:<2} {l:5}: {n} modes, ratio {rj.min():.2f}-{rj.max():.2f}")
        elif w < NARROW:
            cells.append(f"{med:.2f}"); tally[l]["tight"] += 1
        else:
            cells.append(f"{med:.1f}~"); tally[l]["flat"] += 1
            notes.append(f"  {k[0]} soc{k[1]:<2} {l:5}: FLAT, ratio {rj.min():.2f}-{rj.max():.2f}")
    print(f"{k[0]+' soc'+str(k[1]):16}{len(keep):>5}   " + "".join(f"{c:>10}" for c in cells))

if notes:
    print("\nflat / multimodal details (theta_eff/theta_nom range):")
    for line in notes:
        print(line)

print(f"\nsummary across {len(files)} conditions (# of conditions in each category):")
print(f"{'param':6}{'tight-1':>9}{'flat':>7}{'bimodal':>9}{'multi(3+)':>11}")
for l in LAB:
    t = tally[l]
    print(f"{l:6}{t['tight']:>9}{t['flat']:>7}{t['bi']:>9}{t['multi']:>11}")

# %%
