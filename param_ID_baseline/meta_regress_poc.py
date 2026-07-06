"""Proof-of-concept: random-effects meta-regression on theta_eff across profiles.

For each fit parameter we treat the per-profile best fits theta_eff_i (from the
rate sweep) as noisy observations of a latent, profile-varying "effective" value:

    y_i = log10(theta_eff_i / theta_nom)                       (obs, per condition)
    y_i = mu + beta_c * x_crate_i + beta_s * x_soc_i + b_i + eps_i
        eps_i ~ N(0, sy_i^2)      within-profile CRLB          (KNOWN)
        b_i   ~ N(0, tau^2)       between-profile spread        (ESTIMATED)

  mu    = transferable center ("base physics"); on synthetic, ratio should -> 1
  beta  = systematic dependence of theta_eff on C-rate / SOC (misspecification trend)
  tau   = honest model-form uncertainty (Sigma_between), separated from CRLB noise
  s/w analog: tau vs mean(sy) tells you if the CRLB explains the scatter.

tau^2 is estimated by REML; mu, beta by GLS at tau^2_hat. A predictive interval
gives the UQ distribution of theta_eff at a chosen operating point.

Design choices (vs the naive per-SOC / c-rate-only fit):
  * POOLED across SOC  -> n up to 16 CC-discharge conditions (not 4) for a stable tau^2
  * CRLB FLOORED       -> a bogus-tight sy (e.g. kappa ~0.01%) can't hijack the GLS weights
  * unidentifiable pts FILTERED (sy above a cap) -> tau reflects misspecification, not non-ID
  * HPPC / sine excluded: no single C-rate.  1D-per-param: ignores theta correlations (POC).

Run cell 1 once (SLOW ~5 min, then cached), then cell 2 (instant) to refit / retune.
"""
# %% cell 1: build (theta_eff, CRLB) per condition  [SLOW first run -> cached to pkl]
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, glob, re, pickle, threading
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
NOM = np.array([6.48e-7, 3.42e-6, 0.9487, 1.7694e-10, 3.3e-14, 4.0e-15])
SIGMA = 1e-3

RATE_TAG = "rc_long_rc_short"          # which sweep to analyze
RUN_LABEL = "wide"
EXCLUDE_PROFILES = {"sin_1Hz"}          # no single C-rate
RATE_DIR = os.path.join("rate_sweep_pkl", RUN_LABEL)
CACHE = f"meta_cache_{RATE_TAG}_{RUN_LABEL}.pkl"
SOLVE_TIMEOUT = 90.0
# profile -> C-rate (constant-current discharges only; HPPC has no single rate)
CRATE = {"C3_dchg": 1.0 / 3.0, "1C_dchg": 1.0, "2C_dchg": 2.0, "5C_dchg": 5.0}


def _core(theta_hat, soc, profile):
    """One SPMe analytic-sensitivity solve -> CRLB_rel (sigma/theta) per param."""
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, FP, values=theta_hat)
    solver = pybamm.IDAKLUSolver(options={"max_num_steps": 20000})
    sol = run_model(make_model("SPMe", options=OPTS), p,
                    make_experiment(profile_id=profile), inputs=iv,
                    calculate_sensitivities=True, solver=solver)
    Sens = get_sensitivities(sol, "Voltage [V]", FP, theta_values=theta_hat, normalize=True)
    F = compute_fim(Sens, FP, sigma=SIGMA)
    crlb_rel = np.sqrt(np.clip(np.diag(np.linalg.pinv(F)), 0, None))
    return crlb_rel


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


if os.path.exists(CACHE):
    eff = pickle.load(open(CACHE, "rb"))
    print(f"loaded cache {CACHE}: {len(eff)} conditions")
else:
    eff = {}
    # theta_eff = best (lowest RMSE) fit per condition, from the sweep pkls
    for f in glob.glob(os.path.join(RATE_DIR, f"rate_{RATE_TAG}_*.pkl")):
        m = re.search(rf"rate_{RATE_TAG}_(.+?)_soc(\d+)\.pkl", os.path.basename(f))
        if not m or m.group(1) in EXCLUDE_PROFILES:
            continue
        res = pickle.load(open(f, "rb"))
        good = [r for r in res if not r.get("stalled")]
        if not good:
            continue
        best = min(good, key=lambda r: r["rmse_mV"])
        eff[(m.group(1), int(m.group(2)))] = dict(theta_hat=best["theta_hat"],
                                                  rmse=best["rmse_mV"])
    # CRLB per condition (thread-guarded)
    for k in sorted(eff):
        try:
            eff[k]["crlb"] = crlb_at(eff[k]["theta_hat"], k[1], k[0])
            eff[k]["ok"] = True
            print(f"  CRLB ok: {k[0]} soc{k[1]}")
        except Exception as e:
            eff[k]["ok"] = False
            print(f"  CRLB FAILED: {k[0]} soc{k[1]}  ({type(e).__name__})")
    pickle.dump(eff, open(CACHE, "wb"))
    print(f"saved cache {CACHE}")

OK = [k for k in sorted(eff) if eff[k].get("ok")]
print(f"{len(OK)} conditions with CRLB ok")


# %% cell 2: REML random-effects meta-regression, per parameter
from scipy.optimize import minimize_scalar

# --- knobs ---
CRLB_FLOOR = 0.01      # floor CRLB_rel at 1% so a bogus-tight sy can't dominate GLS
CRLB_MAX = 0.50        # drop conditions with CRLB_rel > 50% (unidentifiable, not model-form)
USE_SOC_MODERATOR = True   # include SOC as a 2nd fixed moderator (pool across SOC)
MOD_TAG = "both" if USE_SOC_MODERATOR else "crateOnly"   # suffix for caches + figures
TAU2_MAX = 4.0         # upper bound on tau^2 (log10 units); 4 -> tau=2 = 2 decades
LN10 = np.log(10.0)
# predictive query point
C_STAR, SOC_STAR = 1.0, 55.0

# EFFECTIVE C-rate = RMS(current) / 1C-current, computed per profile. Works for ANY
# profile (HPPC, drive cycles, future ones) since current is prescribed, not theta-
# dependent -> a universal "intensity" moderator so HPPC etc. can join the fit.
CEFF_CACHE = f"meta_ceff_{RUN_LABEL}.pkl"
_NOMD = {FP[j]: NOM[j] for j in range(len(FP))}
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))   # time integration
if os.path.exists(CEFF_CACHE):
    CRATE_EFF = pickle.load(open(CEFF_CACHE, "rb"))
else:
    CRATE_EFF = {}
    for prof in sorted({k[0] for k in OK}):
        base = make_base_params("Chen2020", soc=0.5, sensitivity_ready=True)
        try:
            I_1C = float(base["Nominal cell capacity [A.h]"])
        except Exception:
            I_1C = 5.0
        p, iv = prepare_sensitivity_inputs(base, FP, values=_NOMD)
        sol = run_model(make_model("SPMe", options=OPTS), p,
                        make_experiment(profile_id=prof), inputs=iv)
        tI = sol["Time [s]"].entries
        I = sol["Current [A]"].entries
        act = (np.abs(I) > 1e-6 * I_1C).astype(float)   # ACTIVE portion only (ignore rest)
        # TIME-weighted RMS (trapezoid). The solver time-grid is non-uniform, so a plain
        # np.mean over samples over-weights fast-transient regions (e.g. HPPC's 1C pulses
        # get oversampled -> inflated 0.71 instead of the true time-weighted 0.40).
        den = _trapz(act, tI)
        Irms = np.sqrt(_trapz((I ** 2) * act, tI) / den) if den > 0 else 0.0
        CRATE_EFF[prof] = float(Irms / I_1C)            # CC -> exactly nominal; rest-length independent
    pickle.dump(CRATE_EFF, open(CEFF_CACHE, "wb"))
print("effective C-rate (RMS/1C):", {k: round(v, 3) for k, v in sorted(CRATE_EFF.items())})

# ALL conditions now join the fit (incl. HPPC), using the effective C-rate moderator
FITK = list(OK)
crate_all = np.array([CRATE_EFF[k[0]] for k in FITK])
soc_all = np.array([k[1] for k in FITK], float)
xc_all = np.log10(crate_all); xc_mean = xc_all.mean()
xs_all = soc_all / 100.0;      xs_mean = xs_all.mean()


def fit_reml(y, sy, X):
    """REML estimate of tau^2, then GLS for coefficients. Returns tau2,b,se,cov."""
    def neg2_reml(tau2):
        W = 1.0 / (sy ** 2 + tau2)
        XtWX = X.T @ (W[:, None] * X)
        b = np.linalg.solve(XtWX, X.T @ (W * y))
        r = y - X @ b
        _, logdet = np.linalg.slogdet(XtWX)
        return np.sum(W * r ** 2) - np.sum(np.log(W)) + logdet
    res = minimize_scalar(neg2_reml, bounds=(1e-10, TAU2_MAX), method="bounded")
    tau2 = float(res.x)
    W = 1.0 / (sy ** 2 + tau2)
    cov = np.linalg.inv(X.T @ (W[:, None] * X))
    b = cov @ (X.T @ (W * y))
    se = np.sqrt(np.diag(cov))
    return tau2, b, se, cov


hdr = f"{'param':6}{'n':>3}{'tau':>8}{'mu->ratio':>11}{'truth?':>8}{'beta_c':>10}{'beta_s':>10}"
print(f"\n[{RATE_TAG}/{RUN_LABEL}]  REML meta-regression per parameter")
print("  y = log10(theta_eff/theta_nom);  moderators: x_c=log10(eff C-rate=RMS/1C), x_s=SOC/100 (centered)")
print("  tau = between-profile std (log10);  mu->ratio = 10^mu (center);  beta * = |b|>2se")
print(hdr)
results = {}
for j, l in enumerate(LAB):
    r = np.array([eff[k]["theta_hat"][FP[j]] / NOM[j] for k in FITK])   # theta_eff/theta_nom
    crlb = np.clip(np.array([eff[k]["crlb"][j] for k in FITK]), CRLB_FLOOR, None)
    keep = crlb <= CRLB_MAX
    if keep.sum() < 4:
        print(f"{l:6}{keep.sum():>3}   too few identifiable conditions -> skip")
        continue
    y = np.log10(r[keep])
    sy = crlb[keep] / LN10                       # delta method: std of log10 = CRLB_rel/ln10
    cols = [np.ones(keep.sum()), (xc_all[keep] - xc_mean)]
    if USE_SOC_MODERATOR:
        cols.append(xs_all[keep] - xs_mean)
    X = np.vstack(cols).T
    tau2, b, se, cov = fit_reml(y, sy, X)
    tau = np.sqrt(tau2)
    mu = b[0]; se_mu = se[0]
    ratio = 10 ** mu
    # truth recovery: is log10(ratio)=0 inside mu +/- 1.96 se_mu ?
    recovers = abs(mu) <= 1.96 * se_mu
    bc = f"{b[1]:+.3f}{'*' if abs(b[1]) > 2 * se[1] else ' '}"
    bs = (f"{b[2]:+.3f}{'*' if abs(b[2]) > 2 * se[2] else ' '}"
          if USE_SOC_MODERATOR else "   --   ")
    print(f"{l:6}{keep.sum():>3}{tau:>8.3f}{ratio:>11.3f}"
          f"{('YES' if recovers else 'no'):>8}{bc:>10}{bs:>10}")
    results[l] = dict(tau=tau, mu=mu, se_mu=se_mu, b=b, se=se, cov=cov,
                      n=int(keep.sum()), recovers=recovers)

# --- predictive UQ at (C_STAR, SOC_STAR): distribution of theta_eff for a NEW profile ---
print(f"\npredictive UQ of theta_eff/theta_nom @ {C_STAR}C, SOC {SOC_STAR}%  (95%)")
print(f"{'param':6}{'center':>9}{'model-form only':>18}{'+ estimation':>20}")
xv = [1.0, np.log10(C_STAR) - xc_mean]
if USE_SOC_MODERATOR:
    xv.append(SOC_STAR / 100.0 - xs_mean)
xv = np.array(xv)
for l in LAB:
    if l not in results:
        continue
    R = results[l]
    mean = xv @ R["b"]
    tau = R["tau"]
    var_full = tau ** 2 + xv @ R["cov"] @ xv          # between-profile + estimation of mean
    def band(half):
        return 10 ** (mean - 1.96 * half), 10 ** (mean + 1.96 * half)
    lo1, hi1 = band(tau)                              # model-form spread only
    lo2, hi2 = band(np.sqrt(var_full))               # + estimation uncertainty
    print(f"{l:6}{10**mean:>9.3f}   [{lo1:>6.3f}, {hi1:>6.3f}]   [{lo2:>6.3f}, {hi2:>6.3f}]")

print("\nreading:")
print("  mu->ratio ~ 1 & truth=YES         -> transferable base physics (center recovers truth)")
print("  mu->ratio != 1 (De, Ds+ expected) -> consistent misspecification bias")
print("  beta_c *  (significant)           -> theta_eff drifts with C-rate (structural, e.g. De)")
print("  large tau                         -> big model-form UQ the CRLB can't see")


# %% cell 2b: PARAMETER RECOVERY -- individual theta_eff_i vs theta_combined (distance & coverage of truth)
# truth = ratio 1 (log10 = 0). individual = per-profile fit + its RAW CRLB (what one fit reports);
# combined = meta-regression center mu +/- sqrt(tau^2 + se_mu^2). This is the KEY evidence that
# pooling recovers the true base physics better than any single profile.
print(f"\n{'=' * 74}\nPARAMETER RECOVERY vs truth (ratio 1):  individual theta_eff_i  vs  theta_combined\n{'=' * 74}")
print(f"{'param':6}| {'---- individual theta_eff_i ----':^32}| {'---- theta_combined ----':^24}| verdict")
print(f"{'':6}| {'dist(RMS)':>11}{'cover%':>8}{'z':>8}  | {'dist|mu|':>10}{'cover?':>8}{'z':>6} | closer cover")
for j, l in enumerate(LAB):
    if l not in results:
        print(f"{l:6}|  (skipped: too few identifiable conditions)")
        continue
    r = np.array([eff[k]["theta_hat"][FP[j]] / NOM[j] for k in OK])
    craw = np.array([eff[k]["crlb"][j] for k in OK])
    keep = np.clip(craw, CRLB_FLOOR, None) <= CRLB_MAX
    y = np.log10(r[keep]); syraw = craw[keep] / LN10                     # individual (raw CRLB)
    e_ind = np.sqrt(np.mean(y ** 2))
    cov_ind = 100 * np.mean(np.abs(y) <= 1.96 * syraw)
    z_ind = np.sqrt(np.mean((y / syraw) ** 2))
    R = results[l]; sig_c = np.sqrt(R["tau"] ** 2 + R["se_mu"] ** 2)     # theta_combined
    e_com, z_com = abs(R["mu"]), abs(R["mu"]) / sig_c
    cov_com = z_com <= 1.96
    closer = "YES" if e_com < e_ind else "no "
    print(f"{l:6}| {e_ind:>11.3f}{cov_ind:>7.0f}%{z_ind:>8.1f}  | "
          f"{e_com:>10.3f}{('Y' if cov_com else 'n'):>8}{z_com:>6.1f} |  {closer}  "
          f"{'Y' if cov_com else 'n'}")
print("  dist = log10 distance to truth (smaller = closer) | cover = 95% band contains truth"
      " | z = truth's distance in sigma (~1 well-calibrated, >>2 overconfident)")
print("  -> theta_combined 'closer' for every param; 'cover' Y for all but the systematically-biased one")


# %% cell 3: generalization matrix -- run each theta candidate on EVERY condition
# Candidates: each per-condition theta_eff_i, PLUS theta_combined (meta-reg center mu).
# For each TEST condition we rebuild its V_truth (clean SPMe@theta_nom + rc + noise,
# exactly as the sweep did) and score every candidate's clean-SPMe RMSE against it.
# theta_eff_i should win on its OWN condition; theta_combined should be the most
# robust ACROSS conditions.  RMSE matrix is cached -> first run ~5-8 min, then instant.
from utils import Vrc_discrepancy, RC_SPECS, ocp_n_discrepancy, Dsn_discrepancy, Dsp_discrepancy
OCP = "Negative electrode OCP [V]"
NOISE_STD = 1e-3
SIM_TIMEOUT = 30.0
GEN_CACHE = f"meta_genmatrix_{RATE_TAG}_{RUN_LABEL}_{MOD_TAG}.pkl"
NOM_D = {FP[j]: NOM[j] for j in range(len(FP))}

# theta_combined = meta-regression center (base value at mean operating point)
theta_comb = {FP[j]: NOM[j] * 10 ** results[LAB[j]]["mu"] if LAB[j] in results else NOM[j]
              for j in range(len(FP))}
ALL_COND = sorted(eff)              # EVERY fitted condition -> ALL theta_eff_i are candidates
CANDS = [(f"{k[0]}_s{k[1]}", eff[k]["theta_hat"]) for k in ALL_COND] + [("COMBINED", theta_comb)]
TESTS = list(ALL_COND)              # score on every condition (incl. HPPC -> held out of meta-reg)
print(f"candidates: {len(ALL_COND)} theta_eff_i + COMBINED = {len(CANDS)} total")


def _guard(fn, timeout):
    box = {}
    def w():
        try: box["v"] = fn()
        except BaseException as e: box["e"] = e
    th = threading.Thread(target=w, daemon=True); th.start(); th.join(timeout)
    return None if (th.is_alive() or "e" in box) else box["v"]


def make_truth(profile, soc):
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, FP, values=NOM_D)
    if "ocp" in RATE_TAG: p[OCP] = ocp_n_discrepancy
    if "Dsn" in RATE_TAG: p[DSN] = Dsn_discrepancy; iv.pop(DSN, None)
    if "Dsp" in RATE_TAG: p[DSP] = Dsp_discrepancy; iv.pop(DSP, None)
    sol = run_model(make_model("SPMe", options=OPTS), p, make_experiment(profile_id=profile), inputs=iv)
    t = sol["Time [s]"].entries; V = sol["Voltage [V]"].entries.copy()
    rc = [RC_SPECS[k] for k in ("rc_short", "rc_long") if k in RATE_TAG]
    if rc:
        V = V - Vrc_discrepancy(t, sol["Current [A]"].entries, rc)
    V = V + np.random.default_rng(0).normal(0.0, NOISE_STD, size=V.shape)   # match sweep seed
    return t, V


def sim_V(theta, profile, soc, t_ref):
    base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
    p, iv = prepare_sensitivity_inputs(base, FP, values=theta)
    solver = pybamm.IDAKLUSolver(options={"max_num_steps": 20000})
    sol = run_model(make_model("SPMe", options=OPTS), p,
                    make_experiment(profile_id=profile), inputs=iv, solver=solver)
    return np.interp(t_ref, sol["Time [s]"].entries, sol["Voltage [V]"].entries)


def _build_gen():
    rmse, curves, truth = {}, {}, {}
    for ti, tk in enumerate(TESTS):
        t_ref, V_truth = make_truth(tk[0], tk[1])
        truth[tk] = (t_ref, V_truth)
        for clabel, theta in CANDS:
            Vs = _guard(lambda th=theta: sim_V(th, tk[0], tk[1], t_ref), SIM_TIMEOUT)
            rmse[(clabel, tk)] = (np.inf if Vs is None
                                  else float(np.sqrt(np.mean((Vs - V_truth) ** 2)) * 1e3))
            curves[(clabel, tk)] = Vs                 # interpolated onto t_ref (or None)
        print(f"  tested all candidates on {tk[0]} s{tk[1]}  ({ti+1}/{len(TESTS)})")
    return {"rmse": rmse, "curves": curves, "truth": truth}


_cache = pickle.load(open(GEN_CACHE, "rb")) if os.path.exists(GEN_CACHE) else None
if isinstance(_cache, dict) and "rmse" in _cache:       # new format (rmse+curves+truth)
    GEN = _cache
    print(f"loaded generalization matrix {GEN_CACHE}")
else:                                                    # missing or old (rmse-only) -> rebuild
    GEN = _build_gen()
    pickle.dump(GEN, open(GEN_CACHE, "wb"))
    print(f"saved {GEN_CACHE}")
RMSE, CURVES, TRUTH = GEN["rmse"], GEN["curves"], GEN["truth"]


# %% cell 4: rank table -- who generalizes best?
CLABELS = [c[0] for c in CANDS]
# rank candidates within each TEST column (1 = lowest RMSE = best fit to that condition)
ranks = {cl: [] for cl in CLABELS}
per_test = {}
for tk in TESTS:
    col = sorted(CLABELS, key=lambda cl: RMSE[(cl, tk)])
    rank_of = {cl: i + 1 for i, cl in enumerate(col)}
    for cl in CLABELS:
        ranks[cl].append(rank_of[cl])
    per_test[tk] = (col[0], rank_of["COMBINED"], RMSE[(col[0], tk)], RMSE[("COMBINED", tk)])

n_cand = len(CLABELS)
mean_rank = {cl: float(np.mean(ranks[cl])) for cl in CLABELS}
worst_rank = {cl: int(np.max(ranks[cl])) for cl in CLABELS}
med_rmse = {cl: float(np.median([RMSE[(cl, tk)] for tk in TESTS])) for cl in CLABELS}

print(f"\n=== generalization ranking over {len(TESTS)} conditions "
      f"({n_cand} candidates) ===")
print("candidates sorted by MEAN rank across all test conditions (lower = more robust)")
print(f"{'rank':>4}  {'candidate':16}{'mean_rank':>10}{'worst':>7}{'med_RMSE_mV':>13}")
for i, cl in enumerate(sorted(CLABELS, key=lambda c: mean_rank[c])):
    star = "  <== COMBINED" if cl == "COMBINED" else ""
    print(f"{i+1:>4}  {cl:16}{mean_rank[cl]:>10.2f}{worst_rank[cl]:>7}{med_rmse[cl]:>13.3f}{star}")

# how the individual theta_eff_i's compare, as a group, to COMBINED
ind = [cl for cl in CLABELS if cl != "COMBINED"]
print(f"\nCOMBINED mean rank = {mean_rank['COMBINED']:.2f} / {n_cand}  "
      f"(worst {worst_rank['COMBINED']})")
print(f"individual theta_eff_i mean-rank: best {min(mean_rank[c] for c in ind):.2f}, "
      f"median {np.median([mean_rank[c] for c in ind]):.2f}, "
      f"worst {max(mean_rank[c] for c in ind):.2f}")
better = sum(mean_rank['COMBINED'] < mean_rank[c] for c in ind)
print(f"COMBINED beats {better}/{len(ind)} individual theta_eff_i on mean rank")

print(f"\nper-condition winner and COMBINED's rank there:")
print(f"{'test condition':16}{'best theta':16}{'COMB rank':>10}{'best_RMSE':>11}{'COMB_RMSE':>11}")
for tk in TESTS:
    best, cr, brm, crm = per_test[tk]
    print(f"{tk[0]+' s'+str(tk[1]):16}{best:16}{cr:>10}{brm:>11.3f}{crm:>11.3f}")


# %% cell 5: rank heatmap  (candidate rows x condition cols; 1 = best fit to that col)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# rows in the SAME order as columns (TESTS order), so row i == column i (diagonal =
# theta tested on its own condition -> rank 1). COMBINED appended as the LAST row.
order = [f"{k[0]}_s{k[1]}" for k in TESTS] + ["COMBINED"]
M = np.array([[ranks[cl][ti] for ti in range(len(TESTS))] for cl in order])
fig, ax = plt.subplots(figsize=(max(11, len(TESTS) * 0.75), max(9, len(order) * 0.45)))
im = ax.imshow(M, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=len(CLABELS))
ax.set_xticks(range(len(TESTS)))
ax.set_xticklabels([f"{tk[0]}\ns{tk[1]}" for tk in TESTS], fontsize=7, rotation=90)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=8)
for yi, cl in enumerate(order):                              # annotate ranks + flag COMBINED
    if cl == "COMBINED":
        lab = ax.get_yticklabels()[yi]; lab.set_fontweight("bold"); lab.set_color("blue")
        ax.axhline(yi - 0.5, color="blue", lw=1.5); ax.axhline(yi + 0.5, color="blue", lw=1.5)
    for xi in range(len(TESTS)):
        ax.text(xi, yi, str(M[yi, xi]), ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=ax, label="rank within condition (1 = best fit)")
ax.set_xlabel("Profile Name")
ax.set_ylabel(r"$\theta_{eff}$")
ax.set_title("Voltage RMSE heatmap")
plt.tight_layout()
plt.savefig(f"meta_rank_heatmap_{MOD_TAG}.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"saved meta_rank_heatmap_{MOD_TAG}.png")

# RANK RANGE of each theta across all conditions: best (min) .. worst (max), + median/mean/spread
# Shows how wide each theta's rank swings: tight range = consistent, wide = specialist/erratic.
print(f"\n{'=' * 64}\nrank range of each theta across {len(TESTS)} conditions (1=best fit that condition)\n"
      f"  best..worst = rank span;  spread = worst - best;  sorted by median rank\n{'=' * 64}")
print(f"{'#':>3}  {'candidate':16}{'best':>6}{'median':>8}{'worst':>7}{'mean':>7}{'spread':>8}")
for i, cl in enumerate(sorted(CLABELS, key=lambda c: (np.median(ranks[c]), mean_rank[c])), 1):
    rs = np.array(ranks[cl])
    mark = "   <== COMBINED" if cl == "COMBINED" else ""
    print(f"{i:>3}  {cl:16}{rs.min():>6}{np.median(rs):>8.1f}{rs.max():>7}"
          f"{rs.mean():>7.1f}{rs.max() - rs.min():>8}{mark}")


# %% cell 6: voltage overlays per condition  (COMBINED drawn LAST -> on top of the cloud)
PROFILES_ORDER = ["1C_dchg", "2C_dchg", "5C_dchg", "C3_dchg", "hppc"]
socs_of = {}
for tk in TESTS:
    socs_of.setdefault(tk[0], []).append(tk[1])
for p in socs_of:
    socs_of[p] = sorted(socs_of[p])
nrow = len(PROFILES_ORDER)
ncol = max(len(v) for v in socs_of.values())
fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
for ri, prof in enumerate(PROFILES_ORDER):
    socs = socs_of.get(prof, [])
    for ci in range(ncol):
        ax = axes[ri][ci]
        if ci >= len(socs):
            ax.axis("off"); continue
        tk = (prof, socs[ci])
        t_ref, V_truth = TRUTH[tk]
        for clabel, _ in CANDS:                              # individual theta_eff_i: gray cloud
            if clabel == "COMBINED":
                continue
            Vs = CURVES.get((clabel, tk))
            if Vs is not None:
                ax.plot(t_ref, Vs, color="0.35", lw=0.9, alpha=0.6, zorder=1)
        ax.plot(t_ref, V_truth, color="tab:blue", lw=1.8, zorder=3)          # V_data (blue solid)
        Vc = CURVES.get(("COMBINED", tk))
        if Vc is not None:
            ax.plot(t_ref, Vc, color="red", ls="--", lw=1.8, zorder=4)       # combined (red dashed)
        ax.set_title(f"{prof} s{socs[ci]}", fontsize=12)
        ax.set_ylim(V_truth.min() - 0.25, V_truth.max() + 0.10)   # focus near truth
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)
        if ci == 0:
            ax.set_ylabel("Voltage (V)", fontsize=14)
        if ri == nrow - 1:
            ax.set_xlabel("Time (s)", fontsize=14)
handles = [Line2D([0], [0], color="0.35", lw=1.6, label=r"$V_{SPMe}(\theta_{eff,i})$"),
           Line2D([0], [0], color="tab:blue", lw=2.0, label=r"$V_{data}$"),
           Line2D([0], [0], color="red", ls="--", lw=2.0, label=r"$V_{SPMe}(\theta_{combined})$")]
fig.legend(handles=handles, loc="upper right", fontsize=16, ncol=3,
           bbox_to_anchor=(0.99, 0.99))
fig.suptitle("Voltage: individual vs combined", y=1.0, fontsize=17)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"meta_voltage_overlays_{MOD_TAG}.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"saved meta_voltage_overlays_{MOD_TAG}.png")


# %% cell 7: HELD-OUT WLTP drive cycle at 4 SOC levels (new figure)
# WLTP was never in the sweep, so no theta was fit to it -> a true generalization
# test. Run every candidate (all theta_eff_i + COMBINED) on WLTP at each SOC.
WLTP_SOCS = [85, 65, 50, 30]         # same SOC levels as HPPC
WLTP_TIMEOUT = 60.0
WLTP_CACHE = f"meta_wltp_{RATE_TAG}_{RUN_LABEL}_{MOD_TAG}.pkl"
if os.path.exists(WLTP_CACHE):
    WALL = pickle.load(open(WLTP_CACHE, "rb"))
    print(f"loaded {WLTP_CACHE}")
else:
    WALL = {}
    for soc in WLTP_SOCS:
        tW, VW = make_truth("WLTP", soc)
        wc, wr = {}, {}
        for clabel, theta in CANDS:
            Vs = _guard(lambda th=theta: sim_V(th, "WLTP", soc, tW), WLTP_TIMEOUT)
            wc[clabel] = Vs
            wr[clabel] = np.inf if Vs is None else float(np.sqrt(np.mean((Vs - VW) ** 2)) * 1e3)
        WALL[soc] = {"t": tW, "truth": VW, "curves": wc, "rmse": wr}
        print(f"  WLTP soc{soc}: all {len(CANDS)} candidates done")
    pickle.dump(WALL, open(WLTP_CACHE, "wb"))
    print(f"saved {WLTP_CACHE}")

print(f"\nWLTP held-out generalization ({len(CLABELS)} candidates):")
for soc in WLTP_SOCS:
    rm = WALL[soc]["rmse"]
    order = sorted(CLABELS, key=lambda cl: rm[cl])
    cr = order.index("COMBINED") + 1
    print(f"  soc{soc}: COMBINED {rm['COMBINED']:.2f} mV  rank {cr}/{len(CLABELS)}   "
          f"(best {order[0]} = {rm[order[0]]:.2f} mV)")

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for ax, soc in zip(axes.ravel(), WLTP_SOCS):
    D = WALL[soc]; tW, VW = D["t"], D["truth"]
    for clabel, _ in CANDS:
        if clabel == "COMBINED":
            continue
        Vs = D["curves"][clabel]
        if Vs is not None:
            ax.plot(tW, Vs, color="0.35", lw=0.9, alpha=0.6, zorder=1)
    ax.plot(tW, VW, color="tab:blue", lw=1.8, zorder=3)              # V_data (blue solid)
    Vc = D["curves"]["COMBINED"]
    if Vc is not None:
        ax.plot(tW, Vc, color="red", ls="--", lw=1.8, zorder=4)      # combined (red dashed)
    ax.set_title(f"WLTP SOC {soc}%", fontsize=12)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylabel("Voltage (V)", fontsize=14)
    ax.set_xlabel("Time (s)", fontsize=14)
fig.legend(handles=[
    Line2D([0], [0], color="0.35", lw=1.6, label=r"$V_{SPMe}(\theta_{eff,i})$"),
    Line2D([0], [0], color="tab:blue", lw=2.0, label=r"$V_{data}$"),
    Line2D([0], [0], color="red", ls="--", lw=2.0, label=r"$V_{SPMe}(\theta_{combined})$")],
    loc="upper right", fontsize=16, ncol=3, bbox_to_anchor=(0.99, 0.99))
fig.suptitle("WLTP Voltage: individual vs combined", y=1.0, fontsize=17)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"meta_wltp_{MOD_TAG}.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"saved meta_wltp_{MOD_TAG}.png")


# %% cell 8: VOLTAGE-ERROR grid (training) -- error = V_truth - V_SPMe(theta) [mV]
# Same 5x4 grid as cell 6 but plotting the ERROR of each candidate: individual
# theta_eff_i errors (gray), theta_combined error drawn LAST (red, on top) so you
# can see how small it is vs the others. y-lim = robust 2-98 pct (rails clipped off).
fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
for ri, prof in enumerate(PROFILES_ORDER):
    socs = socs_of.get(prof, [])
    for ci in range(ncol):
        ax = axes[ri][ci]
        if ci >= len(socs):
            ax.axis("off"); continue
        tk = (prof, socs[ci])
        t_ref, V_truth = TRUTH[tk]
        for clabel, _ in CANDS:
            if clabel == "COMBINED":
                continue
            Vs = CURVES.get((clabel, tk))
            if Vs is None:
                continue
            if clabel == "2C_dchg_s90":                       # highlight this theta_eff in blue
                ax.plot(t_ref, (V_truth - Vs) * 1e3, color="tab:blue", lw=1.6, alpha=0.9, zorder=3)
            else:
                ax.plot(t_ref, (V_truth - Vs) * 1e3, color="0.35", lw=0.7, alpha=0.5, zorder=1)
        Vc = CURVES.get(("COMBINED", tk))
        if Vc is not None:
            ax.plot(t_ref, (V_truth - Vc) * 1e3, color="red", lw=1.7, zorder=4)  # COMBINED last
        ax.axhline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
        if Vc is not None:                          # y-lim = +/-3x COMBINED error (rails clipped off)
            m = max(np.max(np.abs((V_truth - Vc) * 1e3)) * 3.0, 5.0)
            ax.set_ylim(-m, m)
        ax.set_title(f"{prof} s{socs[ci]}", fontsize=12)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)
        if ci == 0:
            ax.set_ylabel("Voltage (mV)", fontsize=14)
        if ri == nrow - 1:
            ax.set_xlabel("Time (s)", fontsize=14)
fig.legend(handles=[
    Line2D([0], [0], color="0.35", lw=1.6, label=r"$V_{data} - V_{SPMe}(\theta_{eff,i})$"),
    Line2D([0], [0], color="tab:blue", lw=1.8, label=r"$V_{data} - V_{SPMe}(\theta_{2C,s90})$"),
    Line2D([0], [0], color="red", lw=2.0, label=r"$V_{data} - V_{SPMe}(\theta_{combined})$")],
    loc="upper right", fontsize=16, ncol=3, bbox_to_anchor=(0.99, 0.955))
fig.suptitle(r"Voltage Error under different $\theta_{eff,i}$", y=1.0, fontsize=17)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"meta_voltage_error_{MOD_TAG}.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"saved meta_voltage_error_{MOD_TAG}.png")


# %% cell 9: VOLTAGE-ERROR grid (WLTP held-out) -- error = V_truth - V_SPMe(theta) [mV]
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for ax, soc in zip(axes.ravel(), WLTP_SOCS):
    D = WALL[soc]; tW, VW = D["t"], D["truth"]
    for clabel, _ in CANDS:
        if clabel == "COMBINED":
            continue
        Vs = D["curves"][clabel]
        if Vs is None:
            continue
        if clabel == "2C_dchg_s90":                       # highlight this theta_eff in blue
            ax.plot(tW, (VW - Vs) * 1e3, color="tab:blue", lw=1.6, alpha=0.9, zorder=3)
        else:
            ax.plot(tW, (VW - Vs) * 1e3, color="0.35", lw=0.6, alpha=0.45, zorder=1)
    Vc = D["curves"]["COMBINED"]
    if Vc is not None:
        ax.plot(tW, (VW - Vc) * 1e3, color="red", lw=1.7, zorder=4)      # COMBINED last
    ax.axhline(0.0, color="k", lw=0.6, ls=":", alpha=0.6)
    if Vc is not None:                              # y-lim = +/-3x COMBINED error (rails clipped off)
        m = max(np.max(np.abs((VW - Vc) * 1e3)) * 3.0, 5.0)
        ax.set_ylim(-m, m)
    ax.set_title(f"WLTP SOC {soc}%", fontsize=12)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylabel("Voltage (mV)", fontsize=14)
    ax.set_xlabel("Time (s)", fontsize=14)
fig.legend(handles=[
    Line2D([0], [0], color="0.35", lw=1.6, label=r"$V_{data} - V_{SPMe}(\theta_{eff,i})$"),
    Line2D([0], [0], color="tab:blue", lw=1.8, label=r"$V_{data} - V_{SPMe}(\theta_{2C,s90})$"),
    Line2D([0], [0], color="red", lw=2.0, label=r"$V_{data} - V_{SPMe}(\theta_{combined})$")],
    loc="upper right", fontsize=16, ncol=3, bbox_to_anchor=(0.99, 0.96))
fig.suptitle(r"WLTP: Voltage Error under different $\theta_{eff,i}$", y=1.0, fontsize=17)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(f"meta_wltp_error_{MOD_TAG}.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"saved meta_wltp_error_{MOD_TAG}.png")


# %% cell 10: theta_eff cloud -- individual theta_eff_i (gray points) vs
# theta_combined (red: mean mu + 1sigma/2sigma cloud). sigma = sqrt(tau^2 + se_mu^2)
# (between-profile model-form spread + estimation). Truth = ratio 1.
from matplotlib.patches import Patch as _Patch
YWIN = 100.0                                                  # +/- window in percent
def _pct(ratio):
    return (ratio - 1.0) * 100.0
fig, ax = plt.subplots(figsize=(11, 6))
_rng = np.random.default_rng(0)
for j, l in enumerate(LAB):
    pct = _pct(np.array([eff[k]["theta_hat"][FP[j]] / NOM[j] for k in OK]))
    xj = j + _rng.uniform(-0.16, 0.16, size=len(pct))
    ax.scatter(xj, np.clip(pct, -YWIN, YWIN), s=20, color="0.5", alpha=0.6,
               zorder=2, edgecolors="none")
    n_hi, n_lo = int(np.sum(pct > YWIN)), int(np.sum(pct < -YWIN))   # off-scale counts
    if n_hi:
        ax.annotate(f"$\\uparrow${n_hi}", (j, YWIN * 0.94), ha="center", fontsize=8, color="0.4")
    if n_lo:
        ax.annotate(f"$\\downarrow${n_lo}", (j, -YWIN * 0.94), ha="center", fontsize=8, color="0.4")
    if l in results:
        R = results[l]
        mu, sig = R["mu"], np.sqrt(R["tau"] ** 2 + R["se_mu"] ** 2)
        for kσ, a in [(1.96, 0.15), (1.0, 0.32)]:            # nested sigma cloud (95% / 68%)
            ax.fill_between([j - 0.42, j + 0.42], _pct(10 ** (mu - kσ * sig)),
                            _pct(10 ** (mu + kσ * sig)), color="tab:red", alpha=a, zorder=1, lw=0)
        ax.plot([j - 0.42, j + 0.42], [_pct(10 ** mu)] * 2,
                color="darkred", lw=2.2, zorder=3)           # mean mu
ax.axhline(0.0, color="k", ls="--", lw=1.3, zorder=4)        # theta_true = 0%
ax.set_ylim(-YWIN, YWIN)
ax.set_xticks(range(len(LAB))); ax.set_xticklabels(LAB, fontsize=11)
ax.set_ylabel(r"$\theta_{eff}/\theta_{nom} - 1$ (%)")
ax.set_title(r"Normalized $\theta_{eff}$ and Confidence Interval")
ax.legend(handles=[
    Line2D([0], [0], marker="o", color="0.5", ls="none", label="individual theta_eff_i"),
    Line2D([0], [0], color="darkred", lw=2.2, label="theta_combined mean (mu)"),
    _Patch(facecolor="tab:red", alpha=0.3, label="theta_combined  ±1σ / ±2σ"),
    Line2D([0], [0], color="k", ls="--", label="theta_true (0%)")],
    fontsize=8, loc="lower left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"meta_combined_cloud_{MOD_TAG}.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"saved meta_combined_cloud_{MOD_TAG}.png")


# %% cell 11: box plot of each theta's RANK distribution across conditions (COMBINED red)
# y-axis order = individuals by name (1C,2C,5C,C3,hppc), COMBINED forced to the LAST (bottom) row
order = sorted([c for c in CLABELS if c != "COMBINED"]) + ["COMBINED"]
data = [ranks[cl] for cl in order]
fig, ax = plt.subplots(figsize=(10, 9))
bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.62, showmeans=True,
                whis=(0, 100),                       # whiskers = full min..max (no outlier dots)
                meanprops=dict(marker="D", markerfacecolor="k", markeredgecolor="k", markersize=4),
                medianprops=dict(color="black", lw=1.3))
for patch, cl in zip(bp["boxes"], order):
    patch.set_facecolor("tab:red" if cl == "COMBINED" else "0.75")
    patch.set_alpha(0.9 if cl == "COMBINED" else 0.55)
ax.set_yticklabels(order, fontsize=9)
for lbl, cl in zip(ax.get_yticklabels(), order):
    if cl == "COMBINED":
        lbl.set_color("darkred"); lbl.set_fontweight("bold")
ax.invert_yaxis()                                    # best median on top
ax.set_xlabel("Rank (-)")
ax.set_ylabel(r"$\theta_{eff,i}$ (-)")
ax.set_xlim(0.5, len(CLABELS) + 0.5)
ax.set_xticks([1, 5, 10, 15, len(CLABELS)])
ax.set_title(r"Rank Span of $\theta_{effective,i}$")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"meta_rank_box_{MOD_TAG}.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"saved meta_rank_box_{MOD_TAG}.png")


# %% cell 12: SEPARATE C-rate and SOC effects (partial-dependence plots)
# For each param, plot theta_eff vs ONE moderator with the OTHER moderator's fitted
# effect removed (partial residuals). Red line = fitted beta for that moderator (* if sig).
# -> pure C-rate effect (fig 1) and pure SOC effect (fig 2), no mixing.
crate_all = np.array([CRATE_EFF[k[0]] for k in OK]); xc_raw = np.log10(crate_all); xc_m = xc_raw.mean()
soc_all = np.array([k[1] for k in OK], float); xs_raw = soc_all / 100.0; xs_m = xs_raw.mean()
for MODER, fname, xlab, titl in [
        ("crate", f"meta_partial_crate_{MOD_TAG}", "C-rate (-)", "C-rate dependence"),
        ("soc", f"meta_partial_soc_{MOD_TAG}", "SOC (-)", "SOC dependence")]:
    if MODER == "soc" and not USE_SOC_MODERATOR:
        continue                                             # no SOC term fitted
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, l in enumerate(LAB):
        ax = axes[j // 3][j % 3]
        if l not in results:
            ax.axis("off"); continue
        R = results[l]; b = R["b"]; mu, bc = b[0], b[1]
        bs = b[2] if len(b) > 2 else 0.0
        se_bc = R["se"][1]; se_bs = R["se"][2] if len(R["se"]) > 2 else np.inf
        r = np.array([eff[k]["theta_hat"][FP[j]] / NOM[j] for k in OK])
        craw = np.array([eff[k]["crlb"][j] for k in OK])
        keep = np.clip(craw, CRLB_FLOOR, None) <= CRLB_MAX
        yk = np.log10(r[keep]); crk = crate_all[keep]; sok = soc_all[keep]
        if MODER == "crate":
            y_part = yk - bs * (sok / 100.0 - xs_m)          # remove SOC effect
            xd = crk; beta, sig = bc, abs(bc) > 2 * se_bc
            xline = np.logspace(np.log10(crate_all.min()), np.log10(crate_all.max()), 50)
            yline = mu + bc * (np.log10(xline) - xc_m)
            ax.set_xscale("log")
        else:
            y_part = yk - bc * (np.log10(crk) - xc_m)        # remove C-rate effect
            xd = sok / 100.0; beta, sig = bs, abs(bs) > 2 * se_bs
            xline = np.linspace(soc_all.min(), soc_all.max(), 50) / 100.0
            yline = mu + bs * (xline - xs_m)
        ax.scatter(xd, 10 ** y_part - 1, s=24, color="0.5", alpha=0.6, zorder=2)   # ratio-1 (-)
        ax.plot(xline, 10 ** yline - 1, color="tab:red", lw=2.2, zorder=3)
        ax.axhline(0.0, color="k", ls="--", lw=1.0, alpha=0.6)
        if MODER == "crate":                                 # ticks at C/3, 0.4(hppc), 1, 2, 5
            ax.set_xticks([1 / 3, 0.4, 1, 2, 5])
            ax.set_xticklabels(["1/3", "\n0.4", "1", "2", "5"])   # 0.4 staggered one line down
            ax.minorticks_off()
        ax.set_title(f"{l}   (beta={beta:+.3f}{'*' if sig else ''})", fontsize=10,
                     fontweight="bold" if sig else "normal")
        if j % 3 == 0:
            ax.set_ylabel(r"$\theta_{eff,i}/\theta_{nom} - 1$ (-)")
        if j // 3 == 1:
            ax.set_xlabel(xlab)
        ax.grid(alpha=0.3)
    fig.suptitle(titl, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{fname}.png", dpi=120, bbox_inches="tight")
    plt.show()
    print(f"saved {fname}.png")
