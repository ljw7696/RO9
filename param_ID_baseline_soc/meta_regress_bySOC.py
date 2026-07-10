# %% Per-SOC-region meta-regression (intercept + C-rate slope, with n<3 fallback)
"""Split the 20 conditions into 4 SOC regions (5 profiles each) and run an
INDEPENDENT random-effects meta-regression within each region:

    y_ij = log10(theta_hat_ij / theta_nom_j) = mu + beta_c * x_c + b_ij + eps_ij
      b_ij  ~ N(0, tau^2)      between-profile model-form spread (within this SOC)
      eps_ij~ N(0, sigma_ij^2) within-fit CRLB noise, sigma_ij = CRLB_rel/ln10

Per region -> one theta_comb set (6 params) + tau + se + beta_c.  Total 4 sets.

Guards:
  - drop conditions with relative CRLB > CRLB_MAX (railed / unidentifiable)
  - if kept n < 3  -> fall back to intercept-only (beta_c degenerate on 2 pts)
  - if kept n < 2  -> flag parameter/region UNIDENTIFIABLE (no fit)
"""
import pickle
import numpy as np
from scipy.optimize import minimize_scalar

DSN = "Negative particle diffusivity [m2.s-1]"; DSP = "Positive particle diffusivity [m2.s-1]"
FP = ["Negative electrode exchange-current density [A.m-2]",
      "Positive electrode exchange-current density [A.m-2]",
      "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]", DSN, DSP]
LAB = ["k-", "k+", "kappa", "De", "Ds-", "Ds+"]
NOM = np.array([6.48e-7, 3.42e-6, 0.9487, 1.7694e-10, 3.3e-14, 4.0e-15])
CRLB_FLOOR, CRLB_MAX, TAU2_MAX, LN10 = 0.01, 0.50, 4.0, np.log(10.0)
USE_SLOPE = False  # False -> intercept-only (band = full scatter, matches point cloud)
                   # True  -> add beta_c C-rate slope (band = residual spread after trend)
NMIN_SLOPE = 3   # (if USE_SLOPE) need >=3 kept points to also fit beta_c
NMIN_FIT = 2     # need >=2 kept points to fit at all

eff = pickle.load(open("meta_cache_rc_long_rc_short_wide.pkl", "rb"))
ceff = pickle.load(open("meta_ceff_wide.pkl", "rb"))
OK = [k for k in sorted(eff) if eff[k].get("ok")]


def region(soc):
    return 1 if soc >= 80 else 2 if soc >= 60 else 3 if soc >= 48 else 4


RLAB = {1: "SOC~85-90", 2: "SOC~65-70", 3: "SOC~50-55", 4: "SOC~30-40"}
REGIONS = [1, 2, 3, 4]


def fit(y, sy, X):
    """REML for tau^2 + GLS for coefficients. Returns (beta, tau, se_beta)."""
    def negll(t2):
        W = 1 / (sy**2 + t2); A = X.T @ (W[:, None] * X)
        b = np.linalg.solve(A, X.T @ (W * y)); r = y - X @ b
        _, ld = np.linalg.slogdet(A)
        return np.sum(W * r**2) - np.sum(np.log(W)) + ld
    t2 = minimize_scalar(negll, bounds=(1e-10, TAU2_MAX), method="bounded").x
    W = 1 / (sy**2 + t2); cov = np.linalg.inv(X.T @ (W[:, None] * X))
    b = cov @ (X.T @ (W * y))
    return b, np.sqrt(t2), np.sqrt(np.diag(cov))


# results[region][param] = dict(theta_comb, ratio, mu, tau, se_mu, beta_c, se_bc, n, mode)
results = {rg: {} for rg in REGIONS}
for rg in REGIONS:
    for j, l in enumerate(LAB):
        ks = [k for k in OK if region(k[1]) == rg and eff[k]["crlb"][j] <= CRLB_MAX]
        n = len(ks)
        rec = dict(n=n, param=l, region=rg)
        if n < NMIN_FIT:
            rec.update(mode="UNIDENTIFIABLE", theta_comb=np.nan, ratio=np.nan,
                       mu=np.nan, tau=np.nan, se_mu=np.nan, beta_c=np.nan, se_bc=np.nan)
            results[rg][l] = rec
            continue
        y = np.array([np.log10(eff[k]["theta_hat"][FP[j]] / NOM[j]) for k in ks])
        sy = np.clip([eff[k]["crlb"][j] for k in ks], CRLB_FLOOR, None) / LN10
        xc = np.log10([ceff[k[0]] for k in ks]); xc = xc - xc.mean()
        if USE_SLOPE and n >= NMIN_SLOPE:
            b, tau, se = fit(y, sy, np.vstack([np.ones(n), xc]).T)
            mu, se_mu, beta_c, se_bc, mode = b[0], se[0], b[1], se[1], "mu+beta_c"
        else:  # intercept-only
            b, tau, se = fit(y, sy, np.ones((n, 1)))
            mode = "intercept-only" if (n >= NMIN_SLOPE or not USE_SLOPE) else "intercept-only(fallback)"
            mu, se_mu, beta_c, se_bc = b[0], se[0], np.nan, np.nan
        rec.update(mode=mode, theta_comb=NOM[j] * 10**mu, ratio=10**mu, mu=mu,
                   tau=tau, se_mu=se_mu, beta_c=beta_c, se_bc=se_bc)
        results[rg][l] = rec

# ---- print: one clean table per SOC region ----
def pct(logsig):
    """convert a log10 sigma to a symmetric-ish % band."""
    return (10**logsig - 1) * 100


print(f"\nPer-SOC-region meta-regression  (common beta_c, n<{NMIN_SLOPE} -> intercept-only)")
print("theta_comb = center;  se = CI of center;  tau = model-form spread;  "
      "band = sqrt(tau^2+se^2) = total\n")
for rg in REGIONS:
    print(f"===== region {rg}: {RLAB[rg]} =====")
    print(f"  {'param':6}{'theta_comb':>12}{'dev%':>8}{'se(center)':>12}"
          f"{'tau(m-form)':>13}{'band(tot)':>11}{'beta_c':>9}{'n':>4}")
    for j, l in enumerate(LAB):
        r = results[rg][l]
        if r["mode"] == "UNIDENTIFIABLE":
            print(f"  {l:6}{'--':>12}{'':>8}{'':>12}{'':>13}"
                  f"{'':>11}{'':>9}{r['n']:>4}   UNIDENTIFIABLE")
            continue
        band = np.sqrt(r["tau"]**2 + r["se_mu"]**2)
        bc = "  --  " if np.isnan(r["beta_c"]) else f"{r['beta_c']:+.2f}"
        flag = "  <-fallback" if r["mode"].startswith("intercept") else ""
        print(f"  {l:6}{r['theta_comb']:>12.3e}{(r['ratio']-1)*100:>+7.0f}%"
              f"{pct(r['se_mu']):>+11.0f}%{pct(r['tau']):>+12.0f}%"
              f"{pct(band):>+10.0f}%{bc:>9}{r['n']:>4}{flag}")
    print()

pickle.dump(results, open("results_bySOC.pkl", "wb"))
print("\nsaved results_bySOC.pkl")
