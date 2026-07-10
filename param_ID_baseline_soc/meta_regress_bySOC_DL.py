# %% Per-SOC-region combine with DerSimonian-Laird (DL) tau^2 estimator (closed form).
# Same model/data as meta_regress_bySOC.py (intercept-only), only tau^2 method differs.
# Saves results_bySOC_DL.pkl with the SAME structure as results_bySOC.pkl.
import pickle
import numpy as np

DSN = "Negative particle diffusivity [m2.s-1]"; DSP = "Positive particle diffusivity [m2.s-1]"
FP = ["Negative electrode exchange-current density [A.m-2]",
      "Positive electrode exchange-current density [A.m-2]",
      "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]", DSN, DSP]
LAB = ["k-", "k+", "kappa", "De", "Ds-", "Ds+"]
NOM = np.array([6.48e-7, 3.42e-6, 0.9487, 1.7694e-10, 3.3e-14, 4.0e-15])
CRLB_FLOOR, CRLB_MAX, LN10 = 0.01, 0.50, np.log(10.0)
NMIN_FIT = 2

eff = pickle.load(open("meta_cache_rc_long_rc_short_wide.pkl", "rb"))
OK = [k for k in sorted(eff) if eff[k].get("ok")]
REGIONS = [1, 2, 3, 4]
RLAB = {1: "SOC~85-90", 2: "SOC~65-70", 3: "SOC~50-55", 4: "SOC~30-40"}


def region(soc):
    return 1 if soc >= 80 else 2 if soc >= 60 else 3 if soc >= 48 else 4


def fit_DL(y, sy):
    """DerSimonian-Laird: closed-form tau^2, then inverse-variance weighted mu."""
    k = len(y)
    w0 = 1.0 / sy**2                                   # fixed-effect weights
    yb = np.sum(w0 * y) / np.sum(w0)
    Q = np.sum(w0 * (y - yb)**2)                       # Cochran's Q
    c = np.sum(w0) - np.sum(w0**2) / np.sum(w0)
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0   # DerSimonian & Laird 1986
    w = 1.0 / (sy**2 + tau2)                           # random-effects weights
    mu = np.sum(w * y) / np.sum(w)
    se_mu = 1.0 / np.sqrt(np.sum(w))
    return mu, np.sqrt(tau2), se_mu


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
        mu, tau, se_mu = fit_DL(y, sy)
        rec.update(mode="intercept-only(DL)", theta_comb=NOM[j] * 10**mu, ratio=10**mu,
                   mu=mu, tau=tau, se_mu=se_mu, beta_c=np.nan, se_bc=np.nan)
        results[rg][l] = rec


def pct(logsig):
    return (10**logsig - 1) * 100


print("Per-SOC-region combine (DerSimonian-Laird tau^2)\n")
for rg in REGIONS:
    print(f"===== region {rg}: {RLAB[rg]} =====")
    print(f"  {'param':6}{'dev%':>8}{'se(center)':>12}{'tau':>10}{'n':>4}")
    for l in LAB:
        r = results[rg][l]
        if r["mode"] == "UNIDENTIFIABLE":
            print(f"  {l:6}{'--':>8}{'':>12}{'':>10}{r['n']:>4}  UNIDENTIFIABLE")
            continue
        print(f"  {l:6}{(r['ratio']-1)*100:>+7.0f}%{pct(r['se_mu']):>+11.0f}%"
              f"{pct(r['tau']):>+9.0f}%{r['n']:>4}")
    print()

pickle.dump(results, open("results_bySOC_DL.pkl", "wb"))
print("saved results_bySOC_DL.pkl")
