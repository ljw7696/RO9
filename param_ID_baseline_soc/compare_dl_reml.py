# %% Compare tau^2 estimators: REML (current) vs DerSimonian-Laird (closed form).
# Same random-effects model, same data (log10 space, CRLB as known sigma_i).
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
eff = pickle.load(open("meta_cache_rc_long_rc_short_wide.pkl", "rb"))
OK = [k for k in sorted(eff) if eff[k].get("ok")]


def region(soc):
    return 1 if soc >= 80 else 2 if soc >= 60 else 3 if soc >= 48 else 4


RLAB = {1: "SOC~85-90", 2: "SOC~65-70", 3: "SOC~50-55", 4: "SOC~30-40"}


def fit_reml(y, sy):                      # intercept-only REML + GLS (current method)
    def negll(t2):
        W = 1 / (sy**2 + t2); Sw = W.sum(); mu = (W * y).sum() / Sw; r = y - mu
        return (W * r**2).sum() - np.log(W).sum() + np.log(Sw)
    t2 = minimize_scalar(negll, bounds=(1e-10, TAU2_MAX), method="bounded").x
    W = 1 / (sy**2 + t2); Sw = W.sum(); mu = (W * y).sum() / Sw
    return {"mu": mu, "tau": np.sqrt(t2), "se_mu": np.sqrt(1 / Sw),
            "se_pred": np.sqrt(t2 + 1 / Sw)}


def fit_DL(y, sy):                        # DerSimonian-Laird (closed form)
    k = len(y)
    w0 = 1.0 / sy**2
    yb = np.sum(w0 * y) / np.sum(w0)
    Q = np.sum(w0 * (y - yb)**2)
    c = np.sum(w0) - np.sum(w0**2) / np.sum(w0)
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    w = 1.0 / (sy**2 + tau2)
    mu = np.sum(w * y) / np.sum(w)
    se_mu = 1.0 / np.sqrt(np.sum(w))
    return {"mu": mu, "tau": np.sqrt(tau2), "se_mu": se_mu,
            "se_pred": np.sqrt(tau2 + se_mu**2)}


def pct(logsig):
    return (10**logsig - 1) * 100


print(f"{'':22}{'-- REML (current) --':^30}{'-- DerSimonian-Laird --':^30}")
print(f"{'region':10}{'param':6}{'mu%':>8}{'tau%':>8}{'se_mu%':>8}{'mu%':>10}{'tau%':>8}{'se_mu%':>8}")
for rg in [1, 2, 3, 4]:
    for j, l in enumerate(LAB):
        ks = [k for k in OK if region(k[1]) == rg and eff[k]["crlb"][j] <= CRLB_MAX]
        if len(ks) < 2:
            print(f"{RLAB[rg]:10}{l:6}   n={len(ks)} -> skip")
            continue
        y = np.array([np.log10(eff[k]["theta_hat"][FP[j]] / NOM[j]) for k in ks])
        sy = np.clip([eff[k]["crlb"][j] for k in ks], CRLB_FLOOR, None) / LN10
        R = fit_reml(y, sy); D = fit_DL(y, sy)
        print(f"{RLAB[rg]:10}{l:6}"
              f"{(10**R['mu']-1)*100:>+7.0f}%{pct(R['tau']):>7.0f}%{pct(R['se_mu']):>7.0f}%"
              f"{(10**D['mu']-1)*100:>+9.0f}%{pct(D['tau']):>7.0f}%{pct(D['se_mu']):>7.0f}%")
    print()
