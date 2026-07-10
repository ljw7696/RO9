# %% Region-1 winner cloud, DL with OUTLIERS INCLUDED (no CRLB cutoff) for theta_combined.
# Individual points + theta_joint unchanged; only the red star (theta_comb) uses all 5 conditions.
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DSN = "Negative particle diffusivity [m2.s-1]"; DSP = "Positive particle diffusivity [m2.s-1]"
FP = ["Negative electrode exchange-current density [A.m-2]",
      "Positive electrode exchange-current density [A.m-2]",
      "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]", DSN, DSP]
LAB = ["k-", "k+", "kappa", "De", "Ds-", "Ds+"]
NOM = np.array([6.48e-7, 3.42e-6, 0.9487, 1.7694e-10, 3.3e-14, 4.0e-15])
CRLB_FLOOR, LN10, YWIN = 0.01, np.log(10.0), 100.0

eff = pickle.load(open("meta_cache_rc_long_rc_short_wide.pkl", "rb"))
JOINT = pickle.load(open("joint_fit_results.pkl", "rb"))
JCRLB = pickle.load(open("joint_crlb.pkl", "rb"))
OK = [k for k in sorted(eff) if eff[k].get("ok")]
conds1 = {k[0]: k for k in OK if k[1] >= 80}          # profile -> condition (region 1)
all1 = [k for k in OK if k[1] >= 80]

PROFILES = ["C3_dchg", "hppc", "1C_dchg", "2C_dchg", "5C_dchg"]
PCOL = {"C3_dchg": "tab:blue", "hppc": "tab:orange", "1C_dchg": "tab:green",
        "2C_dchg": "tab:purple", "5C_dchg": "tab:brown"}
PLAB = {"C3_dchg": "C3", "hppc": "hppc", "1C_dchg": "1C", "2C_dchg": "2C", "5C_dchg": "5C"}
OFF = np.linspace(-0.30, 0.30, 7)


def fit_DL(y, sy):                                    # DL: mu, tau, se_mu (NO cutoff -> all 5)
    k = len(y); w0 = 1.0 / sy**2
    yb = np.sum(w0 * y) / np.sum(w0)
    Q = np.sum(w0 * (y - yb)**2)
    c = np.sum(w0) - np.sum(w0**2) / np.sum(w0)
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    w = 1.0 / (sy**2 + tau2); Sw = np.sum(w)
    return np.sum(w * y) / Sw, np.sqrt(tau2), 1.0 / np.sqrt(Sw)


# theta_comb for region 1 with ALL 5 conditions (outliers included)
comb1 = {}
for j in range(6):
    y = np.array([np.log10(eff[k]["theta_hat"][FP[j]] / NOM[j]) for k in all1])
    sy = np.clip([eff[k]["crlb"][j] for k in all1], CRLB_FLOOR, None) / LN10
    mu, tau, se = fit_DL(y, sy)
    comb1[j] = dict(ratio=10**mu, sig=np.sqrt(tau**2 + se**2))


def errbar(ax, xw, ratio, sig_log10, col, ms, marker="o"):
    lo2, hi2 = (ratio*10**(-1.96*sig_log10)-1)*100, (ratio*10**(1.96*sig_log10)-1)*100
    lo1, hi1 = (ratio*10**(-1.0*sig_log10)-1)*100, (ratio*10**(1.0*sig_log10)-1)*100
    ax.plot([xw, xw], [np.clip(lo2, -YWIN, YWIN), np.clip(hi2, -YWIN, YWIN)], color=col, lw=1.1, alpha=0.6, zorder=5)
    ax.plot([xw, xw], [np.clip(lo1, -YWIN, YWIN), np.clip(hi1, -YWIN, YWIN)], color=col, lw=2.4, alpha=0.9, zorder=6)
    ax.scatter(xw, np.clip((ratio-1)*100, -YWIN, YWIN), s=ms, color=col, marker=marker, edgecolors="k", linewidths=0.5, zorder=7)


fig, ax = plt.subplots(figsize=(11, 6.5))
for j, l in enumerate(LAB):
    n_off = 0
    for i, prof in enumerate(PROFILES):
        k = conds1.get(prof)
        if k is None:
            continue
        ratio = eff[k]["theta_hat"][FP[j]] / NOM[j]; crlb = eff[k]["crlb"][j]; pct = (ratio-1)*100
        if crlb > 0.50 or abs(pct) > YWIN:
            ax.scatter(j+OFF[i], np.clip(pct, -YWIN, YWIN), s=45, color=PCOL[prof], marker="o",
                       edgecolors="k", linewidths=0.4, alpha=0.6, zorder=4)
            if abs(pct) > YWIN:
                n_off += 1
        else:
            errbar(ax, j+OFF[i], ratio, crlb/LN10, PCOL[prof], 55)
    if n_off:
        ax.annotate(f"$\\uparrow${n_off}", (j, YWIN*0.9), ha="center", fontsize=10, color="0.4")
    errbar(ax, j+OFF[5], comb1[j]["ratio"], comb1[j]["sig"], "tab:red", 220, marker="*")   # comb (outliers incl)
    tj = JOINT[1]["best"]["theta_hat"][FP[j]] / NOM[j]; pctj = (tj-1)*100
    if abs(pctj) > YWIN:
        ax.scatter(j+OFF[6], np.clip(pctj, -YWIN, YWIN), s=200, color="k", marker="*", edgecolors="w", linewidths=0.6, zorder=8)
        ax.annotate("$\\uparrow$" if pctj > 0 else "$\\downarrow$", (j+OFF[6], YWIN*0.86*np.sign(pctj)), ha="center", fontsize=11, color="k")
    else:
        errbar(ax, j+OFF[6], tj, JCRLB[1][j]/LN10, "k", 200, marker="*")

ax.axhline(0.0, color="k", ls="--", lw=1.3, zorder=2)
ax.set_ylim(-YWIN, YWIN); ax.set_xticks(range(6)); ax.set_xticklabels(LAB, fontsize=13)
ax.set_ylabel(r"$\theta_{eff}/\theta_{nom} - 1$ (%)", fontsize=14)
ax.set_title(r"Normalized $\theta_{eff}$ and Confidence Interval  (SOC ~85-90) [DL, outliers incl.]", fontsize=13)
ax.grid(axis="y", alpha=0.3)
handles = [Line2D([0], [0], marker="o", color=PCOL[p], mec="k", ls="none", ms=8, label=PLAB[p]) for p in PROFILES]
handles += [Line2D([0], [0], marker="*", color="tab:red", mec="k", ls="none", ms=15, label=r"$\theta_{combined}$"),
            Line2D([0], [0], marker="*", color="k", mec="w", ls="none", ms=15, label=r"$\theta_{joint}$ (+CRLB)"),
            Line2D([0], [0], color="0.3", lw=2.4, label=r"$\pm1\sigma$ (thick) / $\pm2\sigma$ (thin)"),
            Line2D([0], [0], color="k", ls="--", lw=1.3, label=r"$\theta_{nom}$ (0%)")]
ax.legend(handles=handles, fontsize=11, ncol=2, loc="lower left")
plt.tight_layout()
plt.savefig("meta_winner_theta_cloud_region1_DL.png", dpi=120, bbox_inches="tight")
print("saved meta_winner_theta_cloud_region1_DL.png (outliers included)")
for j, l in enumerate(LAB):
    print(f"  {l:6} comb(incl) = {(comb1[j]['ratio']-1)*100:+.0f}%   band=+/-{(10**comb1[j]['sig']-1)*100:.0f}%")
