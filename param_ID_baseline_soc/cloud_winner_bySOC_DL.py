# %% Per-SOC-region theta cloud in "winner" style (points + 1/2-sigma error bars, 2x2)
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
CRLB_MAX, LN10, YWIN = 0.50, np.log(10.0), 100.0

eff = pickle.load(open("meta_cache_rc_long_rc_short_wide.pkl", "rb"))
results = pickle.load(open("results_bySOC_DL.pkl", "rb"))   # DL
JOINT = pickle.load(open("joint_fit_results.pkl", "rb"))   # best theta_joint per region
JCRLB = pickle.load(open("joint_crlb.pkl", "rb"))          # joint CRLB (relative) per region
OK = [k for k in sorted(eff) if eff[k].get("ok")]


def region(soc):
    return 1 if soc >= 80 else 2 if soc >= 60 else 3 if soc >= 48 else 4


RLAB = {1: "SOC ~85-90", 2: "SOC ~65-70", 3: "SOC ~50-55", 4: "SOC ~30-40"}
PROFILES = ["C3_dchg", "hppc", "1C_dchg", "2C_dchg", "5C_dchg"]
PCOL = {"C3_dchg": "tab:blue", "hppc": "tab:orange", "1C_dchg": "tab:green",
        "2C_dchg": "tab:purple", "5C_dchg": "tab:brown"}
PLAB = {"C3_dchg": "C3", "hppc": "hppc", "1C_dchg": "1C", "2C_dchg": "2C", "5C_dchg": "5C"}
OFF = np.linspace(-0.30, 0.30, 7)   # 5 profiles + theta_comb + theta_joint


def errbar(ax, xw, ratio, sig_log10, col, ms, marker="o"):
    lo2, hi2 = (ratio * 10 ** (-1.96 * sig_log10) - 1) * 100, (ratio * 10 ** (1.96 * sig_log10) - 1) * 100
    lo1, hi1 = (ratio * 10 ** (-1.0 * sig_log10) - 1) * 100, (ratio * 10 ** (1.0 * sig_log10) - 1) * 100
    ax.plot([xw, xw], [np.clip(lo2, -YWIN, YWIN), np.clip(hi2, -YWIN, YWIN)], color=col, lw=1.1, alpha=0.6, zorder=5)
    ax.plot([xw, xw], [np.clip(lo1, -YWIN, YWIN), np.clip(hi1, -YWIN, YWIN)], color=col, lw=2.4, alpha=0.9, zorder=6)
    ax.scatter(xw, np.clip((ratio - 1) * 100, -YWIN, YWIN), s=ms, color=col, marker=marker,
               edgecolors="k", linewidths=0.5, zorder=7)


handles = [Line2D([0], [0], marker="o", color=PCOL[p], mec="k", ls="none", ms=8, label=PLAB[p]) for p in PROFILES]
handles += [Line2D([0], [0], marker="*", color="tab:red", mec="k", ls="none", ms=15, label=r"$\theta_{combined}$"),
            Line2D([0], [0], marker="*", color="k", mec="w", ls="none", ms=15, label=r"$\theta_{joint}$ (+CRLB)"),
            Line2D([0], [0], color="0.3", lw=2.4, label=r"$\pm1\sigma$ (thick) / $\pm2\sigma$ (thin)"),
            Line2D([0], [0], color="k", ls="--", lw=1.3, label=r"$\theta_{nom}$ (0%)")]

for rg in [1, 2, 3, 4]:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    conds = {k[0]: k for k in OK if region(k[1]) == rg}   # profile -> condition key
    for j, l in enumerate(LAB):
        n_off = 0
        for i, prof in enumerate(PROFILES):
            k = conds.get(prof)
            if k is None:
                continue
            ratio = eff[k]["theta_hat"][FP[j]] / NOM[j]
            crlb = eff[k]["crlb"][j]
            pct = (ratio - 1) * 100
            if crlb > CRLB_MAX or abs(pct) > YWIN:       # railed / off-scale: clip point, no bar
                ax.scatter(j + OFF[i], np.clip(pct, -YWIN, YWIN), s=45, color=PCOL[prof],
                           marker="o", edgecolors="k", linewidths=0.4, alpha=0.6, zorder=4)
                if abs(pct) > YWIN:
                    n_off += 1
            else:
                errbar(ax, j + OFF[i], ratio, crlb / LN10, PCOL[prof], 55)
        if n_off:
            ax.annotate(f"$\\uparrow${n_off}", (j, YWIN * 0.9), ha="center", fontsize=10, color="0.4")
        R = results[rg][l]
        if R["mode"] != "UNIDENTIFIABLE" and not np.isnan(R["mu"]):
            sig = np.sqrt(R["tau"] ** 2 + R["se_mu"] ** 2)
            errbar(ax, j + OFF[5], R["ratio"], sig, "tab:red", 220, marker="*")
        # theta_joint (best mean-RMSE fit): black diamond + joint-CRLB error bars
        tj = JOINT[rg]["best"]["theta_hat"][FP[j]] / NOM[j]
        pctj = (tj - 1) * 100
        if abs(pctj) > YWIN:                                  # railed / off-scale: clip + arrow
            ax.scatter(j + OFF[6], np.clip(pctj, -YWIN, YWIN), s=200, color="k", marker="*",
                       edgecolors="w", linewidths=0.6, zorder=8)
            ax.annotate("$\\uparrow$" if pctj > 0 else "$\\downarrow$",
                        (j + OFF[6], YWIN * 0.86 * np.sign(pctj)), ha="center", fontsize=11, color="k")
        else:
            errbar(ax, j + OFF[6], tj, JCRLB[rg][j] / LN10, "k", 200, marker="*")
    ax.axhline(0.0, color="k", ls="--", lw=1.3, zorder=2)
    ax.set_ylim(-YWIN, YWIN)
    ax.set_xticks(range(6)); ax.set_xticklabels(LAB, fontsize=13)
    ax.set_ylabel(r"$\theta_{eff}/\theta_{nom} - 1$ (%)", fontsize=14)
    ax.set_title(rf"Normalized $\theta_{{eff}}$ and Confidence Interval  ({RLAB[rg]}) [DL]", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=handles, fontsize=11, ncol=2, loc="lower left")
    plt.tight_layout()
    fname = f"meta_winner_theta_cloud_region{rg}_DL.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"saved {fname}")
