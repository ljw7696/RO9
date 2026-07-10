# %% 2x2 cloud plots, one per SOC region (same style as meta_combined_cloud)
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DSN = "Negative particle diffusivity [m2.s-1]"; DSP = "Positive particle diffusivity [m2.s-1]"
FP = ["Negative electrode exchange-current density [A.m-2]",
      "Positive electrode exchange-current density [A.m-2]",
      "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]", DSN, DSP]
LAB = ["k-", "k+", "kappa", "De", "Ds-", "Ds+"]
NOM = np.array([6.48e-7, 3.42e-6, 0.9487, 1.7694e-10, 3.3e-14, 4.0e-15])
YWIN = 100.0

eff = pickle.load(open("meta_cache_rc_long_rc_short_wide.pkl", "rb"))
results = pickle.load(open("results_bySOC.pkl", "rb"))
OK = [k for k in sorted(eff) if eff[k].get("ok")]


def region(soc):
    return 1 if soc >= 80 else 2 if soc >= 60 else 3 if soc >= 48 else 4


RLAB = {1: "SOC ~85-90", 2: "SOC ~65-70", 3: "SOC ~50-55", 4: "SOC ~30-40"}


def _pct(ratio):
    return (ratio - 1.0) * 100.0


fig, axes = plt.subplots(2, 2, figsize=(15, 11))
for rg, ax in zip([1, 2, 3, 4], axes.flat):
    ks = [k for k in OK if region(k[1]) == rg]
    rng = np.random.default_rng(0)
    for j, l in enumerate(LAB):
        pct = _pct(np.array([eff[k]["theta_hat"][FP[j]] / NOM[j] for k in ks]))
        xj = j + rng.uniform(-0.16, 0.16, size=len(pct))
        ax.scatter(xj, np.clip(pct, -YWIN, YWIN), s=20, color="0.5", alpha=0.6,
                   zorder=2, edgecolors="none")
        n_hi, n_lo = int(np.sum(pct > YWIN)), int(np.sum(pct < -YWIN))
        if n_hi:
            ax.annotate(f"$\\uparrow${n_hi}", (j, YWIN * 0.94), ha="center", fontsize=8, color="0.4")
        if n_lo:
            ax.annotate(f"$\\downarrow${n_lo}", (j, -YWIN * 0.94), ha="center", fontsize=8, color="0.4")
        R = results[rg][l]
        if R["mode"] == "UNIDENTIFIABLE" or np.isnan(R["mu"]):
            continue
        mu, sig = R["mu"], np.sqrt(R["tau"] ** 2 + R["se_mu"] ** 2)
        for ks_, a in [(1.96, 0.15), (1.0, 0.32)]:
            ax.fill_between([j - 0.42, j + 0.42], _pct(10 ** (mu - ks_ * sig)),
                            _pct(10 ** (mu + ks_ * sig)), color="tab:red", alpha=a, zorder=1, lw=0)
        ax.plot([j - 0.42, j + 0.42], [_pct(10 ** mu)] * 2, color="darkred", lw=2.2, zorder=3)
    ax.axhline(0.0, color="k", ls="--", lw=1.3, zorder=4)
    ax.set_ylim(-YWIN, YWIN)
    ax.set_xticks(range(len(LAB))); ax.set_xticklabels(LAB, fontsize=11)
    ax.set_ylabel(r"$\theta_{eff}/\theta_{nom} - 1$ (%)")
    ax.set_title(RLAB[rg], fontsize=13)
    ax.grid(alpha=0.3)

handles = [
    Line2D([0], [0], marker="o", color="0.5", ls="none", label=r"individual $\theta_{eff,i}$"),
    Line2D([0], [0], color="darkred", lw=2.2, label=r"$\theta_{combined}$ ($\mu$)"),
    Patch(facecolor="tab:red", alpha=0.3, label=r"$\theta_{combined}$ ($\pm 1\sigma\,/\,\pm 2\sigma$)"),
    Line2D([0], [0], color="k", ls="--", lw=1.3, label=r"$\theta_{nom}$ (0%)")]
fig.legend(handles=handles, fontsize=11, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(r"Normalized $\theta_{eff}$ and Confidence Interval", fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0.02, 1, 0.97])
plt.savefig("meta_combined_cloud_bySOC.png", dpi=120, bbox_inches="tight")
print("saved meta_combined_cloud_bySOC.png")
