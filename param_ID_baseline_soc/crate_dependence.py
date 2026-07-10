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
LAB = [r"$k_-$", r"$k_+$", r"$\kappa$", r"$D_e$", r"$D_{s-}$", r"$D_{s+}$"]
NOM = np.array([6.48e-7, 3.42e-6, 0.9487, 1.7694e-10, 3.3e-14, 4.0e-15])
CRLB_MAX = 0.50
eff = pickle.load(open("meta_cache_rc_long_rc_short_wide.pkl", "rb"))
ceff = pickle.load(open("meta_ceff_wide.pkl", "rb"))
OK = [k for k in sorted(eff) if eff[k].get("ok")]


def region(s):
    return 1 if s >= 80 else 2 if s >= 60 else 3 if s >= 48 else 4


RLAB = {1: "SOC ~85-90", 2: "SOC ~65-70", 3: "SOC ~50-55", 4: "SOC ~30-40"}
RCOL = {1: "tab:red", 2: "tab:orange", 3: "tab:green", 4: "tab:blue"}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for j, ax in enumerate(axes.flat):
    for rg in [1, 2, 3, 4]:
        ks = [k for k in OK if region(k[1]) == rg]
        cr = np.array([ceff[k[0]] for k in ks])
        pct = np.array([(eff[k]["theta_hat"][FP[j]] / NOM[j] - 1) * 100 for k in ks])
        rel = np.array([eff[k]["crlb"][j] for k in ks])
        ident = rel <= CRLB_MAX
        # identifiable points (filled), railed (open)
        ax.scatter(cr[ident], np.clip(pct[ident], -100, 100), s=70, color=RCOL[rg],
                   edgecolors="k", linewidths=0.4, zorder=4)
        ax.scatter(cr[~ident], np.clip(pct[~ident], -100, 100), s=70, facecolors="none",
                   edgecolors=RCOL[rg], linewidths=1.3, zorder=4)
        # weighted linear fit over identifiable
        if ident.sum() >= 3:
            x = np.log10(cr[ident]); y = np.log10((pct[ident] / 100) + 1)
            w = 1 / np.clip(rel[ident], 0.01, None)**2
            X = np.vstack([np.ones(len(x)), x]).T
            beta = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
            xx = np.linspace(np.log10(0.3), np.log10(5.5), 20)
            yy = (10**(beta[0] + beta[1] * xx) - 1) * 100
            ax.plot(10**xx, yy, color=RCOL[rg], lw=1.8, alpha=0.8, zorder=3)
    ax.axhline(0, color="k", ls="--", lw=1.2)
    ax.set_xscale("log")
    ax.set_xticks([0.33, 1, 2, 5]); ax.set_xticklabels(["C/3", "1C", "2C", "5C"])
    ax.set_title(LAB[j], fontsize=17)
    ax.grid(alpha=0.3, which="both")
    if j % 3 == 0:
        ax.set_ylabel(r"$\theta/\theta_{nom}-1$ (%)", fontsize=13)
    if j >= 3:
        ax.set_xlabel("effective C-rate", fontsize=13)

handles = [Line2D([0], [0], marker="o", color=RCOL[r], mec="k", ls="-", lw=1.8, ms=9, label=RLAB[r]) for r in [1, 2, 3, 4]]
handles += [Line2D([0], [0], marker="o", color="0.4", mfc="none", ls="none", ms=9, label="railed (CRLB>50%)")]
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=12, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Effective parameter vs C-rate, within each SOC region", fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig("crate_dependence.png", dpi=120, bbox_inches="tight")
print("saved crate_dependence.png")
