# %% Per-SOC-region rank box plots (rank distribution of each candidate across conditions)
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA = pickle.load(open("rmse_matrices_bySOC.pkl", "rb"))
RLAB = {1: "SOC ~85-90", 2: "SOC ~65-70", 3: "SOC ~50-55", 4: "SOC ~30-40"}


def ranks_of(RM):
    """rank within each column (1=best); return dict row_index -> list of ranks."""
    RM = np.asarray(RM)
    nrow, ncol = RM.shape
    R = np.zeros_like(RM, dtype=int)
    for ci in range(ncol):
        for r, ri in enumerate(np.argsort(RM[:, ci])):
            R[ri, ci] = r + 1
    return R


fig, axes = plt.subplots(2, 2, figsize=(15, 11))
for rg, ax in zip([1, 2, 3, 4], axes.flat):
    rows, conds, RM = DATA[rg]
    R = ranks_of(RM)
    n = len(rows)
    # y positions: first row at top; COMBINED (last) at bottom
    ypos = np.arange(n)[::-1]
    for ri in range(n):
        is_c = (rows[ri] == "COMBINED")
        bp = ax.boxplot(R[ri], positions=[ypos[ri]], vert=False, widths=0.6,
                        patch_artist=True, whis=(0, 100), showmeans=True,
                        medianprops=dict(color="k", lw=1.6),
                        meanprops=dict(marker="D", markerfacecolor="yellow",
                                       markeredgecolor="k", markersize=8),
                        boxprops=dict(facecolor="tab:red" if is_c else "0.6",
                                      alpha=0.75 if is_c else 0.55,
                                      edgecolor="darkred" if is_c else "0.3",
                                      lw=1.8 if is_c else 1.0),
                        whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"))
    ax.set_yticks(ypos)
    lab = [(r"$\theta_{combined}$" if r == "COMBINED"
            else r"$\theta_{" + r.replace("_", r"\_") + "}$") for r in rows]
    ax.set_yticklabels(lab, fontsize=12)
    for tl, r in zip(ax.get_yticklabels(), rows):
        if r == "COMBINED":
            tl.set_color("darkred"); tl.set_fontweight("bold")
    ax.set_xlim(0.5, n + 0.5); ax.set_xticks(range(1, n + 1))
    ax.set_xlabel("rank within condition (1 = best)", fontsize=13)
    ax.set_title(RLAB[rg], fontsize=15)
    ax.grid(axis="x", alpha=0.3)

fig.suptitle(r"Rank span of each $\theta_{eff,i}$  (per SOC region)", fontsize=18, y=0.99)
fig.legend(handles=[
    Line2D([0], [0], color="k", lw=1.6, label="median rank"),
    Line2D([0], [0], marker="D", markerfacecolor="yellow", markeredgecolor="k",
           ls="none", markersize=8, label="mean rank")],
    fontsize=12, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.955))
fig.subplots_adjust(hspace=0.35, wspace=0.30, top=0.90)
plt.savefig("meta_rank_box_bySOC_2x2.png", dpi=120, bbox_inches="tight")
print("saved meta_rank_box_bySOC_2x2.png")
