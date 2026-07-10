# %% Report the joint-fit multistart results per SOC region (like the rate-sweep summary)
import pickle
import numpy as np

FP = ["Negative electrode exchange-current density [A.m-2]",
      "Positive electrode exchange-current density [A.m-2]",
      "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]",
      "Negative particle diffusivity [m2.s-1]", "Positive particle diffusivity [m2.s-1]"]
LAB = ["k-", "k+", "kappa", "De", "Ds-", "Ds+"]
NOM = np.array([6.48e-7, 3.42e-6, 0.9487, 1.7694e-10, 3.3e-14, 4.0e-15])
RLAB = {1: "SOC~85-90", 2: "SOC~65-70", 3: "SOC~50-55", 4: "SOC~30-40"}

OUT = pickle.load(open("joint_fit_results.pkl", "rb"))
comb = pickle.load(open("results_bySOC.pkl", "rb"))


def pdev(theta_hat, j):
    v = theta_hat.get(FP[j], np.nan)
    return (v / NOM[j] - 1) * 100 if np.isfinite(v) else np.nan


for rg in sorted(OUT):
    d = OUT[rg]; R = d["all"]; conds = d["conds"]
    plabs = [c[0].replace("_dchg", "") for c in conds]
    print("=" * 96)
    print(f"REGION {rg}  ({RLAB[rg]})   profiles: {[f'{c[0]}_s{c[1]}' for c in conds]}")
    print("=" * 96)
    # ---- per-start table (best -> worst) ----
    print(f"{'rank':>4} {'start':>5} {'mean_rmse':>10} {'stall':>6}  | per-profile RMSE (mV): "
          + " ".join(f"{p:>6}" for p in plabs))
    for i, r in enumerate(R, 1):
        pr = (" ".join(f"{v:6.1f}" for v in r["rmses"]) if r.get("rmses") else " (stalled)")
        print(f"{i:>4} {r['start_id']:>5} {r['rmse_mV']:>10.3f} {str(r.get('stalled')):>6}  | {pr}")
    best = R[0]["rmse_mV"]
    good = [r for r in R if (not r.get("stalled")) and r["rmse_mV"] <= best * 1.05]
    print(f"\n  best mean_rmse = {best:.3f} mV | {len(good)}/{len(R)} within 5% of best | "
          f"{sum(r.get('stalled') for r in R)}/{len(R)} stalled")
    # ---- parameter spread across the good starts (identifiability) ----
    print(f"\n  {'param':6}{'theta_comb%':>12}{'joint_best%':>12}{'good_min%':>11}{'good_max%':>11}{'spread%':>9}")
    for j, l in enumerate(LAB):
        tc = (comb[rg][LAB[j]]["theta_comb"] / NOM[j] - 1) * 100
        vb = pdev(R[0]["theta_hat"], j)
        vals = np.array([pdev(g["theta_hat"], j) for g in good])
        print(f"  {l:6}{tc:>+11.0f}%{vb:>+11.0f}%{vals.min():>+10.0f}%{vals.max():>+10.0f}%"
              f"{vals.max()-vals.min():>8.0f}%")
    print()
