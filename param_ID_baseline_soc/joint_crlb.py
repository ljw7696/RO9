# %% Joint CRLB at theta_joint per region: FIM summed over the region's 5 profiles.
# (independent profiles -> Fisher information adds). Same normalization as crlb_eff.py.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, pickle
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import numpy as np
import pybamm
pybamm.set_logging_level("ERROR")
from utils import (make_base_params, make_model, run_model, make_experiment,
                   prepare_sensitivity_inputs, get_sensitivities, compute_fim)

FP = ["Negative electrode exchange-current density [A.m-2]",
      "Positive electrode exchange-current density [A.m-2]",
      "Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]",
      "Negative particle diffusivity [m2.s-1]", "Positive particle diffusivity [m2.s-1]"]
LAB = ["k-", "k+", "kappa", "De", "Ds-", "Ds+"]
OPTS = {"surface form": "differential", "contact resistance": "true"}
SIGMA = 1e-3

OUT = pickle.load(open("joint_fit_results.pkl", "rb"))

CR = {}
print(f"{'region':8}" + "".join(f"{l:>9}" for l in LAB) + "  (joint CRLB, relative %)")
for rg in sorted(OUT):
    theta = OUT[rg]["best"]["theta_hat"]
    F_total = np.zeros((len(FP), len(FP)))
    for (prof, soc) in OUT[rg]["conds"]:
        base = make_base_params("Chen2020", soc=soc / 100.0, sensitivity_ready=True)
        p, iv = prepare_sensitivity_inputs(base, FP, values=theta)
        sol = run_model(make_model("SPMe", options=OPTS), p, make_experiment(profile_id=prof),
                        inputs=iv, calculate_sensitivities=True,
                        solver=pybamm.IDAKLUSolver(options={"max_num_steps": 50000}))
        Sens = get_sensitivities(sol, "Voltage [V]", FP, theta_values=theta, normalize=True)
        F_total += compute_fim(Sens, FP, sigma=SIGMA)
    crlb = np.sqrt(np.clip(np.diag(np.linalg.pinv(F_total)), 0, None))
    CR[rg] = crlb
    print(f"region {rg}" + "".join(f"{c*100:>8.1f}%" for c in crlb))

pickle.dump(CR, open("joint_crlb.pkl", "wb"))
print("\nsaved joint_crlb.pkl")
