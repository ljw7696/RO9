# %% Imports
import time
import gc
import sys
from pathlib import Path

import pybamm
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

sys.path.append(str(Path.cwd().parent))
from utils import *

gc.collect()
pybamm.set_logging_level("ERROR")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# %% Setup
log("SCRIPT START")

initial_soc = 0.5
N_STARTS = 20          # 여기 바꾸면 됨
ADD_NOISE = True
NOISE_STD = 1e-3      # 1 mV
MAX_NFEV = 30

MODEL_OPTIONS = {
    "surface form": "differential",
    "contact resistance": "true",
}

experiment = make_experiment(profile_id="hppc")

theta_values_nominal = {
    "Negative electrode exchange-current density [A.m-2]": 6.48e-7,
    "Positive electrode exchange-current density [A.m-2]": 3.42e-6,
    "Negative electrode double-layer capacity [F.m-2]": 0.2,
    "Positive electrode double-layer capacity [F.m-2]": 0.2,
    "Negative electrode active material volume fraction": 0.75,
    "Positive electrode active material volume fraction": 0.665,
    "Cation transference number": 0.2594,
    "Electrolyte conductivity [S.m-1]": 0.9487,
    "Electrolyte diffusivity [m2.s-1]": 1.7694e-10,
    "Positive electrode porosity": 0.335,
    "Negative electrode porosity": 0.25,
    "Separator porosity": 0.47,
    "Negative particle diffusivity [m2.s-1]": 3.3e-14,
    "Positive particle diffusivity [m2.s-1]": 4.0e-15,
    "Negative electrode conductivity [S.m-1]": 215.0,
    "Positive electrode conductivity [S.m-1]": 0.18,
}

PARAM_LABELS_ASCII = {
    "Negative electrode exchange-current density [A.m-2]": "k-",
    "Positive electrode exchange-current density [A.m-2]": "k+",
    "Negative electrode double-layer capacity [F.m-2]": "Cdl-",
    "Positive electrode double-layer capacity [F.m-2]": "Cdl+",
    "Negative electrode active material volume fraction": "eps_s-",
    "Positive electrode active material volume fraction": "eps_s+",
    "Cation transference number": "t+",
    "Electrolyte conductivity [S.m-1]": "kappa",
    "Electrolyte diffusivity [m2.s-1]": "De",
    "Positive electrode porosity": "eps_e+",
    "Negative electrode porosity": "eps_e-",
    "Separator porosity": "eps_e_sep",
    "Negative particle diffusivity [m2.s-1]": "Ds-",
    "Positive particle diffusivity [m2.s-1]": "Ds+",
    "Negative electrode conductivity [S.m-1]": "sigma-",
    "Positive electrode conductivity [S.m-1]": "sigma+",
}

# 16개 전체
fit_params = list(theta_values_nominal.keys())

PARAM_BOUNDS = {
    "Negative electrode exchange-current density [A.m-2]": (1e-9, 1e-5),
    "Positive electrode exchange-current density [A.m-2]": (1e-8, 1e-4),

    "Negative electrode double-layer capacity [F.m-2]": (0.05, 0.8),
    "Positive electrode double-layer capacity [F.m-2]": (0.05, 0.8),

    "Negative electrode active material volume fraction": (0.1, 0.9),
    "Positive electrode active material volume fraction": (0.1, 0.9),

    "Cation transference number": (0.1, 0.45),

    "Electrolyte conductivity [S.m-1]": (0.3, 3.0),
    "Electrolyte diffusivity [m2.s-1]": (1e-12, 1e-9),

    "Positive electrode porosity": (0.1, 0.7),
    "Negative electrode porosity": (0.1, 0.7),
    "Separator porosity": (0.1, 0.7),

    "Negative particle diffusivity [m2.s-1]": (1e-17, 1e-13),
    "Positive particle diffusivity [m2.s-1]": (1e-17, 1e-13),

    "Negative electrode conductivity [S.m-1]": (50.0, 500.0),
    "Positive electrode conductivity [S.m-1]": (0.05, 5.0),
}

log_scaled_params = [
    "Negative electrode exchange-current density [A.m-2]",
    "Positive electrode exchange-current density [A.m-2]",
    "Electrolyte conductivity [S.m-1]",
    "Electrolyte diffusivity [m2.s-1]",
    "Negative particle diffusivity [m2.s-1]",
    "Positive particle diffusivity [m2.s-1]",
    "Negative electrode conductivity [S.m-1]",
    "Positive electrode conductivity [S.m-1]",
]

sigma_V_fit = 1e-3


# %% Transform helpers
def check_bounds(params):
    for p in params:
        if p not in PARAM_BOUNDS:
            raise KeyError(f"No bound defined for: {p}")

        lo, hi = PARAM_BOUNDS[p]
        val = theta_values_nominal[p]

        if not (lo <= val <= hi):
            raise ValueError(
                f"Nominal outside bounds for {p}: "
                f"lo={lo}, nominal={val}, hi={hi}"
            )


def theta_dim_to_bar(theta_dim, params):
    theta_bar = []

    for p in params:
        val = float(theta_dim[p])
        lo, hi = PARAM_BOUNDS[p]

        if p in log_scaled_params:
            z = (np.log(val) - np.log(lo)) / (np.log(hi) - np.log(lo))
        else:
            z = (val - lo) / (hi - lo)

        theta_bar.append(z)

    return np.asarray(theta_bar)


def theta_bar_to_dim(theta_bar, params):
    theta_dim = {}

    for z, p in zip(theta_bar, params):
        lo, hi = PARAM_BOUNDS[p]
        z = float(np.clip(z, 0.0, 1.0))

        if p in log_scaled_params:
            theta_dim[p] = np.exp(np.log(lo) + z * (np.log(hi) - np.log(lo)))
        else:
            theta_dim[p] = lo + z * (hi - lo)

    return theta_dim


def build_param_values(theta_dim):
    p = make_base_params(
        "Chen2020",
        soc=initial_soc,
        sensitivity_ready=False,
    )

    for name, val in theta_values_nominal.items():
        p.update({name: val})

    for name, val in theta_dim.items():
        p.update({name: float(val)})

    return p


def solve_voltage(theta_dim, tag):
    t_start = time.time()

    log(f"{tag}: build parameters")
    p = build_param_values(theta_dim)

    log(f"{tag}: build model")
    model = make_model("SPMe", options=MODEL_OPTIONS)

    log(f"{tag}: solve start")
    sol = run_model(model, p, experiment)

    log(f"{tag}: solve done ({time.time() - t_start:.2f} sec total)")

    t = sol["Time [s]"].entries
    V = sol["Voltage [V]"].entries

    return t, V


# %% Nominal synthetic data
log("Check parameter bounds")
check_bounds(fit_params)

log("Prepare nominal theta_bar")
theta_bar0 = theta_dim_to_bar(theta_values_nominal, fit_params)
theta_dim0 = theta_bar_to_dim(theta_bar0, fit_params)

log("Build nominal synthetic voltage data")
t_data, V_data_clean = solve_voltage(theta_dim0, tag="NOMINAL")
log(f"Nominal data ready: len={len(t_data)}, t_end={t_data[-1]:.2f}")

V_data = V_data_clean.copy()

if ADD_NOISE:
    rng_noise = np.random.default_rng(0)
    V_data = V_data + rng_noise.normal(0.0, NOISE_STD, size=V_data.shape)
    log(f"Added Gaussian voltage noise: std={NOISE_STD * 1e3:.2f} mV")


# %% Residual
eval_count = {"n": 0}


def residual_voltage_only(theta_bar):
    eval_count["n"] += 1
    n = eval_count["n"]

    print("\n" + "=" * 70, flush=True)
    log(f"EVAL {n} START")

    t_eval = time.time()
    theta_dim = theta_bar_to_dim(theta_bar, fit_params)

    for p_name in fit_params:
        label = PARAM_LABELS_ASCII.get(p_name, p_name)
        print(f"    {label:10s}: {theta_dim[p_name]:.4e}", flush=True)

    try:
        t_fit, V_fit = solve_voltage(theta_dim, tag=f"EVAL {n}")

        V_fit_q = interp1d(
            t_fit,
            V_fit,
            bounds_error=False,
            fill_value="extrapolate",
        )(t_data)

        r = (V_fit_q - V_data) / sigma_V_fit

        if np.any(~np.isfinite(r)):
            log(f"EVAL {n}: non-finite residual -> penalty")
            r = np.ones_like(V_data) * 1e6

    except Exception as e:
        log(f"EVAL {n}: failed -> {repr(e)}")
        r = np.ones_like(V_data) * 1e6

    rmse_mV = np.sqrt(np.mean((r * sigma_V_fit) ** 2)) * 1e3

    log(
        f"EVAL {n} DONE | RMSE={rmse_mV:.6f} mV "
        f"| eval_time={time.time() - t_eval:.2f} sec"
    )

    return r


# %% Initial residual check
log("Check initial residual at nominal")
r0 = residual_voltage_only(theta_bar0)
rmse0 = np.sqrt(np.mean((r0 * sigma_V_fit) ** 2)) * 1e3
log(f"Initial nominal RMSE = {rmse0:.6f} mV")


# %% Multi-start TRF
all_results = []

for start_id in range(N_STARTS):

    print("\n" + "#" * 90, flush=True)
    log(f"START {start_id + 1}/{N_STARTS}: random uniform initialization")

    rng = np.random.default_rng(start_id)

    theta_bar_init = rng.uniform(
        low=0.0,
        high=1.0,
        size=len(fit_params),
    )

    print("\nRandom initial guess", flush=True)
    for p, z0, zinit in zip(fit_params, theta_bar0, theta_bar_init):
        print(
            f"{PARAM_LABELS_ASCII.get(p,p):10s} "
            f"theta_bar_nom={z0:.3f} "
            f"theta_bar_init={zinit:.3f}",
            flush=True,
        )

    eval_count["n"] = 0

    log(f"START {start_id + 1}: TRF optimization start")

    res = least_squares(
        residual_voltage_only,
        theta_bar_init,
        bounds=(np.zeros(len(fit_params)), np.ones(len(fit_params))),
        method="trf",
        max_nfev=MAX_NFEV,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        verbose=2,
    )

    log(f"START {start_id + 1}: TRF optimization finished")

    theta_hat = theta_bar_to_dim(res.x, fit_params)

    r_hat = residual_voltage_only(res.x)
    rmse_hat = np.sqrt(np.mean((r_hat * sigma_V_fit) ** 2)) * 1e3

    result = {
        "start_id": start_id,
        "success": res.success,
        "cost": res.cost,
        "rmse_mV": rmse_hat,
        "theta_init": theta_bar_to_dim(theta_bar_init, fit_params),
        "theta_hat": theta_hat,
        "theta_bar_init": theta_bar_init.copy(),
        "theta_bar_hat": res.x.copy(),
    }

    all_results.append(result)

    print("\nEstimated parameters", flush=True)
    for p in fit_params:
        print(
            f"{PARAM_LABELS_ASCII.get(p,p):10s} "
            f"nominal={theta_values_nominal[p]:.4e} "
            f"estimate={theta_hat[p]:.4e} "
            f"ratio={theta_hat[p] / theta_values_nominal[p]:.3f}",
            flush=True,
        )

    log(
        f"START {start_id + 1} DONE | "
        f"success={res.success} | cost={res.cost:.3e} | RMSE={rmse_hat:.6f} mV"
    )


# %% Summary
print("\nInitial -> Final parameter ratios", flush=True)

for result in all_results:
    print("\n" + "-" * 80, flush=True)
    print(
        f"start={result['start_id']} "
        f"success={result['success']} "
        f"cost={result['cost']:.3e} "
        f"RMSE={result['rmse_mV']:.6f} mV",
        flush=True,
    )

    for p in fit_params:
        label = PARAM_LABELS_ASCII.get(p, p)
        init_ratio = result["theta_init"][p] / theta_values_nominal[p]
        final_ratio = result["theta_hat"][p] / theta_values_nominal[p]

        print(
            f"{label:10s} "
            f"init={init_ratio:9.3f} "
            f"final={final_ratio:9.3f}",
            flush=True,
        )

log("SCRIPT END")




# %%
import pickle
import time

fname = f"multistart_results_{time.strftime('%Y%m%d_%H%M%S')}.pkl"

with open(fname, "wb") as f:
    pickle.dump(all_results, f)

print("saved:", fname)




# %% Plot SOC trajectories from multistart results
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 0. SOC endpoint: nominal parameter 기준으로 한 번만 계산
# ============================================================

p_ref = make_base_params(
    "Chen2020",
    soc=initial_soc,
    sensitivity_ready=False,
)

for name, val in theta_values_nominal.items():
    p_ref.update({name: val})

endpoints_ref = get_soc_endpoints(p_ref)

print("SOC endpoints:")
print(endpoints_ref)


# ============================================================
# 1. simulate one parameter set
# ============================================================

def simulate_from_theta_hat(theta_hat, tag=""):
    p = make_base_params(
        "Chen2020",
        soc=initial_soc,
        sensitivity_ready=False,
    )

    # nominal 전체 먼저 넣고
    for name, val in theta_values_nominal.items():
        p.update({name: val})

    # fitted parameter overwrite
    for name, val in theta_hat.items():
        p.update({name: float(val)})

    model = make_model("SPMe", options=MODEL_OPTIONS)
    sol = run_model(model, p, experiment)

    t = sol["Time [s]"].entries
    V = sol["Voltage [V]"].entries

    csn = sol["Average negative particle concentration [mol.m-3]"].entries
    csp = sol["Average positive particle concentration [mol.m-3]"].entries

    # 여기서는 get_soc_endpoints(p)를 다시 호출하지 않음
    soc_n = (csn - endpoints_ref["csn_0"]) / (
        endpoints_ref["csn_100"] - endpoints_ref["csn_0"]
    )

    soc_p = (csp - endpoints_ref["csp_100"]) / (
        endpoints_ref["csp_0"] - endpoints_ref["csp_100"]
    )

    return t, V, soc_n, soc_p


# ============================================================
# 2. nominal trajectory
# ============================================================

t_nom, V_nom, socn_nom, socp_nom = simulate_from_theta_hat(
    theta_values_nominal,
    tag="nominal",
)


# ============================================================
# 3. fitted trajectories
# ============================================================

traj = []

for r in all_results:
    try:
        theta_hat = r["theta_hat"]

        t, V, soc_n, soc_p = simulate_from_theta_hat(
            theta_hat,
            tag=f"start {r['start_id']}",
        )

        traj.append({
            "start_id": r["start_id"],
            "rmse_mV": r["rmse_mV"],
            "t": t,
            "V": V,
            "soc_n": soc_n,
            "soc_p": soc_p,
        })

        print(f"ok start={r['start_id']} RMSE={r['rmse_mV']:.3f} mV")

    except Exception as e:
        print(f"skip start={r['start_id']} error={repr(e)}")


# ============================================================
# 4. Voltage plot
# ============================================================

plt.figure(figsize=(8, 4))

for tr in traj:
    plt.plot(
        tr["t"],
        tr["V"],
        alpha=0.35,
        label=f"start {tr['start_id']}",
    )

plt.plot(
    t_nom,
    V_nom,
    "k--",
    linewidth=2,
    label="nominal",
)

plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("Voltage trajectories from multistart parameter sets")
plt.legend(ncol=2, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 5. SOC_n plot
# ============================================================

plt.figure(figsize=(8, 4))

for tr in traj:
    plt.plot(
        tr["t"],
        tr["soc_n"],
        alpha=0.35,
        label=f"start {tr['start_id']}",
    )

plt.plot(
    t_nom,
    socn_nom,
    "k--",
    linewidth=2,
    label="nominal",
)

plt.xlabel("Time [s]")
plt.ylabel("SOC_n")
plt.title("Negative electrode SOC trajectories")
plt.legend(ncol=2, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 6. SOC_p plot
# ============================================================

plt.figure(figsize=(8, 4))

for tr in traj:
    plt.plot(
        tr["t"],
        tr["soc_p"],
        alpha=0.35,
        label=f"start {tr['start_id']}",
    )

plt.plot(
    t_nom,
    socp_nom,
    "k--",
    linewidth=2,
    label="nominal",
)

plt.xlabel("Time [s]")
plt.ylabel("SOC_p")
plt.title("Positive electrode SOC trajectories")
plt.legend(ncol=2, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
