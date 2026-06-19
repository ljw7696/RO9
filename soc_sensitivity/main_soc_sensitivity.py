# %% Import necessary libraries
import pybamm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
import random
import functools
import matplotlib as mpl
import plotly.graph_objects as go
import pickle
from plotly.subplots import make_subplots
from joblib import Parallel, delayed


from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze


import gc
gc.collect()

import logging
# Set the logging level to ERROR to suppress warnings
pybamm.set_logging_level("ERROR")



## Read supplementary files
from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent))



%load_ext autoreload
%autoreload 2

from utils import *
# from simplify import *




# %% Base Model Run
initial_soc = 0.3


######## Default model run ########
model = make_model("SPMe")

base_param = make_base_params("Chen2020", soc=initial_soc)

experiment = make_experiment(profile_id=4)

sol = run_model(model, base_param, experiment)


t_query = get_t_query(sol)

VIT = get_VIT(sol, t_query)

state_names = [
    "Average negative particle concentration [mol.m-3]",
    "Average positive particle concentration [mol.m-3]",
]
States = get_states(
    sol,
    state_names,
    t_query,
)

plot_results(VIT, kind="VIT")
plot_results(States, kind="states")



# %% Single Theta Sensitivity
initial_soc = 0.3


MODEL_OPTIONS = {
    "surface form": "differential",
    "contact resistance": "true",
}

model = make_model("SPMe", options=MODEL_OPTIONS)


base_param = make_base_params("Chen2020", soc=initial_soc,
                              sensitivity_ready=True)

experiment = make_experiment(profile_id="hppc")


theta_values_nominal = {
    "Negative electrode exchange-current density [A.m-2]": 6.48e-7, ##
    "Positive electrode exchange-current density [A.m-2]": 3.42e-6, ##

    "Negative electrode double-layer capacity [F.m-2]": 0.2,
    "Positive electrode double-layer capacity [F.m-2]": 0.2,

    "Negative electrode active material volume fraction": 0.75,
    "Positive electrode active material volume fraction": 0.665,

    "Cation transference number": 0.2594,

    "Electrolyte conductivity [S.m-1]": 0.9487, ##
    "Electrolyte diffusivity [m2.s-1]": 1.7694e-10, ##

    "Positive electrode porosity": 0.335,
    "Negative electrode porosity": 0.25,
    "Separator porosity": 0.47,

    "Negative particle diffusivity [m2.s-1]": 3.3e-14,
    "Positive particle diffusivity [m2.s-1]": 4.0e-15,

    "Negative electrode conductivity [S.m-1]": 215.0,
    "Positive electrode conductivity [S.m-1]": 0.18,

    # "Negative electrode thickness [m]",
    # "Positive electrode thickness [m]",
    # "Separator thickness [m]",
    
    # "Negative particle radius [m]",
    # "Positive particle radius [m]",
}


sensitivity_targets = list(theta_values_nominal.keys())


target_params, target_values = prepare_sensitivity_inputs(
    base_param,
    sensitivity_targets,
    values=theta_values_nominal,
)


sol = run_model(
    model,
    target_params,
    experiment,
    inputs=target_values,
    calculate_sensitivities=True,
)


t_query = get_t_query(sol)

VIT = get_VIT(sol, t_query)

state_names = [
    "Average negative particle concentration [mol.m-3]",
    "Average positive particle concentration [mol.m-3]",
]
States = get_states(
    sol,
    state_names,
    t_query,
)

Sens_V = get_sensitivities(
    sol,
    "Voltage [V]",
    sensitivity_targets,
    t_query,
    theta_values=target_values,
    normalize=True,
)

Sens_V_scaled = scale_voltage_sens_by_ocv(
    Sens_V,
    base_param,
)

Sens_csp = get_sensitivities(
    sol,
    "Average positive particle concentration [mol.m-3]",
    sensitivity_targets,
    t_query,
    theta_values=target_values,
    normalize=True,
)

Sens_csn = get_sensitivities(
    sol,
    "Average negative particle concentration [mol.m-3]",
    sensitivity_targets,
    t_query,
    theta_values=target_values,
    normalize=True,
)

plot_results(VIT, kind="VIT")
plot_results(States, kind="states")
plot_results(
    Sens_V_scaled,
    kind="sensitivities",
    max_cols=4,
    figsize_per_plot=(4, 3),
)


plot_sensitivity_comparison_yy(
    Sens_V_scaled,
    Sens_csp,
    label_left="Voltage sensitivity",
    label_right="csp sensitivity",
)

plot_sensitivity_comparison_yy(
    Sens_V_scaled,
    Sens_csn,
    label_left="Voltage sensitivity",
    label_right="csn sensitivity",
)


plot_sensitivity_comparison_yy(
    Sens_csn,
    Sens_csp,
    label_left="csn sensitivity",
    label_right="csp sensitivity",
)




# %% Normalized sensitivities plot
endpoints = get_soc_endpoints(base_param)


Sens_soc_n, Sens_soc_p = concentrations_to_soc_sensitivities(
    Sens_csn,
    Sens_csp,
    endpoints,
)


plot_sensitivity_comparison_yy(
    Sens_V_scaled,
    Sens_soc_p,
    label_left="V sensitivity",
    label_right="soc_p sensitivity",
    suptitle="Voltage-SOC_p Sensitivity Comparison",
)

plot_sensitivity_comparison_yy(
    Sens_V_scaled,
    Sens_soc_n,
    label_left="V sensitivity",
    label_right="soc_n sensitivity",
    suptitle="Voltage-SOC_n Sensitivity Comparison",
)

plot_sensitivity_comparison_yy(
    Sens_soc_p,
    Sens_soc_n,
    label_left="soc_p sensitivity",
    label_right="soc_n sensitivity",
    suptitle="SOC Sensitivity Comparison",
)



# %% FD sensitivity on t_query grid

V_span = (
    base_param["Open-circuit voltage at 100% SOC [V]"]
    - base_param["Open-circuit voltage at 0% SOC [V]"]
)

csn_span = endpoints["csn_100"] - endpoints["csn_0"]
csp_span = endpoints["csp_0"] - endpoints["csp_100"]

Sens_V_fd = {"t": t_query}
Sens_soc_n_fd = {"t": t_query}
Sens_soc_p_fd = {"t": t_query}

for name in sensitivity_targets:

    print("FD:", PARAM_LABELS.get(name, name))

    profiles = run_perturbed_profiles_single_update(
        name=name,
        initial_soc=initial_soc,
        experiment=experiment,
        theta_values_nominal=theta_values_nominal,
        model_options=MODEL_OPTIONS,
        frac=0.01,
    )

    def interp_to_tquery(case, key):
        return interp1d(
            profiles[case]["t"],
            profiles[case][key],
            bounds_error=False,
            fill_value=np.nan,
        )(t_query)

    V_low = interp_to_tquery("lower", "V")
    V_up = interp_to_tquery("upper", "V")

    csn_low = interp_to_tquery("lower", "csn_avg")
    csn_up = interp_to_tquery("upper", "csn_avg")

    csp_low = interp_to_tquery("lower", "csp_avg")
    csp_up = interp_to_tquery("upper", "csp_avg")

    S_V = (V_up - V_low) / (2.0 * 0.01)
    S_csn = (csn_up - csn_low) / (2.0 * 0.01)
    S_csp = (csp_up - csp_low) / (2.0 * 0.01)

    Sens_V_fd[name] = S_V / V_span
    Sens_soc_n_fd[name] = S_csn / csn_span
    Sens_soc_p_fd[name] = -S_csp / csp_span

    print(
        "  t_end lower/upper:",
        profiles["lower"]["t"][-1],
        profiles["upper"]["t"][-1],
    )


# %%
for p in sensitivity_targets:
    print(
        PARAM_LABELS.get(p, p),
        len(Sens_V_fd["t"]),
        len(np.asarray(Sens_V_fd[p]).squeeze())
    )




# %%
plot_sensitivity_comparison_yy(
    Sens_V_scaled,
    Sens_V_fd,
    label_left="sODE dV/dtheta",
    label_right="FD dV/dtheta",
    suptitle="Voltage sensitivity: sODE vs FD",
)

plot_sensitivity_comparison_yy(
    Sens_soc_n,
    Sens_soc_n_fd,
    label_left="sODE dSOC_n/dtheta",
    label_right="FD dSOC_n/dtheta",
    suptitle="SOC_n sensitivity: sODE vs FD",
)

plot_sensitivity_comparison_yy(
    Sens_soc_p,
    Sens_soc_p_fd,
    label_left="sODE dSOC_p/dtheta",
    label_right="FD dSOC_p/dtheta",
    suptitle="SOC_p sensitivity: sODE vs FD",
)











# %% Frequency sweep sensitivity + FIM with FD-corrected sigma for Parameter ranking
######################################################################################
######################################################################################
######################################################################################
profile_list = [
    "sin_100Hz",
    "sin_10Hz",
    "sin_1Hz",
    "sin_01Hz",
    "hppc",
]

soc_list = [0.3, 0.5]

freq_results = {}

for initial_soc in soc_list:

    freq_results[initial_soc] = {}

    print("\n" + "#" * 120)
    print(f"Initial SOC = {initial_soc}")
    print("#" * 120)

    for profile_id in profile_list:

        print("\n" + "=" * 100)
        print(profile_id)
        print("=" * 100)

        model = make_model("SPMe", options=MODEL_OPTIONS)

        base_param = make_base_params(
            "Chen2020",
            soc=initial_soc,
            sensitivity_ready=True,
        )

        experiment = make_experiment(profile_id=profile_id)

        target_params, target_values = prepare_sensitivity_inputs(
            base_param,
            sensitivity_targets,
            values=theta_values_nominal,
        )

        sol = run_model(
            model,
            target_params,
            experiment,
            inputs=target_values,
            calculate_sensitivities=True,
        )

        t_query = get_t_query(sol)

        Sens_V = get_sensitivities(
            sol,
            "Voltage [V]",
            sensitivity_targets,
            t_query,
            theta_values=target_values,
            normalize=True,
        )

        Sens_V_scaled = scale_voltage_sens_by_ocv(
            Sens_V,
            base_param,
        )

        Sens_csn = get_sensitivities(
            sol,
            "Average negative particle concentration [mol.m-3]",
            sensitivity_targets,
            t_query,
            theta_values=target_values,
            normalize=True,
        )

        Sens_csp = get_sensitivities(
            sol,
            "Average positive particle concentration [mol.m-3]",
            sensitivity_targets,
            t_query,
            theta_values=target_values,
            normalize=True,
        )

        endpoints = get_soc_endpoints(base_param)

        Sens_soc_n, Sens_soc_p = concentrations_to_soc_sensitivities(
            Sens_csn,
            Sens_csp,
            endpoints,
        )

        freq_results[initial_soc][profile_id] = {
            "Sens_V_scaled": Sens_V_scaled,
            "Sens_soc_n": Sens_soc_n,
            "Sens_soc_p": Sens_soc_p,
        }

        print_relative_sensitivity(
            Sens_V_scaled,
            params=sensitivity_targets,
            title=f"SOC {initial_soc} | {profile_id} | Voltage",
        )

        print_relative_sensitivity(
            Sens_soc_n,
            params=sensitivity_targets,
            title=f"SOC {initial_soc} | {profile_id} | SOC_n",
        )

        print_relative_sensitivity(
            Sens_soc_p,
            params=sensitivity_targets,
            title=f"SOC {initial_soc} | {profile_id} | SOC_p",
        )


####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################



# %%
# plain 라벨 (정렬용)
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

def print_measures_side_by_side(freq_results, profile_id, params, top_n=None):
    soc_keys = sorted(freq_results.keys())
    measures = [("V", "Sens_V_scaled"), ("SOC_n", "Sens_soc_n"), ("SOC_p", "Sens_soc_p")]

    def rms_rel(Sens):
        rms = {p: np.sqrt(np.mean(np.asarray(Sens[p]).squeeze()**2)) for p in params}
        total = sum(rms.values())
        return rms, {p: rms[p] / total for p in params}

    data = {}
    for _, mkey in measures:
        for soc in soc_keys:
            data[(mkey, soc)] = rms_rel(freq_results[soc][profile_id][mkey])

    ref_rms = data[("Sens_V_scaled", soc_keys[0])][0]
    order = sorted(params, key=lambda p: ref_rms[p], reverse=True)
    if top_n:
        order = order[:top_n]

    LABEL_W = 10           # 라벨 칸 고정폭
    CELL    = "{:>9.2e} {:>6.1f}%"   # RMS rel 한 칸

    print("\n" + "#" * 100)
    print(f"PROFILE: {profile_id}")
    print("#" * 100)

    # 헤더 1: measure 이름
    block_w = len(soc_keys) * 18      # 한 measure 블록 폭
    h1 = " " * LABEL_W
    for mname, _ in measures:
        h1 += "| " + mname.center(block_w)
    print(h1)

    # 헤더 2: soc 라벨
    h2 = " " * LABEL_W
    for _ in measures:
        h2 += "|"
        for soc in soc_keys:
            h2 += f"  s{soc}:RMS    rel "
    print(h2)
    print("-" * 100)

    for p in order:
        row = f"{PARAM_LABELS_ASCII.get(p, p):<{LABEL_W}}"
        for _, mkey in measures:
            row += "|"
            for soc in soc_keys:
                rms, rel = data[(mkey, soc)]
                row += " " + CELL.format(rms[p], 100*rel[p]) + " "
        print(row)


for profile_id in profile_list:
    print_measures_side_by_side(freq_results, profile_id, sensitivity_targets, top_n=None)



# %%

import numpy as np
V_span = 4.2 - 2.5
sigma_V, sigma_SOC = 1/1000/V_span, 0.3/100

def active_by_floor(r, floor):
    return [p for p in sensitivity_targets
            if np.sqrt(np.mean(np.asarray(r["Sens_V_scaled"][p]).squeeze()**2)) > floor]

for soc in [0.3, 0.5]:
    print("\n" + "#"*64)
    print(f"#  SOC = {soc}")
    print("#"*64)
    for prof in ["sin_100Hz","sin_10Hz","sin_1Hz","sin_01Hz","hppc"]:
        r = freq_results[soc][prof]
        params = active_by_floor(r, sigma_V)

        F_V = compute_fim(r["Sens_V_scaled"], params, sigma=sigma_V)
        F_n = compute_fim(r["Sens_soc_n"],   params, sigma=sigma_SOC)
        F_p = compute_fim(r["Sens_soc_p"],   params, sigma=sigma_SOC)

        ld_V  = np.linalg.slogdet(F_V)[1]
        ld_n  = np.linalg.slogdet(F_V+F_n)[1]
        ld_p  = np.linalg.slogdet(F_V+F_p)[1]
        ld_np = np.linalg.slogdet(F_V+F_n+F_p)[1]

        c_V  = np.linalg.cond(F_V)
        c_n  = np.linalg.cond(F_V+F_n)
        c_p  = np.linalg.cond(F_V+F_p)
        c_np = np.linalg.cond(F_V+F_n+F_p)

        C_V = np.linalg.pinv(F_V)
        imp_n  = np.diag(C_V)/np.diag(np.linalg.pinv(F_V+F_n))
        imp_p  = np.diag(C_V)/np.diag(np.linalg.pinv(F_V+F_p))
        imp_np = np.diag(C_V)/np.diag(np.linalg.pinv(F_V+F_n+F_p))

        print("\n" + "="*64)
        print(f"{prof}  (SOC {soc})   n={len(params)}")
        print(f"observable: {[PARAM_LABELS_ASCII.get(p,p) for p in params]}")
        print("="*64)
        print(f"{'':10s}{'V':>12s}{'+SOCn':>12s}{'+SOCp':>12s}{'+both':>12s}")
        print(f"{'logdet':10s}{ld_V:12.2f}{ld_n:12.2f}{ld_p:12.2f}{ld_np:12.2f}")
        print(f"{'Δlogdet':10s}{0:12.2f}{ld_n-ld_V:12.2f}{ld_p-ld_V:12.2f}{ld_np-ld_V:12.2f}")
        print(f"{'cond':10s}{c_V:12.1e}{c_n:12.1e}{c_p:12.1e}{c_np:12.1e}")
        print(f"\n  {'param':10s}{'+SOCn':>9s}{'+SOCp':>9s}{'+both':>9s}")
        for i, p in enumerate(params):
            mark = " <--" if imp_np[i] > 2 else ""
            print(f"  {PARAM_LABELS_ASCII.get(p,p):10s}{imp_n[i]:9.1f}{imp_p[i]:9.1f}{imp_np[i]:9.1f}{mark}")
        
####################################################################################
####################################################################################
####################################################################################
####################################################################################


# %%
import numpy as np
V_span = 4.2 - 2.5
sigma_V, sigma_SOC = 1/1000/V_span, 0.3/100

def active_by_floor(r, floor):
    return [p for p in sensitivity_targets
            if np.sqrt(np.mean(np.asarray(r["Sens_V_scaled"][p]).squeeze()**2)) > floor]

for soc in [0.3, 0.5]:
    print("\n" + "#"*64)
    print(f"#  SOC = {soc}")
    print("#"*64)
    for prof in ["sin_100Hz","sin_10Hz","sin_1Hz","sin_01Hz","hppc"]:
        r = freq_results[soc][prof]
        params = active_by_floor(r, sigma_V)
        F_V = compute_fim(r["Sens_V_scaled"], params, sigma=sigma_V)
        F_n = compute_fim(r["Sens_soc_n"], params, sigma=sigma_SOC)
        F_p = compute_fim(r["Sens_soc_p"], params, sigma=sigma_SOC)

        C_V  = np.linalg.pinv(F_V)
        C_n  = np.linalg.pinv(F_V+F_n)
        C_p  = np.linalg.pinv(F_V+F_p)
        C_np = np.linalg.pinv(F_V+F_n+F_p)

        # V 기준 σ/θ 계산 후 내림차순 정렬
        rows = []
        for i, p in enumerate(params):
            sV  = np.sqrt(abs(C_V[i,i]))*100
            sn  = np.sqrt(abs(C_n[i,i]))*100
            sp  = np.sqrt(abs(C_p[i,i]))*100
            snp = np.sqrt(abs(C_np[i,i]))*100
            rows.append((sV, sn, sp, snp, p))
        rows.sort(reverse=True)   # V 기준 내림차순

        print(f"\n{'='*60}")
        print(f"{prof}  (SOC {soc})  n={len(params)}  cond_V={np.linalg.cond(F_V):.1e}")
        print(f"{'='*60}")
        print(f"{'param':10s}{'V':>10s}{'+SOCn':>10s}{'+SOCp':>10s}{'+both':>10s}")
        for sV, sn, sp, snp, p in rows:
            print(f"{PARAM_LABELS_ASCII.get(p,p):10s}{sV:9.1f}%{sn:9.1f}%{sp:9.1f}%{snp:9.1f}%")
####################################################################################
####################################################################################
####################################################################################
####################################################################################




# %%
import numpy as np
import matplotlib.pyplot as plt

V_span = 4.2 - 2.5
sigma_V, sigma_SOC = 1/1000/V_span, 0.3/100

SYM = {
    "k-": r"$k^-$", "k+": r"$k^+$",
    "Cdl-": r"$C_{dl}^-$", "Cdl+": r"$C_{dl}^+$",
    "eps_s-": r"$\varepsilon_s^-$", "eps_s+": r"$\varepsilon_s^+$",
    "t+": r"$t_c^0$", "kappa": r"$\kappa$", "De": r"$D_e$",
    "eps_e+": r"$\varepsilon_e^+$", "eps_e-": r"$\varepsilon_e^-$",
    "eps_e_sep": r"$\varepsilon_e^{sep}$",
    "Ds-": r"$D_s^-$", "Ds+": r"$D_s^+$",
    "sigma-": r"$\sigma^-$", "sigma+": r"$\sigma^+$",
}
def sym(p):
    a = PARAM_LABELS_ASCII.get(p, p)
    return SYM.get(a, a)

def active_by_floor(r, floor):
    return [p for p in sensitivity_targets
            if np.sqrt(np.mean(np.asarray(r["Sens_V_scaled"][p]).squeeze()**2)) > floor]

cases = [("hppc", 0.3), ("hppc", 0.5), ("sin_01Hz", 0.3), ("sin_01Hz", 0.5)]
prof_name = {"hppc": "HPPC", "sin_01Hz": "0.1 Hz sin"}

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes = axes.flatten()

for ax, (prof, soc) in zip(axes, cases):
    r = freq_results[soc][prof]
    params = active_by_floor(r, sigma_V)
    F_V = compute_fim(r["Sens_V_scaled"], params, sigma=sigma_V)
    F_n = compute_fim(r["Sens_soc_n"], params, sigma=sigma_SOC)
    F_p = compute_fim(r["Sens_soc_p"], params, sigma=sigma_SOC)
    C_V  = np.linalg.pinv(F_V)
    C_np = np.linalg.pinv(F_V+F_n+F_p)

    sV  = np.array([np.sqrt(abs(C_V[i,i]))*100 for i in range(len(params))])
    snp = np.array([np.sqrt(abs(C_np[i,i]))*100 for i in range(len(params))])
    syms = [sym(p) for p in params]

    order = np.argsort(-sV)
    sV, snp = sV[order], snp[order]
    syms = [syms[i] for i in order]

    x = np.arange(len(syms)); w = 0.38
    ax.bar(x-w/2, sV,  w, label="V only",  color="#bbbbbb")
    ax.bar(x+w/2, snp, w, label="V + SOC$_n$ + SOC$_p$", color="#c0392b")
    ax.set_xticks(x); ax.set_xticklabels(syms, rotation=0, fontsize=16)
    ax.set_ylabel("σ/θ  (%)", fontsize=15)
    ax.set_title(f"{prof_name[prof]}  (SOC {soc})", fontsize=16, loc='left', fontweight='bold')
    ax.legend(fontsize=13, loc='upper right')
    ax.tick_params(axis='y', labelsize=13)

    for i in range(len(syms)):
        imp = sV[i]/snp[i]
        if imp > 1.5:
            ax.text(x[i], max(sV[i],snp[i])+1.5, f"{imp:.0f}×", ha='center',
                    fontsize=12, color='#c0392b', fontweight='bold')

plt.tight_layout()
plt.savefig("crlb_2x2.png", dpi=150, bbox_inches='tight')
plt.show()





# %% Correlation heatmaps: V only vs V + SOC_n + SOC_p

def cov_to_corr(C):
    d = np.sqrt(np.diag(C))
    d[d == 0] = np.nan
    return C / np.outer(d, d)


def plot_corr_from_fim(F, params, title):
    C = np.linalg.pinv(F)
    Corr = cov_to_corr(C)

    labels = [PARAM_LABELS_ASCII.get(p, p) for p in params]

    plt.figure(figsize=(7, 6))
    plt.imshow(Corr, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(params)), labels, rotation=90)
    plt.yticks(range(len(params)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def active_by_voltage_floor(r, params, sigma_V_scaled):
    active = []
    for p in params:
        s = np.asarray(r["Sens_V_scaled"][p]).squeeze()
        rms = np.sqrt(np.nanmean(s**2))
        if rms > sigma_V_scaled:
            active.append(p)
    return active


# noise setting
V_span = 4.2 - 2.5
sigma_V = 1e-3 / V_span     # 1 mV, scaled voltage
sigma_SOC = 0.3 / 100       # 0.3% SOC

profiles_to_plot = ["sin_01Hz", "hppc"]
socs_to_plot = [0.3, 0.5]

for soc in socs_to_plot:
    for profile_id in profiles_to_plot:

        r = freq_results[soc][profile_id]

        params = active_by_voltage_floor(
            r,
            sensitivity_targets,
            sigma_V_scaled=sigma_V,
        )

        print("\n" + "=" * 80)
        print(f"SOC={soc}, profile={profile_id}, n={len(params)}")
        print([PARAM_LABELS_ASCII.get(p, p) for p in params])

        F_V = compute_fim(
            r["Sens_V_scaled"],
            params=params,
            sigma=sigma_V,
        )

        F_n = compute_fim(
            r["Sens_soc_n"],
            params=params,
            sigma=sigma_SOC,
        )

        F_p = compute_fim(
            r["Sens_soc_p"],
            params=params,
            sigma=sigma_SOC,
        )

        F_both = F_V + F_n + F_p

        plot_corr_from_fim(
            F_V,
            params,
            title=f"{profile_id}, SOC={soc}: V only",
        )

        plot_corr_from_fim(
            F_both,
            params,
            title=f"{profile_id}, SOC={soc}: V + SOC_n + SOC_p",
        )




# %%
# %% Correlation reduction heatmap

# %% HPPC correlation reduction heatmap
for soc in [0.3, 0.5]:

    r = freq_results[soc]["hppc"]

    params = active_by_voltage_floor(
        r,
        sensitivity_targets,
        sigma_V,
    )

    F_V = compute_fim(
        r["Sens_V_scaled"],
        params=params,
        sigma=sigma_V,
    )

    F_n = compute_fim(
        r["Sens_soc_n"],
        params=params,
        sigma=sigma_SOC,
    )

    F_p = compute_fim(
        r["Sens_soc_p"],
        params=params,
        sigma=sigma_SOC,
    )

    F_SOC = F_V + F_n + F_p

    Corr_V = cov_to_corr(np.linalg.pinv(F_V))
    Corr_SOC = cov_to_corr(np.linalg.pinv(F_SOC))

    Delta = np.abs(Corr_V) - np.abs(Corr_SOC)

    np.fill_diagonal(Delta, 0)

    labels = [PARAM_LABELS_ASCII.get(p, p) for p in params]

    plt.figure(figsize=(7, 6))

    plt.imshow(
        Delta,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(
        label=r"$|Corr_V|-|Corr_{V+SOC}|$"
    )

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=90,
    )

    plt.yticks(
        range(len(labels)),
        labels,
    )

    plt.title(
        f"HPPC, SOC={soc}\nCorrelation reduction from SOC measurements"
    )

    plt.tight_layout()
    plt.show()
# %% Normalized FIM computation
######################################################################################
#####################################################################################################################
####################################################################################
#####################################################
V_span = 4.2 - 2.5

# ===== sigma 제외 토글 =====
EXCLUDE_SIGMA = True   # False로 하면 sigma 포함

SIGMA = ["Negative electrode conductivity [S.m-1]",
         "Positive electrode conductivity [S.m-1]"]

if EXCLUDE_SIGMA:
    fim_params = [p for p in sensitivity_targets if p not in SIGMA]
else:
    fim_params = list(sensitivity_targets)
# ===========================

cases = {
    "No noise":      {"sigma_V": None,          "sigma_SOC": None},
    "1mV + 0.3%SOC": {"sigma_V": 1/1000/V_span, "sigma_SOC": 0.3/100},
    "2mV + 0.6%SOC": {"sigma_V": 2/1000/V_span, "sigma_SOC": 0.6/100},
    "3mV + 0.9%SOC": {"sigma_V": 3/1000/V_span, "sigma_SOC": 0.9/100},
}


for case_name, noise in cases.items():

    print("\n" + "="*80)
    print(case_name)
    print("="*80)

    F_V = compute_fim(
        Sens_V_scaled,
        params=fim_params,
        sigma=noise["sigma_V"],
    )

    F_soc_n = compute_fim(
        Sens_soc_n,
        params=fim_params,
        sigma=noise["sigma_SOC"],
    )

    F_soc_p = compute_fim(
        Sens_soc_p,
        params=fim_params,
        sigma=noise["sigma_SOC"],
    )

    F_V_soc_n = F_V + F_soc_n
    F_V_soc_p = F_V + F_soc_p
    F_V_soc_np = F_V + F_soc_n + F_soc_p

    print("Condition numbers")
    print("V       :", np.linalg.cond(F_V))
    print("V+SOC_n :", np.linalg.cond(F_V_soc_n))
    print("V+SOC_p :", np.linalg.cond(F_V_soc_p))
    print("V+Both  :", np.linalg.cond(F_V_soc_np))

    print()

    base = np.linalg.slogdet(F_V)[1]

    print("Δlogdet")
    print("SOC_n :", np.linalg.slogdet(F_V_soc_n)[1] - base)
    print("SOC_p :", np.linalg.slogdet(F_V_soc_p)[1] - base)
    print("Both  :", np.linalg.slogdet(F_V_soc_np)[1] - base)

    print()

    C_V = np.linalg.pinv(F_V)
    C_np = np.linalg.pinv(F_V_soc_np)

    improve = np.diag(C_V) / np.diag(C_np)

    print("CRLB improvement")
    for p, r in zip(fim_params, improve):
        print(
            f"{PARAM_LABELS.get(p,p):25s}"
            f"{r:10.2f}"
        )



# %%
V_span = 4.2 - 2.5
SIGMA = ["Negative electrode conductivity [S.m-1]",
         "Positive electrode conductivity [S.m-1]"]

profile_list = ["sin_100Hz", "sin_10Hz", "sin_1Hz", "sin_01Hz", "hppc"]
soc_list = [0.3, 0.5]
sigma_V, sigma_SOC = 1/1000/V_span, 0.3/100

TOP_N = 8
FORCE_INCLUDE = {
    "sin_01Hz": ["Negative particle diffusivity [m2.s-1]"],
    "hppc":     ["Negative particle diffusivity [m2.s-1]"],
}

def select_params(r, profile_id, top_n):
    cand = [p for p in sensitivity_targets if p not in SIGMA]
    rms = {p: np.sqrt(np.mean(np.asarray(r["Sens_V_scaled"][p]).squeeze()**2))
           for p in cand}
    ranked = sorted(cand, key=lambda p: rms[p], reverse=True)[:top_n]
    for p in FORCE_INCLUDE.get(profile_id, []):
        if p not in ranked:
            ranked.append(p)
    return ranked

def cov_to_corr(C):
    d = np.sqrt(np.diag(C)); return C / np.outer(d, d)

for initial_soc in soc_list:
    for profile_id in profile_list:
        r = freq_results[initial_soc][profile_id]
        fim_params = select_params(r, profile_id, TOP_N)

        F_V = compute_fim(r["Sens_V_scaled"], params=fim_params, sigma=sigma_V)
        F_n = compute_fim(r["Sens_soc_n"], params=fim_params, sigma=sigma_SOC)
        F_p = compute_fim(r["Sens_soc_p"], params=fim_params, sigma=sigma_SOC)
        F_np = F_V + F_n + F_p

        cond_V = np.linalg.cond(F_V)
        dlogdet = np.linalg.slogdet(F_np)[1] - np.linalg.slogdet(F_V)[1]
        use = "OK" if cond_V < 1e12 else "BAD (singular)"

        C_V  = np.linalg.pinv(F_V)
        C_n  = np.linalg.pinv(F_V + F_n)
        C_p  = np.linalg.pinv(F_V + F_p)
        C_np = np.linalg.pinv(F_np)
        imp_n  = np.diag(C_V)/np.diag(C_n)
        imp_p  = np.diag(C_V)/np.diag(C_p)
        imp_np = np.diag(C_V)/np.diag(C_np)

        print("\n" + "="*62)
        print(f"{profile_id}  SOC={initial_soc}   n={len(fim_params)}  "
              f"cond_V={cond_V:.1e}  Δlogdet={dlogdet:.1f}   [{use}]")
        print("="*62)
        print(f"{'param':12s}{'+SOCn':>10s}{'+SOCp':>10s}{'+both':>10s}")
        for p, rn, rp, rnp in zip(fim_params, imp_n, imp_p, imp_np):
            mark = " <--" if rnp > 2 else ""
            print(f"{PARAM_LABELS_ASCII.get(p,p):12s}"
                  f"{rn:10.1f}{rp:10.1f}{rnp:10.1f}{mark}")

        # ===== correlation (루프 안으로 이동) =====
        R_V  = cov_to_corr(C_V)
        R_np = cov_to_corr(C_np)
        try:
            a = fim_params.index("Negative electrode active material volume fraction")
            b = fim_params.index("Positive electrode active material volume fraction")
            print(f"\ncorr(eps_s-,eps_s+):  V={R_V[a,b]:+.3f}  ->  V+SOC={R_np[a,b]:+.3f}")
        except ValueError:
            pass

        nn = len(fim_params)
        changes = []
        for i in range(nn):
            for j in range(i+1, nn):
                drop = abs(R_V[i,j]) - abs(R_np[i,j])
                changes.append((drop, fim_params[i], fim_params[j], R_V[i,j], R_np[i,j]))
        changes.sort(reverse=True)
        print("Most decorrelated pairs:")
        for drop, pi, pj, rv, rnp in changes[:3]:
            print(f"  {PARAM_LABELS_ASCII.get(pi,pi):8s}-{PARAM_LABELS_ASCII.get(pj,pj):8s}"
                  f"  |R|: {abs(rv):.2f} -> {abs(rnp):.2f}  (drop {drop:+.2f})")





# # %% C50_dchg : low-rate (near-OCV) discharge
# profile_id = "C50_dchg"
# initial_soc = 1.0

# model = make_model("SPMe", options=MODEL_OPTIONS)

# base_param = make_base_params(
#     "Chen2020",
#     soc=initial_soc,
#     sensitivity_ready=True,
# )

# experiment = make_experiment(profile_id='C50_dchg')

# target_params, target_values = prepare_sensitivity_inputs(
#     base_param,
#     sensitivity_targets,
#     values=theta_values_nominal,
# )

# sol = run_model(
#     model,
#     target_params,
#     experiment,
#     inputs=target_values,
#     calculate_sensitivities=True,
# )

# t_query = get_t_query(sol)

# Sens_V = get_sensitivities(
#     sol, "Voltage [V]", sensitivity_targets,
#     t_query, theta_values=target_values, normalize=True,
# )
# Sens_V_scaled = scale_voltage_sens_by_ocv(Sens_V, base_param)

# Sens_csn = get_sensitivities(
#     sol, "Average negative particle concentration [mol.m-3]",
#     sensitivity_targets, t_query, theta_values=target_values, normalize=True,
# )
# Sens_csp = get_sensitivities(
#     sol, "Average positive particle concentration [mol.m-3]",
#     sensitivity_targets, t_query, theta_values=target_values, normalize=True,
# )

# endpoints = get_soc_endpoints(base_param)
# Sens_soc_n, Sens_soc_p = concentrations_to_soc_sensitivities(
#     Sens_csn, Sens_csp, endpoints,
# )

# # freq_results에 저장 (initial_soc=1.0 슬롯)
# if initial_soc not in freq_results:
#     freq_results[initial_soc] = {}
# freq_results[initial_soc][profile_id] = {
#     "Sens_V_scaled": Sens_V_scaled,
#     "Sens_soc_n": Sens_soc_n,
#     "Sens_soc_p": Sens_soc_p,
# }

# print(f"C50_dchg done.  t_end = {sol['Time [s]'].entries[-1]:.0f} s")

# # ranking 출력 (다른 profile과 같은 스타일)
# print_relative_sensitivity(Sens_V_scaled, params=sensitivity_targets,
#                            title=f"SOC {initial_soc} | {profile_id} | Voltage")
# print_relative_sensitivity(Sens_soc_n, params=sensitivity_targets,
#                            title=f"SOC {initial_soc} | {profile_id} | SOC_n")
# print_relative_sensitivity(Sens_soc_p, params=sensitivity_targets,
#                            title=f"SOC {initial_soc} | {profile_id} | SOC_p")


# # %% C50_dchg : eps_s 2개만 (capacity 전용)
# ######################################################################################
# V_span = 4.2 - 2.5
# sigma_V, sigma_SOC = 1/1000/V_span, 0.3/100

# eps_s_params = [
#     "Negative electrode active material volume fraction",   # eps_s-
#     "Positive electrode active material volume fraction",   # eps_s+
# ]

# def cov_to_corr(C):
#     d = np.sqrt(np.diag(C)); return C / np.outer(d, d)

# r = freq_results[1.0]["C50_dchg"]

# F_V = compute_fim(r["Sens_V_scaled"], params=eps_s_params, sigma=sigma_V)
# F_n = compute_fim(r["Sens_soc_n"],    params=eps_s_params, sigma=sigma_SOC)
# F_p = compute_fim(r["Sens_soc_p"],    params=eps_s_params, sigma=sigma_SOC)
# F_np = F_V + F_n + F_p

# cond_V  = np.linalg.cond(F_V)
# dlogdet = np.linalg.slogdet(F_np)[1] - np.linalg.slogdet(F_V)[1]
# use = "OK" if cond_V < 1e12 else "BAD (singular)"

# C_V  = np.linalg.pinv(F_V)
# C_n  = np.linalg.pinv(F_V + F_n)
# C_p  = np.linalg.pinv(F_V + F_p)
# C_np = np.linalg.pinv(F_np)
# imp_n  = np.diag(C_V)/np.diag(C_n)
# imp_p  = np.diag(C_V)/np.diag(C_p)
# imp_np = np.diag(C_V)/np.diag(C_np)

# print("="*62)
# print(f"C50_dchg  SOC=1.0  (eps_s only, n=2)  "
#       f"cond_V={cond_V:.1e}  Δlogdet={dlogdet:.1f}  [{use}]")
# print("="*62)
# print(f"{'param':12s}{'+SOCn':>10s}{'+SOCp':>10s}{'+both':>10s}")
# for p, rn, rp, rnp in zip(eps_s_params, imp_n, imp_p, imp_np):
#     print(f"{PARAM_LABELS_ASCII.get(p,p):12s}{rn:10.1f}{rp:10.1f}{rnp:10.1f}")

# R_V, R_np = cov_to_corr(C_V), cov_to_corr(C_np)
# print(f"\ncorr(eps_s-, eps_s+):  V={R_V[0,1]:+.3f}  ->  V+SOC={R_np[0,1]:+.3f}")

# # CRLB 절대값도 (표준편차)
# print(f"\nCRLB (std):")
# print(f"{'param':12s}{'V only':>12s}{'V+SOC':>12s}{'reduction':>11s}")
# for i, p in enumerate(eps_s_params):
#     sV = np.sqrt(abs(C_V[i,i]))
#     sA = np.sqrt(abs(C_np[i,i]))
#     print(f"{PARAM_LABELS_ASCII.get(p,p):12s}{sV:12.3e}{sA:12.3e}{sV/sA:10.1f}x")




# # %% 각 measurement FIM 개별 요약 (V / SOC_n / SOC_p 따로, 안 더함)
# ######################################################################################
# V_span = 4.2 - 2.5
# sigma_V, sigma_SOC = 1/1000/V_span, 0.3/100

# FIXED = ["Negative electrode active material volume fraction",     # eps_s-
#          "Positive electrode active material volume fraction",     # eps_s+
#          "Negative electrode conductivity [S.m-1]",                # sigma-
#          "Positive electrode conductivity [S.m-1]"]                # sigma+

# profile_list = ["sin_100Hz", "sin_10Hz", "sin_1Hz", "sin_01Hz", "hppc"]
# soc_list = [0.3, 0.5]

# def summarize_fim(Sens, params, sigma):
#     """개별 FIM의 요약값: trace, logdet, cond, 최대고유값"""
#     F = compute_fim(Sens, params=params, sigma=sigma)
#     tr = np.trace(F)
#     sign, logdet = np.linalg.slogdet(F)
#     cond = np.linalg.cond(F)
#     return tr, logdet, cond

# print(f"{'profile':10s}{'soc':>5s} | "
#       f"{'trace_V':>10s}{'trace_SOCn':>12s}{'trace_SOCp':>12s} | "
#       f"{'SOCn/V':>9s}{'SOCp/V':>9s}")
# print("-"*80)

# for soc in soc_list:
#     for profile_id in profile_list:
#         r = freq_results[soc][profile_id]
#         params = [p for p in sensitivity_targets if p not in FIXED]  # eps_s 뺀 나머지

#         tr_V, _, _   = summarize_fim(r["Sens_V_scaled"], params, sigma_V)
#         tr_n, _, _   = summarize_fim(r["Sens_soc_n"],    params, sigma_SOC)
#         tr_p, _, _   = summarize_fim(r["Sens_soc_p"],    params, sigma_SOC)

#         ratio_n = tr_n / tr_V if tr_V > 0 else 0
#         ratio_p = tr_p / tr_V if tr_V > 0 else 0

#         print(f"{profile_id:10s}{soc:5.1f} | "
#               f"{tr_V:10.2e}{tr_n:12.2e}{tr_p:12.2e} | "
#               f"{ratio_n:9.1e}{ratio_p:9.1e}")
        


# # %%
# # eps_s 포함 (sigma만 제외) — 위와 같은 표
# FIXED_only_sigma = ["Negative electrode conductivity [S.m-1]",
#                     "Positive electrode conductivity [S.m-1]"]

# print(f"{'profile':10s}{'soc':>5s} | "
#       f"{'trace_V':>10s}{'trace_SOCn':>12s}{'trace_SOCp':>12s} | "
#       f"{'SOCn/V':>9s}{'SOCp/V':>9s}")
# print("-"*80)
# for soc in soc_list:
#     for profile_id in profile_list:
#         r = freq_results[soc][profile_id]
#         params = [p for p in sensitivity_targets if p not in FIXED_only_sigma]  # eps_s 포함
#         tr_V, _, _ = summarize_fim(r["Sens_V_scaled"], params, sigma_V)
#         tr_n, _, _ = summarize_fim(r["Sens_soc_n"],    params, sigma_SOC)
#         tr_p, _, _ = summarize_fim(r["Sens_soc_p"],    params, sigma_SOC)
#         print(f"{profile_id:10s}{soc:5.1f} | {tr_V:10.2e}{tr_n:12.2e}{tr_p:12.2e} | "
#               f"{tr_n/tr_V:9.1e}{tr_p/tr_V:9.1e}")
# # %%



# import numpy as np

# V_span = 4.2 - 2.5
# sigma_V, sigma_SOC = 1/1000/V_span, 0.3/100

# PARAM_LABELS_ASCII = {
#     "Negative electrode exchange-current density [A.m-2]": "k-",
#     "Positive electrode exchange-current density [A.m-2]": "k+",
#     "Negative electrode double-layer capacity [F.m-2]": "Cdl-",
#     "Positive electrode double-layer capacity [F.m-2]": "Cdl+",
#     "Negative electrode active material volume fraction": "eps_s-",
#     "Positive electrode active material volume fraction": "eps_s+",
#     "Cation transference number": "t+",
#     "Electrolyte conductivity [S.m-1]": "kappa",
#     "Electrolyte diffusivity [m2.s-1]": "De",
#     "Positive electrode porosity": "eps_e+",
#     "Negative electrode porosity": "eps_e-",
#     "Separator porosity": "eps_e_sep",
#     "Negative particle diffusivity [m2.s-1]": "Ds-",
#     "Positive particle diffusivity [m2.s-1]": "Ds+",
#     "Negative electrode conductivity [S.m-1]": "sigma-",
#     "Positive electrode conductivity [S.m-1]": "sigma+",
# }

# CAPACITY = ["Negative electrode active material volume fraction",
#             "Positive electrode active material volume fraction"]
# SIGMA    = ["Negative electrode conductivity [S.m-1]",
#             "Positive electrode conductivity [S.m-1]"]

# def active_by_floor(r, floor, exclude):
#     return [p for p in sensitivity_targets
#             if p not in exclude
#             and np.sqrt(np.mean(np.asarray(r["Sens_V_scaled"][p]).squeeze()**2)) > floor]

# for soc in [0.3, 0.5]:
#     r = freq_results[soc]["hppc"]
#     params = active_by_floor(r, sigma_V, exclude=CAPACITY+SIGMA)

#     F_V = compute_fim(r["Sens_V_scaled"], params, sigma=sigma_V)
#     F_n = compute_fim(r["Sens_soc_n"],    params, sigma=sigma_SOC)
#     F_p = compute_fim(r["Sens_soc_p"],    params, sigma=sigma_SOC)

#     C_V  = np.linalg.pinv(F_V)
#     C_np = np.linalg.pinv(F_V + F_n + F_p)
#     imp  = np.diag(C_V) / np.diag(C_np)

#     ld_V  = np.linalg.slogdet(F_V)[1]
#     ld_np = np.linalg.slogdet(F_V + F_n + F_p)[1]

#     print(f"\nHPPC  SOC={soc}   n={len(params)}   Δlogdet={ld_np-ld_V:+.3f}")
#     print(f"observable: {[PARAM_LABELS_ASCII.get(p,p) for p in params]}")
#     print(f"{'param':10s}{'CRLB imp':>10s}")
#     for p, i in zip(params, imp):
#         mark = " <--" if i > 1.5 else ""
#         print(f"  {PARAM_LABELS_ASCII.get(p,p):10s}{i:8.2f}{mark}")
# # %%
