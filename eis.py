import pybamm
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time

from utils import *
from params import *


def run_EIS(battery_info, eis_info, model, param_set="Chen2020", name=None, param=None, soc2theta_n=None, soc2theta_p=None):
    soc      = battery_info["soc"]
    T_kelvin = battery_info["T_kelvin"]

    solutions = {}
    t_total_start = time.time()

    for freq in eis_info["freqs_hz"]:
        t_freq_start = time.time()

        current_fn, t_eval = make_sinusoidal_input(
            freq_hz        = freq,
            n_periods      = eis_info["n_periods"],
            dc_offset_a    = eis_info["dc_offset_a"],
            ac_mag_a       = eis_info["ac_mag_a"],
            pts_per_period = eis_info["pts_per_period"],
        )

        p = param.copy() if param is not None else pybamm.ParameterValues(param_set)

        if soc2theta_n is not None and soc2theta_p is not None:
            c_n, c_p = soc2conc(soc, p, soc2theta_n, soc2theta_p)
        else:
            c_n = soc * p["Maximum concentration in negative electrode [mol.m-3]"]
            c_p = (1 - soc) * p["Maximum concentration in positive electrode [mol.m-3]"]

        p.update({
            "Ambient temperature [K]": T_kelvin,
            "Current function [A]"   : current_fn,
            "Initial concentration in negative electrode [mol.m-3]": c_n,
            "Initial concentration in positive electrode [mol.m-3]": c_p,
        })

        solver = pybamm.CasadiSolver(atol=1e-9, rtol=1e-9, mode="safe")
        sim = pybamm.Simulation(model, parameter_values=p, solver=solver)
        sol = sim.solve(t_eval=t_eval)

        solutions[freq] = sol
        elapsed = time.time() - t_freq_start
        print(f"  SOC={soc:.2f} | T={T_kelvin:.1f}K | {freq:.4g} Hz ✅  ({elapsed:.1f}s)")

    total_elapsed = time.time() - t_total_start
    print(f"\n총 소요 시간: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    return {
        "name"      : name or f"SOC={soc:.2f} | T={T_kelvin:.1f}K",
        "soc"       : soc,
        "T"         : T_kelvin,
        "freqs"     : eis_info["freqs_hz"],
        "solutions" : solutions,
        "ac_mag"    : eis_info["ac_mag_a"],
    }


def compute_impedance(sol, freq_hz):
    t = sol["Time [s]"].entries
    V = sol["Terminal voltage [V]"].entries
    I = sol["Current [A]"].entries

    T_period = 1.0 / freq_hz
    mask = t >= t[-1] - 5 * T_period
    t_ss, V_ss, I_ss = t[mask], V[mask], I[mask]

    V_ss = V_ss - np.mean(V_ss)
    I_ss = I_ss - np.mean(I_ss)

    exp_factor = np.exp(-1j * 2 * np.pi * freq_hz * t_ss)
    V_dft = np.sum(V_ss * exp_factor)
    I_dft = np.sum(I_ss * exp_factor)

    Z = V_dft / I_dft
    return -Z


def plot_nyquist(result_list):
    fig = go.Figure()

    for result in result_list:
        Z_list = [compute_impedance(result["solutions"][f], f) * 1000 for f in result["freqs"]]
        label  = result.get("name", f"SOC={result['soc']:.2f} | T={result['T']:.1f}K")

        fig.add_trace(go.Scatter(
            x    = [z.real for z in Z_list],
            y    = [-z.imag for z in Z_list],
            mode = "lines+markers",
            name = label,
        ))

    font_size = 36
    fig.update_layout(
        title  = dict(text="Nyquist Plot", font=dict(size=font_size)),
        xaxis  = dict(
            title     = dict(text="Re(Z) [mΩ]", font=dict(size=font_size)),
            tickfont  = dict(size=font_size),
            range     = [0, 80],
        ),
        yaxis  = dict(
            title    = dict(text="-Im(Z) [mΩ]", font=dict(size=font_size)),
            tickfont = dict(size=font_size),
            range    = [0, 55],
        ),
        legend = dict(x=0.01, y=0.99, font=dict(size=font_size)),
        width  = 12 * 96,
        height = 9 * 96,
    )

    fig.show()


def plot_raw_signal(result, freq_hz):
    sol = result["solutions"][freq_hz]
    t = sol["Time [s]"].entries
    V = sol["Terminal voltage [V]"].entries
    I = sol["Current [A]"].entries

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    ax1.plot(t, V)
    ax1.set_ylabel("Voltage [V]")
    ax2.plot(t, I)
    ax2.set_ylabel("Current [A]")
    plt.suptitle(f"{result['name']} | {freq_hz:.4g} Hz")
    plt.tight_layout()
    plt.show()


def is_freq_valid(sol, freq_hz, v_min=2.5, v_max=4.3, cs_margin=0.02, min_zero_crossings=4):
    V = sol["Terminal voltage [V]"].entries
    
    # 1. 전압 범위 체크
    if V.min() < v_min or V.max() > v_max:
        return False, "voltage out of range"

    # 2. 사인파 체크 - 마지막 5주기에서 zero crossing 수
    t = sol["Time [s]"].entries
    T_period = 1.0 / freq_hz
    mask = t >= t[-1] - 5 * T_period
    V_ss = V[mask] - np.mean(V[mask])
    zero_crossings = np.sum(np.diff(np.sign(V_ss)) != 0)
    if zero_crossings < min_zero_crossings:
        return False, f"not sinusoidal (zero crossings={zero_crossings})"

    # 3. cs 범위 체크
    cs_neg_min = sol["Minimum negative particle surface concentration"].entries
    cs_neg_max = sol["Maximum negative particle surface concentration"].entries
    cs_pos_min = sol["Minimum positive particle surface concentration"].entries
    cs_pos_max = sol["Maximum positive particle surface concentration"].entries

    if cs_neg_min.min() < cs_margin or cs_neg_max.max() > 1 - cs_margin:
        return False, f"cs_neg out of range [{cs_neg_min.min():.3f}, {cs_neg_max.max():.3f}]"
    if cs_pos_min.min() < cs_margin or cs_pos_max.max() > 1 - cs_margin:
        return False, f"cs_pos out of range [{cs_pos_min.min():.3f}, {cs_pos_max.max():.3f}]"

    return True, "ok"


def plot_nyquist_filtered(result_list, filter_threshold_a=10.0):
    fig = go.Figure()

    for result in result_list:
        valid_Z = []

        for f in reversed(result["freqs"]):
            sol = result["solutions"][f]

            if result.get("ac_mag", 0) >= filter_threshold_a:
                valid, reason = is_freq_valid(sol, f, min_zero_crossings=8)
                if not valid:
                    print(f"  ⚠️ {result['name']} | {f:.4g} Hz → {reason}, stopping")
                    break

            valid_Z.append(compute_impedance(sol, f) * 1000)

        label = result.get("name", f"SOC={result['soc']:.2f} | T={result['T']:.1f}K")
        print(f"  ✅ {label}: {len(valid_Z)}/{len(result['freqs'])} points valid")

        fig.add_trace(go.Scatter(
            x    = [z.real for z in valid_Z],
            y    = [-z.imag for z in valid_Z],
            mode = "lines+markers",
            name = label,
        ))

    font_size = 36
    fig.update_layout(
        title  = dict(text="Nyquist Plot", font=dict(size=font_size)),
        xaxis  = dict(
            title       = dict(text="Re(Z) [mΩ]", font=dict(size=font_size)),
            tickfont    = dict(size=font_size),
            range       = [0, 60],
        ),
        yaxis  = dict(
            title    = dict(text="-Im(Z) [mΩ]", font=dict(size=font_size)),
            tickfont = dict(size=font_size),
            range    = [0, 55],
        ),
        legend = dict(x=0.01, y=0.99, font=dict(size=font_size)),
        width  = 12 * 96,
        height = 9 * 96,
    )
    fig.show()