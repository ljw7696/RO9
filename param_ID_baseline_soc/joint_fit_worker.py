"""Standalone fit worker + per-start subprocess pool for the rate sweep.

Why a standalone script instead of joblib:
  IDAKLU can stall for minutes on isolated interior parameter combinations
  (resonance bands that no fixed bound can exclude). A stalled solve cannot be
  interrupted in-process and kills/deadlocks a joblib/loky worker. Running each
  start in its OWN OS subprocess lets the parent enforce a hard wall-clock
  timeout: a stalled start is killed and recorded as a penalty, so the sweep
  always finishes unattended. Plain multiprocessing.Process can't be used from
  an interactive kernel on Windows (the child re-imports the notebook); a fresh
  `python rate_fit_worker.py --child ...` process avoids that entirely.

Used by main_rate_sweep.py:  from rate_fit_worker import run_one_start, run_multistart_subprocess
"""
import os
# Cap math-library threads to 1 per process BEFORE numpy/pybamm load (the child
# also inherits this from the parent, but set it here too for direct runs).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
# make utils.py (in the repo root, one level up) importable from a clean subprocess
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)


def run_one_start(
    start_id,
    profiles,                     # JOINT: list of (profile_id, initial_soc, t_data, V_data)
    fit_params,
    param_bounds,
    log_scaled_params,
    model_options,
    sigma_V_fit,
    max_nfev,
    jac_mode="fd",
    diff_step=1e-2,
):
    """Run TRF local optimization from one random start (build-once, leak-free).

    JOINT version: identical to the single-profile fit EXCEPT the residual now
    stacks all `profiles`, each weighted 1/sqrt(N_i) so every profile contributes
    equally regardless of length -> least_squares minimizes the MEAN voltage RMSE
    over the profiles. (Only the residual/setup changed; optimizer is unchanged.)"""
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.interpolate import interp1d
    import pybamm

    pybamm.set_logging_level("ERROR")
    from utils import (
        make_base_params,
        prepare_sensitivity_inputs,
        make_model,
        make_experiment,
    )

    log_scaled = set(log_scaled_params)
    n_p = len(fit_params)
    want_sens = (jac_mode == "analytic")

    def theta_bar_to_dim(theta_bar):
        out = {}
        for z, p in zip(theta_bar, fit_params):
            lo, hi = param_bounds[p]
            z = float(np.clip(z, 0.0, 1.0))
            if p in log_scaled:
                out[p] = float(np.exp(np.log(lo) + z * (np.log(hi) - np.log(lo))))
            else:
                out[p] = float(lo + z * (hi - lo))
        return out

    def dtheta_ddz(theta_dim):
        d = np.zeros(n_p)
        for k, p in enumerate(fit_params):
            lo, hi = param_bounds[p]
            if p in log_scaled:
                d[k] = theta_dim[p] * (np.log(hi) - np.log(lo))
            else:
                d[k] = (hi - lo)
        return d

    # per-profile data + equal weight (1/sqrt(N_i)) + ONE Simulation each (build once)
    tdatas = [np.asarray(pr[2]) for pr in profiles]
    vdatas = [np.asarray(pr[3]) for pr in profiles]
    weights = [1.0 / np.sqrt(len(v)) for v in vdatas]
    Ntot = sum(len(v) for v in vdatas)
    sims = []
    for (profile_id, initial_soc, _t, _V) in profiles:
        _base = make_base_params("Chen2020", soc=initial_soc, sensitivity_ready=True)
        _p_struct, _ = prepare_sensitivity_inputs(
            _base, fit_params, values=theta_bar_to_dim(np.full(n_p, 0.5))
        )
        sims.append(pybamm.Simulation(
            make_model("SPMe", options=model_options),
            experiment=make_experiment(profile_id=profile_id),
            parameter_values=_p_struct,
            solver=pybamm.IDAKLUSolver(options={"max_num_steps": 50000}),
        ))

    cache = {"key": None, "Vq": None, "S": None, "ok": False, "theta_dim": None}

    def _solve(theta_bar):
        key = np.asarray(theta_bar).tobytes()
        if cache["key"] == key:
            return
        cache["key"] = key
        theta_dim = theta_bar_to_dim(theta_bar)
        cache["theta_dim"] = theta_dim
        iv = {name: theta_dim[name] for name in fit_params}
        Vqs, Ss, ok = [], [], True
        for sim, t_data in zip(sims, tdatas):
            try:
                sol = sim.solve(inputs=iv, calculate_sensitivities=want_sens)
                t = sol["Time [s]"].entries
                V = sol["Voltage [V]"].entries
                Vq = interp1d(t, V, bounds_error=False, fill_value="extrapolate")(t_data)
                S = None
                if want_sens:
                    sens = sol["Voltage [V]"].sensitivities
                    S = np.zeros((len(t_data), n_p))
                    for k, name in enumerate(fit_params):
                        s_arr = np.asarray(sens[name]).reshape(-1)
                        S[:, k] = interp1d(t, s_arr, bounds_error=False,
                                           fill_value="extrapolate")(t_data)
                Vqs.append(Vq); Ss.append(S)
                ok = ok and bool(np.all(np.isfinite(Vq)))
            except Exception:
                ok = False; Vqs.append(None); Ss.append(None)
        cache.update(Vq=Vqs, S=Ss, ok=ok)

    def residual(theta_bar):
        _solve(theta_bar)
        if not cache["ok"]:
            return np.ones(Ntot) * 1e6
        r = np.concatenate([w * (vq - vd) / sigma_V_fit
                            for w, vq, vd in zip(weights, cache["Vq"], vdatas)])
        if np.any(~np.isfinite(r)):
            return np.ones(Ntot) * 1e6
        return r

    def jac(theta_bar):
        _solve(theta_bar)
        if not cache["ok"] or any(S is None for S in cache["S"]):
            return np.zeros((Ntot, n_p))
        scale = dtheta_ddz(cache["theta_dim"])
        return np.vstack([(w * S / sigma_V_fit) * scale[np.newaxis, :]
                          for w, S in zip(weights, cache["S"])])

    rng = np.random.default_rng(start_id)
    theta_bar_init = rng.uniform(0.0, 1.0, size=n_p)

    ls_kwargs = dict(
        bounds=(np.zeros(n_p), np.ones(n_p)),
        method="trf",
        max_nfev=max_nfev,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        verbose=0,
    )
    if jac_mode == "analytic":
        res = least_squares(residual, theta_bar_init, jac=jac, **ls_kwargs)
    else:
        res = least_squares(residual, theta_bar_init, diff_step=diff_step, **ls_kwargs)

    _solve(res.x)
    rmses = ([1e9] * len(profiles) if not cache["ok"]
             else [float(np.sqrt(np.mean((vq - vd) ** 2)) * 1e3)
                   for vq, vd in zip(cache["Vq"], vdatas)])
    rmse_hat = float(np.mean(rmses))          # MEAN voltage RMSE over the profiles

    return {
        "start_id": start_id,
        "success": bool(res.success),
        "cost": float(res.cost),
        "rmse_mV": rmse_hat,
        "rmses": rmses,
        "jac_mode": jac_mode,
        "theta_init": theta_bar_to_dim(theta_bar_init),
        "theta_hat": theta_bar_to_dim(res.x),
        "theta_bar_init": np.asarray(theta_bar_init),
        "theta_bar_hat": np.asarray(res.x),
        "stalled": False,
    }


def _penalty_result(start_id, fit_params, jac_mode):
    """Result for a start that stalled (killed) or crashed (no output)."""
    import numpy as np
    return {
        "start_id": start_id,
        "success": False,
        "cost": float("inf"),
        "rmse_mV": 1e9,
        "jac_mode": jac_mode,
        "theta_init": {p: float("nan") for p in fit_params},
        "theta_hat": {p: float("nan") for p in fit_params},
        "theta_bar_init": np.full(len(fit_params), np.nan),
        "theta_bar_hat": np.full(len(fit_params), np.nan),
        "stalled": True,
    }


def run_multistart_subprocess(common_args, n_starts, n_jobs, timeout_s, workdir,
                              py_exe=None, verbose=True):
    """Run `n_starts` fits, each in its own subprocess, ≤ n_jobs concurrently.

    A start whose wall-clock exceeds `timeout_s` is killed and recorded as a
    penalty (stalled=True). A start whose process dies without writing output
    (segfault) is also a penalty. Returns a list of result dicts ordered by
    start_id. `common_args` is the dict of run_one_start kwargs EXCEPT start_id.
    """
    import pickle, time, subprocess

    py_exe = py_exe or sys.executable
    os.makedirs(workdir, exist_ok=True)
    fit_params = common_args["fit_params"]
    jac_mode = common_args.get("jac_mode", "fd")

    common_pkl = os.path.join(workdir, "common.pkl")
    with open(common_pkl, "wb") as f:
        pickle.dump(common_args, f)

    results = {}
    active = {}                     # sid -> (proc, out_pkl, t0, err_fh)
    pending = list(range(n_starts))
    script = os.path.abspath(__file__)

    def launch(sid):
        out_pkl = os.path.join(workdir, f"out_{sid}.pkl")
        if os.path.exists(out_pkl):
            os.remove(out_pkl)
        err_fh = open(os.path.join(workdir, f"err_{sid}.log"), "w")
        proc = subprocess.Popen(
            [py_exe, script, "--child", common_pkl, str(sid), out_pkl],
            stdout=subprocess.DEVNULL, stderr=err_fh,
        )
        active[sid] = (proc, out_pkl, time.time(), err_fh)

    def finish(sid, stalled):
        proc, out_pkl, t0, err_fh = active.pop(sid)
        try:
            err_fh.close()
        except Exception:
            pass
        if stalled:
            results[sid] = _penalty_result(sid, fit_params, jac_mode)
        elif os.path.exists(out_pkl):
            try:
                with open(out_pkl, "rb") as f:
                    results[sid] = pickle.load(f)
            except Exception:
                results[sid] = _penalty_result(sid, fit_params, jac_mode)
        else:
            results[sid] = _penalty_result(sid, fit_params, jac_mode)  # crashed

    while pending and len(active) < n_jobs:
        launch(pending.pop(0))

    while active:
        for sid in list(active):
            proc, out_pkl, t0, err_fh = active[sid]
            rc = proc.poll()
            if rc is not None:
                finish(sid, stalled=False)
                if verbose:
                    r = results[sid]
                    tag = "STALL/CRASH" if r.get("stalled") else f"RMSE={r['rmse_mV']:.3f}mV"
                    print(f"    [start {sid:2d}] done ({tag})", flush=True)
                if pending:
                    launch(pending.pop(0))
            elif time.time() - t0 > timeout_s:
                try:
                    proc.kill()
                except Exception:
                    pass
                finish(sid, stalled=True)
                if verbose:
                    print(f"    [start {sid:2d}] KILLED (stalled > {timeout_s:.0f}s)", flush=True)
                if pending:
                    launch(pending.pop(0))
        time.sleep(0.5)

    return [results[s] for s in range(n_starts)]


if __name__ == "__main__":
    # child entry: python rate_fit_worker.py --child <common_pkl> <start_id> <out_pkl>
    import pickle
    if len(sys.argv) >= 5 and sys.argv[1] == "--child":
        _common_pkl, _sid, _out_pkl = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        with open(_common_pkl, "rb") as _f:
            _common = pickle.load(_f)
        _res = run_one_start(start_id=_sid, **_common)
        with open(_out_pkl, "wb") as _f:
            pickle.dump(_res, _f)
