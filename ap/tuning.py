"""Phase 6 - numerical controller tuning on the NONLINEAR model.

Cost, averaged over a small patient set (insulin-sensitivity robustness) and run WITH
a 10 min CGM delay and 3 mg/dL sensor noise:

    J = mean[(G - r)^2] + W_HYPO * mean[max(0, G_LOW - G)^2] + W_MARGIN * margin_penalty

margin_penalty is nonzero when the linear loop has PM < PM_MIN or GM < GM_MIN_DB (or is
unstable), so the optimiser can not buy performance with fragile loops.
"""
import numpy as np
from scipy.optimize import minimize
from .params import DEFAULT_PATIENT, DEFAULT_MEAL, TARGET
from .sim import simulate_closed_loop
from .controllers import PIDController, ConstrainedPI, LQRObserverController
from .analysis import loop_margins

W_HYPO = 40.0
G_LOW = 85.0
PM_MIN, GM_MIN_DB, W_MARGIN = 50.0, 8.0, 40.0
DELAY, NOISE = 10.0, 3.0
PATIENT_SET = [(1.0, "nominal"), (0.6, "insulin resistant"), (1.6, "insulin sensitive")]


def margin_penalty(ctrl):
    pen = 0.0
    for f, _ in PATIENT_SET:
        m = loop_margins(ctrl, DEFAULT_PATIENT.scaled(p3=f))
        if not m["stable"]:
            pen += 400.0; continue
        pen += max(0.0, PM_MIN - m["pm_deg"]) ** 2 / 25.0 if not np.isnan(m["pm_deg"]) else 0.0
        pen += max(0.0, GM_MIN_DB - m["gm_db"]) ** 2 if np.isfinite(m["gm_db"]) else 0.0
    return pen / len(PATIENT_SET)


def cost(make_controller, meal=DEFAULT_MEAL, t_end=600.0):
    J = 0.0
    for f, _ in PATIENT_SET:
        pat = DEFAULT_PATIENT.scaled(p3=f)
        res = simulate_closed_loop(make_controller(DEFAULT_PATIENT), pat, meal, t_end=t_end,
                                   noise_sd=NOISE, delay_min=DELAY)
        G = res["G"]
        J += np.mean((G - TARGET) ** 2) + W_HYPO * np.mean(np.maximum(0, G_LOW - G) ** 2)
    J /= len(PATIENT_SET)
    return J + W_MARGIN * margin_penalty(make_controller(DEFAULT_PATIENT))


def _nm(f, x0, maxiter):
    r = minimize(f, np.log(x0), method="Nelder-Mead",
                 options=dict(maxiter=maxiter, xatol=1e-2, fatol=1e-2))
    return np.exp(r.x), r.fun


TI_MIN, TI_MAX = 60.0, 240.0      # min; keeps real integral action (zero steady-state error)


def _ti(v):
    return float(np.clip(v, TI_MIN, TI_MAX))


def tune_pid(x0=(0.8, 150.0, 20.0), maxiter=150):
    """Parameters: Kp, Ti [min], Kd  ->  Ki = Kp/Ti."""
    def f(lx):
        if np.any(np.abs(lx) > 12):
            return 1e12
        Kp, Ti, Kd = np.exp(lx); Ti = _ti(Ti)
        return cost(lambda p: PIDController(p, Kp, Kp / Ti, Kd))
    x, fv = _nm(f, x0, maxiter)
    return np.array([x[0], x[0] / _ti(x[1]), x[2]]), fv          # (Kp, Ki, Kd)


def tune_pi(x0=(0.8, 150.0), maxiter=120, B_max=900.0):
    def f(lx):
        if np.any(np.abs(lx) > 12):
            return 1e12
        Kp, Ti = np.exp(lx); Ti = _ti(Ti)
        return cost(lambda p: ConstrainedPI(p, Kp, Kp / Ti, B_max=B_max))
    x, fv = _nm(f, x0, maxiter)
    return np.array([x[0], x[0] / _ti(x[1])]), fv                # (Kp, Ki)


def tune_lqr(x0=(0.5, 1.0), maxiter=120):
    """Optimise (qz, rho); the q_g weight is fixed to 1."""
    def f(lx):
        if np.any(np.abs(lx) > 14):
            return 1e12
        qz, rho = np.exp(lx)
        return cost(lambda p: LQRObserverController(p, qz=qz, rho=rho))
    return _nm(f, x0, maxiter)
