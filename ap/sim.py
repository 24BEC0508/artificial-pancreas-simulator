"""Phase 2/6 - Closed-loop simulation and clinical metrics."""
import numpy as np
from .params import Patient, Meal, TARGET, HYPO, HYPER, DEFAULT_PATIENT, DEFAULT_MEAL
from .model import rk4_step, equilibrium


def simulate_closed_loop(controller, patient: Patient = DEFAULT_PATIENT,
                         meal: Meal = DEFAULT_MEAL, t_end=600.0, Ts=5.0, dt=1.0,
                         noise_sd=0.0, delay_min=0.0, rng=None, x0=None):
    """Nonlinear plant + sampled controller.

    noise_sd : CGM white noise [mg/dL];  delay_min : CGM transport delay [min]
    Returns dict(t, G, X, I, u, y, D).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    controller.reset()
    n_sub = int(round(Ts / dt))
    n_samp = int(round(t_end / Ts))
    x = equilibrium(patient) if x0 is None else np.array(x0, dtype=float)
    t_all, G, X, I, U, Y, Dm = [], [], [], [], [], [], []
    hist = [x[0]] * (int(round(delay_min / Ts)) + 1)
    t = 0.0
    for k in range(n_samp):
        hist.append(x[0]); y_true = hist[-1 - int(round(delay_min / Ts))]
        y = y_true + (rng.normal(0, noise_sd) if noise_sd > 0 else 0.0)
        u = controller(t, y)
        for _ in range(n_sub):
            t_all.append(t); G.append(x[0]); X.append(x[1]); I.append(x[2]); U.append(u)
            Y.append(y); Dm.append(meal.D(t, patient))
            x = rk4_step(t, x, u, dt, patient, meal); t += dt
    t_all.append(t); G.append(x[0]); X.append(x[1]); I.append(x[2]); U.append(U[-1]); Y.append(y); Dm.append(0)
    return dict(t=np.array(t_all), G=np.array(G), X=np.array(X), I=np.array(I),
                u=np.array(U), y=np.array(Y), D=np.array(Dm))


def metrics(res, r=TARGET, patient: Patient = DEFAULT_PATIENT, meal: Meal = DEFAULT_MEAL):
    """Clinical + classical performance metrics for one simulation."""
    t, G, u = res["t"], res["G"], res["u"]
    dt = np.median(np.diff(t))
    post = t >= meal.t_meal
    Gp = G[post]; tp = t[post]
    peak = Gp.max(); t_peak = tp[Gp.argmax()]
    gmin = G.min()
    tir = 100.0 * np.mean((G >= HYPO) & (G <= HYPER))
    t_hypo = float(np.sum(G < HYPO) * dt)
    t_hyper = float(np.sum(G > HYPER) * dt)
    # settling: last time |G - r| > 10 mg/dL after the meal start (only after the peak)
    after = tp >= t_peak
    out_band = np.where(np.abs(Gp[after] - r) > 10.0)[0]
    t_settle = float(tp[after][out_band[-1]] - meal.t_meal) if len(out_band) else 0.0
    if len(out_band) and tp[after][out_band[-1]] >= t[-1] - 2 * dt:
        t_settle = np.inf
    return dict(
        peak=float(peak), t_peak=float(t_peak - meal.t_meal), min=float(gmin),
        undershoot=float(max(0.0, r - gmin)),
        tir=float(tir), t_hypo=t_hypo, t_hyper=t_hyper, t_settle=t_settle,
        iae=float(np.sum(np.abs(G - r)) * dt / 60.0),
        insulin_U=float(np.sum(u) * dt / 1000.0), u_max=float(u.max()),
        G_end=float(G[-1]),
    )
