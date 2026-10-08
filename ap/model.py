"""Phase 2 - Nonlinear Bergman minimal model + meal disturbance."""
import numpy as np
from .params import Patient, Meal


def rhs(t, x, u, patient: Patient, meal: Meal):
    """Right-hand side. x = [G, X, I]; u = insulin infusion [mU/min] (absolute)."""
    G, X, I = x
    p = patient
    dG = -p.p1 * (G - p.Gb) - X * G + meal.D(t, p)
    dX = -p.p2 * X + p.p3 * (I - p.Ib)
    dI = -p.n * (I - p.Ib) + (u - p.u_basal) / p.VI   # u = absolute infusion; deck's u(t) = u - u_basal
    return np.array([dG, dX, dI])


def rk4_step(t, x, u, dt, patient, meal):
    """One fixed-step RK4 step with zero-order-hold insulin u."""
    k1 = rhs(t, x, u, patient, meal)
    k2 = rhs(t + dt / 2, x + dt / 2 * k1, u, patient, meal)
    k3 = rhs(t + dt / 2, x + dt / 2 * k2, u, patient, meal)
    k4 = rhs(t + dt, x + dt * k3, u, patient, meal)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def equilibrium(patient: Patient):
    """Basal equilibrium (G, X, I) = (Gb, 0, Ib) with u = u_basal."""
    return np.array([patient.Gb, 0.0, patient.Ib])


def simulate_open_loop(patient: Patient, meal: Meal, u_func=None, t_end=600.0, dt=0.5):
    """Open-loop simulation. u_func(t) -> mU/min, default: constant basal."""
    if u_func is None:
        u_func = lambda t: patient.u_basal
    t = np.arange(0.0, t_end + dt, dt)
    x = equilibrium(patient)
    xs = np.empty((len(t), 3)); us = np.empty(len(t))
    for k, tk in enumerate(t):
        xs[k] = x
        us[k] = u_func(tk)
        x = rk4_step(tk, x, us[k], dt, patient, meal)
    return t, xs, us
