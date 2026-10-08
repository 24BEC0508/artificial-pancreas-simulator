"""Phase 6 - Controllers.

All controllers are sampled at the CGM period Ts and return the TOTAL insulin
infusion u [mU/min] for the next interval (zero-order hold).

    PIDController            baseline PID, filtered derivative, back-calculation
                             anti-windup, hard bounds 0 <= u <= u_max
    ConstrainedPI            PI + insulin-on-board (IOB) projection + low-glucose
                             suspend; guarantees u >= 0 and IOB <= IOB_max by design
    LQRObserverController    integral LQR on [g, x, i, z] + Luenberger/Kalman observer
                             (4 states incl. meal disturbance) -> only CGM is measured

Sign convention: e = G - r.  Insulin lowers glucose, so the controller gain on e is
positive (u_d = Kp e + ...).  The LTI y->u transfer function of each controller is
exposed via .tf() / .ss() so Bode / margin analysis is exactly the same for all.
"""
import numpy as np
from scipy import linalg
from .params import Patient, TARGET
from .linear import state_space, augmented_with_disturbance


# --------------------------------------------------------------------------- PID
class PIDController:
    name = "PID"

    def __init__(self, patient: Patient, Kp, Ki, Kd, Tf=10.0, Ts=5.0, r=TARGET,
                 u_max=200.0, Tt=30.0):
        self.p, self.Kp, self.Ki, self.Kd, self.Tf, self.Ts = patient, Kp, Ki, Kd, Tf, Ts
        self.r, self.u_max, self.Tt = r, u_max, Tt
        self.reset()

    def reset(self):
        self.I = 0.0; self.D = 0.0; self.e_prev = None

    def __call__(self, t, y):
        e = y - self.r
        if self.e_prev is None:
            self.e_prev = e
        Ts, Tf = self.Ts, self.Tf
        self.D = Tf / (Tf + Ts) * self.D + self.Kd / (Tf + Ts) * (e - self.e_prev)
        self.e_prev = e
        u_unsat = self.p.u_basal + self.Kp * e + self.I + self.D
        u = float(np.clip(u_unsat, 0.0, self.u_max))
        self.I += Ts * (self.Ki * e + (u - u_unsat) / self.Tt)          # back-calculation
        return u

    def tf(self):
        """(num, den) of Kp + Ki/s + Kd s/(Tf s + 1)  (deviation form, u_d = C(s) e)."""
        Tf = self.Tf
        num = np.polyadd(np.polyadd(self.Kp * np.array([Tf, 1, 0]), self.Ki * np.array([Tf, 1])),
                         self.Kd * np.array([1.0, 0, 0]))
        return num, np.array([Tf, 1.0, 0.0])


# ------------------------------------------------------------- constrained PI + IOB
class ConstrainedPI:
    """PI controller with insulin-on-board projection.

    IOB state B [mU] tracks excess insulin pool: B' = -n B + u_d, B = VI * (I - Ib).
    Exact ZOH projection keeps B(k+1) <= B_max:
        u_d <= n (B_max - B e^{-nTs}) / (1 - e^{-nTs})
    Low-glucose suspend: u = 0 if G < g_suspend or if G is falling fast toward it.
    """
    name = "Constrained PI (IOB)"

    def __init__(self, patient: Patient, Kp, Ki, B_max=900.0, Ts=5.0, r=TARGET,
                 u_max=200.0, g_suspend=80.0, Tt=30.0, predict_min=20.0):
        self.p, self.Kp, self.Ki, self.Ts, self.r = patient, Kp, Ki, Ts, r
        self.B_max, self.u_max, self.g_suspend, self.Tt = B_max, u_max, g_suspend, Tt
        self.predict_min = predict_min
        self.reset()

    def reset(self):
        self.I = 0.0; self.B = 0.0; self.y_prev = None
        self.iob_active = 0

    def __call__(self, t, y):
        p, Ts = self.p, self.Ts
        e = y - self.r
        u_unsat = p.u_basal + self.Kp * e + self.I
        u_hi = self.u_max
        # IOB projection
        a = np.exp(-p.n * Ts)
        ud_max = p.n * (self.B_max - self.B * a) / (1 - a)
        u_hi = min(u_hi, p.u_basal + ud_max)
        u = float(np.clip(u_unsat, 0.0, max(u_hi, 0.0)))
        if u_unsat > u_hi:
            self.iob_active += 1
        # low-glucose suspend with short-horizon prediction
        slope = 0.0 if self.y_prev is None else (y - self.y_prev) / Ts
        if y < self.g_suspend or (y + self.predict_min * slope) < self.g_suspend - 10:
            u = 0.0
        self.y_prev = y
        self.I += Ts * (self.Ki * e + (u - u_unsat) / self.Tt)
        # propagate IOB with the applied u
        self.B = self.B * a + (u - p.u_basal) * (1 - a) / p.n
        return u

    def tf(self):
        return np.array([self.Kp, self.Ki]), np.array([1.0, 0.0])      # linear region


# --------------------------------------------------------------- LQR + observer
class LQRObserverController:
    """Integral LQR with observer on the 4-state model [g, x, i, d].

    Controller states  : x_hat (4, incl. meal disturbance estimate), z = integral of e
    Control law        : u_d = -Kx x_hat[:3] - Kz z
    Observer           : continuous Kalman-Bucy gain L, implemented with exact ZOH steps.
    """
    name = "LQR + observer"

    def __init__(self, patient: Patient, q=(1.0, 0.0, 0.0), qz=1e-3, rho=1.0,
                 qw=(1e-2, 1e-6, 1e-4, 1e-1), rv=4.0, Ts=5.0, r=TARGET, u_max=200.0,
                 Tt=30.0, poles=None):
        self.p, self.Ts, self.r, self.u_max, self.Tt = patient, Ts, r, u_max, Tt
        A, B, E, C = state_space(patient)
        # --- integral LQR on [g,x,i,z],  z' = g
        Aaug = np.zeros((4, 4)); Aaug[:3, :3] = A; Aaug[3, :3] = C
        Baug = np.vstack([B, [[0.0]]])
        Q = np.diag([q[0], q[1], q[2], qz]); R = np.array([[rho]])
        P = linalg.solve_continuous_are(Aaug, Baug, Q, R)
        Kfull = np.linalg.solve(R, Baug.T @ P)            # u_d = -Kfull [g x i z]
        if poles is not None:                                   # pole-placement variant
            from scipy.signal import place_poles
            Kfull = place_poles(Aaug, Baug, poles).gain_matrix
            self.name = "Pole placement + observer"
        self.Kx, self.Kz = Kfull[0, :3], float(Kfull[0, 3])
        self.Aaug, self.Baug, self.Q, self.R = Aaug, Baug, Q, R
        # --- Kalman-Bucy observer on [g,x,i,d]
        Aa, Ba, Ca = augmented_with_disturbance(patient)
        W = np.diag(qw); V = np.array([[rv]])
        S = linalg.solve_continuous_are(Aa.T, Ca.T, W, V)
        self.L = (S @ Ca.T / rv)                           # (4,1)
        self.Aa, self.Ba, self.Ca = Aa, Ba, Ca
        # exact ZOH discretisation of  xh' = (Aa - L Ca) xh + Ba u + L y
        Ao = Aa - self.L @ Ca
        M = np.zeros((6, 6)); M[:4, :4] = Ao * Ts
        M[:4, 4:5] = Ba * Ts; M[:4, 5:6] = self.L * Ts
        E_ = linalg.expm(M)
        self.Phi, self.Gu, self.Gy = E_[:4, :4], E_[:4, 4:5], E_[:4, 5:6]
        self.reset()

    def reset(self):
        self.xh = np.zeros((4, 1)); self.z = 0.0; self.u_prev_d = 0.0

    def poles(self):
        return (np.linalg.eigvals(self.Aaug - self.Baug @ np.hstack([self.Kx, [self.Kz]])[None, :]),
                np.linalg.eigvals(self.Aa - self.L @ self.Ca))

    def __call__(self, t, y):
        p, Ts = self.p, self.Ts
        yd = y - p.Gb                                       # deviation measurement
        # observer update with last applied input (done first: uses y(k))
        self.xh = self.Phi @ self.xh + self.Gu * self.u_prev_d + self.Gy * yd
        e = y - self.r
        ud = -float(self.Kx @ self.xh[:3, 0]) - self.Kz * self.z
        u_unsat = p.u_basal + ud
        u = float(np.clip(u_unsat, 0.0, self.u_max))
        self.z += Ts * e
        # conditional integration (stop integrating when saturated in the wrong direction)
        if (u_unsat > self.u_max and e > 0) or (u_unsat < 0 and e < 0):
            self.z -= Ts * e
        self.u_prev_d = u - p.u_basal
        return u

    def ss(self):
        """Controller as LTI system y_dev -> u_d:  states [xh(4), z]."""
        n = 4
        Ac = np.zeros((5, 5)); Bc = np.zeros((5, 1)); Cc = np.zeros((1, 5))
        Kx_ext = np.hstack([self.Kx, [0.0]])[None, :]
        Ac[:4, :4] = self.Aa - self.L @ self.Ca - self.Ba @ Kx_ext
        Ac[:4, 4:5] = -self.Ba * self.Kz
        Ac[4, :4] = 0.0
        Bc[:4, :] = self.L; Bc[4, 0] = 1.0                  # z' = y_dev  (r = Gb offset ignored)
        Cc[0, :4] = -Kx_ext[0]; Cc[0, 4] = -self.Kz
        return Ac, Bc, Cc, np.zeros((1, 1))
