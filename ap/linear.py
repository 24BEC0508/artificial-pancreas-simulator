"""Phase 3/4/7 - Linearization, transfer function, stability & state-space tools."""
import numpy as np
from scipy import signal
from .params import Patient


def state_space(p: Patient):
    """Linearize about (Gb, 0, Ib).  Deviation states [g, x, i], input u_d, disturbance d.

        g' = -p1 g - Gb x + d
        x' = -p2 x + p3 i
        i' = -n i + u_d/VI          (X*G -> Gb*x because X_eq = 0)
    """
    A = np.array([[-p.p1, -p.Gb, 0.0],
                  [0.0,   -p.p2, p.p3],
                  [0.0,    0.0, -p.n]])
    B = np.array([[0.0], [0.0], [1.0 / p.VI]])
    E = np.array([[1.0], [0.0], [0.0]])
    C = np.array([[1.0, 0.0, 0.0]])
    return A, B, E, C


def plant_tf(p: Patient):
    """G(s) = g/u_d  and the positive-gain plant P(s) = -G(s).

    P(s) = K / ((s+p1)(s+p2)(s+n)),   K = Gb*p3/VI      (G(s) = -P(s))
    Returns (num, den) of P(s).  Insulin LOWERS glucose, so G(s) has negative DC gain.
    """
    K = p.Gb * p.p3 / p.VI
    den = np.poly([-p.p1, -p.p2, -p.n])
    return np.array([K]), den


def disturbance_tf(p: Patient):
    """g/d = (s+p2)(s+n) / ((s+p1)(s+p2)(s+n) + ...)  -> from ss: C (sI-A)^-1 E."""
    A, B, E, C = state_space(p)
    num, den = signal.ss2tf(A, E, C, np.zeros((1, 1)))
    return num[0], den


def poly_mul(*polys):
    out = np.array([1.0])
    for q in polys:
        out = np.polymul(out, q)
    return out


def routh_table(coeffs):
    """Routh array for polynomial coefficients (highest power first).

    Returns (table, n_sign_changes_in_first_column).  Handles a zero in the first
    column with the epsilon method.
    """
    c = np.array(coeffs, dtype=float)
    n = len(c) - 1
    cols = (n // 2) + 1
    R = np.zeros((n + 1, cols))
    R[0, :len(c[0::2])] = c[0::2]
    R[1, :len(c[1::2])] = c[1::2]
    eps = 1e-9
    for i in range(2, n + 1):
        if abs(R[i - 1, 0]) < 1e-14:
            R[i - 1, 0] = eps
        for j in range(cols - 1):
            R[i, j] = (R[i - 1, 0] * R[i - 2, j + 1] - R[i - 2, 0] * R[i - 1, j + 1]) / R[i - 1, 0]
    first = R[:, 0]
    changes = int(np.sum(np.sign(first[:-1]) * np.sign(first[1:]) < 0))
    return R, changes


def controllability(A, B):
    n = A.shape[0]
    M = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])
    return M, np.linalg.matrix_rank(M)


def observability(A, C):
    n = A.shape[0]
    M = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])
    return M, np.linalg.matrix_rank(M)


def augmented_with_disturbance(p: Patient):
    """4-state model [g, x, i, d] with the meal modelled as a slowly varying d' = 0.

    Used by the observer so that an unmeasured meal does not bias x_hat / i_hat.
    """
    A, B, E, C = state_space(p)
    Aa = np.zeros((4, 4)); Aa[:3, :3] = A; Aa[:3, 3:] = E
    Ba = np.vstack([B, [[0.0]]])
    Ca = np.hstack([C, [[0.0]]])
    return Aa, Ba, Ca
