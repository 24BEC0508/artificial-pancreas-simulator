"""PHASE 3 - Linearization about the basal state, state-space and transfer functions."""
from _common import *
from scipy import signal
from ap.params import *
from ap.model import *
from ap.linear import *

banner("PHASE 3 - Linearization & transfer function derivation")
p = DEFAULT_PATIENT
A, B, E, C = state_space(p)
np.set_printoptions(precision=5, suppress=False, linewidth=120)
print("States (deviation): [g, x, i]   input u_d [mU/min]   disturbance d [mg/dL/min]   output y = g\n")
print("A =\n", A, "\nB =\n", B.T, "(transposed)\nE =\n", E.T, "(transposed)\nC =", C)

Pn, Pd = plant_tf(p)
K = Pn[0]
print(f"\nPlant (insulin LOWERS glucose):")
print(f"  G(s) = g/u_d = -{K:.4e} / ((s+{p.p1})(s+{p.p2})(s+{p.n}))")
print(f"  P(s) = -G(s) (positive-gain form used for controller design)")
print(f"  characteristic polynomial: s^3 + {Pd[1]:.5f} s^2 + {Pd[2]:.6f} s + {Pd[3]:.4e}")
dc = K / Pd[-1]
print(f"  DC gain G(0) = -{dc:.3f} mg/dL per mU/min   (=> +10 mU/min sustained -> -{10*dc:.1f} mg/dL; matches phase 2 check)")
dn, dd = disturbance_tf(p)
print(f"  disturbance TF g/d = {np.round(dn,6)} / {np.round(dd,6)}  (DC {dn[-1]/dd[-1]:.2f} mg/dL per mg/dL/min = 1/p1 - 'meal gain')")
print(f"  plant has no zeros; relative degree 3; poles at -p1, -p2, -n (all real, stable)")

# ---- validation: linear vs nonlinear with a 60 g meal AND a sustained insulin step
# (the only nonlinearity is the bilinear term X*G, which is active only when insulin deviates)
dt = 0.5; t = np.arange(0, 600 + dt, dt)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
res = {}
mm = Meal(carbs_g=60)
for ud, a in [(5.0, ax[0]), (20.0, ax[1]), (60.0, ax[2])]:
    tn, xn, _ = simulate_open_loop(p, mm, u_func=lambda tt: p.u_basal + ud, t_end=600, dt=dt)
    sysx = signal.StateSpace(A, np.hstack([B, E]), C, np.zeros((1, 2)))
    _, gl, _ = signal.lsim(sysx, np.column_stack([np.full_like(t, ud), mm.D(t, p)]), t)
    err = np.abs(gl + p.Gb - xn[:, 0]).max()
    a.plot(tn, xn[:, 0], label="nonlinear"); a.plot(t, gl + p.Gb, "--", label="linearized")
    a.set(title=f"60 g meal + {ud:g} mU/min insulin", xlabel="min", ylabel="G [mg/dL]"); a.legend()
    res[ud] = dict(max_error=float(err), min_nl=float(xn[:, 0].min()))
    print(f"  meal + insulin step {ud:>4g} mU/min: max |linear - nonlinear| = {err:.2f} mg/dL")
fig.suptitle("Validity of the linearization about the basal state")
f1 = savefig(fig, "phase3_linear_vs_nonlinear")
print("  (without insulin deviation X stays 0, the X*G term vanishes and linear == nonlinear exactly)")

# insulin step response
fig, ax = plt.subplots(figsize=(6, 3.6))
ud = 5.0
tn, xn, _ = simulate_open_loop(p, Meal(carbs_g=0), u_func=lambda t: p.u_basal + ud, t_end=800, dt=dt)
ts = np.arange(0, 800 + dt, dt)
_, gl, _ = signal.lsim(signal.StateSpace(A, B, C, [[0]]), np.full_like(ts, ud), ts)
ax.plot(tn, xn[:, 0], label="nonlinear"); ax.plot(ts, gl + p.Gb, "--", label="linear G(s)")
ax.set(title=f"Glucose response to +{ud} mU/min insulin step", xlabel="min", ylabel="mg/dL"); ax.legend()
f2 = savefig(fig, "phase3_insulin_step")

# pole-zero & sensitivity of the poles to parameters
print("\nEigenvalues of A:", np.round(np.linalg.eigvals(A), 5))
save_json("phase3", dict(A=A, B=B, E=E, C=C, plant_num=Pn, plant_den=Pd, dc_gain=-dc,
                         validation=res, figures=[f1, f2]))
