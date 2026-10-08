"""PHASE 7 - State-space analysis: controllability, observability, state feedback & observer."""
from _common import *
from scipy import linalg
from ap.params import *
from ap.linear import *
from ap.sim import simulate_closed_loop
from ap.controllers import LQRObserverController
from ap import tuned

banner("PHASE 7 - State-space analysis, state feedback & observer")
p = DEFAULT_PATIENT
A, B, E, C = state_space(p)
fmt = lambda M: np.array2string(np.atleast_2d(M), formatter={'float_kind': lambda v: f'{v:.3g}'})

# ---- controllability / observability
Mc, rc = controllability(A, B)
Mo, ro = observability(A, C)
print(f"Controllability matrix [B AB A^2B] rank = {rc} / 3  ->  {'CONTROLLABLE' if rc==3 else 'NOT controllable'}")
sv = np.linalg.svd(Mc / np.linalg.norm(Mc, axis=0), compute_uv=False)
print(f"  normalised-column singular values {fmt(sv)}  (condition {sv[0]/sv[-1]:.1e}) -> controllable but strongly ill-conditioned (slow, weak coupling i->x->g)")
Wc = linalg.solve_continuous_lyapunov(A, -B @ B.T)
print(f"  controllability Gramian eigenvalues {fmt(np.linalg.eigvalsh(Wc))}")
print(f"Observability  matrix [C; CA; CA^2] rank = {ro} / 3  ->  {'OBSERVABLE from glucose (CGM)' if ro==3 else 'NOT observable'}")
Wo = linalg.solve_continuous_lyapunov(A.T, -C.T @ C)
print(f"  observability Gramian eigenvalues {fmt(np.linalg.eigvalsh(Wo))}")

# PBH test for each eigenvalue
print("\nPBH tests (rank [lambda I - A, B] and [lambda I - A; C]):")
for lam in np.linalg.eigvals(A):
    r1 = np.linalg.matrix_rank(np.hstack([lam * np.eye(3) - A, B]), tol=1e-10)
    r2 = np.linalg.matrix_rank(np.vstack([lam * np.eye(3) - A, C]), tol=1e-12)
    print(f"  lambda={lam.real:+.3f}: controllability rank {r1}, observability rank {r2}")

# what if other sensors were used?
print("\nObservability with other single outputs (for discussion):")
for name, Cx in [("glucose g", [1, 0, 0]), ("remote action x", [0, 1, 0]), ("plasma insulin i", [0, 0, 1])]:
    print(f"  y = {name:18s}: rank {observability(A, np.array([Cx], float))[1]}")
# controllability from meal channel
Md, rd = controllability(A, E)
print(f"\nMeal channel E is controllable? rank = {rd} (irrelevant for design, but shows meals excite only g directly)")
print("Key fact: E is NOT in span(B) -> the meal cannot be cancelled instantaneously; insulin acts through i -> x -> g with lag ~ 1/n + 1/p2 = "
      f"{1/p.n + 1/p.p2:.0f} min.  This lag limits the achievable peak reduction.")

# augmented observer model
Aa, Ba, Ca = augmented_with_disturbance(p)
_, rao = observability(Aa, Ca)
print(f"\nAugmented model [g, x, i, d] (meal as constant disturbance): observability rank {rao} / 4  ->  disturbance observable from CGM")

# ---- state feedback: LQR and pole placement
lqr = LQRObserverController(p, **tuned.LQR)
pp = LQRObserverController(p, poles=tuned.POLES)
plqr_c, plqr_o = lqr.poles()
ppp_c, ppp_o = pp.poles()
fm = lambda v: np.array2string(np.sort_complex(v), precision=4, suppress_small=True)
print("\nState feedback on [g, x, i, z] (z = integral of error):")
print(f"  LQR   Kx={fmt(lqr.Kx)}  Kz={lqr.Kz:.4f}\n        closed-loop poles {fm(plqr_c)}")
print(f"  Place Kx={fmt(pp.Kx)}  Kz={pp.Kz:.4f}\n        closed-loop poles {fm(ppp_c)}")
print(f"  Observer (Kalman-Bucy) poles {fm(plqr_o)}  (observer is not much faster than controller -> limited by weak x/i visibility from g)")
print(f"  Observer gain L = {fmt(lqr.L.T)}")
print("  Separation principle: closed-loop poles = state-feedback poles  U  observer poles ->",
      "all LHP" if np.all(np.concatenate([plqr_c, plqr_o]).real < 0) else "NOT stable")
Ac, Bc, Cc, Dc = lqr.ss()
print(f"  Combined LQG controller (5 states) eigenvalues: {fm(np.linalg.eigvals(Ac))}")

# ---- observer performance on the nonlinear plant
class Rec:
    def __init__(self, c): self.c, self.rec = c, []; self.name = c.name
    def reset(self): self.c.reset(); self.rec = []
    def __call__(self, t, y):
        u = self.c(t, y); self.rec.append((t, *self.c.xh[:, 0])); return u
rc_ = Rec(LQRObserverController(p, **tuned.LQR))
res = simulate_closed_loop(rc_, p, DEFAULT_MEAL, t_end=720)
R = np.array(rc_.rec); ts = R[:, 0]
idx = (ts / 1.0).astype(int)
true = dict(g=res["G"][idx] - p.Gb, x=res["X"][idx], i=res["I"][idx] - p.Ib, d=res["D"][idx])
fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
for a, k, j, ttl in [(ax[0, 0], "g", 1, "glucose deviation g [mg/dL]"), (ax[0, 1], "x", 2, "remote insulin action x [1/min]"),
                     (ax[1, 0], "i", 3, "plasma insulin deviation i [uU/mL]"), (ax[1, 1], "d", 4, "meal disturbance d [mg/dL/min]")]:
    a.plot(ts, true[k], "k", label="true"); a.plot(ts, R[:, j], "--", color=COLORS["LQR + observer"], label="estimate")
    a.set_title(ttl)
ax[0, 0].legend(); ax[1, 0].set_xlabel("min"); ax[1, 1].set_xlabel("min")
fig.suptitle("Observer performance (only CGM glucose is measured)")
f_obs = savefig(fig, "phase7_observer")
rmse = {k: float(np.sqrt(np.mean((R[:, j] - true[k]) ** 2))) for k, j in [("g", 1), ("x", 2), ("i", 3), ("d", 4)]}
print("\nObserver RMSE:", {k: float(f"{v:.3g}") for k, v in rmse.items()})

# ---- LQR weight study (design trade-off)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
study = []
for qz in (0.05, 0.43, 3.0, 20.0):
    c = LQRObserverController(p, qz=qz, rho=1.0)
    r_ = simulate_closed_loop(c, p, DEFAULT_MEAL, t_end=720)
    ax[0].plot(r_["t"], r_["G"], label=f"qz={qz}"); ax[1].plot(r_["t"], r_["u"])
    study.append(dict(qz=qz, peak=float(r_["G"].max()), umax=float(r_["u"].max())))
ax[0].set(title="LQR integral weight qz: peak vs aggressiveness", xlabel="min", ylabel="G"); ax[0].legend(fontsize=8)
ax[1].set(xlabel="min", ylabel="u [mU/min]")
f_q = savefig(fig, "phase7_lqr_weights")
print("LQR weight study:", study)
save_json("phase7", dict(rank_c=rc, rank_o=ro, rank_aug_o=rao, cond=sv[0] / sv[-1],
                         lqr=dict(Kx=lqr.Kx, Kz=lqr.Kz, poles_cl=plqr_c, poles_obs=plqr_o),
                         place=dict(Kx=pp.Kx, Kz=pp.Kz, poles_cl=ppp_c), observer_rmse=rmse,
                         lqr_study=study, figures=[f_obs, f_q]))
