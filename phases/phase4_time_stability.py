"""PHASE 4 - Time-domain & stability analysis: poles, Routh-Hurwitz, root locus, meal response."""
from _common import *
from scipy import signal
from ap.params import *
from ap.linear import *
from ap.analysis import *
from ap import tuned
from ap.controllers import PIDController, ConstrainedPI

banner("PHASE 4 - Time response & stability analysis")
p = DEFAULT_PATIENT
Pn, Pd = plant_tf(p)

# ---- 1. open-loop stability: Routh-Hurwitz
R, ch = routh_table(Pd)
print("Open-loop characteristic polynomial s^3 + a2 s^2 + a1 s + a0 :", np.round(Pd, 6))
print("Routh array first column:", np.array2string(R[:, 0], formatter={'float_kind': lambda v: f'{v:.3g}'}), f"-> sign changes: {ch}  => open loop {'STABLE' if ch==0 else 'UNSTABLE'}")

# ---- 2. closed-loop with PID / PI: Routh-Hurwitz and poles
pid = PIDController(p, **tuned.PID); pi = ConstrainedPI(p, **tuned.PI)
summ = {}
for ctrl in (pid, pi):
    cn, cd = ctrl.tf()
    Ln, Ld = np.polymul(cn, Pn), np.polymul(cd, Pd)
    ch_poly = closed_loop_poly(Ln, Ld)
    R, ch = routh_table(ch_poly)
    poles = np.roots(ch_poly)
    print(f"\n{ctrl.name}: closed-loop char. polynomial {np.array2string(ch_poly, formatter={'float_kind': lambda v: f'{v:.3g}'})}")
    print(f"  Routh first column {np.array2string(R[:,0], formatter={'float_kind': lambda v: f'{v:.3g}'})} -> {ch} sign changes -> {'STABLE' if ch==0 else 'UNSTABLE'}")
    print(f"  closed-loop poles: {np.round(poles, 4)}   (slowest real part {poles.real.max():.4f} /min => tau ~ {-1/poles.real.max():.0f} min)")
    summ[ctrl.name] = dict(poly=ch_poly, routh_changes=ch, poles=poles)

# ---- 3. critical (ultimate) gain of P-control by Routh: 1 + k P(s)
# s^3 + a2 s^2 + a1 s + a0 + k K = 0 ; stable iff 0 < a0 + kK  and  a2 a1 > a0 + kK
a2, a1, a0 = Pd[1], Pd[2], Pd[3]; K = Pn[0]
k_crit = (a2 * a1 - a0) / K
w_u = np.sqrt(a1)
print(f"\nP-only ultimate gain (Routh): k_u = (a2*a1 - a0)/K = {k_crit:.3f} mU/min per mg/dL,  ultimate freq {w_u:.4f} rad/min (period {2*np.pi/w_u:.0f} min)")

# ---- 4. root locus plots
ks = np.logspace(-3, np.log10(k_crit * 3), 600)
rl = root_locus(Pn, Pd, ks)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for j in range(rl.shape[1]):
    ax[0].plot(rl[:, j].real, rl[:, j].imag, ".", ms=2)
ax[0].plot(np.roots(Pd).real, np.roots(Pd).imag, "kx", ms=8, label="open-loop poles")
ax[0].axvline(0, color="r", lw=0.8); ax[0].set(title=f"Root locus - P control (k_u = {k_crit:.2f})", xlabel="Re", ylabel="Im"); ax[0].legend()
# PI root locus: scale PI gain keeping Ti
Ti = tuned.PI["Kp"] / tuned.PI["Ki"]
Cn = np.array([1.0, 1.0 / Ti]); Cd = np.array([1.0, 0.0])
ks2 = np.logspace(-3, 1, 600)
rl2 = root_locus(np.polymul(Cn, Pn), np.polymul(Cd, Pd), ks2)
for j in range(rl2.shape[1]):
    ax[1].plot(rl2[:, j].real, rl2[:, j].imag, ".", ms=2)
cp = np.roots(closed_loop_poly(tuned.PI["Kp"] * np.polymul(Cn, Pn), np.polymul(Cd, Pd)))
ax[1].plot(cp.real, cp.imag, "g*", ms=11, label=f"chosen Kp={tuned.PI['Kp']:.2f}")
ax[1].axvline(0, color="r", lw=0.8); ax[1].set(title=f"Root locus - PI (Ti={Ti:.0f} min) vs Kp", xlabel="Re", ylabel="Im"); ax[1].legend()
f_rl = savefig(fig, "phase4_root_locus")

# ---- 5. pole-zero map of plant + closed loops
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.plot(np.roots(Pd).real, np.roots(Pd).imag, "kx", ms=9, label="plant poles")
for ctrl, c in ((pid, "#d95f02"), (pi, "#1b9e77")):
    pl = summ[ctrl.name]["poles"]; ax.plot(pl.real, pl.imag, "o", color=c, mfc="none", ms=8, label=f"{ctrl.name} closed loop")
ax.axvline(0, color="r", lw=0.8); ax.set(title="Pole map", xlabel="Re [1/min]", ylabel="Im [1/min]"); ax.legend(fontsize=7)
f_pz = savefig(fig, "phase4_pole_map")

# ---- 6. linear meal response vs open loop and PID/PI closed loop
t = np.arange(0, 720, 1.0); d = DEFAULT_MEAL.D(t, p)
dn, dd = disturbance_tf(p)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
_, g0, _ = signal.lsim((dn, dd), d, t)
ax[0].plot(t, g0 + p.Gb, color=COLORS["open loop"], label="open loop")
resp = {}
for ctrl in (pid, pi):
    cn, cd = ctrl.tf()
    chp = closed_loop_poly(np.polymul(cn, Pn), np.polymul(cd, Pd))
    _, g, _ = signal.lsim((np.polymul(dn, cd), chp), d, t)
    _, ud, _ = signal.lsim((cn, cd), g, t)
    ax[0].plot(t, g + p.Gb, color=COLORS[ctrl.name], label=ctrl.name)
    ax[1].plot(t, ud + p.u_basal, color=COLORS[ctrl.name], label=ctrl.name)
    resp[ctrl.name] = dict(peak=float(g.max() + p.Gb), min=float(g.min() + p.Gb), u_min=float(ud.min() + p.u_basal))
    print(f"\nLinear meal response ({ctrl.name}): peak G {g.max()+p.Gb:.1f}, min G {g.min()+p.Gb:.1f}, u_min {ud.min()+p.u_basal:.1f} mU/min")
ax[0].set(title="Linear meal response (no constraints; ref = Gb)", xlabel="min", ylabel="G [mg/dL]"); ax[0].legend()
ax[1].axhline(0, color="r", lw=0.8); ax[1].set(title="Insulin infusion (linear, unconstrained)", xlabel="min", ylabel="u [mU/min]"); ax[1].legend()
f_ts = savefig(fig, "phase4_linear_meal_response")

# ---- 7. unit step response of plant (rise/settling)
tt = np.arange(0, 1000, 1.0)
_, y = signal.step((Pn, Pd), T=tt)
ts = tt[np.where(np.abs(y - y[-1]) > 0.02 * abs(y[-1]))[0][-1]]
print(f"\nPlant step response (insulin step -> glucose drop): final {y[-1]:.3f} mg/dL per mU/min, 2% settling {ts:.0f} min, no overshoot (real poles)")
save_json("phase4", dict(open_loop_routh_changes=ch, k_ultimate=k_crit, w_ultimate=w_u,
                         closed_loop=summ, linear_meal=resp, plant_settling=ts, figures=[f_rl, f_pz, f_ts]))
