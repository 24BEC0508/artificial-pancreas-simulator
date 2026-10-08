"""PHASE 6 - Controller design, tuning and comparison on the NONLINEAR plant.

    python phase6_controllers.py            # use frozen gains from ap/tuned.py
    python phase6_controllers.py --retune   # re-run the numerical tuning (~30 s) and print new gains
"""
from _common import *
from ap.params import *
from ap.sim import simulate_closed_loop, metrics
from ap.controllers import PIDController, ConstrainedPI, LQRObserverController
from ap.analysis import loop_margins
from ap import tuned

banner("PHASE 6 - Controller design & comparison")
p = DEFAULT_PATIENT

if "--retune" in sys.argv:
    from ap import tuning
    print("Retuning (Nelder-Mead on the nonlinear model, 3 patients, 10 min CGM delay, 3 mg/dL noise)...")
    x, f = tuning.tune_pid();  print(f"  PID  Kp={x[0]:.4f} Ki={x[1]:.6f} Kd={x[2]:.3f}   J={f:.1f}")
    x, f = tuning.tune_pi();   print(f"  PI   Kp={x[0]:.4f} Ki={x[1]:.6f}   J={f:.1f}")
    x, f = tuning.tune_lqr();  print(f"  LQR  qz={x[0]:.5f} rho={x[1]:.5f}   J={f:.1f}")
    print("Copy these into ap/tuned.py to make them the defaults.\n")

ctrls = tuned.make_all(p)
pp = LQRObserverController(p, poles=tuned.POLES)        # pole-placement alternative
ctrls.append(pp)
print("Gains:")
print(f"  PID : {tuned.PID}\n  PI  : {tuned.PI}\n  LQR : Kx={np.round(ctrls[2].Kx,4)}  Kz={ctrls[2].Kz:.4f}  (qz={tuned.LQR['qz']}, rho={tuned.LQR['rho']})")
print(f"  PP  : poles {tuned.POLES}  Kx={np.round(pp.Kx,4)}  Kz={pp.Kz:.4f}")
print(f"  pump limits: 0 <= u <= 200 mU/min (12 U/h);  IOB cap (PI) {tuned.PI['B_max']:.0f} mU;  low-glucose suspend < 80 mg/dL")

keys = ["peak", "t_peak", "min", "tir", "t_hypo", "t_settle", "iae", "insulin_U"]
hdr = ["controller", "peak mg/dL", "t_peak min", "min mg/dL", "TIR %", "hypo min", "settle min", "IAE g.h/dL", "insulin U"]


def fmt(m):
    return [f"{m['peak']:.0f}", f"{m['t_peak']:.0f}", f"{m['min']:.0f}", f"{m['tir']:.0f}", f"{m['t_hypo']:.0f}",
            ("inf" if not np.isfinite(m['t_settle']) else f"{m['t_settle']:.0f}"), f"{m['iae']:.0f}", f"{m['insulin_U']:.1f}"]


scen = {"nominal": dict(), "CGM noise 3 mg/dL + 10 min delay": dict(noise_sd=3.0, delay_min=10.0)}
all_res, all_met, md_tables = {}, {}, {}
for sname, kw in scen.items():
    rows = []
    # open loop reference (basal only): run with a fixed-basal 'controller'
    class Basal:
        name = "open loop"
        def reset(self): pass
        def __call__(self, t, y): return p.u_basal
    for c in [Basal()] + ctrls:
        res = simulate_closed_loop(c, p, DEFAULT_MEAL, t_end=720, **kw)
        m = metrics(res); all_res[(sname, c.name)] = res; all_met.setdefault(sname, {})[c.name] = m
        rows.append([c.name] + fmt(m))
    txt, md = table(rows, hdr); md_tables[sname] = md
    print(f"\n--- Scenario: {sname}  ({DEFAULT_MEAL.carbs_g:.0f} g meal, target {TARGET:.0f}, 12 h horizon) ---\n{txt}")

# ---- figure: nominal and noisy time responses
for sname, tag in [("nominal", "nominal"), ("CGM noise 3 mg/dL + 10 min delay", "noisy")]:
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for (sn, cn), res in all_res.items():
        if sn != sname: continue
        ax[0].plot(res["t"], res["G"], color=COLORS[cn], label=cn, lw=1.6 if cn != "open loop" else 1.2, ls="--" if cn == "open loop" else "-")
        ax[1].plot(res["t"], res["u"], color=COLORS[cn], lw=1.2)
        ax[2].plot(res["t"], res["I"], color=COLORS[cn], lw=1.2)
    ax[0].axhspan(HYPO, HYPER, color="g", alpha=0.08); ax[0].axhline(TARGET, color="k", lw=0.6, ls=":")
    ax[0].set(ylabel="G [mg/dL]", title=f"Meal rejection ({DEFAULT_MEAL.carbs_g:.0f} g at t={DEFAULT_MEAL.t_meal:.0f} min) - {sname}"); ax[0].legend(fontsize=8)
    ax[1].set(ylabel="insulin u [mU/min]"); ax[2].set(ylabel="plasma insulin [uU/mL]", xlabel="min")
    fp = savefig(fig, f"phase6_meal_response_{tag}")

# ---- insulin constraint behaviour (PID baseline vs constrained PI vs LQR), large meal 100 g
big = Meal(carbs_g=100)
fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
big_rows, big_m = [], {}
for c in ctrls:
    res = simulate_closed_loop(c, p, big, t_end=720); m = metrics(res, meal=big)
    ax[0].plot(res["t"], res["G"], color=COLORS[c.name], label=c.name); ax[1].plot(res["t"], res["u"], color=COLORS[c.name])
    big_rows.append([c.name] + fmt(m)); big_m[c.name] = m
ax[0].axhspan(HYPO, HYPER, color="g", alpha=0.08); ax[0].set(ylabel="G", title="Stress test: 100 g meal (saturation / anti-windup / IOB cap)"); ax[0].legend(fontsize=8)
ax[1].axhline(200, color="r", ls="--", lw=0.8); ax[1].axhline(0, color="r", ls="--", lw=0.8); ax[1].set(ylabel="u [mU/min]", xlabel="min")
f_big = savefig(fig, "phase6_stress_100g")
print("\n--- Stress test 100 g meal ---\n" + table(big_rows, hdr)[0])

# ---- robustness sweep: insulin sensitivity p3 x0.4 ... x2.5 (plant changes, controllers fixed)
factors = [0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5]
sweep = {c.name: [] for c in ctrls}
for c in ctrls:
    for f in factors:
        res = simulate_closed_loop(c, p.scaled(p3=f), DEFAULT_MEAL, t_end=720)
        m = metrics(res); sweep[c.name].append(dict(factor=f, peak=m["peak"], min=m["min"], tir=m["tir"], t_hypo=m["t_hypo"]))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for c in ctrls:
    s = sweep[c.name]
    ax[0].plot(factors, [x["peak"] for x in s], "o-", color=COLORS[c.name], label=c.name)
    ax[1].plot(factors, [x["min"] for x in s], "o-", color=COLORS[c.name])
    ax[2].plot(factors, [x["tir"] for x in s], "o-", color=COLORS[c.name])
ax[0].axhline(HYPER, color="r", ls="--", lw=0.8); ax[0].set(title="Peak glucose", xlabel="p3 factor", ylabel="mg/dL"); ax[0].legend(fontsize=7)
ax[1].axhline(HYPO, color="r", ls="--", lw=0.8); ax[1].set(title="Glucose nadir", xlabel="p3 factor")
ax[2].set(title="Time in range 70-180 (%)", xlabel="p3 factor")
f_sw = savefig(fig, "phase6_patient_sweep")
print("\n--- Insulin-sensitivity sweep (p3 factor): peak / nadir ---")
for c in ctrls:
    print(f"  {c.name:28s}", "  ".join(f"{x['factor']}x:{x['peak']:.0f}/{x['min']:.0f}" for x in sweep[c.name]))

# ---- meal size sweep
sizes = [20, 40, 60, 80, 100]
msw = {c.name: [] for c in ctrls}
for c in ctrls:
    for s in sizes:
        mm = Meal(carbs_g=s); res = simulate_closed_loop(c, p, mm, t_end=720); m = metrics(res, meal=mm)
        msw[c.name].append(dict(carbs=s, peak=m["peak"], min=m["min"], t_hypo=m["t_hypo"]))
print("\n--- Meal-size sweep: peak / nadir ---")
for c in ctrls:
    print(f"  {c.name:28s}", "  ".join(f"{x['carbs']}g:{x['peak']:.0f}/{x['min']:.0f}" for x in msw[c.name]))

# ---- constraint verification
print("\n--- Safety constraint checks (all scenarios above, nominal patient) ---")
chk = {}
for c in ctrls:
    res = all_res[("nominal", c.name)]
    chk[c.name] = dict(u_min=float(res["u"].min()), u_max=float(res["u"].max()))
    print(f"  {c.name:28s} u in [{res['u'].min():.1f}, {res['u'].max():.1f}] mU/min   (limits 0 .. 200)")
pi_ctrl = ctrls[1]
simulate_closed_loop(pi_ctrl, p, big, t_end=720)
print(f"  Constrained PI: IOB cap was active in {pi_ctrl.iob_active} of 144 samples during the 100 g stress test")

save_json("phase6", dict(gains=dict(PID=tuned.PID, PI=tuned.PI, LQR=tuned.LQR, POLES=tuned.POLES),
                         scenarios=all_met, tables_md=md_tables, stress=big_m, stress_md=table(big_rows, hdr)[1],
                         patient_sweep=sweep, meal_sweep=msw, constraints=chk,
                         figures=["figures/phase6_meal_response_nominal.png", "figures/phase6_meal_response_noisy.png", f_big, f_sw]))
