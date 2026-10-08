"""PHASE 2 - Nonlinear simulation (open loop) and model validation checks."""
from _common import *
from ap.params import *
from ap.model import *

banner("PHASE 2 - Nonlinear Bergman model simulation")
p = DEFAULT_PATIENT

# validation 1: equilibrium is preserved with basal insulin and no meal
t, x, u = simulate_open_loop(p, Meal(carbs_g=0))
drift = float(np.abs(x - equilibrium(p)).max())
print(f"Check 1  basal equilibrium drift over 10 h with no meal  : {drift:.2e}  (must be ~0)")

# validation 2: mass balance - area under D equals total meal glucose rise potential
mm = DEFAULT_MEAL
tt = np.linspace(0, 1500, 150001)
area = np.trapezoid(mm.D(tt, p), tt)
exp_area = mm.bioavail * mm.carbs_g * 1000 / p.VG
print(f"Check 2  meal appearance integral {area:.2f} mg/dL vs expected {exp_area:.2f} mg/dL")

# validation 3: sustained insulin lowers steady glucose (sign of the model)
t3, x3, _ = simulate_open_loop(p, Meal(carbs_g=0), u_func=lambda t: p.u_basal + 10.0, t_end=2000)
print(f"Check 3  +10 mU/min extra insulin -> steady glucose {x3[-1,0]:.1f} mg/dL (< Gb = {p.Gb})")

# meal-size sweep, open loop (basal insulin only)
fig, ax = plt.subplots(2, 2, figsize=(10, 6))
data = {}
for c in (20, 40, 60, 80):
    t, x, u = simulate_open_loop(p, Meal(carbs_g=c), t_end=720)
    data[c] = dict(peak=float(x[:, 0].max()), t_peak=float(t[x[:, 0].argmax()] - mm.t_meal),
                   t_above_180=float((x[:, 0] > HYPER).sum() * 0.5))
    ax[0, 0].plot(t, x[:, 0], label=f"{c} g")
    ax[0, 1].plot(t, x[:, 1] * 1e3)
    ax[1, 0].plot(t, x[:, 2])
    ax[1, 1].plot(t, Meal(carbs_g=c).D(t, p))
ax[0, 0].axhspan(HYPO, HYPER, color="g", alpha=0.08); ax[0, 0].set(title="Plasma glucose G", ylabel="mg/dL"); ax[0, 0].legend(title="carbs")
ax[0, 1].set(title="Remote insulin action X", ylabel="1e-3 /min")
ax[1, 0].set(title="Plasma insulin I (basal only)", ylabel="uU/mL", xlabel="time [min]")
ax[1, 1].set(title="Meal glucose appearance D(t)", ylabel="mg/dL/min", xlabel="time [min]")
fig.suptitle("Open-loop (basal insulin only) meal response - nonlinear Bergman model")
f = savefig(fig, "phase2_open_loop_meals")
print("\nOpen-loop peaks:", {k: round(v["peak"], 1) for k, v in data.items()})
save_json("phase2", dict(drift=drift, area=area, expected_area=exp_area, steady_with_extra=x3[-1, 0],
                         meals=data, figure=f))
