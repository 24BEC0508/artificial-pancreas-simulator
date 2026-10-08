"""PHASE 5 - Frequency-domain analysis: Bode, margins, sensitivity, robustness to patient gain."""
from _common import *
from ap.params import *
from ap.linear import *
from ap.analysis import *
from ap import tuned

banner("PHASE 5 - Frequency-domain analysis")
p = DEFAULT_PATIENT
w = np.logspace(-4, 1, 3000)
ctrls = tuned.make_all(p)
Pn, Pd = plant_tf(p)

# ---- Bode of plant and loops
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
P = freqresp(Pn, Pd, w)
ax[0].semilogx(w, 20 * np.log10(abs(P)), color=COLORS["open loop"], label="plant P(s)")
ax[1].semilogx(w, np.unwrap(np.angle(P)) * 180 / np.pi, color=COLORS["open loop"])
rows, out = [], {}
for c in ctrls:
    cn, cd = controller_tf(c)
    L = freqresp(np.polymul(cn, Pn), np.polymul(cd, Pd), w)
    m = loop_margins(c, p)
    S = 1 / (1 + L); T = L / (1 + L)
    Ms = float(np.max(abs(S))); Mt = float(np.max(abs(T)))
    bw = w[np.where(abs(T) < 1 / np.sqrt(2))[0][0]] if np.any(abs(T) < 1 / np.sqrt(2)) else np.nan
    dm = delay_margin(m["pm_deg"], m["wgc"])
    ax[0].semilogx(w, 20 * np.log10(abs(L)), color=COLORS[c.name], label=c.name)
    ax[1].semilogx(w, np.unwrap(np.angle(L)) * 180 / np.pi, color=COLORS[c.name])
    rows.append([c.name, f"{m['gm_db']:.1f}", f"{m['pm_deg']:.0f}", f"{m['wgc']:.3f}", f"{dm:.0f}", f"{Ms:.2f}", f"{Mt:.2f}", f"{bw:.3f}"])
    out[c.name] = dict(**m, delay_margin_min=dm, Ms=Ms, Mt=Mt, bandwidth=bw)
ax[0].axhline(0, color="k", lw=0.7); ax[1].axhline(-180, color="r", lw=0.7)
ax[0].set(ylabel="|L| [dB]", title="Bode - plant and loop gains L = C P (positive-feedback sign absorbed in P)"); ax[0].legend(fontsize=8)
ax[1].set(ylabel="phase [deg]", xlabel="rad/min"); ax[0].set_ylim(-120, 60)
f_bode = savefig(fig, "phase5_bode")
txt, md = table(rows, ["controller", "GM dB", "PM deg", "wgc rad/min", "delay margin min", "Ms", "Mt", "bw rad/min"])
print(txt)
print("\nDelay margin = PM/wgc: largest CGM + pump delay before the loop destabilises.")
print("Ms = peak sensitivity (<2 is good).  A CGM lag of ~5-15 min is typical.")

# ---- Sensitivity functions
fig, ax = plt.subplots(figsize=(7, 3.8))
for c in ctrls:
    cn, cd = controller_tf(c)
    L = freqresp(np.polymul(cn, Pn), np.polymul(cd, Pd), w)
    ax.semilogx(w, 20 * np.log10(abs(1 / (1 + L))), color=COLORS[c.name], label=f"S  {c.name}")
ax.axhline(0, color="k", lw=0.7); ax.axvline(2 * np.pi / 1440, color="gray", ls=":", lw=1)
ax.set(title="Sensitivity S(jw): disturbance rejection (below 0 dB) vs amplification", xlabel="rad/min", ylabel="dB"); ax.legend(fontsize=7)
f_s = savefig(fig, "phase5_sensitivity")

# ---- Robustness to patient gain (insulin sensitivity p3) and other physiology
factors = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
print("\nRobustness to insulin sensitivity (p3 scaled), controllers designed on nominal patient:")
rrows, rob = [], {c.name: [] for c in ctrls}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for c in ctrls:
    gms, pms = [], []
    for f in factors:
        m = loop_margins(c, p.scaled(p3=f))
        gms.append(m["gm_db"] if m["stable"] else np.nan); pms.append(m["pm_deg"] if m["stable"] else np.nan)
        rob[c.name].append(dict(p3_factor=f, gm_db=m["gm_db"], pm_deg=m["pm_deg"], stable=m["stable"]))
    ax[0].semilogx(factors, gms, "o-", color=COLORS[c.name], label=c.name)
    ax[1].semilogx(factors, pms, "o-", color=COLORS[c.name])
    for f, g_, p_ in zip(factors, gms, pms):
        rrows.append([c.name, f, f"{g_:.1f}", f"{p_:.0f}"])
ax[0].axhline(6, color="r", ls="--", lw=0.8); ax[0].set(title="Gain margin [dB]", xlabel="p3 factor"); ax[0].legend(fontsize=7)
ax[1].axhline(30, color="r", ls="--", lw=0.8); ax[1].set(title="Phase margin [deg]", xlabel="p3 factor")
f_rob = savefig(fig, "phase5_robustness_patient_gain")
print(table(rrows, ["controller", "p3 x", "GM dB", "PM deg"])[0])

# max tolerated gain increase (gain margin as a patient-sensitivity ratio)
print("\nGain margin tells how much MORE insulin-sensitive a patient can get before instability:")
for c in ctrls:
    gm = out[c.name]["gm_db"]; print(f"  {c.name:28s} GM = {gm:.1f} dB  ->  factor {10**(gm/20):.1f}x")
save_json("phase5", dict(nominal=out, nominal_md=md, robustness=rob, figures=[f_bode, f_s, f_rob]))
