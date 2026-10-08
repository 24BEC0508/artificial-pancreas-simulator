"""PHASE 1 - Literature & parameters.  Documents the model, units and basal equilibrium."""
from _common import *
from ap.params import *
from dataclasses import asdict

banner("PHASE 1 - Literature & parameters")
p, m = DEFAULT_PATIENT, DEFAULT_MEAL
rows = [
    ["p1", p.p1, "1/min", "glucose effectiveness (insulin-independent uptake)"],
    ["p2", p.p2, "1/min", "decay of remote insulin action  (time const. 1/p2 = %.0f min)" % (1 / p.p2)],
    ["p3", p.p3, "1/(min^2 uU/mL)", "insulin sensitivity (gain from plasma insulin to action)"],
    ["n", p.n, "1/min", "plasma insulin clearance (time const. %.1f min)" % (1 / p.n)],
    ["Gb", p.Gb, "mg/dL", "basal glucose"],
    ["Ib", p.Ib, "uU/mL", "basal insulin"],
    ["VI", p.VI, "L", "insulin distribution volume"],
    ["VG", p.VG, "dL", "glucose distribution volume (1.6 dL/kg x 70 kg)"],
    ["u_basal", round(p.u_basal, 3), "mU/min", "n*Ib*VI = %.2f U/h basal infusion" % (p.u_basal * 60 / 1000)],
]
txt, md = table(rows, ["param", "value", "unit", "meaning"])
print(txt)
print(f"\nControl objective: keep G in [{HYPO:.0f}, {HYPER:.0f}] mg/dL, set point {TARGET:.0f} mg/dL")
print(f"Test meal: {m.carbs_g:.0f} g carbs at t={m.t_meal:.0f} min, absorption peak after {m.tau:.0f} min, "
      f"bioavailability {m.bioavail:.0%}  ->  total {m.bioavail*m.carbs_g*1000/p.VG:.0f} mg/dL of glucose rise potential")
print("\nReferences: Bergman, Ider, Bowden, Cobelli (1979) Am J Physiol 236:E667;  "
      "Bergman RN (2021) Front Endocrinol 'Origins and history of the minimal model';  "
      "Fisher ME (1991) IEEE TBME 38:57 (closed-loop glucose control);  "
      "Hovorka et al. (2004) Physiol Meas 25:905 (MPC artificial pancreas).")
save_json("phase1", dict(params=asdict(p), u_basal=p.u_basal, meal=asdict(m), table_md=md,
                         target=TARGET, hypo=HYPO, hyper=HYPER))
