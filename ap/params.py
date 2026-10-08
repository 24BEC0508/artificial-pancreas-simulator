"""Phase 1 - Model parameters (single source of truth).

Bergman minimal model, glucose-insulin dynamics:
    dG/dt = -p1 (G - Gb) - X G + D(t)
    dX/dt = -p2 X + p3 (I - Ib)
    dI/dt = -n (I - Ib) + u_d(t)/VI ,   u_d = u - u_basal

Units
-----
G  : plasma glucose            [mg/dL]
X  : remote insulin action     [1/min]
I  : plasma insulin            [uU/mL]
u  : TOTAL insulin infusion    [mU/min]   (u >= 0, cannot remove insulin)
u_d: deviation from basal      [mU/min]   (the u(t) of the project slides; u_d >= -u_basal)
D  : meal glucose appearance   [mg/dL/min] (already divided by glucose volume)

The values below are typical textbook / literature values for the minimal
model (Bergman et al. 1979; Bergman 2021 review; Fisher 1991 for the
insulin-infusion form).  They are NOT patient specific -- the robustness
studies (phases 5-6) deliberately sweep them.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Patient:
    p1: float = 0.028      # 1/min   glucose effectiveness (insulin independent)
    p2: float = 0.025      # 1/min   insulin action decay
    p3: float = 1.3e-5     # 1/min^2 per (uU/mL)  insulin sensitivity
    n: float = 0.093       # 1/min   insulin clearance rate
    Gb: float = 90.0       # mg/dL   basal glucose
    Ib: float = 7.0        # uU/mL   basal insulin
    VI: float = 12.0       # L       insulin distribution volume
    VG: float = 112.0      # dL      glucose distribution volume (1.6 dL/kg * 70 kg)

    @property
    def u_basal(self) -> float:
        """Basal infusion [mU/min] that holds (Gb, 0, Ib) as an equilibrium."""
        return self.n * self.Ib * self.VI

    def scaled(self, **factors) -> "Patient":
        """Return a copy with chosen parameters multiplied, e.g. scaled(p3=0.5)."""
        return replace(self, **{k: getattr(self, k) * f for k, f in factors.items()})


@dataclass(frozen=True)
class Meal:
    carbs_g: float = 60.0      # grams of carbohydrate
    t_meal: float = 30.0       # min, meal start
    tau: float = 40.0          # min, time-to-peak of gut absorption
    bioavail: float = 0.8      # fraction of carbs reaching plasma

    def D(self, t, patient: Patient):
        """Glucose appearance rate [mg/dL/min]: gamma-shaped (Tmax = tau)."""
        import numpy as np
        s = np.asarray(t, dtype=float) - self.t_meal
        total = self.bioavail * self.carbs_g * 1000.0 / patient.VG   # mg/dL
        out = np.where(s > 0, total * s / self.tau**2 * np.exp(-s / self.tau), 0.0)
        return out if out.ndim else float(out)


# Clinical targets
TARGET = 100.0        # mg/dL control set point
HYPO = 70.0           # mg/dL
HYPER = 180.0         # mg/dL

DEFAULT_PATIENT = Patient()
DEFAULT_MEAL = Meal()
