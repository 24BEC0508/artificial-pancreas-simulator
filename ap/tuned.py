"""Frozen controller gains produced by `python phases/phase6_controllers.py --retune`
(see ap/tuning.py).  Re-tune any time; these are the defaults used by all phases."""
PID = dict(Kp=0.9655, Ki=0.004023, Kd=16.86, Tf=10.0)       # Ki = Kp/Ti, Ti ~ 240 min
PI = dict(Kp=0.8133, Ki=0.003389, B_max=900.0)             # Ti ~ 240 min, IOB cap 900 mU
LQR = dict(qz=0.43148, rho=0.99988)
# Pole-placement alternative (closed-loop poles of [g, x, i, z] augmented system)
POLES = [-0.06, -0.07, -0.10, -0.12]


def make_all(patient):
    from .controllers import PIDController, ConstrainedPI, LQRObserverController
    return [PIDController(patient, **PID), ConstrainedPI(patient, **PI),
            LQRObserverController(patient, **LQR)]
