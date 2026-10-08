"""Unit tests.  Run:  python -m unittest discover tests   (or: pytest)"""
import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ap.params import *
from ap.model import rhs, equilibrium, simulate_open_loop
from ap.linear import *
from ap.analysis import loop_margins
from ap.controllers import PIDController, ConstrainedPI, LQRObserverController
from ap.sim import simulate_closed_loop, metrics
from ap import tuned

p = DEFAULT_PATIENT


class TestModel(unittest.TestCase):
    def test_equilibrium(self):
        self.assertTrue(np.allclose(rhs(0, equilibrium(p), p.u_basal, p, Meal(carbs_g=0)), 0))

    def test_meal_mass_balance(self):
        t = np.linspace(0, 2000, 200001)
        self.assertAlmostEqual(np.trapezoid(DEFAULT_MEAL.D(t, p), t), DEFAULT_MEAL.bioavail * DEFAULT_MEAL.carbs_g * 1000 / p.VG, places=1)

    def test_insulin_lowers_glucose(self):
        _, x, _ = simulate_open_loop(p, Meal(carbs_g=0), u_func=lambda t: p.u_basal + 10, t_end=1500)
        self.assertLess(x[-1, 0], p.Gb)


class TestLinear(unittest.TestCase):
    def test_linearization_matches_jacobian(self):
        A, B, E, C = state_space(p); x0 = equilibrium(p); h = 1e-6
        J = np.zeros((3, 3))
        for k in range(3):
            dx = np.zeros(3); dx[k] = h
            J[:, k] = (rhs(0, x0 + dx, p.u_basal, p, Meal(carbs_g=0)) - rhs(0, x0 - dx, p.u_basal, p, Meal(carbs_g=0))) / (2 * h)
        self.assertTrue(np.allclose(J, A, atol=1e-8))

    def test_tf_matches_state_space(self):
        A, B, E, C = state_space(p)
        from scipy import signal
        n, d = signal.ss2tf(A, B, C, [[0]])
        Pn, Pd = plant_tf(p)
        self.assertTrue(np.allclose(-n[0][-1], Pn[-1], rtol=1e-9)); self.assertTrue(np.allclose(d, Pd))

    def test_controllable_observable(self):
        A, B, E, C = state_space(p)
        self.assertEqual(controllability(A, B)[1], 3); self.assertEqual(observability(A, C)[1], 3)
        Aa, Ba, Ca = augmented_with_disturbance(p); self.assertEqual(observability(Aa, Ca)[1], 4)

    def test_routh(self):
        _, ch = routh_table([1, 3, 3, 1]); self.assertEqual(ch, 0)           # (s+1)^3
        _, ch = routh_table([1, 1, 1, 10]); self.assertEqual(ch, 2)          # unstable


class TestControllers(unittest.TestCase):
    def test_constraints_respected(self):
        for c in tuned.make_all(p):
            r = simulate_closed_loop(c, p, Meal(carbs_g=100), t_end=600)
            self.assertGreaterEqual(r["u"].min(), 0.0); self.assertLessEqual(r["u"].max(), 200.0 + 1e-9)

    def test_loops_stable_with_margins(self):
        for c in tuned.make_all(p):
            m = loop_margins(c, p); self.assertTrue(m["stable"]); self.assertGreater(m["gm_db"], 6); self.assertGreater(m["pm_deg"], 30)

    def test_closed_loop_beats_open_loop_and_no_hypo(self):
        class Basal:
            name = "b"
            def reset(self): pass
            def __call__(self, t, y): return p.u_basal
        peak_ol = metrics(simulate_closed_loop(Basal(), p, DEFAULT_MEAL, t_end=600))["peak"]
        for c in tuned.make_all(p):
            m = metrics(simulate_closed_loop(c, p, DEFAULT_MEAL, t_end=600))
            self.assertLess(m["peak"], peak_ol); self.assertEqual(m["t_hypo"], 0.0)

    def test_zero_steady_state_error(self):
        for c in tuned.make_all(p):
            r = simulate_closed_loop(c, p, DEFAULT_MEAL, t_end=1500)
            self.assertLess(abs(r["G"][-1] - TARGET), 2.0, msg=c.name)

    def test_iob_cap(self):
        c = ConstrainedPI(p, **tuned.PI); simulate_closed_loop(c, p, Meal(carbs_g=100), t_end=600)
        self.assertLessEqual(c.B, tuned.PI["B_max"] + 1e-6)


if __name__ == "__main__":
    unittest.main()
