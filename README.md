# Artificial Pancreas: Feedback Control of Blood Glucose

Control Systems (BECE302L) project. Bergman minimal model + PID / constrained PI (insulin-on-board) /
LQR-with-observer, following the 8-phase roadmap of the project slides.
Pure Python: **numpy + scipy + matplotlib only** (no `control` package, no MATLAB needed).

## Setup (any OS, Python 3.9+)

```
pip install -r requirements.txt
```

## Run it - one phase at a time

```
cd phases
python phase1_parameters.py     # 1 Literature & parameters, units, basal equilibrium
python phase2_nonlinear.py      # 2 Nonlinear simulation + model validation checks
python phase3_linearization.py  # 3 Linearization, state space, transfer functions
python phase4_time_stability.py # 4 Poles, Routh-Hurwitz, root locus, meal response
python phase5_frequency.py      # 5 Bode, GM/PM, delay margin, sensitivity, patient-gain robustness
python phase6_controllers.py    # 6 PID vs constrained PI vs LQR/pole-placement (add --retune to re-tune)
python phase7_state_space.py    # 7 Controllability, observability, state feedback, observer
python phase8_report.py         # 8 Builds results/REPORT.md from phases 1-7
```

or everything at once (~1 min): `python run_all.py`.
Add `--show` to any phase to pop up the figures; otherwise they are saved to `results/figures/`.
Tests: `python -m unittest discover tests`.

## Layout

```
ap/params.py       model parameters, meal model, targets (single source of truth)
ap/model.py        nonlinear Bergman model, RK4 integrator
ap/linear.py       linearization, transfer functions, Routh table, controllability/observability
ap/analysis.py     frequency response, margins, root locus helpers
ap/controllers.py  PIDController, ConstrainedPI (IOB), LQRObserverController (LQR or pole placement)
ap/sim.py          sampled closed-loop simulation (noise, CGM delay) + clinical metrics
ap/tuning.py       numerical tuning on the nonlinear model;  ap/tuned.py = frozen gains
phases/            one script per project phase;   results/ = figures, JSON, REPORT.md
```

## Key modelling choices (worth stating in your report)

- Units: u in mU/min, I in uU/mL, VI in L, so u/VI is uU/mL/min. `u` is the TOTAL infusion;
  the slides' u(t) is `u - u_basal`, with `u_basal = n*Ib*VI`, so the physical constraint is u >= 0.
- Plant gain is negative (insulin lowers glucose); controllers use positive gain on e = G - r.
- Set point 100 mg/dL (basal 90), target range 70-180, pump limit 200 mU/min.
- Controllers run at the CGM period (5 min); simulation can add 3 mg/dL noise and a 10 min delay.
- Linearization is only a local design model (Phase 3 quantifies its error); every performance
  number is measured on the nonlinear plant.

## Caveats

- Parameters are typical literature values, not patient data. Check the citations in Phase 1
  against the originals before putting them in a submission. This is a coursework simulation,
  not a medical device.
- The slides mention `python-control` and an optional MATLAB cross-check; neither is required here
  (all analysis is done with scipy), and the MATLAB check is not included.
