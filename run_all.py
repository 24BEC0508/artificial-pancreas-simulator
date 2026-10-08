"""Run every phase in order:  python run_all.py   (about 1 minute)."""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PHASES = ["phase1_parameters", "phase2_nonlinear", "phase3_linearization", "phase4_time_stability",
          "phase5_frequency", "phase6_controllers", "phase7_state_space", "phase8_report"]
t0 = time.time()
for ph in PHASES:
    r = subprocess.run([sys.executable, os.path.join(HERE, "phases", ph + ".py")] + sys.argv[1:], cwd=os.path.join(HERE, "phases"))
    if r.returncode:
        sys.exit(f"{ph} failed")
print(f"\nAll phases done in {time.time()-t0:.0f} s.  Open results/REPORT.md and results/figures/")
