"""Shared helpers for the phase scripts."""
import json, os, sys
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="Badly conditioned filter coefficients")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "figure.autolayout": True})
COLORS = {"PID": "#d95f02", "Constrained PI (IOB)": "#1b9e77", "LQR + observer": "#7570b3",
          "Pole placement + observer": "#e7298a", "open loop": "#666666"}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else str(v)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (complex, np.complexfloating)):
        return f"{complex(o).real:.5g}{complex(o).imag:+.5g}j"
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def save_json(name, data):
    with open(os.path.join(RESULTS, f"{name}.json"), "w") as f:
        json.dump(_clean(data), f, indent=2)


def load_json(name):
    with open(os.path.join(RESULTS, f"{name}.json")) as f:
        return json.load(f)


def savefig(fig, name):
    path = os.path.join(FIGS, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved results/figures/{name}.png")
    if "--show" in sys.argv:
        plt.show()
    plt.close(fig)
    return f"figures/{name}.png"


def table(rows, header):
    """Plain-text + markdown table helper. rows: list of lists."""
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    line = lambda r: " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r))
    txt = [line(header), "-+-".join("-" * x for x in w)] + [line(r) for r in rows]
    md = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + \
         ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(txt), "\n".join(md)


def banner(s):
    print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70)
