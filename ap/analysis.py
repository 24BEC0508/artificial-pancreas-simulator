"""Phase 4/5 - frequency-domain & stability helpers (numpy/scipy only)."""
import numpy as np
from scipy import signal


def freqresp(num, den, w):
    """Complex frequency response of num(s)/den(s) at angular frequencies w [rad/min]."""
    s = 1j * w
    return np.polyval(num, s) / np.polyval(den, s)


def margins_from_response(w, L):
    """Gain/phase margins and crossover frequencies from L(jw) samples.

    Returns dict(gm_db, pm_deg, wgc, wpc).  Missing crossover -> np.inf / np.nan.
    Phase is unwrapped and referenced so that a standard negative-feedback loop
    is unstable at -180 deg.
    """
    mag = np.abs(L)
    ph = np.unwrap(np.angle(L)) * 180 / np.pi
    out = dict(gm_db=np.inf, pm_deg=np.nan, wgc=np.nan, wpc=np.nan)
    # gain crossover |L| = 1 (last downward crossing -> the relevant one)
    idx = np.where((mag[:-1] >= 1) & (mag[1:] < 1))[0]
    if len(idx):
        i = idx[-1]
        f = (np.log(1) - np.log(mag[i])) / (np.log(mag[i + 1]) - np.log(mag[i]))
        out["wgc"] = float(np.exp(np.log(w[i]) + f * (np.log(w[i + 1]) - np.log(w[i]))))
        phg = ph[i] + f * (ph[i + 1] - ph[i])
        out["pm_deg"] = float(180 + phg)
    # phase crossover -180
    idx = np.where((ph[:-1] > -180) & (ph[1:] <= -180))[0]
    if len(idx):
        i = idx[0]
        f = (-180 - ph[i]) / (ph[i + 1] - ph[i])
        out["wpc"] = float(np.exp(np.log(w[i]) + f * (np.log(w[i + 1]) - np.log(w[i]))))
        mp = np.exp(np.log(mag[i]) + f * (np.log(mag[i + 1]) - np.log(mag[i])))
        out["gm_db"] = float(-20 * np.log10(mp))
    return out


def delay_margin(pm_deg, wgc):
    """Largest pure time delay [min] before the loop loses stability (PM / wgc)."""
    if np.isnan(pm_deg) or np.isnan(wgc):
        return np.inf
    return np.deg2rad(pm_deg) / wgc


def closed_loop_poly(num_L, den_L):
    """Characteristic polynomial den + num of the unity negative-feedback loop."""
    n = max(len(num_L), len(den_L))
    return np.polyadd(np.pad(den_L, (n - len(den_L), 0)), np.pad(num_L, (n - len(num_L), 0)))


def root_locus(num, den, gains):
    """Closed-loop poles of 1 + k*num/den = 0 for each k.  Returns array (len(gains), order)."""
    out = []
    for k in gains:
        out.append(np.roots(closed_loop_poly(k * np.asarray(num), np.asarray(den))))
    return np.array(out)


def controller_tf(controller):
    """(num, den) of the controller y->u_d for any controller in ap.controllers."""
    if hasattr(controller, "ss"):
        Ac, Bc, Cc, Dc = controller.ss()
        n, d = signal.ss2tf(Ac, Bc, Cc, Dc)
        return n[0], d
    return controller.tf()


def loop_margins(controller, patient, w=None):
    """Margins of L(s) = C(s) P(s) with P(s) the (positive-gain) plant of `patient`."""
    from .linear import plant_tf
    w = np.logspace(-4, 1, 3000) if w is None else w
    Pn, Pd = plant_tf(patient)
    cn, cd = controller_tf(controller)
    Ln, Ld = np.polymul(cn, Pn), np.polymul(cd, Pd)
    m = margins_from_response(w, freqresp(Ln, Ld, w))
    m["stable"] = bool(np.all(np.roots(closed_loop_poly(Ln, Ld)).real < 0))
    return m
