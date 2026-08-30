"""Viscoelastic constitutive validation: three homogeneous rheometric tests, each
against a closed-form analytic target, for the log-conformation update used by the
solid solver (pyRMT.viscoelastic.logconf_local_step_strang).

  (a) Startup of steady shear     -- transient sigma_xy(t), N1(t) vs UCM analytic
  (b) Small-amplitude oscillatory -- storage/loss moduli G'(w), G''(w) vs Maxwell
      shear (SAOS)
  (c) Steady planar extension     -- b_e^xx, b_e^yy vs 1/(1-2Wi), 1/(1+2Wi),
                                     including the coil-stretch approach to Wi=1/2

The upper-convected Maxwell (UCM) branch sigma = G b_e is exercised; G=1, tau=1 so
Wi = (rate)*tau reads directly off the imposed rate. The point is that the matrix-
logarithm transport reproduces the analytic viscometric functions across transient,
frequency and high-Weissenberg regimes while keeping b_e symmetric positive-definite.

Usage: python benchmarks/viscoelastic_validation.py  ->  data/viscoelastic_validation.png
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyRMT.viscoelastic import logconf_local_step_strang, sym_exp

G, TAU = 1.0, 1.0


def _evolve(Lfun, t_end, dt):
    """Integrate psi=log(b_e) from rest (b_e=I) under a (possibly time-dependent)
    homogeneous velocity gradient L(t). Returns arrays t, b11, b12, b22."""
    p11 = p12 = p22 = 0.0                       # psi = log(I) = 0
    n = int(round(t_end / dt))
    ts = np.empty(n + 1); b11 = np.empty(n + 1); b12 = np.empty(n + 1); b22 = np.empty(n + 1)
    for k in range(n + 1):
        t = k * dt
        c11, c12, c22 = sym_exp(p11, p12, p22)
        ts[k], b11[k], b12[k], b22[k] = t, c11, c12, c22
        L11, L12, L21, L22 = Lfun(t + 0.5 * dt)   # midpoint L for the Strang stretch
        p11, p12, p22 = logconf_local_step_strang(p11, p12, p22, L11, L12, L21, L22, TAU, dt)
    return ts, b11, b12, b22


# ── (a) startup of steady shear ─────────────────────────────────────────────
def startup_shear(gdot=1.0, t_end=6.0, dt=2e-3):
    ts, b11, b12, b22 = _evolve(lambda t: (0.0, gdot, 0.0, 0.0), t_end, dt)
    sxy = G * b12; N1 = G * (b11 - b22)
    sxy_an = G * TAU * gdot * (1.0 - np.exp(-ts / TAU))
    N1_an = 2.0 * G * (TAU * gdot)**2 * (1.0 - (1.0 + ts / TAU) * np.exp(-ts / TAU))
    err = max(np.max(np.abs(sxy - sxy_an)), np.max(np.abs(N1 - N1_an))) / (G * TAU * gdot)
    return ts, sxy, N1, sxy_an, N1_an, err


# ── (b) small-amplitude oscillatory shear (SAOS) ────────────────────────────
def saos(gamma0=0.02, n_dec=25):
    wtau = np.logspace(-1.3, 1.3, n_dec)          # omega*tau sweep
    Gp = np.empty_like(wtau); Gpp = np.empty_like(wtau)
    for i, wt in enumerate(wtau):
        w = wt / TAU
        period = 2.0 * np.pi / w
        dt = period / 400.0
        t_end = 24.0 * period                     # discard transient, fit last 4 periods
        Lfun = lambda t: (0.0, gamma0 * w * np.cos(w * t), 0.0, 0.0)
        ts, _, b12, _ = _evolve(Lfun, t_end, dt)
        sxy = G * b12
        mask = ts >= t_end - 4.0 * period         # steady oscillation window
        t, s = ts[mask], sxy[mask]
        # sigma = gamma0 (G' sin wt + G'' cos wt); least-squares projection
        A = np.column_stack([np.sin(w * t), np.cos(w * t)])
        coef, *_ = np.linalg.lstsq(A, s / gamma0, rcond=None)
        Gp[i], Gpp[i] = coef
    Gp_an = G * wtau**2 / (1.0 + wtau**2)
    Gpp_an = G * wtau / (1.0 + wtau**2)
    err = max(np.max(np.abs(Gp - Gp_an)), np.max(np.abs(Gpp - Gpp_an))) / G
    return wtau, Gp, Gpp, Gp_an, Gpp_an, err


# ── (c) steady planar extension ─────────────────────────────────────────────
def planar_extension(wi_max=0.49, n=18):
    Wi = np.linspace(0.02, wi_max, n)
    bxx = np.empty_like(Wi); byy = np.empty_like(Wi)
    for i, w in enumerate(Wi):
        edot = w / TAU
        # relax toward steady state; time to steady grows as Wi->1/2, so scale t_end
        t_end = 12.0 * TAU / (1.0 - 2.0 * w + 0.05)
        ts, b11, _, b22 = _evolve(lambda t: (edot, 0.0, 0.0, -edot), t_end, 2e-3 * TAU)
        bxx[i], byy[i] = b11[-1], b22[-1]
    bxx_an = 1.0 / (1.0 - 2.0 * Wi); byy_an = 1.0 / (1.0 + 2.0 * Wi)
    err = np.max(np.abs(bxx - bxx_an) / bxx_an)
    return Wi, bxx, byy, bxx_an, byy_an, err


def main():
    import matplotlib; matplotlib.use("Agg")
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    ts, sxy, N1, sxy_an, N1_an, e_a = startup_shear()
    wtau, Gp, Gpp, Gp_an, Gpp_an, e_b = saos()
    Wi, bxx, byy, bxx_an, byy_an, e_c = planar_extension()
    print(f"startup shear  max err = {e_a:.2e}")
    print(f"SAOS moduli    max err = {e_b:.2e}")
    print(f"planar ext.    max rel err = {e_c:.2e}")

    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.7))

    # (a) startup of shear (normalized by steady-state values)
    g = G * TAU * 1.0
    ax[0].plot(ts, sxy_an / g, "--", color="k", lw=1.4, label="UCM analytic")
    ax[0].plot(ts, N1_an / (2 * G * TAU**2), "--", color="k", lw=1.4)
    ax[0].plot(ts[::25], sxy[::25] / g, "o", ms=4.5, color="b",
               label=r"$\sigma_{xy}/G\tau\dot\gamma$")
    ax[0].plot(ts[::25], N1[::25] / (2 * G * TAU**2), "s", ms=4.5, color="r",
               label=r"$N_1/2G\tau^2\dot\gamma^2$")
    ax[0].set_xlabel(r"$t/\tau$"); ax[0].set_ylabel("normalized stress")
    ax[0].set_title(r"(a) startup of steady shear ($\mathrm{Wi}=1$)")
    ax[0].legend(loc="lower right"); ax[0].set_xlim(0, ts[-1])

    # (b) SAOS moduli
    ax[1].loglog(wtau, Gp_an, "--", color="k", lw=1.4, label="Maxwell analytic")
    ax[1].loglog(wtau, Gpp_an, "--", color="k", lw=1.4)
    ax[1].loglog(wtau, Gp, "o", ms=5, color="b", label=r"$G'(\omega)/G$")
    ax[1].loglog(wtau, Gpp, "s", ms=5, color="r", label=r"$G''(\omega)/G$")
    ax[1].set_xlabel(r"$\omega\tau$"); ax[1].set_ylabel(r"$G',\,G''\ /\ G$")
    ax[1].set_title("(b) oscillatory shear (SAOS)")
    ax[1].legend(loc="lower right")

    # (c) planar extension
    ax[2].plot(Wi, bxx_an, "--", color="k", lw=1.4, label="UCM analytic")
    ax[2].plot(Wi, byy_an, "--", color="k", lw=1.4)
    ax[2].plot(Wi, bxx, "o", ms=5, color="b", label=r"$b_e^{xx}$")
    ax[2].plot(Wi, byy, "s", ms=5, color="r", label=r"$b_e^{yy}$")
    ax[2].axvline(0.5, ls=":", color="0.5", lw=1.2)
    ax[2].set_yscale("log")
    ax[2].set_xlabel(r"$\mathrm{Wi}=\dot\varepsilon\,\tau$")
    ax[2].set_ylabel(r"steady $b_e$ eigenvalues")
    ax[2].set_title("(c) planar extension")
    ax[2].legend(loc="upper left")

    fig.tight_layout()
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data, exist_ok=True)
    out = os.path.join(data, "viscoelastic_validation.png")
    fig.savefig(out); print("saved", out)
    return e_a, e_b, e_c


if __name__ == "__main__":
    main()
