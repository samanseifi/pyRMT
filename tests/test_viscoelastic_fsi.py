"""Viscoelastic fluid-structure interaction validation (the paper's headline capability,
reviewer M3). A viscoelastic (Maxwell-branch, modulus G, relaxation tau) disc in a
lid-driven cavity:

  * tau -> infinity recovers the neo-Hookean soft-disc trajectory (which is itself
    validated against Sugiyama 2011) -- a two-way-coupled viscoelastic FSI tied to a
    validated benchmark;
  * finite tau runs stably and relaxes (physically sensible).
"""
import numpy as np
from benchmarks.mac_soft_disc_lid import run


def _reached(tr, t_end):
    return len(tr) > 0 and tr[-1, 0] >= t_end - 1e-9


def test_viscoelastic_tau_inf_recovers_neohookean():
    """tau=inf viscoelastic (b_e -> F F^T) reproduces the neo-Hookean soft-disc centroid."""
    N, t_end = 40, 0.6
    dx = 1.0 / N; dt = 0.2 * dx * dx / 0.01
    kw = dict(N=N, t_end=t_end, dt=dt, integrator="explicit", write=False)
    tn = run(mu_s=0.1, **kw)
    tv = run(mu_s=0.1, viscoelastic=True, G_ve=0.1, tau=np.inf, **kw)
    assert _reached(tn, t_end) and _reached(tv, t_end)
    n = min(len(tn), len(tv))
    d = max(np.abs(tn[:n, 1] - tv[:n, 1]).max(), np.abs(tn[:n, 2] - tv[:n, 2]).max())
    assert d < 1.5e-2, f"tau=inf viscoelastic vs neo-Hookean centroid diff {d:.2e}"


def test_viscoelastic_finite_tau_stable():
    """A finite-tau viscoelastic disc runs stably to t_end (relaxing solid)."""
    N, t_end = 40, 0.6
    dx = 1.0 / N; dt = 0.2 * dx * dx / 0.01
    tv = run(N=N, t_end=t_end, dt=dt, integrator="explicit", write=False,
             mu_s=0.1, viscoelastic=True, G_ve=0.1, tau=1.0)
    assert _reached(tv, t_end) and np.all(np.isfinite(tv[-1]))
