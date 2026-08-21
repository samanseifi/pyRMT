"""Linearly-implicit elastic stabilizer (#14.4, stage 1): lifts the elastic-wave
CFL dt < dx/cs that survives the IMEX viscous+relaxation lift.

Theory (1D von-Neumann on the linear elastic wave u_t = cs^2 d_xx, d_t = u): with the
elastic force inside the implicit RHS and an added O(dt^2) wave operator dt^2 cs^2 Lap,
the amplification is |lambda| = 1/sqrt(1 + dt^2 cs^2 k^2) <= 1 -> unconditionally stable.

Checks on the four-roll extensional blob:
  1. CONSISTENCY -- at small dt the elastic-IMEX result reproduces the explicit
     reference (the stabilizer is O(dt^2) and vanishes).
  2. CFL LIFT -- at a dt well above the elastic-wave CFL the plain IMEX driver (elastic
     force explicit) diverges, while elastic-IMEX stays finite and completes.

Stage-1 caveat (documented, not hidden): the isotropic-Laplacian stabilizer adds
numerical damping, so at large dt elastic-IMEX is stable but progressively over-damped;
an accurate-at-large-dt scheme needs the consistent material tangent (stage 2, JFNK).
"""
import numpy as np
from benchmarks.mac_viscoelastic_extension import run

_CFG = dict(N=40, tau=0.5, G=0.3, eps_rate=0.5, mu_f=0.05, save_frames=False, write=False)
_DT_EL = 0.3 * (1.0 / 40) / np.sqrt(0.3)        # elastic-wave CFL of this config


def _bxx(h):
    return h[-1][2] if h else float("nan")


def _reached(h, t_end):
    return bool(h) and h[-1][0] >= t_end - 1e-9


def test_elastic_imex_consistent_at_small_dt():
    """At small dt the elastic-IMEX stabilizer vanishes -> reproduces explicit b_xx."""
    t_end = 0.6
    _, h_ref = run(imex=False, t_end=t_end, **_CFG)
    _, h_el = run(elastic_imex=True, dt=0.2 * _DT_EL, t_end=t_end, **_CFG)
    assert _reached(h_ref, t_end) and _reached(h_el, t_end)
    br, be = _bxx(h_ref), _bxx(h_el)
    assert abs(be - br) / br < 0.025, f"elastic-IMEX small-dt rel diff {abs(be-br)/br:.2e}"


def test_elastic_imex_lifts_elastic_cfl():
    """At 8x the elastic-wave CFL, plain IMEX (explicit elastic force) diverges while
    elastic-IMEX stays finite and completes."""
    t_end = 0.8
    dt = 8.0 * _DT_EL
    with np.errstate(over="ignore", invalid="ignore"):
        _, h_plain = run(imex=True, dt=dt, t_end=t_end, **_CFG)
    _, h_el = run(elastic_imex=True, dt=dt, t_end=t_end, **_CFG)
    assert not _reached(h_plain, t_end), "plain IMEX unexpectedly survived 8x the elastic CFL"
    assert _reached(h_el, t_end) and np.isfinite(_bxx(h_el)), "elastic-IMEX did not complete"
