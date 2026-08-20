"""IMEX viscoelastic FSI integration (#14.3): the combined implicit-viscosity
(#14.1) + exact-relaxation (#14.2) integrator, wired into the coupled four-roll
extensional-blob driver.

Two checks, per the request that each new time-integration methodology must first
reproduce the previous results on a Taylor-Green-family flow before it earns its
extra stability:

  1. REGRESSION -- at a matched (viscous-limited) dt the IMEX driver reproduces the
     explicit b_xx(centre) trajectory.
  2. CAPABILITY -- at a dt above the explicit viscous CFL the explicit driver
     diverges (never reaches t_end) while the IMEX driver stays finite, reaches
     t_end, and matches the small-dt reference to a few percent.

The four-roll mill u = U0 (sin kx cos ky, -cos kx sin ky) is the same sin/cos
vortex family as Taylor-Green; the soft-disc-in-lid-driven regression is the
planned follow-on (needs the wall/DCT Helmholtz variant of #14.1).
"""
import numpy as np
from benchmarks.mac_viscoelastic_extension import run

_CFG = dict(N=40, tau=0.5, t_end=1.0, eps_rate=0.5, G=0.3, mu_f=0.05,
            save_frames=False, write=False)
_DX = 1.0 / 40
_DT_VISC = 0.2 * _DX * _DX / 0.05          # the driver's explicit viscous CFL


def _bxx(hist):
    return hist[-1][2] if hist else float("nan")


def _reached_end(hist):
    return bool(hist) and hist[-1][0] >= _CFG["t_end"] - 1e-9


def test_imex_fsi_reproduces_explicit_at_matched_dt():
    """REGRESSION: matched viscous-limited dt -> IMEX reproduces explicit b_xx."""
    _, h_exp = run(imex=False, dt=_DT_VISC, **_CFG)
    _, h_imex = run(imex=True, dt=_DT_VISC, **_CFG)
    assert _reached_end(h_exp) and _reached_end(h_imex), "a run terminated early"
    be, bi = _bxx(h_exp), _bxx(h_imex)
    assert abs(bi - be) / be < 0.03, f"IMEX vs explicit rel diff {abs(bi - be) / be:.2e}"


def test_imex_fsi_stable_where_explicit_diverges():
    """CAPABILITY: at dt = 3x the explicit viscous CFL the explicit driver diverges
    (does not reach t_end) while IMEX stays finite, completes, and matches the
    small-dt reference to a few percent."""
    dt = 3.0 * _DT_VISC
    _, h_ref = run(imex=False, dt=_DT_VISC, **_CFG)
    with np.errstate(over="ignore", invalid="ignore"):
        _, h_exp_big = run(imex=False, dt=dt, **_CFG)
    _, h_imex_big = run(imex=True, dt=dt, **_CFG)
    assert not _reached_end(h_exp_big), "explicit unexpectedly survived 3x the viscous CFL"
    assert _reached_end(h_imex_big), "IMEX did not reach t_end at 3x the viscous CFL"
    bi, br = _bxx(h_imex_big), _bxx(h_ref)
    assert np.isfinite(bi) and abs(bi - br) / br < 0.06, \
        f"IMEX big-dt b_xx {bi:.3f} vs reference {br:.3f}"
