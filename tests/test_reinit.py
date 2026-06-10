"""Level-set reinitialization: dispatcher + FMM signed-distance accuracy."""
import numpy as np
import pytest
from pyRMT.functions import create_grid, reinitialize_level_set


def _disc(N, R=0.25):
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    sdf = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - R   # exact signed distance
    return X, Y, dx, dy, sdf


def test_none_is_identity():
    _, _, dx, dy, sdf = _disc(65)
    phi = sdf * 1.0
    out = reinitialize_level_set(phi, dx, dy, method='none')
    assert np.array_equal(out, phi)


def test_unknown_method_raises():
    _, _, dx, dy, sdf = _disc(33)
    with pytest.raises(ValueError):
        reinitialize_level_set(sdf, dx, dy, method='bogus')


def test_fmm_recovers_signed_distance():
    skfmm = pytest.importorskip("skfmm")
    N = 129
    _, _, dx, dy, sdf = _disc(N)
    # corrupt the SDF (keep the zero level set), then redistance
    phi = np.sign(sdf) * (sdf ** 2 + 0.3)
    out = reinitialize_level_set(phi, dx, dy, method='fmm')
    band = np.abs(sdf) < 0.05
    # |grad(phi)| ~ 1 and matches the true SDF near the interface
    gy, gx = np.gradient(out, dy, dx)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    assert abs(mag[band].mean() - 1.0) < 0.05
    assert np.max(np.abs((out - sdf)[band])) < 0.02


def test_fmm_agrees_with_pde_near_interface():
    pytest.importorskip("skfmm")
    N = 129
    _, _, dx, dy, sdf = _disc(N)
    phi = np.sign(sdf) * (sdf ** 2 + 0.3)
    fmm = reinitialize_level_set(phi, dx, dy, method='fmm')
    pde = reinitialize_level_set(phi.copy(), dx, dy, method='pde',
                                 num_iters=200, dt_reinit_factor=0.2)
    band = np.abs(sdf) < 0.03
    assert np.max(np.abs((fmm - pde)[band])) < 0.03
