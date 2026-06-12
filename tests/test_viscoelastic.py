"""Unit tests for the viscoelastic (upper-convected Maxwell) constitutive update."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.viscoelastic import ucm_local_step, viscoelastic_stress


def test_step_strain_relaxation():
    """sigma_xy relaxes as G*gamma*exp(-t/tau) after a held step shear strain."""
    G, tau, gamma, dt = 1.0, 2.0, 0.5, 2e-4
    b11, b12, b22 = 1.0 + gamma**2, gamma, 1.0      # F F^T for shear gamma
    for k in range(1, int(round(8.0 / dt)) + 1):
        b11, b12, b22 = ucm_local_step(b11, b12, b22, 0., 0., 0., 0., tau, dt)
        assert abs(b12 - gamma * np.exp(-k * dt / tau)) < 1e-4


def test_steady_shear_viscometric():
    """UCM steady simple shear: b_xy=tau*gdot, b_xx=1+2 tau^2 gdot^2, N1=2 G tau^2 gdot^2."""
    G, tau, gdot, dt = 1.0, 2.0, 0.3, 2e-4
    b11, b12, b22 = 1.0, 0.0, 1.0
    for _ in range(int(round(80.0 / dt))):
        b11, b12, b22 = ucm_local_step(b11, b12, b22, 0., gdot, 0., 0., tau, dt)
    assert abs(b12 - tau * gdot) < 1e-3
    assert abs(b11 - (1.0 + 2.0 * tau**2 * gdot**2)) < 1e-3
    assert abs(b22 - 1.0) < 1e-3
    _, sxy, _ = viscoelastic_stress(b11, b12, b22, G)
    N1 = G * (b11 - b22)
    assert abs(sxy - G * tau * gdot) < 1e-3
    assert abs(N1 - 2.0 * G * tau**2 * gdot**2) < 1e-3


def test_elastic_limit():
    """tau -> infinity: b_e tracks F F^T exactly under constant L."""
    gdot, dt = 0.4, 2e-4
    b11, b12, b22 = 1.0, 0.0, 1.0
    F = np.array([[1.0, 0.0], [0.0, 1.0]]); L = np.array([[0.0, gdot], [0.0, 0.0]])
    for _ in range(int(round(2.0 / dt))):
        b11, b12, b22 = ucm_local_step(b11, b12, b22, 0., gdot, 0., 0., np.inf, dt)
        k1 = L @ F; k2 = L @ (F + dt * k1); F = F + 0.5 * dt * (k1 + k2)
    bb = F @ F.T
    assert abs(b11 - bb[0, 0]) < 1e-6
    assert abs(b12 - bb[0, 1]) < 1e-6
    assert abs(b22 - bb[1, 1]) < 1e-6
