"""Regression tests for the homogeneous viscoelastic (UCM) validation suite:
startup of steady shear, SAOS moduli, and steady planar extension all match the
closed-form analytic targets. Guards the log-conformation constitutive update used
by the paper's viscoelastic validation figure."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
from viscoelastic_validation import startup_shear, saos, planar_extension


def test_startup_shear_matches_ucm():
    *_, err = startup_shear()
    assert err < 1e-4, f"startup-shear error {err:.2e}"


def test_saos_matches_maxwell():
    *_, err = saos(n_dec=5)          # reduced sweep for speed
    assert err < 5e-3, f"SAOS moduli error {err:.2e}"


def test_planar_extension_matches_ucm():
    *_, err = planar_extension(n=8)
    assert err < 5e-2, f"planar-extension rel error {err:.2e}"
