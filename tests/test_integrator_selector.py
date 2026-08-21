"""Unified time-integration strategy selector (resolve_integrator): one `integrator=`
knob across the MAC drivers, back-compatible with the legacy per-driver booleans."""
import numpy as np
import pytest
from pyRMT.mac import resolve_integrator, INTEGRATORS


def test_canonical_names_and_flags():
    assert resolve_integrator("explicit") == ("explicit", False, False)
    assert resolve_integrator("imex") == ("imex", True, False)
    assert resolve_integrator("imex-elastic") == ("imex-elastic", True, True)


def test_aliases():
    assert resolve_integrator("IMEX_Elastic")[0] == "imex-elastic"
    assert resolve_integrator("elastic")[0] == "imex-elastic"
    assert resolve_integrator("implicit-viscosity")[0] == "imex"
    assert resolve_integrator("semi-implicit")[0] == "imex"
    assert resolve_integrator("forward-euler")[0] == "explicit"


def test_legacy_booleans_when_integrator_none():
    assert resolve_integrator(None) == ("explicit", False, False)
    assert resolve_integrator(None, imex=True) == ("imex", True, False)
    assert resolve_integrator(None, implicit_visc=True) == ("imex", True, False)
    assert resolve_integrator(None, elastic_imex=True) == ("imex-elastic", True, True)


def test_errors():
    with pytest.raises(ValueError):
        resolve_integrator("bogus")
    with pytest.raises(ValueError):                 # driver without a solid-stress path
        resolve_integrator("imex-elastic", supports_elastic=False)


def test_string_selector_matches_legacy_flag_on_taylor_green():
    """integrator="imex" is byte-identical to the legacy implicit_visc=True path."""
    from benchmarks.mac_taylor_green_convergence import run_one
    e_flag, _ = run_one(32, nu=0.1, T=0.05, dt=1e-3, implicit_visc=True)
    e_str, _ = run_one(32, nu=0.1, T=0.05, dt=1e-3, integrator="imex")
    assert e_flag == e_str
    e_exp0, _ = run_one(32, nu=0.1, T=0.05, dt=1e-3)
    e_exp1, _ = run_one(32, nu=0.1, T=0.05, dt=1e-3, integrator="explicit")
    assert e_exp0 == e_exp1
