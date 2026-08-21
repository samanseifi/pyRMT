"""Stage-2 core: energy-conserving implicit elastic-wave integrator (#14.4 stage 2).

The stage-1 stabilizer lifts the elastic CFL but damps the elastic wave; stage-2 uses a
trapezoidal (implicit-midpoint) coupling of velocity and displacement, which is
unconditionally stable AND energy-conserving (undamped). Verified on the linear elastic
standing shear wave d_tt = cs^2 nabla^2 d (the wave that sets the RMT elastic CFL).
"""
import numpy as np
from benchmarks.implicit_elastic_wave import run_one


def test_stage2_energy_conserving_where_stage1_damps():
    """At 2x the explicit CFL the stage-2 integrator conserves energy while the
    stage-1 stabilizer damps the wave away."""
    s2 = run_one("stage2", dt_fac=2.0, n_periods=5.0)
    s1 = run_one("stage1", dt_fac=2.0, n_periods=5.0)
    assert s2["edrift"] < 1e-2, f"stage-2 energy drift {s2['edrift']:.2e} too large"
    assert s1["edrift"] > 0.5, "stage-1 expected to damp strongly"
    assert s2["err"] < 0.2 < s1["err"], "stage-2 should be far more accurate"


def test_stage2_stable_above_cfl_where_explicit_diverges():
    """At 5x the explicit CFL the explicit scheme diverges; stage-2 stays finite and
    energy-conserving (mild phase error only)."""
    ex = run_one("explicit", dt_fac=5.0, n_periods=5.0)
    s2 = run_one("stage2", dt_fac=5.0, n_periods=5.0)
    assert not np.isfinite(ex["err"]) or ex["err"] > 1e2, "explicit should diverge at 5x CFL"
    assert np.isfinite(s2["err"]) and s2["edrift"] < 5e-2, "stage-2 should stay bounded/undamped"


def test_stage2_accurate_at_moderate_dt():
    """At the explicit CFL the stage-2 integrator is accurate (small L2 error) and
    essentially energy-neutral."""
    s2 = run_one("stage2", dt_fac=1.0, n_periods=5.0)
    assert s2["err"] < 5e-2 and s2["edrift"] < 1e-3
