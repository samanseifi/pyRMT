# pyRMT API refactoring strategy

Goal: a cleaner, smaller public API and the end of copy-pasted time loops, **without
breaking the test suite** and without over-engineering a research code.

## What's wrong today (from the current tree)
1. **`functions.py` is a ~1450-line grab-bag** — collocated solver, advection
   schemes, neo-Hookean stress, extrapolation, three Poisson solvers (DCT/FFT/AMG),
   curvature, contact force, reinit, *and* deprecated aliases all in one file.
2. **Two solver families with overlapping names** — `divergence`, `project`,
   `poisson_*` exist in both `mac.py` (staggered) and `functions.py` (collocated)
   with different conventions; easy to import the wrong one.
3. **Leaky `__init__`** — it re-exports private `_precompute_*` helpers and
   deprecated aliases; there is no `__all__`, so the public surface is undefined.
4. **No high-level driver** — every benchmark hand-rolls the same loop (advect ξ →
   extrapolate → stress → assemble force → momentum predictor → project). This was
   copy-pasted ~10× while building the demos. It is the biggest source of friction.
5. **Long argument lists** — e.g. `momentum_step_rk4(...)` (~20 args),
   `compute_timestep(...)`, `solid_cauchy_stress(...)` — scalars threaded everywhere.
6. **Mixed concerns / kernels** — operators and physics live in the same modules;
   Numba kernels are interleaved with the pure-Python API.

## Target layout (move, don't rewrite)
```
pyRMT/
  grid.py            # GridSpec, mac_grid, cell/face coordinate helpers
  operators.py       # divergence / gradient / laplacian (staggered + periodic)
  poisson.py         # neumann(DCT), periodic(FFT), amg; one solve() entry
  projection.py      # project(), project_per()
  momentum.py        # predictors: lid / freeslip / periodic (+ face forces)
  advection.py       # semilagrangian(+cubic) / central2 / weno5 / conservative + dispatch
  interface.py       # smoothed_heaviside, extrapolate, rebuild_phi, level-set reinit
  forces.py          # surface tension (CSF), contact_stress
  constitutive/
    elastic.py       # neo-Hookean solid_cauchy_stress
    viscoelastic.py  # UCM + log-conformation (already clean -> move as-is)
  reinit.py          # reference-map F-storage; level-set FMM/PDE
  io.py              # energy / dissipation / HDF5 output (from output.py)
  interpolators.py   # Numba kernels (private)
  utils.py           # finite-difference helpers
  collocated.py      # legacy collocated solver (the rest of functions.py), namespaced
  sim.py             # the high-level driver (see Stage 3)
  __init__.py        # curated public API + __all__
```
Keep `mac.py` and `functions.py` as **thin re-export shims** during the transition
so imports keep working; emit `DeprecationWarning` and remove in a later version.

## Config objects (tame the arg lists)
Small frozen dataclasses replace scalar soup:
```python
GridSpec(n=128, lx=1.0, ly=1.0)
Fluid(rho=1.0, mu=0.01)
Elastic(mu_s=1.0, kappa=0.0)
Viscoelastic(G=0.3, tau=0.5, G_inf=0.0)
SurfaceTension(gamma=1.0)
Contact(eta=2.5, eta_wall=3.0, eps_cells=3.0)
TimeControl(cfl=0.3, t_end=8.0, frame_dt=0.1)
```
Functions/driver take these instead of 10–20 positional scalars.

## Stage 3 — the high-level driver (the big win)
One class encapsulates the loop that every benchmark duplicates:
```python
from pyRMT import Simulation, GridSpec, Fluid, Elastic, Viscoelastic, Contact

sim = Simulation(GridSpec(128), Fluid(rho=1, mu=0.01),
                 bc="lid", u_lid=1.0, advection="semilagrangian")
sim.add_solid(disc_sdf, material=Elastic(mu_s=1.0))
sim.add_solid(bar_sdf,  material=Viscoelastic(G=0.3, tau=0.5))
sim.contact = Contact(eta=2.5)

for state in sim.run(t_end=8.0, frame_dt=0.1):   # generator of StepState
    render(state)            # state.phis, state.u, state.stress(), state.minJ ...
```
Internals:
- per-solid state `(xi, phi, b_e/psi, F_stored)`; `sim.step(dt)` runs
  advect → extrapolate → stress → forces (ST + all-pairs/wall contact) → momentum → project.
- **strategy seams** (the point): `bc`, `advection`, `material`, and crucially the
  **time integrator** are pluggable — so the implicit work (backlog #14) slots in by
  swapping the integrator, not rewriting the loop.
- built-in **frame streaming to disk** (the OOM fix becomes a feature) and
  **diagnostics hooks** (energy, max|div|, minJ) reused by tests.

Benchmarks then shrink to a few lines + their own rendering/CLI.

## Curate the public surface
- `__init__` exports only the documented API; add `__all__`.
- Move deprecated aliases + private `_`-helpers out of the public namespace
  (`pyRMT.compat` with `DeprecationWarning`).
- Type hints; consistent naming (`advect_*`, `project*`, `stress_*`); Numba kernels
  private in `_kernels`/`interpolators` behind pure-Python wrappers.

## Sequencing (keep tests green at every step)
0. **Safety net** — characterization tests pinning a few benchmark golden values +
   the existing operator/projection/stress/viscoelastic tests.
1. **Split `functions.py`** into the modules above (pure moves + shims). Run `pytest`.
2. **Config dataclasses** — adopt incrementally; keep scalar overloads temporarily.
3. **`Simulation` driver** — extract from one benchmark, port the rest onto it.
4. **Curate `__init__`/`__all__`**, route aliases through `compat`.
5. **Consolidate benchmarks** onto the driver; delete duplicated loops.
6. **Docs/tests** — update the manuals to the new API; add driver-level tests.

## Guardrails
- **Move before rewrite**; `pytest` after every stage.
- Back-compat shims with deprecation warnings; remove only in a tagged release.
- Don't over-abstract — the **driver (#3) + `functions.py` split (#1) + config
  objects (#2)** are ~80% of the value; resist deeper hierarchies.
- Design the driver's **integrator seam** now so the implicit/IMEX work (backlog
  #14) is a plug-in, not a fork.

## Priority
| step | value | effort |
|---|---|---|
| 3. `Simulation` driver | ★★★ (kills duplication, enables implicit swap) | medium |
| 1. split `functions.py` | ★★ | medium |
| 4. curate `__init__`/`__all__` | ★★ | small |
| 2. config dataclasses | ★ | small |
| 5. consolidate benchmarks | ★ | medium |
