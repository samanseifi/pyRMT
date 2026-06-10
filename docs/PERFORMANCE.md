# Performance notes

## Profile (soft disc in lid-driven, N=128, semi-Lagrangian, steady state)

~31 ms/step on CPU (8 threads). Per-step breakdown:

| cost | component |
|---|---|
| ~16.7 ms | `momentum_step_rk4` — RK4 calls `velocity_rhs_blended_optimized` 4× (gradients, stress blend, upwind advection, divergence) |
| ~8.3 ms  | pressure projection (`_solve_poisson_dct` ~6 ms) |
| ~6.4 ms  | semi-Lagrangian advection of X1, X2 (bilinear interp) |
| ~3.5 ms  | reference-map extrapolation (serial least-squares) |

The DCT Poisson solve and the JIT stencil kernels are already near CPU-optimal;
the RK4 right-hand side dominates because it is evaluated four times per step.

## Applied

- **`cache=True` on all Numba kernels** — persists compiled code, so the ~18 s
  cold-start recompile is paid once, not every run (**~8× faster startup**:
  18 s → 2 s on the test suite). No effect on results (numerics unchanged).

## Next levers (not yet done — each changes numerics, needs re-validation)

1. **njit-fuse `velocity_rhs_blended_optimized`** — compile the whole RHS so the
   ~15 NumPy temporaries fuse into fewer passes. Biggest steady-state CPU win;
   may shift results at fma level (re-validate the disc within tolerance). Note:
   the surface-tension force arg must be made array-typed for njit.
2. **Parallelize the extrapolation** — the per-frontier least-squares fit is
   independent within a layer, but the current code marks cells "known" mid-loop
   (a race under `parallel=True`). Restructure to compute-then-mark; this changes
   which neighbours are available, so results shift slightly — re-validate.
3. **CG warm-start** (variable-density only) — start from the previous pressure
   correction instead of zero; migrate off the deprecated `tol=` kwarg.

## The real jump

GPU acceleration (issue #8): the structured grid + stencils + DCT/FFT map
directly to CuPy/cuFFT for ~10–50× at 256²–512², and it is the prerequisite for
3D (#10).
