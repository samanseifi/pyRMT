# Staggered (MAC) solver — plan

Goal: a staggered Marker-and-Cell incompressible solver where divergence,
pressure-gradient and the Laplacian are **consistent by construction** (div = grad^T,
Laplacian = div∘grad), so the projection is exact (machine-zero divergence), there is
no checkerboard / Rhie-Chow workaround, and interfacial forcing is balanced. This is
the principled fix for the collocated issues (#1 parasitic currents, #2 contact, #18
2nd-order, #22).

Kept on a separate branch (`mac-staggered`); collocated stays on `main` as fallback.

## Layout (Nx x Ny cells, dx=Lx/Nx, dy=Ly/Ny)
- p : (Ny, Nx)      cell centres
- u : (Ny, Nx+1)    x-faces  (u[:,0], u[:,Nx] are the left/right walls)
- v : (Ny+1, Nx)    y-faces  (v[0,:], v[Ny,:] are the bottom/top walls)

## Phases (unit tests are the priority at every phase)
1. **Operators** (`pyRMT/mac.py`): divergence, pressure-gradient (to u/v faces),
   vector Laplacian, face interpolations.
   tests: exactness on manufactured fields; **div∘grad == Laplacian** (consistency).
2. **Projection**: Neumann Poisson via DCT-II; correct u*,v* -> divergence-free.
   tests: Poisson round-trip; **projection -> divergence machine-zero** (the headline).
3. **Advection + diffusion + time step** (momentum).
   tests: operator order; Taylor-Green decay vs analytic.
4. **Lid-driven cavity** driver -> Ghia Re=100 & 1000; spatial convergence.
5. **Reference map** (semi-Lagrangian, reused) on MAC; rebuild phi; stress.
6. **FSI**: soft disc in lid-driven vs Sugiyama; disc-in-TG energy.

Proceed phase by phase; do not advance until the phase's tests pass.

## Results so far

### Phase 1-2 (operators + projection) — DONE, 5 unit tests pass
- div(grad p) == DCT Laplacian to machine precision (consistency).
- projection -> divergence < 1e-11 (vs collocated ~0.07% residual).

### Phase 3-4 (momentum + lid-driven cavity) — DONE, validated vs Ghia (1982)
| case | MAC RMS vs Ghia | collocated (main) | max\|div\| |
|---|---|---|---|
| Re=100, N=128  | 6.6e-3 | 1.7e-3* | ~3e-14 |
| Re=1000, N=128 | **1.57e-2** | 2.78e-2 | ~3e-14 |
- *collocated used N=129 nodes aligned with Ghia's grid; MAC N=128 cell-centres
  need interpolation, which dominates the Re=100 RMS difference.
- MAC Re=1000 is **better** than collocated (non-dissipative central advection +
  exact projection) and stable with NO upwinding.
- convergence: Re=100 RMS 1.27e-2 (N=64) -> 6.6e-3 (N=128).
- divergence is machine-zero at every step (the structural win).

### Next: Phase 5 (reference map, semi-Lagrangian) then Phase 6 (FSI).

### Phase 5-6 (reference map + soft-disc FSI) — DONE, validated vs Sugiyama
- Reference map (X1,X2) at cell centres, advected semi-Lagrangian on a 0-based index
  grid with the cell-centre velocity (faces->centres). Cell-centred RMT machinery
  (phi rebuild, extrapolation, neo-Hookean stress) reused from collocated.
- Soft disc in lid-driven cavity (N=128, t=8): the centroid trajectory tracks the
  Sugiyama 1024^2 and Kolahduz reference data closely along the spiral; the disc
  deforms (J in [0.87, 2.26]); divergence ~1e-14 throughout; stable.

### Bug fixed along the way
The long-standing nondeterministic semi-Lagrangian segfault (which also limited
collocated FSI / contact / coupled surface tension) was numba `parallel=True` on
`bilinear_interpolate`. Made it serial -> segfault gone, all tests pass. This fix
benefits the collocated solver on `main` too (worth cherry-picking).

### Remaining for full FSI parity: disc-in-Taylor-Green energy; variable-density
projection (rho_s != rho_f) for the general case; then re-test surface tension &
contact in the consistent MAC setting (expected to behave far better).
