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
