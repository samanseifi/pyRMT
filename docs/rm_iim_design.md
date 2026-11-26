# Reference-map immersed-interface method (RM-IIM): design note

## Why

With the solid stress blended over a band of width ε = 2Δx, every interface quantity is first
order: the smeared traction jump is an O(ε) model error. Evidence (vortex disc, N = 32–256):

- At a fixed physical band width with a continuum-defined band field (material band), the
  discretization converges at about second order (strain energy 2.0–3.6, bulk pressure 1.6–2.0).
- With ε ∝ Δx the strain energy, full-domain velocity and pressure fall to about first order
  (pressure about half order in L2).
- Extrapolating in the band width did not remove the error. Neither did the material band.

The error is the smearing of the jump itself, so the jump has to be applied sharply.

## Idea

The reference map gives what a sharp method needs, at sub-cell precision and without
reinitialization:

- interface location: φ = φ₀(ξ) with φ₀ the initial signed distance; the crossing point between
  two cell centres is found by interpolating φ (θ = φᵢ / (φᵢ − φᵢ₊₁));
- normal: n = ∇φ / |∇φ|;
- the solid stress at the interface: σ_s from F = (∇ξ)⁻¹ (or b_e = exp ψ), extrapolated from
  the solid side only.

Momentum with a sharp interface: σ_total = σ_f + χ σ_s, χ the sharp indicator. Its divergence
splits into a bulk part (χ ∇·σ_s, inside the solid only) and a surface traction t = σ_s·n on Γ.
Instead of smearing t, impose it as jump conditions on the fluid variables (velocity continuous,
viscosity uniform, as in the one-fluid model):

    [p] = n·σ_s·n,     [μ ∂u/∂n] = −(I − n⊗n)·σ_s·n   (tangential traction)

## Discretization (constant viscosity, MAC grid)

1. Pressure: the projection Poisson problem gets the jump [p] (and [∂p/∂n] = 0 to leading order)
   as ghost-fluid / IIM corrections on the faces crossed by Γ. They enter only the right-hand side,
   so the DCT/FFT solve is unchanged. The velocity correction uses the same jump-corrected face
   gradient, so a purely normal, isotropic traction is absorbed exactly.
2. Tangential traction: jump corrections in the viscous Laplacian stencils that cross Γ, again
   right-hand side only, so the DST Helmholtz solve is unchanged.
3. Bulk solid force: compact face divergence of χ σ_s with one-sided stencils at the interface.
4. Order: ghost-fluid (Liu–Fedkiw–Kang) corrections give a sharp jump and first-order fluxes; full
   IIM (LeVeque–Li; Lee & LeVeque for Navier–Stokes) corrections using θ give second order. Start
   with ghost-fluid, then add the θ-dependent terms.

## Test ladder (each has an exact answer)

1. Static isotropic stress: disc with uniform σ_s = s I in quiescent fluid. Exact: u = 0,
   p = s inside, 0 outside (plus a constant). Tests the pressure jump alone.
2. Static deviatoric stress balanced by geometry: a disc stretched by an affine map and held at
   rest by a matching far field. Tests the normal and tangential jump together.
3. Rigid rotation of a stress-free disc in an annular Couette flow (exact profile). Tests the
   velocity-gradient jump with moving material.
4. The vortex disc convergence study: target second order in the strain energy and pressure as
   Δx → 0.

## Risks

- Cut cells with tiny θ: use the standard θ floor and one-sided stencils.
- The interface traction comes from extrapolated solid data; extrapolate from the solid side
  only, over at most 1–2 cells.
- Moving interface: cells changing side between steps need a fresh-cell treatment (as in
  sharp-interface IB methods).

## Status (2026-11-26): pressure

- Static eigenstress ellipse, direct steady solve (`benchmarks/iim_eigenstress_pressure.py`):
  the sharp pressure is second order both far from the interface and up to 1.5 cells from it
  (near-interface max error 3.8e-3 at N = 128, 9.7e-4 at N = 256); the smeared pressure is second
  order only outside its band (5.4e-2 at N = 128 at the same distance). Velocity: sharp max error
  3.7e-4, 9.5e-5 at N = 64, 128 (order 2), smeared 2.5e-3, 1.5e-3 (order 0.8).
- An earlier impression that the sharp bulk pressure was much worse than the smeared one was a
  test artefact: the smeared scheme's non-incremental projection scalar is a Helmholtz-smoothed
  pressure, and the reference pressure had a Gibbs error of 1e-4. Second-derivative pressure
  jumps in the cut-face gradient were tried and gave no gain (not kept).
- Moving disc: the pressure level inside the solid jittered in time with amplitude ~ 1/dt (the
  classic moving sharp-interface pressure noise). Cause: the cut-cell divergence correction took
  the side of the cell centre separately along x and y. Fixed by taking it from the level set
  at the centre (`DIV_CENTRE_SIDE`); jitter at N = 64, quarter step 4.3e-3 -> 2.3e-4, now
  independent of dt (`benchmarks/iim_pressure_jitter.py`).
- Bulk solid force now pointwise (every face inside the solid, using the extended stress),
  which is what the jump corrections assume: the momentum-conservation fix-up force is 3x
  smaller and the kinetic-energy order rises from 1.6 to 2.2 (E = KE + W: 2.7; no energy injection).
- With both changes (dt ~ h^2, 32/64 vs 64/128, t = 0.2): bulk pressure order 2.2 (3.0e-4,
  smeared 3.3e-4), velocity L2 order 2.1 (1.6e-4, smeared 4.7e-4 at order 1.4).
- Open: late in the decay (t = 1, energy down 300x) the pressure inside the solid keeps a
  non-converging error of about 5e-4 (0.3% of the initial pressure scale; smeared 1e-4). It
  changes with the time step alone at fixed N, i.e. it is noise in the interface stress taken
  from the reference map near the interface, not a discretisation error of the jump conditions.
