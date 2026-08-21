# pyRMT backlog (MAC / reference-map FSI)

Working notes for the staggered-grid (MAC) reference-map solver on `mac-staggered`.
Priority: **P1** = needed for the planned Computers & Fluids paper, **P2** = strong
add / reviewer-proofing, **P3** = nice-to-have. Effort is rough.

---

## Method / constitutive (paper headline)

### 1. Viscoelastic relaxation (P1, large) — *headline contribution* ✅ CLOSED (viscoelastic-extension, merged)
Carry an elastic Finger tensor `b_e` evolved by upper-convected Maxwell with
relaxation time τ:
`∂b_e/∂t + (u·∇)b_e − L·b_e − b_e·Lᵀ = −(1/τ)(b_e − I)`, `σ = G∞·dev(FFᵀ) + G_v·dev(b_e)`.
- **Why:** the principled fix for *unbounded shear* — a purely hyperelastic solid
  sheared without bound stores infinite energy (ill-posed); reinit only treats the
  symptom. Decouples the stress-carrying strain from the folding-prone reference map.
- τ→∞ must recover the current neo-Hookean RMT (limiting check).
- Standard-linear-solid form (keep `G∞` elastic branch) = stays a solid; drop it = Maxwell fluid.
- Needs **log-conformation** (Fattal–Kupferman) advection to keep `b_e` SPD.
- See [[mac-reinit-scope]]: this is the fix reinit can't be.
- **Done:** `pyRMT/viscoelastic.py` (UCM + SLS, log-conformation `ψ=log b_e`, closed-form 2×2
  eigen-update, SPD by construction); extensional money figures (`mac_viscoelastic_extension.py`,
  `mac_viscoelastic_uniform.py`); τ→∞ elastic limit verified.
- **Remaining follow-up (new item #15):** promote the hand-rolled benchmark loop into a
  first-class solver routine in `mac.py` with a full-solver integration test.

### 2. Stress-relaxation / creep verification (P1, small) ✅ CLOSED
Step-strain and step-stress tests vs the analytical Maxwell/SLS response; verifies τ
quantitatively. First, self-contained de-risking step for #1.
- **Done:** `tests/test_viscoelastic.py` — step-strain relaxation `σ_xy=Gγe^{-t/τ}`, steady-shear
  viscometric (`N1=2Gτ²γ̇²`), τ→∞ elastic limit, SPD-under-extreme-shear (log-conformation).

### 3. Elastoplastic (von Mises yield) variant (P3, medium)
Multiplicative split with a yield criterion (Kamrin connection). Bonus capability;
bounds strain by flowing above yield.

## Contact

### 4. Lubrication-corrected near-contact (P2, medium) — *adapt Fai & Rycroft 2018*
Sub-grid thin-film resistance in the near-contact band: normal squeeze `~ μ Δuₙ/h³`,
tangential `~ μ Δu_t/h`, applied **as a stress** (curl-free body force would be
projection-nullified — same lesson as the contact stress).
- **Why:** fixes the *contact pinch-fold* (bar/discs dying when gap < contact band)
  at coarse resolution instead of by brute-force refinement (N=128 folded, N=256
  didn't — lubrication would fix N=128).
- **Synergy:** their main cost is the height function `h`; RMT gives it nearly free
  from the signed-distance level sets (`h ≈ φ_i + φ_j`, normals `∇φ`).
- Frame as *adapting* the lubrication idea to Eulerian RMT, NOT porting the IBM.
- Ref: `refs/1-s2.0-S0021999117308677-am.pdf` (Fai & Rycroft, JCP 2018).

## Robustness / numerics (Jain et al. 2019 improvements)

### 5. Least-squares extrapolation of the reference map (P2, medium)
Replace the current constant/linear ghost extrapolation with higher-order
least-squares (Jain 2019). **Biggest single robustness lever** for corners/thin
features — raises the fold threshold at *every* resolution. Localized change to
`extrapolate_reference_map`. Ref: `refs/1-s2.0-S0021999119306278-am.pdf`.

### 6. Momentum-consistent / non-dissipative reference-map advection (P3, medium)
Advect ξ with the same discrete flux as momentum (Jain Eq. 26 conservative form is
in `mac.py` as `advect_xi_conservative`; make it the default & validate). Reduces the
numerical diffusion that seeds wrinkles → folds.

## Time integration

### 14. Implicit / monolithic time integration (P2→P1, large) — *potential JCP flagship*
The solver is fully **explicit** (forward-Euler predictor); `dt` is CFL-limited by the
elastic-wave speed `c_s=√(μ_s/ρ)`, viscosity `Δx²/ν`, and advection. Lift this.
The one-fluid RMT is **already monolithic in space** (one velocity field, no
partitioned added-mass instability) — only the *time* discretization is explicit.

**Why it could be a flagship:** the dominant RMT papers (Kamrin–Rycroft–Nave,
Rycroft 2020, Jain 2019) are all explicit; an *implicit monolithic* RMT appears to
be an open gap (needs a lit check). It would unify viscoelasticity ([[1]]), contact
([[4]]) and surface tension under one framework and enable **stiff solids** and
**high-Weissenberg** (small τ) regimes where explicit dies. Likely JCP/JFM if novel
AND robustness + net speedup are shown.

**Staging (de-risk):**
1. **IMEX (low risk):** implicit viscous diffusion `(I−νΔt∇²)` — diagonalized by the
   same DCT/FFT used for pressure — + implicit/exponential relaxation `−(1/τ)(b_e−I)`
   (local). Lifts the viscous + small-τ CFL with the existing transform solvers;
   keep semi-Lagrangian advection (already unconditionally stable) explicit.
2. **Fully-implicit elastic kernel (hard):** Newton/JFNK over `(u,v,p,ξ,b_e)`;
   implicit `∇·σ(ξ)` (analytic material tangent) coupled to flux-form ξ-advection;
   saddle-point preconditioner (reuse the projection as the pressure preconditioner).
   Lifts the elastic-wave CFL.

**Obstacles:** the saddle-point solve (preconditioner needed); and the **non-smooth
operators** (semi-Lagrangian interp, `extrapolate_reference_map`, level-set rebuild,
contact `min`) break a clean Newton — switch ξ-advection to the differentiable
conservative flux form ([[6]]) and regularize the interface ops, or use JFNK.
**Must demonstrate:** `dt` 10–100× the explicit limit AND net speedup (not just
feasibility), else the contribution is thin. Novelty gated by a lit check
("implicit reference map technique", "monolithic Eulerian FSI hyperelastic implicit").
Distinct from the GPR / Peshkov–Romenski–Dumbser unified hyperbolic model (that is
**compressible**; the incompressible implicit RMT is the open gap) and from the
adaptive-reference-map relaxation of Kamrin–Rycroft (that relaxes the *geometry* map;
here `ψ=log b_e` carries stress, decoupled from a geometry-only ξ).

**Tracked sub-issues (branches `feat/imex-*`, `feat/implicit-elastic-kernel`):**

- **#14.1 — Implicit viscosity (Helmholtz-DCT/FFT). ✅ DONE** (`feat/imex-implicit-viscosity`)
  Backward-Euler viscous term: `(I − Δt·ν∇²)u* = u − Δt(u·∇)u`, advection explicit.
  The Helmholtz operator diagonalizes under the SAME FFT/DCT as the pressure Poisson
  solve → one extra transform-solve per component per step; removes `dt<dx²/4ν`.
  - **Done (periodic solver):** `mac.py` `lap_eigs_periodic`, `solve_helmholtz_periodic`,
    `momentum_predictor_periodic_imex`; `run_one(..., implicit_visc=True)`.
  - **Verified:** Helmholtz round-trip to machine precision; Taylor–Green stable & accurate
    (div ~1e-14) up to **dt = 50× the explicit viscous CFL**, where explicit → NaN;
    2nd-order spatial accuracy preserved at small dt.
  - **Remaining for #14.3:** the wall/free-slip (DCT-Neumann) Helmholtz variant needed by
    the coupled FSI drivers (with the constant-μ₀ frozen-coefficient split for the
    one-fluid variable viscosity).
- **#14.2 — Exact/exponential relaxation. ✅ DONE** (`feat/imex-exact-relaxation`)
  Strang-split the log-conformation update; integrate the relaxation half-steps
  analytically `b_e ← I + (b_e−I)e^{−Δt/2τ}` (A-stable ∀τ) → removes the small-τ /
  high-Weissenberg stiffness.
  - **Done:** `viscoelastic.py` `relax_exact`, `logconf_local_step_strang`
    (relax dt/2 → explicit stretch dt → relax dt/2). For τ=∞ it is byte-for-byte the
    explicit stretch step (exact neo-Hookean-limit recovery).
  - **Verified:** step-strain is now exact to machine precision at ANY dt; at dt/τ=2 the
    explicit step errs by 0.29 (blows up at dt/τ=5) while Strang stays exact; steady-shear
    viscometric functions reproduced; SPD preserved. Taylor-Green field regression
    (`benchmarks/viscoelastic_taylor_green.py`): `||ψ_explicit−ψ_strang||=1.2e-5` at small
    dt (old physics reproduced), explicit ψ diverges for dt≳τ. Tests in
    `tests/test_viscoelastic.py`; full suite green (no regression to TG/MAC/FSI).
  - Note: log-conformation keeps `b_e=exp(ψ)` SPD for both steppers, so the explicit
    failure mode is ψ accuracy/divergence, not loss of positive-definiteness.
- **#14.3 — IMEX combined (= #14.1 + #14.2). ✅ DONE** (`feat/imex-combined`)
  Wire both into the full FSI loop; measure lifted `Δt` and net speedup.
  - **Done:** `benchmarks/mac_viscoelastic_extension.py run(..., imex=True)` swaps in the
    IMEX viscous predictor + Strang relaxation; the stiff fluid-locking penalization
    (rate β) is also made exact (local exponential relaxation), so *every* stiff term
    (viscous, relaxation, penalization) is implicit/exact and only advection + the
    elastic wave stay explicit. `compare_integrators()` prints the reproducibility table.
  - **Verified** (four-roll extensional blob, a Taylor-Green-family flow): at matched
    dt the IMEX driver reproduces explicit `b_xx(centre)` to **0.55%**; at 2–4× the
    explicit viscous CFL the explicit driver **diverges** while IMEX completes and stays
    within **~2%** of the reference. Tests: `tests/test_ve_imex_fsi.py`. Full suite green.
  - **Residual ceiling:** the elastic-wave dt (`0.3 dx/cs`, ≈5.5× the viscous CFL here) —
    the still-explicit `∇·σ_el` blows up beyond it. That is exactly what **#14.4** targets.
  - **Follow-on ✅ DONE:** wall-bounded implicit viscosity + soft-disc-in-lid regression
    (`feat/imex-wall-helmholtz`). The lid velocity BCs are Dirichlet (no-slip + moving lid),
    which the DCT (Neumann) pressure transform does not diagonalize, so the viscous
    Helmholtz is solved **matrix-free by CG** reusing the exact ghost-cell viscous stencil
    (`momentum_predictor_lid_imex`, `_lap_u_lid_hom`/`_lap_v_lid_hom`, `_cg_helmholtz` in
    `mac.py`); the lid enters as an RHS inhomogeneity. This is SPD and generalizes to
    variable (one-fluid) viscosity — and is the Helmholtz/Stokes solve #14.4 will reuse.
  - **Verified:** lid cavity IMEX reproduces the Ghia steady state (RMS 1.68e-2 vs explicit
    1.71e-2 at N=48) and stays correct at 5× the explicit viscous CFL; soft-disc-in-lid
    IMEX reproduces the explicit (Sugiyama-validated) centroid trajectory to **2e-3**
    (identical minJ). Tests: `tests/test_imex_wall.py`; `mac_lid_driven.py`/`mac_soft_disc_lid.py`
    gain `imex=True`. Full suite green.
- **#14.4 — Fully-implicit elastic kernel.** (`feat/implicit-elastic-kernel`)
  Lifts the elastic-wave CFL `dt < dx/cs` that remains after the IMEX viscous+relaxation
  lift. Staged:
  - **Stage 1 ✅ DONE — linearly-implicit elastic stabilizer.** The nonlinear elastic
    force stays explicit (physics), but an O(dt²) wave operator `dt²·cs²·χ·∇²` (χ=solid
    indicator) is added implicitly, with the force moved INSIDE the implicit RHS. A 1D
    von-Neumann analysis (`u_t=cs²d_xx, d_t=u`) gives amplification `|λ|=1/√(1+dt²cs²k²)≤1`
    → unconditionally stable (adding the force *after* the solve is unstable — verified
    by analysis). FFT frozen-coefficient split keeps it one transform solve.
    `mac.py momentum_predictor_periodic_imex_elastic`, `_lap_per`; driver
    `run(elastic_imex=True)`. **Verified:** consistent (0.44% vs explicit at 0.1× the
    elastic CFL) and lifts the CFL — at **8× the elastic CFL plain IMEX diverges while
    elastic-IMEX completes**. Tests: `tests/test_elastic_imex.py`. Full suite 60 passed.
    *Caveat (honest):* the isotropic-Laplacian stabilizer adds numerical damping, so at
    large dt it is stable but over-damped (4%→17% drift over 1×→8× the elastic CFL) —
    accuracy-at-large-dt needs the consistent tangent (stage 2).
  - **Stage 2 — energy-conserving implicit elastic kernel (the JCP-class part). ⏳ IN PROGRESS.**
    - **Stage 2a ✅ DONE — core numerics de-risked.** The stage-1 damping comes from an
      explicit force + O(dt²) stabilizer; the fix is a **trapezoidal (implicit-midpoint)
      coupling** of velocity and displacement, one Helmholtz solve per step, `|λ|=1`
      (energy-conserving) instead of stage-1's `1/√(1+a)`. Verified on the linear elastic
      standing shear wave (`benchmarks/implicit_elastic_wave.py`,
      `tests/test_implicit_elastic.py`): at 2× the explicit CFL stage-2 energy drift
      `5.5e-5`–`1.8e-3` while stage-1 damps the wave away (drift →1.0); stable at 5× CFL
      where explicit diverges; accurate (`err<5e-2`) to ~1–2× CFL, phase error only beyond.
    - **Stage 2b — wire trapezoidal elastic coupling into the nonlinear FSI loop** (advance
      ξ/ψ with the midpoint velocity, elastic force with the midpoint stress; Helmholtz/CG
      solve). Neo-Hookean soft disc first, then viscoelastic.
    - **Stage 2c — full JFNK** over `(u,p,ξ,ψ)` with analytic material tangent, conservative
      differentiable ξ/ψ advection ([[6]]), projection as pressure preconditioner,
      frozen-interface linearization. Clears the JCP bar *if* 10–100× dt AND net speedup
      with accuracy retained.

### 15. Promote viscoelastic loop to a first-class solver routine (P1, small) — *follow-up to #1*
The log-conformation FSI loop is currently hand-rolled inside
`benchmarks/mac_viscoelastic_extension.py`. Lift the advect→extrapolate→relax→stress
sequence into a reusable routine in `mac.py` (or a `driver.py`) with a full-solver
integration test, so the headline is an API a reviewer can rerun, not a demo script.

## Capability gaps (reviewers will ask)

### 7. Variable-density projection ρ_s ≠ ρ_f (P2, medium)
The settling demo uses a single-density "reduced gravity" body force on the solid
fraction. Real buoyancy/inertia contrast needs a variable-density (variable-coeff)
pressure Poisson solve. Enables proper sedimentation/rising.

### 8. 3D extension (P3, large)
RMT papers are 3D; solver is currently 2D only. Out of scope for the first paper but
note it as a limitation.

## Validation / benchmarks

### 9. Oscillating-drop Lamb frequency (P1, small)
Dynamic surface-tension benchmark: perturbed drop oscillates at the Lamb frequency
with viscous decay. Pairs with viscoelastic damping (#1).

### 10. Quantitative contact / sedimentation benchmark (P2, medium)
Compare a settling/collision case against published data (drafting-kissing-tumbling,
or a single-particle settling velocity) for a quantitative contact validation.

## Paper / software

### 11. Write the Computers & Fluids / IJNMF paper (P1, large)
Headline = viscoelasticity (#1); pillars = MAC exact projection, balanced-force
surface tension (~5.5× fewer parasitic currents), stress-based + lubrication contact
(#4), reinit-vs-relaxation discussion, convergence-cap result. Full benchmark suite
(Ghia, Taylor-Green order, Sugiyama, Laplace, #2, #9).

### 12. Clean pyRMT release (P3, medium)
Docs, tests, install, reproducible benchmark scripts → enables a JOSS/CPC software
paper and reproducibility. Many one-off `benchmarks/*_N*.py` variants to consolidate.

### 13. README figure-gallery index (P3, small)
Index `docs/img/mac/` (lid Ghia, TG/disc convergence, surface tension, contact,
multi-shape, bar N256, settling) with one-line captions.

---
*Compiled from the MAC-staggered design sessions. Items #1, #4, #5 trace directly to
the failure modes characterized in [[mac-reinit-scope]] (folding / crushing / unbounded shear).*
