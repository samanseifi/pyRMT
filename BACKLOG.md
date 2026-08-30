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
    - **Stage 2b ✅ DONE — trapezoidal elastic coupling in the FSI + preconditioned solve.**
      (i) **Preconditioned CG** (`_pcg_helmholtz`, DST spectral preconditioner) fixes the
      net-speedup bottleneck the N=128 crossover exposed: iterations `17→5` (N=128 viscous),
      `27→7` (elastic), staying ~5–7 as N grows (vs CG's O(N)). (ii) **Trapezoidal
      implicit-elastic** in the soft-disc loop (`integrator="imex-elastic"`, `cs2` in
      `momentum_predictor_lid_imex`): elastic force inside the implicit solve + the
      `(dt²/4)cs²∇²` stabilizer, reference map advanced with the **implicit-midpoint
      velocity** (required — explicit-displacement gives det>1, unstable). Verified:
      reproduces the explicit centroid to `2e-3` (regression), runs a stiff disc stably.
      Tests: `tests/test_imex_wall.py` (PCG), `tests/test_elastic_imex_fsi.py`.
      - **Honest finding:** the lid-driven case is *advection*-limited (lid flow U=1, momentum
        advection explicit), so lifting the elastic CFL gives no speedup here — the elastic
        lift pays off in stiff/quiescent regimes (shown on the standing wave, 2a). A net
        speedup in flow-driven FSI additionally needs **implicit advection** (semi-Lagrangian
        momentum) — folded into 2c.
    - **Stage 2c ✅ DONE — semi-Lagrangian momentum advection (lifts the LAST CFL).** The
      binding constraint after 2a/2b was advection (explicit central momentum advection).
      Semi-Lagrangian (RK2 backtrace, **cubic** interpolation — bilinear leaves a ~1e-2
      diffusion floor) makes advection unconditionally stable; combined with implicit
      viscosity + PCG the scheme has no explicit stability limit.
      `mac.py momentum_predictor_periodic_semilag`, `momentum_predictor_lid_semilag`,
      `_interp_per` (cubic); drivers gain `integrator="imex-sl"` / `"imex-elastic-sl"`.
      **Verified — Taylor-Green:** accurate at small dt (`<8e-3`), stable & accurate above
      the advection CFL, machine-zero divergence (`tests/test_semilag.py`).
      **Verified — soft disc (the payoff):** reproduces the explicit centroid at matched dt
      (`2.2e-3`); at 6× the explicit dt it completes with **3.05× net wall-clock speedup**
      (27 vs 161 steps) within ~6% of the reference trajectory
      (`tests/test_elastic_imex_fsi.py`). **This is the net-speedup result the whole
      implicit program targeted** — every explicit CFL (viscous, relaxation, elastic wave,
      advection) is now lifted, and the PCG keeps the implicit solve cheap.
    - **Stage 2d — FULLY MONOLITHIC Newton–Krylov. ✅ IMPLEMENTED** (`pyRMT/monolithic.py`,
      `tests/test_monolithic.py`, `benchmarks/monolithic_disc_tg.py`; the numerical
      realization of `paper/monolithic_aml.tex`). The reference map is **eliminated**, not
      carried as an unknown: an *implicit* semi-Lagrangian step `ξ(x)=ξ̂ⁿ(x−Δt·uⁿ⁺¹(x))`
      with a **C¹ (unclamped Catmull–Rom)** interpolant makes ξ — hence `F=(∇ξ)⁻¹`, σ_s and
      the interface `φ=φ₀∘ξ` — a differentiable function of the velocity, so one
      `(u,p)` Newton solve advances the whole state with no CFL of any kind.
      - **Three enablers.** (i) The C¹ interpolant: `interpolators.bicubic_interpolate`
        clamps to the stencil min/max, and min/max is not differentiable — that limiter
        would break Newton exactly at the interface. (ii) **`extrapolate_reference_map`
        made differentiable**: its layer flags + weighted LSQ plane fit are frozen at φⁿ
        into an explicit linear "plan" (`ExtrapPlan`), so the band extrapolation is a fixed
        sparse *linear* map on ξ (verified linear to 1e-15). It is also now deterministic —
        the original reads *and writes* `known_flag` inside a `prange` loop. (iii) The
        interface needs **no lagging**: `smoothed_heaviside` is C¹ and φ₀∘ξ analytic, so
        `H(φ(ξ(u)))` differentiates through.
      - **Preconditioner.** On the free-slip box the velocity block is *exactly* separable
        (Dirichlet across the wall-normal direction, Neumann along it) → **DST-I ⊗ DCT-II**
        for u and the transpose for v, machine-exact. With the DCT pressure Poisson this is
        an exact projection preconditioner + a Cahouet–Chabard Schur term. Krylov is a
        hand-rolled **FGMRES** (Givens rotations); it beat `scipy.gmres` ~2.5× on matvecs.
      - **Verified.** Newton converges **quadratically** (`|R|` 2.3e-1 → 4.1e-5 → 8.0e-13);
        the **closed-form tangent** (draft Eqs. 9–10) matches a central FD to round-off;
        at **40× the elastic-wave CFL** (N=48, μ_s=100) it runs with `min det(∇ξ)=1.001`
        and **0.66%** velocity error vs a resolved explicit reference, where the explicit
        scheme **diverges at 10×** and yields `det=−128` (folded) at 40×. 13 tests.
      - **Honest gaps (both measured, both open):**
        1. **Order in dt is 1, not 2.** The velocity–pressure kernel *is* second-order-capable
           — θ=½ + Eulerian advection with no solid gives order **2.00** at ~1e-9 error — but
           the reference-map band (masked/extrapolated field on a cell-quantized solid region)
           caps the coupled scheme at first order. The outer interface fixed point
           (`n_outer`) buys ~2.5× in the constant, not the order. **The draft's
           second-order claim is not supported as written.**
        2. **NET SPEEDUP ACHIEVED at resolution — the regime matters.** Which explicit
           limit binds decides everything, and the two routes behave *oppositely*:
           - *Elastic/advective route*: both scale as dx, so `dt_mono/dt_exp ~ c_s/U` is
             **resolution-independent**. Worse, chasing stiffness raises `mu_s*dt`, exactly
             what degrades conditioning, so the bigger step is eaten by extra iterations.
             **No crossover in stiffness found up to mu_s=1e5.**
           - *Viscous route*: `dt_visc ~ dx^2`, so refining makes it bind (N>75 here) and the
             gain grows **∝ N** while leaving `mu_s*dt` alone, so the Krylov count stays
             bounded. **This is where the method pays.** Measured (disc-in-TG, mu_s=0.1,
             T=0.1, both timed in one process so the ratio is machine-independent):

             | N | binds | expl steps/wall | mono steps/matvec | speedup | rel err |
             |---|---|---|---|---|---|
             | 32 | adv | 13 / 0.46 s | 13 / 10 | 0.27x | 1.8e-3 |
             | 64 | adv | 25 / 1.70 s | 25 / 9 | 0.29x | 9.3e-4 |
             | 128 | **visc** | 82 / 26.2 s | 49 / 7 | **1.11x** | 4.6e-4 |
             | 192 | **visc** | 185 / 130.7 s | 73 / 6 | **1.77x** | 3.1e-4 |
             | 256 | **visc** | 328 / 387.5 s | 97 / 6 | **2.40x** | 2.3e-4 |

             Step gain tracks the predicted `nu/(U dx)` to two digits (1.67 vs 1.70, 2.53 vs
             2.55, 3.38 vs 3.40) and **matvecs FALL with N (10→6)** — the preconditioner is at
             least mesh-independent. Not saturating at N=256.
           - *Caveat*: this is vs a **fully explicit** solver (as all published RMT solvers
             are). A fractional-step scheme with implicit viscosity would also remove the
             viscous limit; against that, the advantage would have to come from the elastic
             wave, which is still preconditioner-limited.

        2b. **The old "no win" finding, for reference (soft lid cavity, advection-limited).** On the **lid-driven
           cavity** (N=64, T=2, matched deviatoric stress, serial, threads pinned) the
           monolithic solver is **5–200× slower** than the explicit driver, and slower than
           the repo's own `imex-elastic-sl` (which reaches 1.3–3.4×). Two independent causes:
           - *Iteration count grows with dt* — **largely FIXED (L3).** The projection
             preconditioner did not contain the elastic stiffness K (11 matvecs at 1× the
             elastic CFL → 202 at 10× → 2211 at 40×). Linearizing the elastic force for a
             near-undeformed map gives `d(sigma_s) ~ 2 mu_s dt dev(D(du))` — an extra
             viscosity `mu_s*dt` inside the solid — so `pc_mode='sparse'` assembles that
             variable-coefficient block (~5 nnz/row) and factors it with `splu`.
             **Measured (N=48, mu_s=100): matvecs at 40× 2211→522, Newton 8.0→2.7, wall
             49.4→13.7 s (3.6×), identical errors, and cost per unit simulated time now
             FLAT in dt** (25.6, 14.4, 12.9, 13.2, 13.7 s for 1×…40×). Below
             `mu_s*dt ~ mu_f` the sparse block needs *more* iterations than the exact
             transform solve (16 vs 11), so `pc_mode='auto'` (default) switches on that
             criterion. Lid supported (no-slip ghost `-u` contributes `2*mu` at wall rows).
           - *The remaining gap is the constant.* Even with the Krylov growth tamed the
             scheme is **~25× slower than explicit** at its largest stable step, and **no
             crossover was found up to mu_s = 1e5** (at mu_s ≥ 1e4 the solve fails when
             pushed to the advective step). The isotropic proxy captures only part of K —
             missing: the **deviatoric tensor structure** (u–v coupling of `dev D`), the
             wider bicubic+extrapolation stencil, and the interface term. **A more
             faithful assembled K is the next step.**
           - *Constant-factor work is not the answer* (tried and measured): fused numba
             kernels for stress/tangent/force-assembly, precomputed constant-offset
             staggered stencils (`ShiftInterp`), and shared kinematics cut the residual
             **2.8×** and the Jacobian **1.7×** — yet the **step** cost fell only **~1.15×**.
             Rejected on measurement: `pc_simple` (halves transforms, nearly triples the
             Krylov count at large dt) and `extrapolate_guess` (no Newton-count reduction).
           - *Timing hygiene*: this machine varies by >2× between sessions (explicit @1×
             on the lid case measured 2.07 s and 4.39 s for identical work). **Always
             normalise wall-clock against an explicit run measured in the same session.**
           **What it does buy:** the monolithic error is nearly *independent of dt* — at
           μ_s=0.1 the cavity centroid error moves only 8.6e-4 → 1.05e-3 while dt grows 8×,
           where the explicit scheme diverges past 1×. **An elastic-aware saddle-point
           preconditioner + a lower constant in the tangent is the open problem.**
        3. **Unresolved:** at μ_s=100 on the cavity, monolithic and explicit centroids differ
           by 2.5%, *flat in dt* — a **spatial** band difference (monolithic uses central
           stencils throughout the blend region; the explicit driver uses one-sided stencils
           on φ≤0 only), not a time error. Which is closer to the converged answer is not
           settled. Needs a grid-refinement study.

      - **Boundary conditions.** Both the free-slip box (disc-in-Taylor-Green) and the
        lid-driven cavity are supported, and *both are exactly separable*: a velocity
        component is Dirichlet across the walls it is normal to (DST-I) and, along the other
        pair, mirrors for free slip (Neumann → DCT-II) or reflects with a sign flip for
        no-slip (cell-centred Dirichlet → **DST-II**). So the preconditioner's velocity block
        is a direct transform solve in both cases (verified machine-exact). Gotcha pinned by
        a test: the lid enters the residual as a *constant* ghost `2·U_lid − u`, so a
        **perturbation** must be padded with `−δu`; using the inhomogeneous ghost in the
        tangent left it wrong by 7e-3 and cost Newton its quadratic rate.

### 16. Non-deviatoric solid stress creates energy for stiff solids (P1, small) — *found via #14.4/2d*
`solid_cauchy_stress(..., isochoric=False)` returns the Rycroft-convention `σ = μ_s b`,
whose isotropic part `μ_s tr(b)/2` is meant to be absorbed by the pressure. It is not:
`∇·[(1−H)qI]` is discretized with the **wide 2Δ central stencil + face averaging**, which
is *not* the consistent partner of the **compact face gradient** used for the pressure, so
the isotropic part is not in the range of the discrete pressure gradient and survives the
exact projection. **Measured:** a purely isotropic solid stress `qI` leaves **6.4% of its
magnitude as spurious solenoidal velocity** after projection, scaling linearly with `q`.
Consequence on the disc-in-Taylor-Green benchmark at N=48, T=0.05: with `σ=μ_s b` the total
energy goes `0.2221 → 0.5048` (**+127%**, `min J → 0.84`) at μ_s=100, at *any* dt; with the
deviatoric `μ_s dev(b)` it is `0.2173` (correct viscous decay). Invisible at the μ_s≈0.1 of
the standard benchmarks, fatal for stiff solids.
**Fix:** use the deviatoric (or `isochoric=True`, which already subtracts the trace) form by
default. `benchmarks/mac_disc_in_tg_convergence.py` defaults to `isochoric=False` and is
affected. Cross-check whether this also explains part of the 1st-order field convergence
reported in `paper/main.tex`.

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
