# pyRMT backlog (MAC / reference-map FSI)

Working notes for the staggered-grid (MAC) reference-map solver on `mac-staggered`.
Priority: **P1** = needed for the planned Computers & Fluids paper, **P2** = strong
add / reviewer-proofing, **P3** = nice-to-have. Effort is rough.

---

## Method / constitutive (paper headline)

### 1. Viscoelastic relaxation (P1, large) — *headline contribution*
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

### 2. Stress-relaxation / creep verification (P1, small)
Step-strain and step-stress tests vs the analytical Maxwell/SLS response; verifies τ
quantitatively. First, self-contained de-risking step for #1.

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
