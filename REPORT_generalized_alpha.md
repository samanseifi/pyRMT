# Report: generalized-α elastic-wave coupling for the IMEX RMT

*Branch:* `feature/generalized-alpha-elastic` (worktree at `../pyrmt-genalpha`), commit
`7628dd8`. Issue spec: `ISSUE_generalized_alpha.md` on that branch.

## 1. Why this came up

Investigating the colliding-disc contact, we found the IMEX **elastic-wave stabilizer** was
over-damping the collision: switching it off raised the coefficient of restitution by up to
~75% (e.g. `e(St≈14)`: 0.13 → 0.23) and it matched the explicit result. The root cause is
that the stabilizer, in its backward-Euler form, damps **every** elastic mode, not just the
unresolved ones. That raised the real question: **can we step past the elastic-wave CFL
without over-killing the elastic waves?** Generalized-α is the principled answer.

## 2. The gap in the current scheme

Stepping past the elastic CFL (`dt < dx/c_s`) needs an implicit treatment of the wave. The
two forms already in the code sit at opposite extremes:

| scheme | `|λ|` (per-step amplitude) | verdict |
|---|---|---|
| backward-Euler stabilizer | `1/√(1+(c_s k dt)²)` — **all** modes damped | over-damps resolved deformation |
| trapezoidal / midpoint | `1` for **all** modes | no control of under-resolved modes |

You can *damp everything* or *damp nothing*. Neither lets you keep the resolved deformation
while removing only the unresolvable high-frequency elastic noise.

## 3. What generalized-α achieves

The generalized-α method (Chung & Hulbert 1993), the standard implicit integrator in
structural dynamics, closes that gap with **one parameter** `ρ∞ ∈ [0,1]` — the amplitude
factor of the infinite-frequency mode. It is unconditionally stable, second-order accurate,
and costs one FFT/DST solve per step (the same constant-coefficient Helmholtz as the
stabilizer). Measured per-mode amplitude factor `|λ|` vs `c_s k dt` (from the benchmark):

```
 c_s k dt        0.1     0.3      1       3      10     100
 backward-E     0.995   0.958   0.707   0.316   0.099   0.010   <- damps even resolved modes
 trapezoidal    1.000   1.000   1.000   1.000   1.000   1.000   <- damps nothing
 gen-α ρ∞=0.9   1.000   1.000   1.000   0.998   0.984   0.928   <- resolved kept, noise -> 0.9
 gen-α ρ∞=0.5   1.000   1.000   0.989   0.876   0.683   0.537
 gen-α ρ∞=0.0   1.000   0.997   0.907   0.562   0.243   0.048
```

Read the `ρ∞=0.9` row against backward-Euler: at the *resolved* mode `c_s k dt = 1`,
generalized-α leaves `|λ| = 1.000` while backward-Euler has already crushed it to `0.707`;
at the *unresolvable* mode `c_s k dt = 100`, generalized-α still damps it (`0.928`, and
`0.537` at `ρ∞=0.5`). That is exactly "past the CFL without over-killing the waves":

- `ρ∞ = 1` → the non-dissipative midpoint rule (energy-conserving);
- `ρ∞ ≈ 0.9` → resolved deformation essentially untouched, grid-scale noise gently removed;
- smaller `ρ∞` → progressively stronger, still frequency-selective, damping.

On the linear standing wave at **5× the explicit CFL** (where explicit diverges),
generalized-α stays finite with energy drift `1.8×10⁻³` (`ρ∞=0.9`) — versus the
backward-Euler stabilizer, which has damped the wave essentially to zero (drift → 1).

## 4. What was implemented (with tests and a benchmark)

- **`pyRMT/genalpha.py`** — `genalpha_params(ρ∞)`, `genalpha_wave_step(...)` (one periodic
  step of `d_tt = c_s² ∇² d`), plus `genalpha_amplification_matrix` / `_spectral_radius`
  for the per-mode analysis.
- **`benchmarks/implicit_elastic_wave.py`** — adds the `genalpha` stepper and the
  amplification-spectrum table above (extends the existing explicit/stabilizer/trapezoidal
  comparison).
- **`tests/test_genalpha.py`** — 20 unit tests, all passing: high-frequency limit equals
  `ρ∞`; resolved modes undamped (`|λ|→1`); selectivity beats backward-Euler; unconditional
  stability; second-order accuracy; `ρ∞=1` reproduces the trapezoidal coupling; stable and
  near-energy-conserving at 5× CFL. The existing elastic-wave tests still pass.

## 5. What it does *not* yet do

The deliverable is the validated **core integrator** on the model wave problem — the novel,
testable piece. It is **not yet wired into the FSI momentum predictors**: doing so means
carrying the elastic acceleration field through the reference-map coupling and evaluating the
elastic force at the `α_f`-intermediate configuration, then benchmarking on the stiff-solid
lid (μ_s=16). That is the documented next step (see the issue). Expected payoff there: the
same large stable step as today's stabilizer, but with the resolved deformation undamped
(min-J and centroid path closer to a CFL-resolved reference), tunable via `ρ∞`.

## 6. How it fits the paper

Two honest options:

1. **Outlook paragraph (recommended now).** The paper already establishes that the IMEX
   speedup comes from the viscous/advective/relaxation CFLs and that the elastic-wave
   stabilizer is a separate, dissipation-vs-step tradeoff. A short outlook can point to
   generalized-α as the principled way to make that tradeoff tunable and
   frequency-selective. Draft below.

2. **A new results subsection (after FSI wiring).** Once generalized-α is in the FSI
   predictor and benchmarked on the stiff-solid lid, it can become a short methods/results
   subsection showing tunable dissipation on a real FSI run.

### Drop-in LaTeX outlook paragraph

```latex
\paragraph{Stepping past the elastic-wave CFL without over-damping.}
The elastic-wave stabilizer used here trades wave accuracy for stability: the
backward-Euler form damps every elastic mode, while the trapezoidal form damps none and so
cannot control the under-resolved high-frequency modes. A generalized-$\alpha$ coupling
\cite{ChungHulbert1993} removes this dichotomy: a single parameter $\rho_\infty\in[0,1]$
sets the amplitude factor of the infinite-frequency mode, damping the unresolvable
grid-scale elastic noise toward $|\lambda|=\rho_\infty$ while leaving the resolved
deformation essentially undamped ($|\lambda|\to1$ as $c_s k\,\dt\to0$), at second order and
one constant-coefficient solve per step. On the linear elastic standing wave it is stable
and near-energy-conserving well past the explicit elastic CFL, with tunable,
frequency-selective dissipation; wiring it into the finite-strain FSI coupling is left to
future work.
```

Add to `refs.bib`:

```bibtex
@article{ChungHulbert1993,
  author  = {Chung, J. and Hulbert, G. M.},
  title   = {A time integration algorithm for structural dynamics with improved numerical
             dissipation: the generalized-$\alpha$ method},
  journal = {Journal of Applied Mechanics},
  volume  = {60}, number = {2}, pages = {371--375}, year = {1993},
  doi     = {10.1115/1.2900803}}
```
