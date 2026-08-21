# pyRMT — A Guided Explanation

*A plain-language companion to the code and the method paper. It explains what the
solver does, why each piece exists, and how the "implicit monolithic" time-integration
story fits together — then points you to references to go deeper.*

---

## 1. What problem are we solving?

**Fluid–structure interaction (FSI):** a soft solid moving in, and pushed around by, a
fluid — a rubbery disc tumbling in a stirred cavity, a red blood cell in plasma, a gel
particle in a polymer melt. The hard part is that the solid *deforms a lot* and the fluid
*flows around it*, and the two are coupled: the fluid pushes the solid, the solid pushes
back, continuously.

Traditional approaches put a **mesh on the solid** that moves and deforms with it, and a
separate mesh for the fluid, then stitch them together at the moving interface. That
stitching is painful: the solid mesh tangles under large deformation, and passing forces
back and forth between two solvers can go unstable (the "added-mass" instability).

## 2. The big idea: one fixed grid for everything

This code is **fully Eulerian**: both fluid and solid live on the *same fixed grid* (like
a fixed camera watching material flow past), sharing **one velocity field** and **one
pressure**. Nothing remeshes; nothing tangles. The challenge becomes: *how do you
remember the solid's shape and elastic stress when it's just flowing across a fixed
grid?* That is what the **Reference Map Technique (RMT)** answers.

## 3. Tracking the solid: the reference map

Give every particle of the solid a permanent name — its original position `X` (its
"reference" location). As the solid moves, each grid point that is currently inside the
solid remembers *which particle is here right now*, i.e. it stores that particle's
name `ξ(x,t) = X`. This field `ξ` is the **reference map**.

Why is that enough? Because if you know the label field `ξ`, you know how stretched the
material is: the **deformation gradient** `F = (∇ξ)⁻¹` measures how a tiny neighborhood
has been distorted, and from `F` you get the elastic stress (a stretched rubber band
pulls back). The map is just carried along by the flow (`∂ξ/∂t + u·∇ξ = 0`), like a dye
pattern advected by the velocity.

**The catch:** if you shear a purely elastic solid forever, the reference map *folds*
(the label field develops creases) and the method breaks. The physical fix is to let the
material *forget* old deformation — i.e. make it **viscoelastic**.

## 4. Making the solid realistic: viscoelasticity and the log-conformation trick

Real soft matter is **viscoelastic**: elastic on short timescales (bounces back), fluid
on long ones (flows and forgets). We model this with a *relaxing elastic strain* — a
tensor `b_e` that stores stress but decays toward "unstressed" over a relaxation time `τ`
(the upper-convected Maxwell / standard-linear-solid model). This is the principled cure
for the folding problem: the stress-carrying strain relaxes instead of creasing forever.

**The numerical trick — log-conformation.** The strain tensor `b_e` must stay
*positive-definite* (you can't have negative stretch), but naive numerical transport can
violate that and blow up, especially at high Weissenberg number (fast, stiff relaxation).
Fattal & Kupferman's fix is to evolve `ψ = log(b_e)` instead; then `b_e = exp(ψ)` is
positive-definite *by construction*, no matter what the numerics do. In 2-D this
`log`/`exp` of a 2×2 symmetric tensor has a closed form, so it's cheap.

## 5. Solving the fluid: staggered grid and exact projection

Incompressible flow has to satisfy `∇·u = 0` (no compression) at every instant, enforced
through the pressure. The classic robust way is the **projection method** (Chorin): take
a provisional velocity, then subtract off a pressure gradient that removes any divergence.

We use a **staggered (MAC) grid** — velocities live on cell faces, pressure at cell
centers. This arrangement makes the discrete operators *consistent* (`divergence` and
`gradient` are exact transposes), so the projection is **exact**: the corrected velocity
is divergence-free to machine precision, with none of the "checkerboard" pressure noise a
collocated grid suffers. The pressure solve itself is a Poisson equation, done fast with a
cosine/Fourier transform (`O(N log N)`).

## 6. The time-step bottleneck: CFL conditions (in plain terms)

An explicit time step is only stable if `Δt` is small enough that information doesn't
cross a grid cell in one step. Each physical process imposes its own limit:

| process | limit on Δt | why it's small |
|---|---|---|
| **advection** | `Δx / U` | stuff shouldn't move more than a cell per step |
| **viscous diffusion** | `Δx² / ν` | *quadratic* in Δx — brutal on fine grids |
| **elastic waves** | `Δx / c_s` | stiff solids have fast waves `c_s=√(μ/ρ)` |
| **relaxation** | `~ τ` | fast-relaxing (high-Weissenberg) materials |

The worst of these caps your step. On a fine grid the viscous `Δx²` limit dominates; for a
stiff solid the elastic-wave limit does; for fast polymers the relaxation limit does. An
**explicit** solver is a hostage to all four. The whole "implicit" program below is about
removing them one at a time.

## 7. The journey: explicit → IMEX → monolithic

Think of it as a **stack**, each level making one more stiff term *implicit* (solved for,
rather than stepped forward blindly) so its CFL limit disappears:

1. **Explicit** — everything stepped forward; all four limits bite. (Baseline.)
2. **Implicit viscosity** — treat diffusion with backward Euler. The resulting
   "Helmholtz" solve is done with the *same transform* as the pressure, so it's nearly
   free. Removes the `Δx²/ν` limit. *(Stable to 50× the viscous limit on Taylor–Green.)*
3. **Exact relaxation** — integrate the stiff relaxation term *analytically*
   (`b_e ← I + (b_e−I)e^{−Δt/τ}`), which is stable for any `τ`. Removes the `τ` limit.
   *(Step-strain becomes exact at any Δt.)*
4. **Implicit elastic waves** — advance the elastic force with a *trapezoidal* (midpoint)
   rule so the elastic wave is unconditionally stable **and energy-conserving** (no
   artificial damping — see the two propositions in the paper). Removes the `Δx/c_s`
   limit. A subtle but essential detail: the elastic force must sit *inside* the implicit
   solve; applying it afterward is unstable.
5. **Implicit/semi-Lagrangian advection** — the last limit. We advect by tracing
   characteristics backward (**semi-Lagrangian**), which is stable at any Δt. Because
   semi-Lagrangian adds a little numerical smearing, we use it *adaptively*: accurate
   central differencing while the advection CFL is satisfied, semi-Lagrangian only beyond
   it — so we pay for stability only when we actually need it.

Once all four are lifted, `Δt` is limited by *accuracy*, not stability.

## 8. Why "preconditioning" matters (the part that makes it actually fast)

Lifting a CFL only helps if the implicit *solve* is cheap. For the wall-bounded velocity
the solve is done by **conjugate gradients (CG)** — an iterative method whose cost is the
number of iterations. Unpreconditioned, that number grows with resolution (we saw an
implicit run become *3× slower* than explicit despite taking half as many steps). A
**preconditioner** — here a fast sine-transform approximation of the operator — collapses
the iteration count to ~5–7 *regardless of grid size*. That is what turns "bigger steps"
into an actual wall-clock **speedup**.

## 9. What the results say (plain terms)

- **It reproduces the classics.** Lid-driven cavity matches Ghia (1982); the soft disc
  matches Sugiyama (2011); Taylor–Green converges at 2nd order in velocity, pressure and
  energy — and *all three integrators give identical accuracy*, so the implicit machinery
  doesn't cost accuracy.
- **The elastic coupling conserves energy** (a standing elastic wave neither grows nor
  decays numerically), unlike the cruder stabilizer which damps it away.
- **Net speedup:** on the soft disc the monolithic scheme reaches **6× the explicit step
  for a ~3× wall-clock speedup**, tracing the same trajectory in a quarter of the steps.
- **Honest caveat:** at very large steps the semi-Lagrangian advection adds a few-percent
  trajectory drift; removing that (2nd-order accuracy *at* large Δt) is the fully
  consistent Newton–Krylov version, noted as future work.

## 10. How to run it

```bash
pip install -e ".[test]"
python benchmarks/mac_lid_driven.py 100 128                 # Ghia validation
python benchmarks/mac_soft_disc_lid.py 128 8.0              # soft disc vs Sugiyama
python benchmarks/mac_soft_disc_lid.py                      # (Python) integrator="imex-sl"
python benchmarks/overlay_integrators.py                    # the comparison figures
python benchmarks/tg_convergence_methods.py                 # convergence figure
pytest                                                      # the test suite
```
Every driver takes one `integrator=` knob: `"explicit"`, `"imex"`, `"imex-elastic"`,
`"imex-sl"` (see the user manual, §"Choosing a time-integration strategy").

---

## 11. Reading list (annotated)

### The Reference Map Technique
- **Kamrin, Rycroft & Nave (2012)**, *Reference map technique for finite-strain
  elasticity and fluid–solid interaction*, JMPS. — The original RMT; start here.
- **Valkov, Rycroft & Kamrin (2015)**, JAM. — Multiple soft bodies in fluid.
- **Rycroft, Wu, Yu & Kamrin (2020)**, *Reference map technique for incompressible FSI*,
  JFM. — The incompressible, projection-based RMT closest to this code.
- **Jain, Kamrin & Mani (2019)**, JCP. — A conservative, low-dissipation Eulerian
  formulation; the basis for much of the discretization here.

### Viscoelasticity and the conformation tensor
- **Fattal & Kupferman (2004)**, JNNFM. — The log-conformation representation (the SPD
  trick). Short and essential.
- **Bird, Armstrong & Hassager**, *Dynamics of Polymeric Liquids* (textbook). — The
  constitutive-model background (UCM, Oldroyd-B, Giesekus, FENE-P).
- **Reese & Govindjee (1998)**, IJSS. — Finite-strain viscoelasticity in solid mechanics
  (the standard-linear-solid split).

### Unified / alternative Eulerian continuum methods (context & contrast)
- **Sugiyama et al. (2011)**, JCP. — Full-Eulerian FSI; the soft-disc benchmark source.
- **Peshkov & Romenski (2016)** and **Dumbser et al. (2016)**, JCP. — The GPR unified
  hyperbolic model (viscous fluids + elastoplastic solids in one system). *Compressible*
  and mostly explicit/semi-implicit — the closest "rival" framing; good to understand how
  the incompressible-implicit approach here differs.

### Incompressible flow, staggered grids, projection
- **Harlow & Welch (1965)**, Phys. Fluids. — The original MAC staggered grid.
- **Chorin (1968)**, Math. Comp. — The projection (fractional-step) method.
- **Ferziger & Perić**, *Computational Methods for Fluid Dynamics* (textbook). — Solid,
  readable grounding in all of the above.

### Implicit time integration and fast solvers
- **Ascher, Ruuth & Spiteri (1997)**, Appl. Numer. Math. — IMEX Runge–Kutta methods (the
  "treat stiff terms implicitly, the rest explicitly" idea, made rigorous).
- **Saad**, *Iterative Methods for Sparse Linear Systems* (textbook, free online). —
  Conjugate gradients, GMRES, preconditioning; the "why the solve is fast" chapter.
- **Knoll & Keyes (2004)**, JCP. — Jacobian-free Newton–Krylov (JFNK); the recipe for the
  fully consistent monolithic solver (the future-work direction).

### Advection: semi-Lagrangian and its accuracy
- **Staniforth & Côté (1991)**, Mon. Weather Rev. — Semi-Lagrangian schemes (the
  unconditionally-stable backtrace).
- **Kim, Liu, Llamas & Rossignac (2005)** / **Dupont & Liu**. — BFECC, the standard fix
  for semi-Lagrangian numerical diffusion (a route to removing the large-Δt drift).

### Interfaces, surface tension, contact
- **Osher & Fedkiw**, *Level Set Methods and Dynamic Implicit Surfaces* (textbook). — The
  level-set machinery used to reconstruct the interface from the reference map.
- **Brackbill, Kothe & Zemach (1992)**, JCP. — The continuum-surface-force (CSF) model.
- **Fai & Rycroft (2018)**, JCP. — Lubricated near-contact (the planned contact upgrade).

### Foundations you can lean on
- **Gonzalez & Stuart**, *A First Course in Continuum Mechanics*. — Deformation gradient,
  stress, objective rates — the language of §3–§4.
- **LeVeque**, *Finite Difference Methods for ODEs and PDEs*. — CFL, stability, von
  Neumann analysis — the language of §6–§7 (including the propositions).

---

## 12. Mini-glossary

- **Eulerian / Lagrangian** — watching fixed points in space / following material
  particles. This code is Eulerian.
- **Reference map `ξ`** — the field that stores each grid point's material label; the
  inverse of the motion.
- **Deformation gradient `F`** — local stretch/rotation of the material; `F=(∇ξ)⁻¹`.
- **Projection** — the step that removes divergence via a pressure Poisson solve.
- **CFL condition** — the stability limit on the explicit time step.
- **IMEX** — implicit–explicit: stiff terms implicit, the rest explicit.
- **Helmholtz solve** — solving `(I − cΔ)x = b`; what backward-Euler diffusion needs.
- **Preconditioner** — an approximate inverse that makes an iterative solve converge in a
  few steps.
- **Semi-Lagrangian** — advection by tracing characteristics backward; unconditionally
  stable, mildly diffusive.
- **Weissenberg number** — dimensionless measure of how elastic/stiff the viscoelastic
  response is (`Wi = τ · strain rate`); high Wi = hard.
```
