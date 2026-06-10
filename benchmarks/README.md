# pyRMT benchmark suite

Three canonical validation cases for the collocated-grid Reference Map Technique
solver. All drivers share `common.py` (boundary conditions, narrow-band checks,
post-processing) and keep the **pressure BC consistent with the velocity BC**.

## Cases

| Driver | Physics | Velocity BC | Pressure BC | Reference data |
|---|---|---|---|---|
| `lid_driven_cavity.py` | pure fluid | no-slip walls + moving lid | Neumann (DCT) | `data/plot_u_y_Ghia{100,1000}.csv` |
| `disc_in_taylor_green.py` | soft disc in TG vortex | free-slip impermeable box | Neumann (DCT) | Jain et al. (2019) energy curves |
| `soft_disc_in_lid_driven.py` | soft disc in cavity (**primary FSI**) | no-slip walls + lid | Neumann (DCT) | `data/Sugiyama_1024x1024.csv`, `data/Kolahduz_2023.csv` |

The Taylor–Green streamfunction `sin(2πx)·sin(2πy)` has **zero normal velocity on
every wall**, so the consistent pairing is free-slip no-penetration walls +
Neumann pressure (not periodic). A periodic FFT Poisson solver
(`bc_type='periodic'`) is also available for genuinely periodic problems; its BC
convention (`a[:,-1]=a[:,0]`) must match the FFT overlap grid.

## Running

```bash
# Fluid solver vs Ghia (1982)
python benchmarks/lid_driven_cavity.py 100      # Re=100, N=129
python benchmarks/lid_driven_cavity.py 1000     # Re=1000, N=129

# Soft disc in Taylor–Green (energy conservation / oscillation)
python benchmarks/disc_in_taylor_green.py 128 semilagrangian

# Primary FSI case vs Sugiyama (2011)
python benchmarks/soft_disc_in_lid_driven.py 128 semilagrangian 8.0
```

Outputs (CSV time series + comparison PNG) are written under `outputs/<case>/`.

## Switchable reference-map advection

All FSI drivers take a `scheme` argument routed through
`pyRMT.functions.advect_reference_map`:

- `semilagrangian` — RK4 semi-Lagrangian backtrace (bilinear). **Default**;
  robust and keeps the reconstructed level set well-behaved.
- `central2` — 2nd-order central + SSP-RK3, restricted to the solid band.
- `weno5` — WENO5 + SSP-RK3, restricted to the solid band.

The semi-Lagrangian scheme is retained as the default because the Eulerian
schemes can distort the level set; switch only when you specifically want the
non-dissipative central/WENO behaviour.

## Convergence & the accuracy-vs-order tradeoff (`stress_band`)

Spatial convergence on the disc-in-TG case (`convergence_taylor_green.py`):

| quantity | default (one-sided stress) | `stress_band=True` (banded central + detG clamp) |
|---|---|---|
| strain energy SE | ~2.0 | ~2.1 |
| kinetic energy KE | ~2nd (converged) | ~1.5 |
| reference map ξ | ~1.6 | ~1.45 |
| velocity \|u\| | ~1.2 | ~1.25 |
| pressure p | **~0.76 (sub-1st)** | **~1.08** |

`stress_band=True` computes the solid stress over the whole `(1-H)>0` blend band
with central ∇ξ and a localized detG clamp. It improves **pressure** convergence
(0.76 → 1.08) and is the more "correct" central discretisation (Jain et al. 2019),
**but** it changes the interface force enough to shrink the soft-disc orbit in the
lid-driven cavity (centroid extent x→0.60 instead of 0.70; Sugiyama is 0.70) at
both clamp=3 and clamp=10. So it trades **FSI accuracy for pressure order** — it is
**OFF by default**.

The default (one-sided, interior-only stress) reproduces the validated soft-disc
trajectory at N=64 and N=128 (matches Sugiyama) and leaves Re=1000 unchanged, at
the cost of sub-1st-order pressure convergence. Getting 2nd-order *fields* without
sacrificing the disc needs the deeper combined work (higher-order reference-map
extrapolation + conservative central momentum), not a single interface tweak.

Enable the higher-order interface stress per-run:
```python
run(N=128, stress_band=True, detg_clamp=3.0)   # higher p-order, smaller disc orbit
```

## Narrow-band / transition-width coupling

`common.check_narrow_band(w_t, dx, num_layers)` enforces
`num_layers >= ceil(w_t/dx) + 1` so the extrapolated solid stress covers the
whole `(1-H) > 0` blend region. Drivers use `w_t = 2·dx` with ≥3 layers.
