# pyRMT method paper

Draft manuscript: **"An implicit, monolithic, incompressible Reference Map Technique
for viscoelastic fluid–structure interaction."**

## Target venue
- **Primary: Journal of Computational Physics (JCP)** — the novelty is the *implicit,
  monolithic, incompressible* RMT (an open gap vs. explicit RMT and vs. the
  compressible GPR model). Gated on the monolithic solver (§7 / stage 2) clearing the
  10–100× dt + net-speedup bar.
- **Fallback: Computers & Fluids / IJNMF** — if framed as the integrated IMEX
  incompressible viscoelastic RMT (stages 1–IMEX only), without the full monolithic
  result. Also a natural home for the constitutive + validation content.
- Both use the Elsevier `elsarticle` class, so the source needs no reformatting to
  switch.

## Build
```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Produces `main.pdf`. Requires `elsarticle`, `amsthm`, `mathtools`, `cleveref`.

## Structure
`main.tex` pulls in `sections/*.tex`:

| file | status |
|---|---|
| `intro` | drafted — positioning + contributions |
| `governing` | drafted — one-fluid incompressible equations |
| `rmt` | drafted — reference map, F, neo-Hookean |
| `constitutive` | drafted — UCM/SLS + log-conformation + exact relaxation (full math) |
| `discretization` | drafted — MAC, exact projection, transform Poisson |
| `timeintegration` | drafted — IMEX stack + von-Neumann proof; **monolithic = design + TODO** |
| `verification` | drafted with measured numbers; some TODO |
| `validation` | drafted with measured numbers; figure TODOs |
| `results` | tables partly filled (measured); **monolithic/high-Wi/fold-cure = TODO** |
| `conclusion` | drafted |

## What is real vs. placeholder
- **Real (measured, in-repo):** constitutive analytic errors, MAC projection/order,
  Taylor–Green IMEX to 50×, four-roll IMEX regression 0.55%, soft-disc regression 2e-3,
  elastic stabilizer to 8× the elastic CFL, N=64 soft-disc integrator comparison.
- **Placeholder (`\TODO`, `\PLACE`, `\phfig`):** the monolithic (stage-2 JFNK) solver
  and its speedup/iteration-count results, N=128/256 crossover numbers, high-Wi regime
  diagram, fold-cure figure, final validation figures, author/affiliation metadata.

Search the source for `TODO`, `PLACE`, and `phfig` to find every open item.

## Reproducing the numbers
The measured results come from the benchmark drivers with the `integrator=` selector
(see the user manual §"Choosing a time-integration strategy"), e.g.
`benchmarks/mac_soft_disc_lid.py` (`integrator="explicit"|"imex"`),
`benchmarks/mac_viscoelastic_extension.py` (`compare_integrators()`), and the test
suite `tests/test_{mac,ve_imex_fsi,imex_wall,elastic_imex,integrator_selector}.py`.
