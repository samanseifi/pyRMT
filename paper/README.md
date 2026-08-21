# pyRMT method paper

Draft manuscript: **"An implicit, monolithic, incompressible Reference Map Technique
for viscoelastic fluid–structure interaction."**

## Target venue
- **Computers & Fluids** (Elsevier, `elsarticle`). The contribution is the
  log-conformation viscoelastic RMT on a MAC staggered grid with a staged semi-implicit
  (IMEX) time-integration stack, validated (incl. viscoelastic FSI and FSI convergence),
  positioned honestly against Richter/Dunne (monolithic hyperelastic Eulerian FSI),
  Kolahdouz (sharp partitioned), and Sugiyama/Ii. The fully-coupled monolithic solver is
  presented as a foundation + roadmap (fluid JFNK + validated elastic tangent), not a
  finished result — the full `(u,p,ξ,ψ)` Newton is future work.

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
