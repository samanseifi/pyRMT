# Re-runs of November 2026

Scripts that regenerated results found unreliable in the review of the paper draft. All write to
`outputs/rerun_2026-11/` and are resumable (finished runs are cached and skipped). Run them in
this order, one at a time:

1. `shear_and_fourroll.py` — shear benchmark (bicubic transport, IMEX, t = 8, D(t) history kept,
   Wi→0 as a genuine SLS run) and the three four-roll cases (IMEX, default step).
2. `shear_long_highWi.py` — Wi = 2, 3, 4, 8 shear runs to t = 20 (slow relaxation).
3. `fourroll_elastic_quarter_dt.py` — elastic four-roll case at a quarter of the default step,
   showing its time-step sensitivity.

Figures: `benchmarks/disc_in_shear_figure.py` and `benchmarks/ve_extension_figure.py`.
