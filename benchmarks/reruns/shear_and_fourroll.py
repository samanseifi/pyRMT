"""Re-run queue for the results flagged unreliable. Writes ONLY to outputs/rerun_2026-11/.
Resumable: each finished run is cached; re-launching skips it.
  1. shear benchmark: bicubic transport (as the paper states), t_end=8 with the D(t) history kept
     so a plateau can be checked, and Wi->0 as a genuine SLS run (tau tiny), not the elastic run.
  2. four-roll extension: all three cases, curves and frames from the SAME runs, bicubic."""
import os, sys, time, io, contextlib, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "outputs", os.environ.get("RERUN_DIR", "rerun_2026-11")); os.makedirs(OUT, exist_ok=True)
NPZ = os.path.join(OUT, "shear.npz")
def say(*a): print(time.strftime("%H:%M"), *a, flush=True)

from benchmarks.disc_in_shear import run as shear
SCH, NRES, TEND = "semilagrangian_cubic", 96, 8.0
store = dict(np.load(NPZ)) if os.path.isfile(NPZ) else {}
def go(key, **kw):
    if key not in store:
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            r = shear(N=NRES, t_end=TEND, scheme=SCH, history=True, **kw)
        h = r["hist"]; store[key] = np.c_[h["t"], h["D"], h["th"]]; np.savez(NPZ, **store)
        say(f"{key}: D={r['D']:.4f} th={np.degrees(r['th']):.1f} minJ={r['minJ']:.2f} ({time.perf_counter()-t0:.0f}s)")
for c in [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    go(f"el_Ca{c}", Ca=c, model="elastic")
gdot = 2 * 0.5 / 1.0
for Wi in [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
    go(f"sls_Wi{Wi}", Ca=0.2, model="sls", tau=(1e-8 if Wi == 0 else Wi / gdot))
say("shear done")

from benchmarks.mac_viscoelastic_extension import run as ext
for tau in (np.inf, 0.25, 0.5):
    tag = "elastic" if tau == np.inf else f"tau{tau}"
    if os.path.isfile(os.path.join(OUT, "mac_ve_extension", f"hist_{tag}.npz")):
        continue
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        ext(N=128, t_end=12.0, tau=tau, out_root=OUT, imex=True)   # IMEX: the paper's integrator
    say(f"four-roll {tag} done ({time.perf_counter()-t0:.0f}s)")
say("QUEUE DONE")
