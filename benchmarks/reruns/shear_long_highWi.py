"""Follow-up: high-Wi SLS shear runs to t=20 to see where D settles. Writes outputs/rerun_2026-11/shear_long.npz."""
import os, sys, time, io, contextlib, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from benchmarks.disc_in_shear import run as shear
NPZ = os.path.join(ROOT, "outputs", os.environ.get("RERUN_DIR", "rerun_2026-11"), "shear_long.npz")
store = dict(np.load(NPZ)) if os.path.isfile(NPZ) else {}
gdot = 1.0
for Wi in (2.0, 3.0, 4.0, 8.0):
    key = f"sls_Wi{Wi}"
    if key in store: continue
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        r = shear(N=96, t_end=20.0, scheme="semilagrangian_cubic", history=True, Ca=0.2, model="sls", tau=Wi / gdot)
    h = r["hist"]; store[key] = np.c_[h["t"], h["D"], h["th"]]; np.savez(NPZ, **store)
    print(time.strftime("%H:%M"), f"long {key}: D(20)={r['D']:.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)
print(time.strftime("%H:%M"), "LONG DONE", flush=True)
