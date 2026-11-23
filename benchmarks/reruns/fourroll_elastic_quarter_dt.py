"""Four-roll elastic case: IMEX at 1/4 of its default step, to separate time-step error from the
transport-scheme change (new IMEX dt0 peak 6.5 vs old explicit/bilinear peak 9.8)."""
import os, sys, time, io, contextlib, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from benchmarks.mac_viscoelastic_extension import run as ext
OUT = os.path.join(ROOT, "outputs", "rerun_2026-11", "dtquarter"); os.makedirs(OUT, exist_ok=True)
dx = 1 / 128; dt0 = 0.3 * dx / np.sqrt(0.3)          # default IMEX step (elastic-wave limit, G=0.3)
t0 = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()) as b:
    ext(N=128, t_end=12.0, tau=np.inf, out_root=OUT, imex=True, dt=dt0 / 4, save_frames=False)
print([l for l in b.getvalue().splitlines() if "dt=" in l][:1], flush=True)
h = np.load(os.path.join(OUT, "mac_ve_extension", "hist_elastic.npz")); t, x = h["t"], h["bxx"]
print(time.strftime("%H:%M"), f"dtquarter elastic done: bxx t=2,3,4 = {np.interp(2,t,x):.2f} {np.interp(3,t,x):.2f} {np.interp(4,t,x):.2f}, peak {x.max():.2f} at t={t[x.argmax()]:.1f} ({time.perf_counter()-t0:.0f}s)", flush=True)
