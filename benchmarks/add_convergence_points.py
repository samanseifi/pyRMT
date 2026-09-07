"""Add intermediate resolutions to the bulk-convergence sweep, reusing the cached N=512 reference
(outputs/overlay/bulk_conv.npz) so 512 need not be recomputed. Runs the new coarse grids, resamples
their fields onto the 512 grid, and merges everything back into bulk_conv.npz. Then re-plot with
`python benchmarks/bulk_convergence.py --plot-only`.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.viscoelastic_disc_tg import run_one
from benchmarks.mac_disc_in_tg_convergence import _sample
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOFT = dict(R=0.15, x0=0.5, y0=0.7, mu_f=0.01, U0=0.3, kw=1)
NPZ = os.path.join(ROOT, "outputs", "overlay", "bulk_conv.npz")
NEW = [48, 96, 192]   # fill-in grids, all >=2.6x coarser than the N=512 reference


def main():
    d = dict(np.load(NPZ))
    g = d["g"]; ref = int(d["ref"])
    NS_old = list(d["NS"])
    ke = dict(zip(NS_old, d["KE"])); se = dict(zip(NS_old, d["SE"]))
    coarse = set(int(x) for x in d["coarse"])
    for N in NEW:
        if f"sp_{N}" in d:
            print(f"  N={N} already present, skipping", flush=True); coarse.add(N); continue
        r = run_one(N, 0.6, model="maxwell", G_ve=0.1, tau=np.inf, integrator="imex",
                    scheme="semilagrangian_cubic", mu_s=0.1, history=True, **SOFT)
        d[f"sp_{N}"] = _sample(r["xc"], r["speed"], g)
        d[f"pr_{N}"] = _sample(r["xc"], r["p"], g)
        ke[N] = r["KE"]; se[N] = float(r["hist"]["Eel"][-1]); coarse.add(N)
        print(f"  N={N} done  KE={ke[N]:.6f} SE={se[N]:.6f}", flush=True)
    coarse = sorted(coarse)
    allN = sorted(coarse + [ref])
    d["coarse"] = np.array(coarse)
    d["NS"] = np.array(allN)
    d["KE"] = np.array([ke[N] for N in allN])
    d["SE"] = np.array([se[N] for N in allN])
    np.savez(NPZ, **d)
    print("merged; resolutions now:", allN, flush=True)


if __name__ == "__main__":
    main()
