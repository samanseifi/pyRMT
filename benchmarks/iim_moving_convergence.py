"""RM-IIM stage 3b: self-convergence of the moving pre-stretched disc, sharp vs smeared.

Grids N = 32, 48, 64, 96, 128 (small on purpose: shared machine), t_end = 1.0. Successive
differences between N and 2N: fields restricted to the coarse grid by 2x2 averaging (velocity
perturbation, pressure: full domain and bulk = farther than 0.06 from the interface), the disc
centroid, and the stored elastic energy. Usage: python benchmarks/iim_moving_convergence.py
"""
import os, sys
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import iim_moving_disc as T
T.T_END = 1.0
NS = (32, 48, 64, 96, 128); PAIRS = ((32, 64), (48, 96), (64, 128))


def restrict(f):
    return 0.25 * (f[0::2, 0::2] + f[1::2, 0::2] + f[0::2, 1::2] + f[1::2, 1::2])


def main():
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    for meth in ("smeared", "sharp"):
        F = {N: T.run(N, meth, fields=True)[1] for N in NS}
        rows = {}
        for a, b in PAIRS:
            fa, fb = F[a], F[b]; phi = fa["phi"]; s = phi <= 0
            dist = np.where(s, ndi.distance_transform_edt(s), ndi.distance_transform_edt(~s)) / a
            mb = dist > 0.06
            dv = np.hypot(fa["uc"] - restrict(fb["uc"]), fa["vc"] - restrict(fb["vc"]))
            dp = fa["p"] - restrict(fb["p"])
            rows[(a, b)] = (rms(dv), rms(dv[mb]), rms(dp), rms(dp[mb]),
                            np.hypot(fa["cx"] - fb["cx"], fa["cy"] - fb["cy"]), abs(fa["W"] - fb["W"]))
        names = ("vel full", "vel bulk", "p full", "p bulk", "centroid", "energy W")
        print(f"{meth}:  successive differences (32/64, 48/96, 64/128) and order 32/64 -> 64/128")
        for k, nm in enumerate(names):
            d = [rows[p][k] for p in PAIRS]
            print(f"  {nm:9s} " + "  ".join(f"{x:.2e}" for x in d) + f"   order {np.log2(d[0] / d[2]):.2f}")
        print(f"  W at N=128: {F[128]['W']:.4e}", flush=True)


if __name__ == "__main__":
    main()
