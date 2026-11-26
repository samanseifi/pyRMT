"""RM-IIM: time jitter of the solid's pressure level in the moving-disc test.

Sharp-interface methods on fixed grids are prone to pressure noise as the interface crosses
cells, growing like 1/dt (Seo & Mittal 2011; "shots of pressure" in Valkov et al. 2015). Here the
measure is the mean pressure inside the disc minus outside (both farther than 0.05 from the
interface), sampled every 0.005 up to t = 0.3; jitter = RMS deviation from a degree-5 polynomial
in time (the smeared method's value, about 3e-4, is the residual of that fit, not noise).

Cause found with this test: the cut-cell divergence correction chose the side of the cell centre
separately along x and along y (theta < 1/2 on each line). When the interface passes close to
a cell centre the two can disagree, leaving an O(1) error in that cell's divergence constraint
for a few steps and a pressure pulse proportional to 1/dt. Taking the side from the level set at
the centre (pyRMT.iim.DIV_CENTRE_SIDE) removes it.

Small on purpose (N <= 128, shared machine). Usage: python benchmarks/iim_pressure_jitter.py
"""
import os, sys
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import iim_moving_disc as T
import pyRMT.iim as I

TS = tuple(np.round(np.arange(0.03, 0.3001, 0.005), 4))


def level(sn, N):
    s = sn["phi"] <= 0
    dist = np.where(s, ndi.distance_transform_edt(s), ndi.distance_transform_edt(~s)) / N
    return sn["p"][s & (dist > 0.05)].mean() - sn["p"][(~s) & (dist > 0.05)].mean()


def jitter(N, dt_factor, method="sharp"):
    T.T_END = 0.3; T.DT_FACTOR = dt_factor
    _, sn = T.run(N, method, snap_times=TS)
    L = np.array([level(s, N) for s in sn]); t = np.array(TS)
    r = L - np.polyval(np.polyfit(t, L, 5), t)
    return float(np.sqrt(np.mean(r ** 2))), float(np.abs(r).max())


def main():
    cases = [(64, 1.0), (64, 0.25), (64, 0.125), (128, 1.0), (128, 0.25)]
    for lab, meth, side in (("smeared", "smeared", True), ("sharp, side per direction (old)", "sharp", False),
                            ("sharp, side from the centre", "sharp", True)):
        I.DIV_CENTRE_SIDE = side
        print(lab)
        for N, f in cases:
            j = jitter(N, f, meth)
            print(f"  N={N:4d} dt x{f:5.3f}: level jitter rms {j[0]:.2e} max {j[1]:.2e}", flush=True)
    I.DIV_CENTRE_SIDE = True


if __name__ == "__main__":
    main()
