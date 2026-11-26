"""RM-IIM: energy convergence and energy budget of the moving pre-stretched disc.

Kinetic energy of the perturbation flow KE(t), stored elastic energy of the disc W(t), and
E = KE + W. Viscosity is the only sink, so E must decrease monotonically; an increase is
numerical energy injection. Grids N = 32, 64, 128 with dt ~ h^2 (so the spatial order shows).
Reported: successive differences of KE, W and E at several times, their order, the largest
single-step increase of E, and the energy remaining at the end.
Usage: python benchmarks/iim_energy_convergence.py [methods, e.g. sharp] [bulk force mode: cells|face]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import iim_moving_disc as T
T.T_END = 1.0
TS = (0.1, 0.2, 0.4, 0.7, 1.0)


def main():
    if len(sys.argv) > 2:
        T.BULK_MODE = sys.argv[2]
    for meth in (sys.argv[1].split(",") if len(sys.argv) > 1 else ("smeared", "sharp")):
        H = {}
        for N, f in ((32, 1.0), (64, 0.5), (128, 0.25)):
            T.DT_FACTOR = f; H[N] = T.run(N, meth)
        print(f"{meth}:")
        for nm, col in (("KE", 2), ("W", 5), ("E=KE+W", None)):
            ser = {N: (H[N][:, 0], H[N][:, 2] + H[N][:, 5] if col is None else H[N][:, col]) for N in H}
            val = lambda N, t: float(np.interp(t, *ser[N]))
            d1 = [abs(val(32, t) - val(64, t)) for t in TS]; d2 = [abs(val(64, t) - val(128, t)) for t in TS]
            rel = [d2[i] / abs(val(128, t)) for i, t in enumerate(TS)]
            print(f"  {nm:7s} |64-128|: " + "  ".join(f"t={t}: {d:.1e} ({r*100:.1f}%)" for t, d, r in zip(TS, d2, rel)))
            print(f"          order:    " + "  ".join(f"{np.log2(a / b):5.2f}" for a, b in zip(d1, d2))
                  + f"   | time-integrated: {np.log2(np.mean(d1) / np.mean(d2)):.2f}")
        for N in H:
            E = H[N][:, 2] + H[N][:, 5]; dE = np.diff(E)
            print(f"  N={N:4d}: E(0+) {E[0]:.4e}  E(end) {E[-1]:.2e}  largest single-step increase of E "
                  f"{max(dE.max(), 0):.1e} ({max(dE.max(), 0) / E[0] * 100:.3f}% of E0), steps with dE>0: {(dE > 0).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
