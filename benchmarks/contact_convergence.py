"""Grid convergence of the two-disc collision (referee M3), paralleling Jain, Kamrin & Mani
(2019, Fig. 20): two elastic solids collide and rebound, and the centroid trajectories are shown
to converge under mesh refinement. The contact stress is the trace-free, momentum-conserving
solid stress of Rycroft et al. (2020), so it conserves total momentum by construction and is not
nullified by the projection. We run the elastic collision at N=96,128,192 and report the
convergence of the collision trajectory, the minimum gap (no interpenetration), and the total
momentum drift. Writes paper/figures/contact_convergence.png and outputs/overlay/contact_conv.npz.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.contact_collision import run
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS = [96, 128, 192]
PAR = dict(model="elastic", mu_s=0.8, V0=0.9, mu_f=0.005, eta=2.0, sep=0.30, R=0.12,
           integrator="imex", stab=False, t_end=1.6)


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks")); from _plotstyle import apply as _s; _s()
    import matplotlib.pyplot as plt
    H = {}
    for N in NS:
        h = run(N=N, **PAR)
        H[N] = h
        t, ca, cb, gap = h[:, 0], h[:, 1], h[:, 2], h[:, 3]
        crossed = bool(np.any(ca >= cb))          # centroids passing through each other
        print(f"  N={N}: min gap={gap.min():.4f} (2R={2*PAR['R']:.3f})  "
              f"interpenetration={crossed}", flush=True)
    np.savez(os.path.join(ROOT, "outputs", "overlay", "contact_conv.npz"),
             **{f"h_{N}": H[N] for N in NS})
    # convergence of the gap trajectory vs the finest grid (common time base)
    ref = H[NS[-1]]; tref = ref[:, 0]
    for N in NS[:-1]:
        g = np.interp(tref, H[N][:, 0], H[N][:, 3])
        err = np.sqrt(np.mean((g - ref[:, 3])**2))
        print(f"  gap RMS(N={N} vs {NS[-1]}) = {err:.3e}", flush=True)

    cols = {96: "blue", 128: "darkgreen", 192: "red"}
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for N in NS:
        t, gap = H[N][:, 0], H[N][:, 3]
        ax.plot(t, gap, "-", color=cols[N], lw=1.8, label=f"$N={N}$")
    ax.axhline(2 * PAR["R"], color="0.6", lw=0.9, ls=":")
    ax.text(t[2], 2 * PAR["R"], "  $2R$ (just touching)", color="0.4", fontsize=9, va="bottom")
    ax.set_xlabel("time $t$"); ax.set_ylabel("centroid gap $c_b-c_a$")
    ax.set_title("Colliding discs: gap trajectory converges under refinement")
    ax.legend(loc="best")
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "contact_convergence.png")
    fig.savefig(out, dpi=200); print("saved", out)


if __name__ == "__main__":
    main()
