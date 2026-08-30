"""Viscoelastic FSI validation figure: a viscoelastic disc in the lid-driven cavity.
Centroid trajectories for tau=inf (recovers neo-Hookean, validated vs Sugiyama) and
finite tau (relaxation). Writes outputs/overlay/viscoelastic_fsi.png."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmarks._plotstyle import apply as _apply_style; _apply_style()
from benchmarks.mac_soft_disc_lid import run

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "overlay")
os.makedirs(OUT, exist_ok=True)


def make(N=64, t_end=8.0):
    dx = 1.0 / N; dt = min(0.3 * dx, 0.2 * dx * dx / 0.01, 0.3 * dx / np.sqrt(0.1))
    kw = dict(N=N, t_end=t_end, dt=dt, integrator="explicit", write=False)
    runs = {
        r"neo-Hookean ($G_\infty$, elastic)": (run(mu_s=0.1, **kw), "k--"),
        r"viscoelastic $\tau=\infty$": (run(mu_s=0.1, viscoelastic=True, G_ve=0.1, tau=np.inf, **kw), "-"),
        r"viscoelastic $\tau=2$": (run(mu_s=0.1, viscoelastic=True, G_ve=0.1, tau=2.0, **kw), "-"),
        r"viscoelastic $\tau=0.5$": (run(mu_s=0.1, viscoelastic=True, G_ve=0.1, tau=0.5, **kw), "-"),
    }
    plt.figure(figsize=(6.2, 6.0))
    colors = ["#000000", "#00798c", "#edae49", "#d1495b"]
    for (lbl, (tr, sty)), col in zip(runs.items(), colors):
        c = "k" if sty == "k--" else col
        plt.plot(tr[:, 1], tr[:, 2], sty if sty == "k--" else "-", color=c, lw=2,
                 label=lbl, alpha=0.9)
    plt.xlabel("centroid $x$"); plt.ylabel("centroid $y$"); plt.axis("equal")
    plt.title(f"Viscoelastic disc in lid-driven cavity (N={N})")
    plt.legend(frameon=False, fontsize=9); plt.tight_layout()
    p = os.path.join(OUT, "viscoelastic_fsi.png"); plt.savefig(p, dpi=150); plt.close()
    print(f"saved {p}")
    for lbl, (tr, _s) in runs.items():
        print(f"  {lbl}: final centroid=({tr[-1,1]:.4f},{tr[-1,2]:.4f}) steps={len(tr)}")


if __name__ == "__main__":
    make()
