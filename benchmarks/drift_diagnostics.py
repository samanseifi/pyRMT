"""Interface-integrity / conservation diagnostics for the reference-map transport (referee C4/C5).
A viscoelastic (SLS) disc orbiting in a single Taylor-Green vortex (bounded b_e, sustained
transport and deformation) is advanced at three resolutions; we track, vs time, (a) the solid area
relative to its initial value -- the RMT is
non-conservative, so this drifts, and we show the drift SHRINKS with resolution; (b) the level-set
band distortion mean||grad phi|-1| over |phi|<2dx -- the diffuse-interface flatness; and
(c) det(b_e) and the interior min-J -- boundedness / SPD retention.
Writes paper/figures/drift_diagnostics.png and outputs/overlay/drift_diag.npz.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.viscoelastic_disc_tg import run_one
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# single vortex (kw=1) -> disc orbits and deforms with bounded b_e; SLS solid
PAR = dict(model="sls", G_ve=0.3, tau=1.0, R=0.15, x0=0.5, y0=0.7, mu_s=0.3,
           mu_f=1e-2, U0=0.3, kw=1, integrator="imex", scheme="semilagrangian_cubic")


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks")); from _plotstyle import apply as _s; _s()
    import matplotlib.pyplot as plt
    NS = tuple(int(x) for x in os.environ.get("DRIFT_NS", "64,128").split(",")); t_end = 3.0
    npz = os.path.join(ROOT, "outputs", "overlay", "drift_diag.npz")
    store = {}
    H = {}
    for N in NS:
        H[N] = run_one(N, t_end, history=True, **PAR)["hist"]
        h = H[N]
        print(f"  N={N}: area drift={abs(h['area'][-1]-1)*100:.2f}%  "
              f"max det be={h['detbe_max'].max():.3f}  min J={h['minJ'].min():.3f}  "
              f"max band |grad phi|-1={h['gphi'].max():.3f}", flush=True)
        for k in ("t", "area", "areas", "gphi", "detbe_max", "minJ"):
            store[f"{k}_{N}"] = h[k]
        np.savez(npz, **store)   # incremental: partial results survive an interrupt

    cols = {64: "blue", 128: "darkgreen", 256: "red"}
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))
    for N in NS:
        h = H[N]
        ax[0].plot(h["t"], (h["areas"] - 1) * 100, "-", color=cols[N], lw=1.7, label=f"$N={N}$")
        ax[1].plot(h["t"], h["gphi"], "-", color=cols[N], lw=1.7, label=f"$N={N}$")
        ax[2].plot(h["t"], h["detbe_max"], "-", color=cols[N], lw=1.7, label=f"$N={N}$ ($\\det b_e$)")
        ax[2].plot(h["t"], h["minJ"], "--", color=cols[N], lw=1.3, alpha=0.8)
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_ylabel("solid-area change (\\%)"); ax[0].set_title("(a) area conservation")
    ax[1].set_ylabel(r"median $\big||\nabla\phi|-1\big|$ in band")
    ax[1].set_title(r"(b) interface-band distortion")
    ax[2].axhline(1, color="0.6", lw=0.8)
    ax[2].set_ylabel(r"$\det b_e$ (solid), interior $\min J$ (dashed)")
    ax[2].set_title(r"(c) boundedness / SPD")
    for a in ax:
        a.set_xlabel("time $t$"); a.set_xlim(0, t_end); a.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "drift_diagnostics.png")
    fig.savefig(out, dpi=200); print("saved", out)


if __name__ == "__main__":
    main()
