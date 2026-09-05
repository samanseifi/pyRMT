"""Plot the drift diagnostics (referee C4/C5) from the cached outputs/overlay/drift_diag.npz
(written incrementally by drift_diagnostics.py). Decoupled from the run so it can plot whatever
resolutions have completed. Writes paper/figures/drift_diagnostics.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks")); from _plotstyle import apply as _s; _s()
    import matplotlib.pyplot as plt
    d = np.load(os.path.join(ROOT, "outputs", "overlay", "drift_diag.npz"))
    NS = sorted({int(k.split("_")[-1]) for k in d.files})
    cols = {64: "blue", 128: "darkgreen", 256: "red", 192: "red", 96: "darkgreen"}
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))
    for N in NS:
        t = d[f"t_{N}"]; c = cols.get(N, "black")
        ak = f"areas_{N}" if f"areas_{N}" in d.files else f"area_{N}"
        ax[0].plot(t, (d[ak] - 1) * 100, "-", color=c, lw=1.7, label=f"$N={N}$")
        ax[1].plot(t, d[f"gphi_{N}"], "-", color=c, lw=1.7, label=f"$N={N}$")
        ax[2].plot(t, d[f"detbe_max_{N}"], "-", color=c, lw=1.7, label=f"$N={N}$ ($\\det b_e$)")
        ax[2].plot(t, d[f"minJ_{N}"], "--", color=c, lw=1.3, alpha=0.8)
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_ylabel("solid-area change (\\%)"); ax[0].set_title("(a) area conservation")
    ax[1].set_ylabel(r"median $\big||\nabla\phi|-1\big|$ in band")
    ax[1].set_title(r"(b) interface-band distortion")
    ax[2].axhline(1, color="0.6", lw=0.8)
    ax[2].set_ylabel(r"$\det b_e$ (solid), interior $\min J$ (dashed)")
    ax[2].set_title(r"(c) boundedness / SPD")
    tmax = max(d[f"t_{N}"][-1] for N in NS)
    for a in ax:
        a.set_xlabel("time $t$"); a.set_xlim(0, tmax); a.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "drift_diagnostics.png")
    fig.savefig(out, dpi=200); print("saved", out, "resolutions", NS)


if __name__ == "__main__":
    main()
