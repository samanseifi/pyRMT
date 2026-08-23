"""Publication figure for the viscoelastic-implementation demonstration: a soft blob held
at the extensional stagnation point of a four-roll mill, ELASTIC vs VISCOELASTIC.

Rebuilt (no re-run) from the saved frames/histories in outputs/mac_ve_extension/ produced
by benchmarks/mac_viscoelastic_extension.py. Under sustained planar extension the purely
elastic Finger tensor grows ~e^{2*eps*t} without bound (the run degrades), whereas the
upper-convected Maxwell relaxation balances the stretching so b_e saturates near the
analytic planar-extension value b_e^xx -> 1/(1-2 Wi). This is the demonstration that most
directly exercises the viscoelastic (log-conformation) implementation.

Usage: python benchmarks/ve_extension_figure.py  ->  data/ve_extension.png
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "outputs", "mac_ve_extension")


def _frame(tag, idx):
    return np.load(os.path.join(D, f"frames_{tag}", f"f{idx:04d}.npz"))


def _panel(ax, fr, title):
    lam = fr["lam"]; phi = fr["phi"]; N = lam.shape[0]
    ext = [0, 1, 0, 1]
    ax.imshow(lam, origin="lower", extent=ext, cmap="magma",
              vmin=1.0, vmax=6.0, aspect="equal")
    xc = (np.arange(N) + 0.5) / N; X, Y = np.meshgrid(xc, xc)
    ax.contour(X, Y, phi, levels=[0.0], colors="w", linewidths=1.4)
    ax.set_xlim(0.2, 0.8); ax.set_ylim(0.2, 0.8)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def main():
    import matplotlib; matplotlib.use("Agg")
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    times = [2.0, 5.0, 9.0]
    idx = [int(round(t / 0.15)) for t in times]

    fig = plt.figure(figsize=(13.2, 5.2))
    gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 1.5], wspace=0.32, hspace=0.12)
    for j, (i, t) in enumerate(zip(idx, times)):
        _panel(fig.add_subplot(gs[0, j]), _frame("elastic", i),
               r"$t=%.0f$" % t if True else "")
        _panel(fig.add_subplot(gs[1, j]), _frame("tau0.5", i), "")
    # row labels
    fig.text(0.065, 0.72, "elastic\n($\\tau\\!\\to\\!\\infty$)", ha="center", va="center", fontsize=10)
    fig.text(0.065, 0.30, "viscoelastic\n($\\mathrm{Wi}=0.25$)", ha="center", va="center", fontsize=10)

    # b_e^xx(t) at the centre: elastic diverges, viscoelastic saturates to 1/(1-2Wi)
    ax = fig.add_subplot(gs[:, 3])
    cols = {"elastic": "C0", "tau0.25": "C1", "tau0.5": "C3"}
    labs = {"elastic": r"elastic ($\mathrm{Wi}=\infty$)",
            "tau0.25": r"$\mathrm{Wi}=0.125$", "tau0.5": r"$\mathrm{Wi}=0.25$"}
    for tag in ("elastic", "tau0.25", "tau0.5"):
        h = np.load(os.path.join(D, f"hist_{tag}.npz"))
        ax.plot(h["t"], h["bxx"], "-", color=cols[tag], label=labs[tag])
        ba = float(h["bxx_analytic"])
        if np.isfinite(ba):
            ax.axhline(ba, ls=":", color=cols[tag], lw=1.0)
    ax.set_xlabel("time $t$"); ax.set_ylabel(r"$b_e^{xx}$ at blob centre")
    ax.set_title(r"elastic diverges; viscoelastic $\to 1/(1-2\mathrm{Wi})$", fontsize=10)
    ax.legend(loc="center right", fontsize=9)
    ax.text(0.5, 0.9, r"dotted: analytic $1/(1-2\mathrm{Wi})$",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="0.4")

    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data, exist_ok=True)
    out = os.path.join(data, "ve_extension.png"); fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
