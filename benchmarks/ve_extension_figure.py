"""Publication figure for the viscoelastic-implementation demonstration: a soft blob held
at the extensional stagnation point of a four-roll mill, ELASTIC vs VISCOELASTIC.

Rebuilt (no re-run) from the saved frames/histories in outputs/mac_ve_extension/ produced
by benchmarks/mac_viscoelastic_extension.py. The elastic blob (tau->inf) accumulates strain
until its elastic stress arrests the stretch (b_e^xx ~ 9.8, steady filament ~10 cells wide).
The relaxing (pure Maxwell, no equilibrium branch) blobs saturate in STRAIN, b_e^xx just below
the homogeneous plateau 1/(1-2 Wi), while their SHAPE keeps elongating: the filament thins
below 4 cells (the 2*w_t blend band) at t ~ 5.6-5.9 and the capture then degrades (the solid
area grows), so the relaxing curves are drawn only while the centre width is >= MIN_CELLS.

Usage: python benchmarks/ve_extension_figure.py  ->  data/ve_extension.png + paper/figures/
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 2026-11 re-runs: IMEX at its default step dt0 (elastic-wave limit), bicubic transport; and the
# elastic case again at dt0/4 to show its time-step sensitivity (curves only, no frames).
D = os.path.join(_ROOT, "outputs", os.environ.get("RERUN_DIR", "rerun_2026-11"), "mac_ve_extension")
D_Q = os.path.join(_ROOT, "outputs", os.environ.get("RERUN_DIR", "rerun_2026-11"), "dtquarter", "mac_ve_extension")


def _frame(tag, idx):
    return np.load(os.path.join(D, f"frames_{tag}", f"f{idx:04d}.npz"))


MIN_CELLS = 4      # filament resolved while its centre width is at least the 2*w_t blend band


def _t_resolved(tag):
    """Last stored frame time at which the blob is still >= MIN_CELLS wide at the centre."""
    tt = np.load(os.path.join(D, f"meta_{tag}.npz"))["t"]
    for i, t in enumerate(tt):
        phi = _frame(tag, i)["phi"]; c = phi.shape[0] // 2
        if (phi[:, c - 1:c + 1] <= 0).any(1).sum() < MIN_CELLS:
            return float(tt[max(i - 1, 0)])
    return float(tt[-1])


def _panel(ax, fr, title, color="b"):
    # white background: four-roll streamlines (the flow) + blob OUTLINE (the particle)
    phi = fr["phi"]; N = phi.shape[0]
    xc = (np.arange(N) + 0.5) / N; X, Y = np.meshgrid(xc, xc)
    ax.set_facecolor("white")
    # four-roll mill velocity: u_mill = U0(sin kx cos ky, -cos kx sin ky), k = 2 pi
    k = 2.0 * np.pi
    U = np.sin(k * X) * np.cos(k * Y); V = -np.cos(k * X) * np.sin(k * Y)
    ax.streamplot(xc, xc, U, V, color="0.72", linewidth=0.5, density=0.7,
                  arrowsize=0.6, arrowstyle="->")
    ax.contour(X, Y, phi, levels=[0.0], colors=color, linewidths=2.0)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("0.7")
    if title:
        ax.set_title(title, fontsize=10)


def main():
    import matplotlib; matplotlib.use("Agg")
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # times chosen before the (fast-elongating) viscoelastic blob reaches the domain walls,
    # so no spurious boundary deformation is shown
    times = [2.0, 3.0, 4.0]
    idx = [int(round(t / 0.15)) for t in times]

    fig = plt.figure(figsize=(13.2, 5.2))
    gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 1.5], wspace=0.32, hspace=0.12)
    for j, (i, t) in enumerate(zip(idx, times)):
        _panel(fig.add_subplot(gs[0, j]), _frame("elastic", i),
               r"$t=%.0f$" % t, color="b")
        _panel(fig.add_subplot(gs[1, j]), _frame("tau0.5", i), "", color="r")
    # row labels
    fig.text(0.065, 0.72, "elastic\n($\\tau\\!\\to\\!\\infty$)", ha="center", va="center", fontsize=10)
    fig.text(0.065, 0.30, "viscoelastic\n($\\mathrm{Wi}=0.25$)", ha="center", va="center", fontsize=10)

    # b_e^xx(t) at the centre: elastic is arrested at large stretch, relaxing cases saturate early
    ax = fig.add_subplot(gs[:, 3])
    cols = {"elastic": "b", "tau0.25": "g", "tau0.5": "r"}
    labs = {"elastic": r"elastic ($\mathrm{Wi}=\infty$), $\Delta t_0$",
            "tau0.25": r"$\mathrm{Wi}=0.125$", "tau0.5": r"$\mathrm{Wi}=0.25$"}
    hq = np.load(os.path.join(D_Q, "hist_elastic.npz"))
    ax.plot(hq["t"], hq["bxx"], "--", color="b", lw=1.4, label=r"elastic, $\Delta t_0/4$")
    print(f"  elastic dt0/4: peak {hq['bxx'].max():.2f} at t={hq['t'][hq['bxx'].argmax()]:.1f}")
    for tag in ("elastic", "tau0.25", "tau0.5"):
        h = np.load(os.path.join(D, f"hist_{tag}.npz"))
        tres = _t_resolved(tag) if tag != "elastic" else np.inf; keep = h["t"] <= tres
        ax.plot(h["t"][keep], h["bxx"][keep], "-", color=cols[tag], lw=1.8, label=labs[tag])
        ba = float(h["bxx_analytic"])
        print(f"  {tag}: resolved to t={tres:.2f}, bxx there {h['bxx'][keep][-1]:.3f}, plateau {ba:.3f}")
        if np.isfinite(ba):     # homogeneous-extension plateau 1/(1-2Wi) at the nominal rate
            ax.axhline(ba, ls=":", color=cols[tag], lw=1.8)
            ax.plot(h["t"][keep][-1], h["bxx"][keep][-1], "o", color=cols[tag], ms=5)
    ax.set_xlabel("time $t$"); ax.set_ylabel(r"$b_e^{xx}$ at blob centre")
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data, exist_ok=True)
    root = os.path.dirname(data)
    for out in (os.path.join(data, "ve_extension.png"),
                os.path.join(root, "paper", "figures", "ve_extension.png")):
        fig.savefig(out, dpi=200, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main()
