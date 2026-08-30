"""Fig.: two-way FSI in plane shear -- a deformable disc perturbs the ambient linear
shear into a closed recirculation. Elastic (soft, Ca=0.4) vs a strongly-relaxing
viscoelastic SLS (Ca=0.4, Wi=2): the viscoelastic disc is markedly stiffer and less
deformed. Streamlines only, with the solid interface outlined.

Writes paper/figures/fsi_shear_snapshot.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.disc_in_shear import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def panel(ax, f, color, title):
    Xc, Yc = f["Xc"], f["Yc"]
    xc = Xc[0, :]; yc = Yc[:, 0]
    ax.streamplot(xc, yc, f["u"], f["v"], color="0.55", linewidth=0.5,
                  density=1.1, arrowsize=0.6, arrowstyle="->")
    ax.contour(Xc, Yc, f["phi"], levels=[0.0], colors=color, linewidths=2.2)
    ax.set_xlim(xc.min(), xc.max()); ax.set_ylim(yc.min(), yc.max())
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    for s in ax.spines.values():
        s.set_edgecolor("0.6")


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    Ca, N, T = 0.4, 128, 8.0
    gdot = 2 * 0.5 / 1.0
    el = run(N=N, Ca=Ca, model="elastic", t_end=T, return_fields=True)
    ve = run(N=N, Ca=Ca, model="sls", tau=2.0 / gdot, t_end=T, return_fields=True)
    print(f"elastic D={el['D']:.3f} th={np.degrees(el['th']):.0f}   "
          f"viscoelastic(Wi=2) D={ve['D']:.3f} th={np.degrees(ve['th']):.0f}")

    np.savez(os.path.join(ROOT, "outputs", "overlay", "fsi_shear_fields.npz"),
             **{f"el_{k}": v for k, v in el["fields"].items() if np.ndim(v) > 0},
             **{f"ve_{k}": v for k, v in ve["fields"].items() if np.ndim(v) > 0},
             el_D=el["D"], ve_D=ve["D"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.0))
    panel(axes[0], el["fields"], "blue",
          rf"elastic ($\mathrm{{Wi}}\!\to\!\infty$): $D={el['D']:.2f}$")
    panel(axes[1], ve["fields"], "red",
          rf"viscoelastic ($\mathrm{{Wi}}=2$): $D={ve['D']:.2f}$")
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "fsi_shear_snapshot.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main()
