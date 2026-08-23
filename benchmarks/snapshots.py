"""Simulation snapshots for the three FSI examples: the elastic disc in the lid-driven
cavity, the viscoelastic disc in shear (the benchmark), and the SLS disc in a Taylor-Green
vortex. Each row shows the deformed solid interface (phi=0, white) over the flow speed at
three instants/parameters.

Usage: python benchmarks/snapshots.py  ->  data/snapshots.png
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.mac_soft_disc_lid import run as run_lid
from benchmarks.viscoelastic_disc_tg import run_one as run_tg
from benchmarks.disc_in_shear import run as run_shear


def _panel(ax, snap, title, Lx=1.0, Ly=1.0):
    sp = snap["speed"]; phi = snap["phi"]
    ax.imshow(sp, origin="lower", extent=[0, Lx, 0, Ly], cmap="viridis", aspect="equal")
    ax.contour(snap["Xc"], snap["Yc"], phi, levels=[0.0], colors="w", linewidths=1.6)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])


def main():
    import matplotlib; matplotlib.use("Agg")
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    print("[snap] elastic disc in lid ...")
    _, lid = run_lid(N=64, t_end=8.0, integrator="explicit", mu_s=0.1, write=False,
                     snap_times=[2.0, 5.0, 8.0])
    print("[snap] viscoelastic disc in shear ...")
    shear = [run_shear(N=48, Ca=c, model="elastic", t_end=4.0, scheme="semilagrangian",
                       return_fields=True)["fields"] for c in (0.1, 0.2, 0.4)]
    print("[snap] SLS disc in Taylor-Green ...")
    tg = run_tg(64, 1.05, tau=1.0, model="sls", snap_times=[0.3, 0.6, 1.0])["snaps"]

    fig, ax = plt.subplots(3, 3, figsize=(9.6, 9.8))
    for j, s in enumerate(lid):
        _panel(ax[0, j], s, r"lid cavity, $t=%.0f$" % s["t"])
    for j, (f, c) in enumerate(zip(shear, (0.1, 0.2, 0.4))):
        _panel(ax[1, j], dict(speed=f["speed"], phi=f["phi"], Xc=f["Xc"], Yc=f["Yc"]),
               r"shear, $\mathrm{Ca}=%.1f$" % c, Lx=f["Lx"], Ly=f["Ly"])
    for j, s in enumerate(tg):
        _panel(ax[2, j], s, r"SLS in TG, $t=%.1f$" % s["t"])
    for a, lab in zip(ax[:, 0], ["(a) elastic disc,\nlid-driven cavity",
                                 "(b) viscoelastic disc,\nsimple shear",
                                 "(c) SLS disc,\nTaylor--Green"]):
        a.set_ylabel(lab, fontsize=10)
    fig.tight_layout()
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data, exist_ok=True)
    out = os.path.join(data, "snapshots.png"); fig.savefig(out, dpi=200); print("saved", out)


if __name__ == "__main__":
    main()
