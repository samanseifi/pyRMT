"""Fluid-solver validation vs Ghia et al. (1982): lid-driven cavity at Re=100 and
Re=1000. Two figures, each with both Re:
  (a) u along the vertical centreline x=0.5   vs Ghia Table I
  (b) v along the horizontal centreline y=0.5 vs Ghia Table II
Writes paper/figures/ghia_uy.png and paper/figures/ghia_vx.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.mac_lid_driven import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ghia et al. (1982) Table I: u on the vertical centreline (y, u)
GHIA_Y = np.array([0.0,0.0547,0.0625,0.0703,0.1016,0.1719,0.2813,0.4531,0.5,
                   0.6172,0.7344,0.8516,0.9531,0.9609,0.9688,0.9766,1.0])
GHIA_U = {
    100:  np.array([0,-0.03717,-0.04192,-0.04775,-0.06434,-0.10150,-0.15662,-0.21090,
                    -0.20581,-0.13641,0.00332,0.23151,0.68717,0.73722,0.78871,0.84123,1.0]),
    1000: np.array([0,-0.18109,-0.20196,-0.22220,-0.29730,-0.38289,-0.27805,-0.10648,
                    -0.06080,0.05702,0.18719,0.33304,0.46604,0.51117,0.57492,0.65928,1.0]),
}
# Ghia et al. (1982) Table II: v on the horizontal centreline (x, v)
GHIA_X = np.array([0.0,0.0625,0.0703,0.0781,0.0938,0.1563,0.2266,0.2344,0.5,
                   0.8047,0.8594,0.9063,0.9453,0.9531,0.9609,0.9688,1.0])
GHIA_V = {
    100:  np.array([0,0.09233,0.10091,0.10890,0.12317,0.16077,0.17507,0.17527,0.05454,
                    -0.24533,-0.22445,-0.16914,-0.10313,-0.08864,-0.07391,-0.05906,0]),
    1000: np.array([0,0.27485,0.29012,0.30353,0.32627,0.37095,0.33075,0.32235,0.02526,
                    -0.31966,-0.42665,-0.51550,-0.39188,-0.33714,-0.27669,-0.21388,0]),
}


def profiles(Re, N=128):
    f = run(Re=Re, N=N, imex=True, return_fields=True)
    u, v, dx, dy = f["u"], f["v"], f["dx"], f["dy"]
    yc = (np.arange(N) + 0.5) * dy
    xc = (np.arange(N) + 0.5) * dx
    u_line = u[:, N // 2]          # x=0.5 is an x-face
    v_line = v[N // 2, :]          # y=0.5 is a y-face
    return yc, u_line, xc, v_line


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    res = {Re: profiles(Re) for Re in (100, 1000)}
    col = {100: "tab:blue", 1000: "tab:red"}

    # (a) u(y)
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    for Re in (100, 1000):
        yc, u_line, _, _ = res[Re]
        ax.plot(u_line, yc, "-", color=col[Re], lw=1.6, label=f"MAC, Re={Re}")
        ax.plot(GHIA_U[Re], GHIA_Y, "o", mfc="none", mec=col[Re], ms=5, mew=1.0,
                label=f"Ghia (1982), Re={Re}")
    ax.set_xlabel(r"$u$ at $x=0.5$"); ax.set_ylabel(r"$y$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left"); fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "paper", "figures", "ghia_uy.png"), dpi=200)
    print("saved ghia_uy.png")

    # (b) v(x)
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    for Re in (100, 1000):
        _, _, xc, v_line = res[Re]
        ax.plot(xc, v_line, "-", color=col[Re], lw=1.6, label=f"MAC, Re={Re}")
        ax.plot(GHIA_X, GHIA_V[Re], "o", mfc="none", mec=col[Re], ms=5, mew=1.0,
                label=f"Ghia (1982), Re={Re}")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$v$ at $y=0.5$")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left"); fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "paper", "figures", "ghia_vx.png"), dpi=200)
    print("saved ghia_vx.png")


if __name__ == "__main__":
    main()
