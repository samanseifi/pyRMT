"""Stiff-solid FSI: a stiff neo-Hookean disc (mu_s=16, c_s=4 U_lid) in the lid-driven
cavity, where the elastic wave binds the explicit step. The explicit scheme diverges once
the step exceeds its combined elastic/advective CFL; the energy-conserving trapezoidal
coupling (Prop. 2) is unconditionally stable and advances the same disc at several times
that step with the body intact. The price is an O(dt) operator-split accuracy cost at large
steps (consistent with the work-precision delineation of Sec. 9), so this establishes the
stability that ENABLES the stiff regime, not a matched-accuracy speedup.

Usage: python benchmarks/stiff_solid_fsi.py  ->  data/stiff_solid_fsi.png
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.mac_soft_disc_lid import run

N, MU_S, T_END = 48, 16.0, 3.0


def _run(dt_fac, integ):
    dx = 1.0 / N; cs = np.sqrt(MU_S); dt_el = 0.3 * dx / cs
    return run(N=N, t_end=T_END, dt=dt_fac * dt_el, integrator=integ, mu_s=MU_S, write=False)


def main():
    import matplotlib; matplotlib.use("Agg")
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    ref = _run(1.0, "explicit")            # converged explicit reference (dt = elastic CFL)
    exp6 = _run(6.0, "explicit")           # explicit beyond its limit -> diverges
    tz4 = _run(4.0, "imex-elastic-sl")     # trapezoidal, 4x the elastic CFL
    tz8 = _run(8.0, "imex-elastic-sl")     # trapezoidal, 8x
    print(f"explicit 1x : reached t={ref[-1,0]:.2f}, steps={len(ref)}")
    print(f"explicit 6x : reached t={exp6[-1,0]:.2f}  (diverges early)")
    print(f"trapz 4x    : reached t={tz4[-1,0]:.2f}, steps={len(tz4)}, minJ={tz4[:,3].min():.2f}")
    print(f"trapz 8x    : reached t={tz8[-1,0]:.2f}, steps={len(tz8)}, minJ={tz8[:,3].min():.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # (a) stability: min J over time -- explicit 6x blows up, trapezoidal stays bounded
    ax[0].plot(ref[:, 0], ref[:, 3], "-", color="0.5", lw=2, label=r"explicit $1\times$ (ref.)")
    ax[0].plot(exp6[:, 0], exp6[:, 3], "x-", color="C3", ms=5, label=r"explicit $6\times$ (diverges)")
    ax[0].plot(tz4[:, 0], tz4[:, 3], "-", color="C0", label=r"trapezoidal $4\times$")
    ax[0].plot(tz8[:, 0], tz8[:, 3], "-", color="C1", label=r"trapezoidal $8\times$")
    ax[0].axhline(0.0, ls=":", color="0.6", lw=0.8)
    ax[0].set_xlabel("time $t$"); ax[0].set_ylabel(r"$\min J$ (solid)")
    ax[0].set_ylim(-0.1, 1.05); ax[0].set_xlim(0, T_END)
    ax[0].set_title(r"(a) stability at stiff $c_s=4\,U$ ($\mu_s=16$)")
    ax[0].legend(loc="lower left", fontsize=9)

    # (b) trajectory: trapezoidal tracks the converged explicit path (with O(dt) drift)
    ax[1].plot(ref[:, 1], ref[:, 2], "-", color="0.5", lw=2.5, label=r"explicit $1\times$ (ref.)")
    ax[1].plot(tz4[:, 1], tz4[:, 2], "-", color="C0", lw=1.4, label=r"trapezoidal $4\times$")
    ax[1].plot(tz8[:, 1], tz8[:, 2], "-", color="C1", lw=1.4, label=r"trapezoidal $8\times$")
    ax[1].set_xlabel("centroid $x$"); ax[1].set_ylabel("centroid $y$"); ax[1].axis("equal")
    ax[1].set_title(r"(b) trajectory: stable, with $O(\Delta t)$ split error")
    ax[1].legend(loc="best", fontsize=9)

    fig.tight_layout()
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data, exist_ok=True)
    out = os.path.join(data, "stiff_solid_fsi.png"); fig.savefig(out); print("saved", out)
    return dict(exp6_t=float(exp6[-1, 0]), tz8_minJ=float(tz8[:, 3].min()))


if __name__ == "__main__":
    main()
