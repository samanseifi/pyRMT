"""Two diagnostics on the lid-driven soft disc (Sugiyama case, N=128, cubic SL transport):

  (2) STABILIZER OFF vs ON: the money run (imex-sl, large step) uses the trapezoidal
      elastic-wave stabilizer to step past the elastic CFL. Rerun with the wave resolved
      (stab off) at a step below the elastic CFL and check the centroid path is unchanged
      -- i.e. the (energy-conserving) stabilizer does not distort the validation.

  (3) NEO-HOOKEAN vs SLS(tau->inf): the viscoelastic branch is sigma = G dev(b_e); with no
      relaxation (tau->inf) b_e -> F F^T, so with G_ve = mu_s it must reproduce the pure
      neo-Hookean stress sigma = mu_s dev(F F^T). Overlay the two centroid paths.

Writes paper/figures/lid_tauinf.png and outputs/overlay/lid_stab_tauinf.npz.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.mac_soft_disc_lid import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, T, MU = 128, 24.0, 0.1
dx = 1.0 / N
dt_ref = min(0.3 * dx, 0.2 * dx * dx / 0.01)      # money reference step (viscous-limited)
dt_big = 4.0 * dt_ref                              # money imex-sl step
SCH = "semilagrangian_cubic"


def arr(traj):
    return np.asarray(traj)                        # cols: t, cx, cy, minJ, maxJ


def maxsep(a, b):
    n = min(len(a), len(b))
    return float(np.max(np.hypot(a[:n, 1] - b[:n, 1], a[:n, 2] - b[:n, 2])))


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    # (3) neo-Hookean vs SLS(tau->inf) -- both at the money imex-sl config
    neo = arr(run(N=N, t_end=T, dt=dt_big, integrator="imex-sl", scheme=SCH,
                  mu_s=MU, write=False))
    print(f"[3] neo done ({len(neo)} steps)", flush=True)
    sls = arr(run(N=N, t_end=T, dt=dt_big, integrator="imex-sl", scheme=SCH,
                  mu_s=MU, viscoelastic=True, G_ve=MU, tau=np.inf, write=False))
    print(f"[3] sls(tau=inf) done ({len(sls)} steps)  maxsep(neo,sls)={maxsep(neo,sls):.4e}", flush=True)

    # (2) stabilizer on (imex-sl money) vs off (imex, wave resolved below elastic CFL)
    cs = np.sqrt(MU); dt_el = 0.25 * dx / cs
    dt_off = min(dt_big, 0.8 * dt_el)             # resolve the elastic wave
    on = neo                                       # imex-sl with stabilizer (== money elastic run)
    off = arr(run(N=N, t_end=T, dt=dt_off, integrator="imex", scheme=SCH,
                  mu_s=MU, stab=False, write=False))
    print(f"[2] stab-off done ({len(off)} steps, dt={dt_off:.2e}<dt_el={dt_el:.2e})  "
          f"maxsep(on,off)={maxsep(on,off):.4e}", flush=True)

    np.savez(os.path.join(ROOT, "outputs", "overlay", "lid_stab_tauinf.npz"),
             neo=neo, sls=sls, off=off)

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 5.0))
    # (3)
    ax[0].plot(neo[:, 1], neo[:, 2], "-", color="black", lw=2.4, label="neo-Hookean")
    ax[0].plot(sls[:, 1], sls[:, 2], "--", color="red", lw=1.8, label=r"SLS, $\tau\!\to\!\infty$")
    ax[0].set_title(rf"(a) $\tau\!\to\!\infty$ recovers neo-Hookean  (max sep {maxsep(neo,sls):.1e})")
    # (2)
    ax[1].plot(on[:, 1], on[:, 2], "-", color="black", lw=2.4, label="stabilizer on (imex-sl)")
    ax[1].plot(off[:, 1], off[:, 2], "--", color="tab:blue", lw=1.8, label="stabilizer off (resolved)")
    ax[1].set_title(rf"(b) stabilizer off matches on  (max sep {maxsep(on,off):.1e})")
    for a in ax:
        a.set_xlabel("centroid $x$"); a.set_ylabel("centroid $y$")
        a.set_aspect("equal"); a.legend(loc="best")
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "lid_tauinf.png")
    fig.savefig(out, dpi=200); print("saved", out)


def replot_split():
    """Re-render two single-panel figures from the cached trajectories (no re-run):
    figures/lid_tauinf.png  -- tau->inf vs neo-Hookean (main text)
    figures/lid_staboff.png -- stabilizer off vs on      (supplementary)."""
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt
    d = np.load(os.path.join(ROOT, "outputs", "overlay", "lid_stab_tauinf.npz"))
    neo, sls, off = d["neo"], d["sls"], d["off"]

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot(neo[:, 1], neo[:, 2], "-", color="black", lw=2.4, label="neo-Hookean")
    ax.plot(sls[:, 1], sls[:, 2], "--", color="red", lw=1.8, label=r"SLS, $\tau\!\to\!\infty$")
    ax.set_xlabel("centroid $x$"); ax.set_ylabel("centroid $y$")
    ax.set_aspect("equal"); ax.legend(loc="lower left"); fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "paper", "figures", "lid_tauinf.png"), dpi=200)
    print("saved lid_tauinf.png  (max sep %.1e)" % maxsep(neo, sls))

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot(neo[:, 1], neo[:, 2], "-", color="black", lw=2.4, label="stabilizer on (imex-sl)")
    ax.plot(off[:, 1], off[:, 2], "--", color="tab:blue", lw=1.8, label="stabilizer off (resolved)")
    ax.set_xlabel("centroid $x$"); ax.set_ylabel("centroid $y$")
    ax.set_aspect("equal"); ax.legend(loc="lower left"); fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "paper", "figures", "lid_staboff.png"), dpi=200)
    print("saved lid_staboff.png  (max sep %.1e)" % maxsep(neo, off))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "replot":
        replot_split()
    else:
        main()
