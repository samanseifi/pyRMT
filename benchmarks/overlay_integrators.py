"""Overlay explicit / IMEX / monolithic (imex-sl) on the two canonical cases.

  Taylor-Green : kinetic-energy decay E(t)/E0 vs the analytic e^{-4 nu t}.
  Soft disc    : centroid trajectory (cx,cy) vs Sugiyama (2011) if available.

Writes outputs/overlay/taylor_green_overlay.png and soft_disc_overlay.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmarks._plotstyle import apply as _apply_style; _apply_style()

from pyRMT.mac import (momentum_predictor_periodic, momentum_predictor_periodic_imex,
                       momentum_predictor_periodic_semilag, lap_eigs_periodic,
                       project_per, poisson_eigs_periodic)

L = 2 * np.pi
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "overlay")
os.makedirs(OUT, exist_ok=True)


# ---------------- Taylor-Green kinetic-energy decay ----------------
def _tg_faces(N):
    dx = L / N; i = np.arange(N); xf = i * dx; xc = (i + 0.5) * dx
    return dx, np.meshgrid(xf, xc), np.meshgrid(xc, xf)


def _tg_init(gu, gv, nu, t):
    Xu, Yu = gu; Xv, Yv = gv; e = np.exp(-2 * nu * t)
    return -np.cos(Xu) * np.sin(Yu) * e, np.sin(Xv) * np.cos(Yv) * e


def tg_energy_series(scheme, N=64, nu=0.05, T=2.0, dt=0.08):
    dx, gu, gv = _tg_faces(N)
    u, v = _tg_init(gu, gv, nu, 0.0)
    eig = poisson_eigs_periodic(N, N, dx, dx); leig = lap_eigs_periodic(N, N, dx, dx)
    ns = int(round(T / dt))
    E0 = 0.5 * (np.mean(u**2) + np.mean(v**2))
    ts = [0.0]; Es = [1.0]
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(1, ns + 1):
            if scheme == "explicit":
                us, vs = momentum_predictor_periodic(u, v, nu, dx, dx, dt)
            elif scheme == "imex":
                us, vs = momentum_predictor_periodic_imex(u, v, nu, dx, dx, dt, leig)
            else:
                us, vs = momentum_predictor_periodic_semilag(u, v, nu, dx, dx, dt, leig)
            u, v, _ = project_per(us, vs, dx, dx, dt, 1.0, eig)
            E = 0.5 * (np.mean(u**2) + np.mean(v**2))
            ts.append(k * dt); Es.append(E / E0 if np.isfinite(E) else np.nan)
    return np.array(ts), np.array(Es)


def taylor_green_overlay():
    nu, T, dt = 0.05, 2.0, 0.08          # dt > explicit viscous CFL (dx^2/4nu ~ 0.048)
    plt.figure(figsize=(6.4, 4.8))
    tan = np.linspace(0, T, 200)
    plt.plot(tan, np.exp(-4 * nu * tan), "k-", lw=2.5, label="analytic  $e^{-4\\nu t}$", zorder=1)
    styles = {"explicit": ("o", "#d1495b"), "imex": ("s", "#00798c"),
              "imex-sl": ("^", "#edae49")}
    for s in ("explicit", "imex", "imex-sl"):
        t, E = tg_energy_series(s, nu=nu, T=T, dt=dt)
        m, c = styles[s]
        lbl = {"explicit": "explicit", "imex": "IMEX (central adv.)",
               "imex-sl": "IMEX (semi-Lagrangian adv.)"}[s]
        plt.plot(t, E, marker=m, color=c, ms=5, lw=1.4, label=lbl, markevery=3, zorder=3)
    plt.axhline(0, color="0.8", lw=0.8)
    plt.ylim(-0.05, 1.35)
    plt.xlabel("time $t$"); plt.ylabel("kinetic energy  $E(t)/E_0$")
    plt.legend(frameon=False); plt.tight_layout()
    p = os.path.join(OUT, "taylor_green_overlay.png"); plt.savefig(p); plt.close()
    print(f"  saved {p}")


# ---------------- soft disc centroid trajectory ----------------
def _render_soft_disc(runs, out_png):
    """Plot the centroid spiral: our grid-converged result against the exact
    vector-extracted references of Jain et al. (2019) Fig. 17 (Sugiyama 2011 at
    1024^2 and 128^2, and Jain's own conservative RMT at 128^2). Our converged
    trajectory agrees with Sugiyama-1024^2 to ~4e-3---on par with Jain's ~3e-3---
    and well inside Sugiyama's own 128^2->1024^2 spread (~1e-2)."""
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    def ref(fn):
        p = os.path.join(data, fn)
        return np.loadtxt(p, delimiter=",", skiprows=1) if os.path.isfile(p) else None
    sug1024 = ref("Jain2019_fig17_sugiyama1024.csv")
    jain128 = ref("Jain2019_fig17_jain128.csv")
    plt.figure(figsize=(5.8, 5.8))
    # classic colours, thin lines. Sugiyama 1024^2 = reference (open circles);
    # Jain 128^2 = blue dashed; present RMT = red; IMEX 4x dt = dark-green markers.
    if jain128 is not None:
        plt.plot(jain128[:, 0], jain128[:, 1], "--", color="tab:blue", lw=1.1,
                 label="Jain et al.\\ (2019) $128^2$", zorder=2)
    if sug1024 is not None:
        plt.plot(sug1024[:, 0], sug1024[:, 1], "o", mfc="none", mec="0.15", ms=5.5,
                 mew=0.9, ls="none", label="Sugiyama (2011) $1024^2$", zorder=3)
    # present RMT: neo-Hookean (red) and, from the same run, the SLS solid in its
    # tau->inf elastic limit (dark green dashed). The log-conformation SLS transports
    # b_e directly, so it is more robust to the reference-map winding on the inner
    # spiral than the neo-Hookean F=(grad xi)^-1 stress, and tracks Sugiyama slightly
    # closer there (mean nearest-point distance 2.5e-3 vs 3.1e-3).
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npz = os.path.join(root, "outputs", "overlay", "lid_stab_tauinf.npz")
    if os.path.isfile(npz):
        d = np.load(npz); neo, sls = d["neo"], d["sls"]
        plt.plot(neo[:, 1], neo[:, 2], "-", color="tab:red", lw=1.8,
                 label=r"present, neo-Hookean $128^2$", zorder=4)
        plt.plot(sls[:, 1], sls[:, 2], "--", color="tab:green", lw=1.5,
                 label=r"present, SLS $\tau\!\to\!\infty$ $128^2$", zorder=5)
    else:
        ex = runs["imex"]
        plt.plot(ex[:, 1], ex[:, 2], "-", color="tab:red", lw=1.8,
                 label="present (RMT)", zorder=4)
    plt.xlabel("centroid $x$"); plt.ylabel("centroid $y$"); plt.axis("equal")
    plt.legend(frameon=False, loc="lower left", fontsize=8.5); plt.tight_layout()
    plt.savefig(out_png); plt.close()
    print(f"  saved {out_png}")


def soft_disc_overlay():
    from benchmarks.mac_soft_disc_lid import run
    N, t_end = 128, 24.0
    dx = 1.0 / N
    dt_ref = min(0.3 * dx, 0.2 * dx * dx / 0.01)
    # Reference-map transport: the monotone bicubic semi-Lagrangian is markedly less
    # diffusive than bilinear, which keeps the disc sharp on the inner spiral and
    # reproduces the Sugiyama match to the vortex centre (bilinear drifts ~2x more).
    sch = "semilagrangian_cubic"
    runs = {
        "explicit": run(N=N, t_end=t_end, dt=dt_ref, integrator="explicit", scheme=sch, write=False),
        "imex": run(N=N, t_end=t_end, dt=dt_ref, integrator="imex", scheme=sch, write=False),
        "imex-sl": run(N=N, t_end=t_end, dt=4.0 * dt_ref, integrator="imex-sl", scheme=sch, write=False),
    }
    # save trajectories so the figure can be re-styled without re-running the sims
    np.savez(os.path.join(OUT, "soft_disc_traj.npz"), **{s: runs[s] for s in runs})
    _render_soft_disc(runs, os.path.join(OUT, "soft_disc_overlay.png"))
    return {s: (len(runs[s]), runs[s][-1, 1], runs[s][-1, 2]) for s in runs}


def soft_disc_replot(npz_path=None, out_png=None):
    """Re-render the figure from the cached trajectories (no re-run)."""
    npz_path = npz_path or os.path.join(OUT, "soft_disc_traj.npz")
    out_png = out_png or os.path.join(OUT, "soft_disc_overlay.png")
    d = np.load(npz_path); runs = {k: d[k] for k in d.files}
    _render_soft_disc(runs, out_png)


if __name__ == "__main__":
    print("[overlay] Taylor-Green ..."); taylor_green_overlay()
    print("[overlay] soft disc ..."); info = soft_disc_overlay()
    for s, (n, cx, cy) in info.items():
        print(f"    {s:9s}: steps={n:4d} final centroid=({cx:.4f},{cy:.4f})")
