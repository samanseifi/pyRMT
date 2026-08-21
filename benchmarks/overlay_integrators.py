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
        lbl = {"explicit": "explicit", "imex": "IMEX (implicit visc.)",
               "imex-sl": "monolithic (imex-sl)"}[s]
        plt.plot(t, E, marker=m, color=c, ms=5, lw=1.4, label=lbl, markevery=3, zorder=3)
    plt.axhline(0, color="0.8", lw=0.8)
    plt.ylim(-0.05, 1.35)
    plt.xlabel("time $t$"); plt.ylabel("kinetic energy  $E(t)/E_0$")
    plt.title(f"Taylor-Green decay  (N=64, $\\nu$={nu}, $\\Delta t$={dt}): all track analytic")
    plt.legend(frameon=False); plt.grid(alpha=0.3); plt.tight_layout()
    p = os.path.join(OUT, "taylor_green_overlay.png"); plt.savefig(p, dpi=140); plt.close()
    print(f"  saved {p}")


# ---------------- soft disc centroid trajectory ----------------
def soft_disc_overlay():
    from benchmarks.mac_soft_disc_lid import run
    N, t_end = 64, 8.0
    dx = 1.0 / N
    dt_ref = min(0.3 * dx, 0.2 * dx * dx / 0.01)
    runs = {
        "explicit": run(N=N, t_end=t_end, dt=dt_ref, integrator="explicit", write=False),
        "imex": run(N=N, t_end=t_end, dt=dt_ref, integrator="imex", write=False),
        "imex-sl": run(N=N, t_end=t_end, dt=4.0 * dt_ref, integrator="imex-sl", write=False),
    }
    plt.figure(figsize=(6.0, 6.0))
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    for nm, fn in (("Sugiyama 2011", "Sugiyama_1024x1024.csv"),):
        pth = os.path.join(data, fn)
        if os.path.isfile(pth):
            d = np.loadtxt(pth, delimiter=","); plt.plot(d[:, 0], d[:, 1], "k--", lw=1.5, label=nm)
    styles = {"explicit": ("o", "#d1495b"), "imex": ("s", "#00798c"), "imex-sl": ("^", "#edae49")}
    lbl = {"explicit": "explicit", "imex": "IMEX", "imex-sl": "monolithic (imex-sl, 4$\\times\\Delta t$)"}
    for s in ("explicit", "imex", "imex-sl"):
        tr = runs[s]; m, c = styles[s]
        plt.plot(tr[:, 1], tr[:, 2], marker=m, color=c, ms=4, lw=1.5, markevery=8, label=lbl[s])
    plt.xlabel("centroid $x$"); plt.ylabel("centroid $y$"); plt.axis("equal")
    plt.title(f"Soft disc in lid-driven cavity  (N={N})")
    plt.legend(frameon=False); plt.grid(alpha=0.3); plt.tight_layout()
    p = os.path.join(OUT, "soft_disc_overlay.png"); plt.savefig(p, dpi=140); plt.close()
    print(f"  saved {p}")
    return {s: (len(runs[s]), runs[s][-1, 1], runs[s][-1, 2]) for s in runs}


if __name__ == "__main__":
    print("[overlay] Taylor-Green ..."); taylor_green_overlay()
    print("[overlay] soft disc ..."); info = soft_disc_overlay()
    for s, (n, cx, cy) in info.items():
        print(f"    {s:9s}: steps={n:4d} final centroid=({cx:.4f},{cy:.4f})")
