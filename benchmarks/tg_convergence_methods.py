"""Spatial convergence of the MAC solver on Taylor-Green, comparing the integrators
(explicit / IMEX / monolithic). Measures the L2 error at time T against the analytic
solution for velocity, pressure, and kinetic energy (= total energy for the pure fluid),
and writes a publication-quality multi-panel convergence figure.

    u = -cos x sin y e^{-2 nu t},  v = sin x cos y e^{-2 nu t},
    p = -1/4 (cos 2x + cos 2y) e^{-4 nu t},   KE(t)/KE0 = e^{-4 nu t}.
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


def _faces(N):
    dx = L / N; i = np.arange(N); xf = i * dx; xc = (i + 0.5) * dx
    return dx, np.meshgrid(xf, xc), np.meshgrid(xc, xf), np.meshgrid(xc, xc)


def _tg(gu, gv, gc, nu, t):
    Xu, Yu = gu; Xv, Yv = gv; Xc, Yc = gc; e2 = np.exp(-2 * nu * t); e4 = np.exp(-4 * nu * t)
    u = -np.cos(Xu) * np.sin(Yu) * e2
    v = np.sin(Xv) * np.cos(Yv) * e2
    p = -0.25 * (np.cos(2 * Xc) + np.cos(2 * Yc)) * e4
    return u, v, p


def run_case(scheme, N, nu=0.05, T=0.1, dt=2e-5):
    dx, gu, gv, gc = _faces(N)
    u, v, _ = _tg(gu, gv, gc, nu, 0.0)
    eig = poisson_eigs_periodic(N, N, dx, dx); leig = lap_eigs_periodic(N, N, dx, dx)
    ns = int(round(T / dt)); phi = np.zeros((N, N))
    for _ in range(ns):
        if scheme == "explicit":
            us, vs = momentum_predictor_periodic(u, v, nu, dx, dx, dt)
        elif scheme == "imex":
            us, vs = momentum_predictor_periodic_imex(u, v, nu, dx, dx, dt, leig)
        else:
            us, vs = momentum_predictor_periodic_semilag(u, v, nu, dx, dx, dt, leig)
        u, v, phi = project_per(us, vs, dx, dx, dt, 1.0, eig)
    ue, ve, pe = _tg(gu, gv, gc, nu, ns * dt)
    err_u = np.sqrt(np.mean((u - ue) ** 2 + (v - ve) ** 2))
    p0 = phi - phi.mean(); pe0 = pe - pe.mean()
    err_p = np.sqrt(np.mean((p0 - pe0) ** 2))
    ke = 0.5 * np.mean(u ** 2 + v ** 2); ke_e = 0.5 * np.mean(ue ** 2 + ve ** 2)
    err_ke = abs(ke - ke_e)
    return err_u, err_p, err_ke


def run(Ns=(16, 24, 32, 48, 64, 96, 128), nu=0.05, T=0.1, dt=2e-5):
    schemes = ["explicit", "imex", "imex-sl"]
    res = {s: {"u": [], "p": [], "ke": []} for s in schemes}
    for N in Ns:
        for s in schemes:
            eu, ep, ek = run_case(s, N, nu, T, dt)
            res[s]["u"].append(eu); res[s]["p"].append(ep); res[s]["ke"].append(ek)
        print(f"  N={N:4d}  u:{res['explicit']['u'][-1]:.2e}/{res['imex']['u'][-1]:.2e}/"
              f"{res['imex-sl']['u'][-1]:.2e}  p:{res['explicit']['p'][-1]:.2e}")
    _plot(Ns, res)
    return res


def _plot(Ns, res):
    h = np.array([1.0 / N for N in Ns])
    labels = {"explicit": "explicit", "imex": "IMEX", "imex-sl": "monolithic"}
    colors = {"explicit": "#d1495b", "imex": "#00798c", "imex-sl": "#edae49"}
    markers = {"explicit": "o", "imex": "s", "imex-sl": "^"}
    panels = [("u", r"velocity $\|\mathbf{u}-\mathbf{u}_{\rm ex}\|_2$"),
              ("p", r"pressure $\|p-p_{\rm ex}\|_2$"),
              ("ke", r"kinetic energy $|E-E_{\rm ex}|$")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax, (key, ttl) in zip(axes, panels):
        for s in ("explicit", "imex", "imex-sl"):
            e = np.array(res[s][key])
            ax.loglog(h, e, marker=markers[s], color=colors[s], ms=6, lw=1.6,
                      label=labels[s], mfc="none" if s != "imex-sl" else colors[s])
        e0 = np.array(res["imex"][key])
        ax.loglog(h, e0[0] * (h / h[0]) ** 2, "k--", lw=1.0, label=r"2nd order")
        ax.loglog(h, e0[0] * (h / h[0]) ** 1, "k:", lw=1.0, label=r"1st order")
        ax.set_xlabel(r"$h = 1/N$"); ax.set_ylabel(ttl)
        ax.invert_xaxis()
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Taylor-Green spatial convergence: explicit / IMEX / monolithic", y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, "tg_convergence_methods.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}")


if __name__ == "__main__":
    print("[TG convergence] explicit / imex / monolithic ...")
    run()
