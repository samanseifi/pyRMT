"""Controlled demonstration that DIRECT b_e transport loses positive-definiteness where the
log-conformation update does not (referee C7).

The failure of direct conformation transport is a *spatial* phenomenon: a high-order
(non-monotone) advection of the b_e components overshoots at the steep gradients that build up
as the solid strains, and the overshoot drives the compressive eigenvalue negative -- b_e becomes
indefinite and the elastic stress unphysical. Transporting psi = log(b_e) instead guarantees
b_e = exp(psi) is symmetric-positive-definite for ANY psi, so the same advection stays admissible.

We freeze a four-cell Taylor-Green straining velocity, seed a disc with b_e = I, and transport with
the SAME bicubic semi-Lagrangian scheme and the SAME local UCM relaxation, once on the b_e
components (direct) and once on psi (log-conformation). We track the minimum eigenvalue of b_e over
the solid vs time. Writes paper/figures/logconf_vs_direct.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyRMT.functions import (advect_reference_map, extrapolate_reference_map,
                             rebuild_phi_from_reference_map, grad_central_x_2nd, grad_central_y_2nd)
from pyRMT.viscoelastic import ucm_local_step, logconf_local_step_strang, sym_exp, sym_log, _sym_eig2
from pyRMT.mac import mac_grid
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(mode, N=128, tau=1.0, U0=0.35, kw=2, R=0.2, t_end=2.0, scheme="semilagrangian_cubic"):
    """Transport b_e (mode='direct') or psi=log b_e (mode='log') under a frozen straining flow.
    Returns (t, min-eigenvalue-of-b_e-over-solid) history."""
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    k = kw * np.pi
    u_c = U0 * k * np.sin(k * Xc) * np.cos(k * Yc)          # frozen cell-centred velocity
    v_c = -U0 * k * np.cos(k * Xc) * np.sin(k * Yc)
    L11 = grad_central_x_2nd(u_c, dx); L12 = grad_central_y_2nd(u_c, dy)
    L21 = grad_central_x_2nd(v_c, dx); L22 = grad_central_y_2nd(v_c, dy)
    phi = np.sqrt((Xc - 0.5)**2 + (Yc - 0.5)**2) - R    # fixed disc (frozen velocity)
    Z = np.zeros((N, N))
    b11 = np.ones((N, N)); b12 = Z.copy(); b22 = np.ones((N, N))    # b_e = I
    p11 = Z.copy(); p12 = Z.copy(); p22 = Z.copy()                  # psi = log I = 0
    dt = 0.25 * dx / (U0 * k); wc = 4.0 * dx
    ts, mineig = [], []
    t = 0.0
    while t < t_end - 1e-12:
        if t + dt > t_end:
            dt = t_end - t
        if mode == "direct":
            b11 = advect_reference_map(b11, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc)
            b12 = advect_reference_map(b12, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc)
            b22 = advect_reference_map(b22, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc)
            b11, b12, b22 = ucm_local_step(b11, b12, b22, L11, L12, L21, L22, tau, dt)
        else:
            p11 = advect_reference_map(p11, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc)
            p12 = advect_reference_map(p12, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc)
            p22 = advect_reference_map(p22, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc)
            p11, p12, p22 = logconf_local_step_strang(p11, p12, p22, L11, L12, L21, L22, tau, dt)
            b11, b12, b22 = sym_exp(p11, p12, p22)
        t += dt
        msk = phi < -2.0 * dx                                   # solid interior
        if msk.any():
            l1, l2, _ = _sym_eig2(b11[msk], b12[msk], b22[msk])
            ts.append(t); mineig.append(float(np.min(np.minimum(l1, l2))))
        if not np.all(np.isfinite(b11)):
            ts.append(t); mineig.append(np.nan); break
    return np.array(ts), np.array(mineig)


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks")); from _plotstyle import apply as _s; _s()
    import matplotlib.pyplot as plt
    tau = 1.0
    td, md = run("direct", tau=tau)
    tl, ml = run("log", tau=tau)
    tcross = td[md < 0][0] if np.any(md < 0) else None
    print(f"  direct: min eig reaches {np.nanmin(md):.3e} (SPD lost if < 0)"
          + (f"; first goes negative at t={tcross:.2f}" if tcross is not None else ""))
    print(f"  log:    min eig reaches {np.nanmin(ml):.3e} (SPD by construction)")
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot(td, md, "-", color="red", lw=2.0, label="direct $b_e$ transport")
    ax.plot(tl, ml, "-", color="blue", lw=2.0, label=r"log-conformation ($\psi=\log b_e$)")
    if tcross is not None:
        ax.axvline(tcross, color="red", ls=":", lw=1.1)
        ax.text(tcross, 1.5, r"  $b_e$ indefinite", color="red", fontsize=10,
                rotation=90, va="bottom", ha="left")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$\min$ eigenvalue of $b_e$ over the solid (symlog)")
    ax.set_title(r"Positive-definiteness under strained transport (four-cell TG)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "logconf_vs_direct.png")
    fig.savefig(out, dpi=200); print("saved", out)


if __name__ == "__main__":
    main()
