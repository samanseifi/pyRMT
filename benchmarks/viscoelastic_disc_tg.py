"""Two-way-coupled VISCOELASTIC FSI: a Maxwell-branch disc in a Taylor-Green vortex.

This is the coupled counterpart of the neo-Hookean disc-in-Taylor-Green of
benchmarks/mac_disc_in_tg_convergence.py (paper Sec. 8.2). The solid carries the
elastic Finger tensor as a log-conformation field psi=log(b_e), transported with the
flow, stretched by the local velocity gradient and relaxed with time constant tau; the
Cauchy stress is sigma_s = G dev(b_e). As tau->inf the Maxwell branch reduces to the
neo-Hookean solid (b_e -> F F^T), so the tau=inf run recovers the elastic baseline;
finite tau dissipates stored elastic energy (reduced kinetic<->elastic exchange, faster
settling). b_e stays symmetric positive-definite by construction (log-conformation), so
det(b_e)=exp(tr psi) and min J are reported as robustness diagnostics.

Produces:
  (a) mesh convergence at fixed finite tau  (L2 vs finest grid)
  (b) tau-sweep: stored elastic energy and kinetic energy vs time; tau=inf vs elastic
  (c) det(b_e) and min J histories (boundedness / SPD preservation)

Usage: python benchmarks/viscoelastic_disc_tg.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import (mac_grid, momentum_predictor_freeslip, project,
                       poisson_eigs_neumann, divergence)
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map,
    rebuild_phi_from_reference_map, solid_cauchy_stress, smoothed_heaviside,
    grad_central_x_2nd, grad_central_y_2nd)
from pyRMT.viscoelastic import logconf_local_step_strang, sym_exp
from pyRMT.output import compute_strain_energy

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "ve_disc_tg")
os.makedirs(OUT, exist_ok=True)


def run_one(N, t_end, tau=np.inf, G_ve=0.1, R=0.15, x0=0.5, y0=0.7, mu_s=0.1,
            mu_f=0.01, U0=0.3, rho=1.0, scheme="semilagrangian", w_cut_fac=4.0,
            elastic_baseline=False, history=False):
    """One viscoelastic (Maxwell-branch) disc-in-TG run. elastic_baseline=True instead
    uses the reference-map neo-Hookean stress (paper Sec. 8.2), for the tau->inf check."""
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    xf = np.arange(N + 1) * dx
    Xu, Yu = np.meshgrid(xf, xc); Xv, Yv = np.meshgrid(xc, xf)

    u = U0 * np.pi * np.sin(np.pi * Xu) * np.cos(np.pi * Yu)
    v = -U0 * np.pi * np.cos(np.pi * Xv) * np.sin(np.pi * Yv)
    u[:, 0] = 0.0; u[:, -1] = 0.0; v[0, :] = 0.0; v[-1, :] = 0.0

    phi_init = lambda X, Y: np.sqrt((X - x0)**2 + (Y - y0)**2) - R
    phi = phi_init(Xc, Yc); sm = (phi <= 0).astype(float)
    X1, X2 = extrapolate_reference_map(Xc * sm, Yc * sm, phi, dx, dy, 3)
    p11 = np.zeros((N, N)); p12 = np.zeros((N, N)); p22 = np.zeros((N, N))  # psi=log I=0
    w_t = 2.0 * dx; nu = mu_f / rho; cs = np.sqrt(mu_s / rho)
    eig = poisson_eigs_neumann(N, N, dx, dy)
    dt = min(0.25 * dx / (U0 * np.pi), 0.2 * dx * dx / nu, 0.25 * dx / (cs + 1e-9))
    p = np.zeros((N, N))

    hist = {"t": [], "KE": [], "Eel": [], "detbe_max": [], "minJ": [], "cx": [], "cy": []}
    t = 0.0
    while t < t_end:
        if t + dt > t_end:
            dt = t_end - t
        u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
        phi = rebuild_phi_from_reference_map(X1, X2, phi_init); sm = (phi <= 0).astype(float)
        wc = w_cut_fac * dx
        X1 = advect_reference_map(X1, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
        X2 = advect_reference_map(X2, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
        X1, X2 = extrapolate_reference_map(X1, X2, phi, dx, dy, 3)
        phi = rebuild_phi_from_reference_map(X1, X2, phi_init)
        sxx, sxy, syy, J = solid_cauchy_stress(X1, X2, dx, dy, mu_s, 0.0, phi)

        detbe_max = 1.0
        if not elastic_baseline:
            p11 = advect_reference_map(p11, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
            p12 = advect_reference_map(p12, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
            p22 = advect_reference_map(p22, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
            p11, p12 = extrapolate_reference_map(p11, p12, phi, dx, dy, 3)
            p22, _ = extrapolate_reference_map(p22, np.zeros((N, N)), phi, dx, dy, 3)
            L11 = grad_central_x_2nd(u_c, dx); L12 = grad_central_y_2nd(u_c, dy)
            L21 = grad_central_x_2nd(v_c, dx); L22 = grad_central_y_2nd(v_c, dy)
            p11, p12, p22 = logconf_local_step_strang(p11, p12, p22, L11, L12, L21, L22, tau, dt)
            be11, be12, be22 = sym_exp(p11, p12, p22)
            trbe = 0.5 * (be11 + be22)
            sxx = G_ve * (be11 - trbe); sxy = G_ve * be12; syy = G_ve * (be22 - trbe)
            detbe = np.exp(p11 + p22)                     # det(exp psi)=exp(tr psi)
            detbe_max = float(np.max(detbe[phi <= 0])) if np.any(phi <= 0) else 1.0

        H = smoothed_heaviside(phi, w_t)
        Sxx = (1 - H) * sxx; Sxy = (1 - H) * sxy; Syy = (1 - H) * syy
        divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dy)
        divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dy)
        fu = np.zeros((N, N + 1)); fu[:, 1:-1] = 0.5 * (divx[:, 1:] + divx[:, :-1])
        fv = np.zeros((N + 1, N)); fv[1:-1, :] = 0.5 * (divy[1:, :] + divy[:-1, :])
        us, vs = momentum_predictor_freeslip(u, v, nu, dx, dy, dt, fu=fu, fv=fv, rho=rho)
        u, v, p = project(us, vs, dx, dy, dt, rho, eig)
        t += dt

        if history:
            u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
            KE = 0.5 * np.sum(u_c**2 + v_c**2) * dx * dy
            if elastic_baseline:
                # stored energy from the reference-map F, using the SAME neo-Hookean
                # form as the Maxwell branch so the tau->inf comparison is consistent.
                g11 = grad_central_x_2nd(X1, dx); g12 = grad_central_y_2nd(X1, dy)
                g21 = grad_central_x_2nd(X2, dx); g22 = grad_central_y_2nd(X2, dy)
                det = g11 * g22 - g12 * g21
                det = np.where(np.abs(det) < 1e-9, 1.0, det)
                F11 = g22 / det; F12 = -g12 / det          # F = inv(grad xi)
                F21 = -g21 / det; F22 = g11 / det
                trb = F11**2 + F12**2 + F21**2 + F22**2     # tr(F F^T)
                lndet = -2.0 * np.log(np.abs(det))          # ln det(F F^T)=ln(1/det^2)
                w = 0.5 * mu_s * (trb - 2.0 - lndet)
                Eel = float(np.sum(np.where(phi <= 0, w, 0.0)) * dx * dy)
            else:
                be11, be12, be22 = sym_exp(p11, p12, p22)
                # neo-Hookean-type stored energy of the Maxwell branch:
                # w = (G/2)(tr b_e - 2 - ln det b_e), integrated over the solid
                trb = be11 + be22; lndet = p11 + p22
                w = 0.5 * G_ve * (trb - 2.0 - lndet)
                Eel = float(np.sum(np.where(phi <= 0, w, 0.0)) * dx * dy)
            msk = phi <= 0
            # min J over the solid INTERIOR (exclude the diffuse blend band, where the
            # non-stress-bearing reference map is extrapolated and J is meaningless)
            msk_int = phi < -2.0 * dx
            minJ = float(J[msk_int].min()) if msk_int.any() else 1.0
            hist["t"].append(t); hist["KE"].append(KE); hist["Eel"].append(Eel)
            hist["detbe_max"].append(detbe_max); hist["minJ"].append(minJ)
            hist["cx"].append(float(Xc[msk].mean()) if msk.any() else np.nan)
            hist["cy"].append(float(Yc[msk].mean()) if msk.any() else np.nan)

    u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
    speed = np.sqrt(u_c**2 + v_c**2)
    KE = 0.5 * np.sum(u_c**2 + v_c**2) * dx * dy
    out = dict(N=N, dx=dx, xc=xc, speed=speed, p=p, KE=KE,
               divmax=float(np.max(np.abs(divergence(u, v, dx, dy)))))
    if history:
        out["hist"] = {k: np.array(val) for k, val in hist.items()}
    return out


def _sample(field_xc, field, grid):
    tmp = np.array([np.interp(grid, field_xc, field[j, :]) for j in range(field.shape[0])])
    return np.array([np.interp(grid, field_xc, tmp[:, k]) for k in range(tmp.shape[1])]).T


def _order(Ns, e):
    e = np.array(e)
    return np.log(e[:-1] / e[1:]) / np.log(np.array(Ns[1:]) / np.array(Ns[:-1]))


def main():
    import matplotlib; matplotlib.use("Agg")
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    # (a) mesh convergence at fixed finite tau
    NS = (32, 48, 64, 96, 128); tau_c = 1.0; tconv = 0.6
    conv = {N: run_one(N, tconv, tau=tau_c) for N in NS}
    g = conv[NS[-1]]["xc"]; ref = NS[-1]; coarse = list(NS[:-1])
    spref = _sample(conv[ref]["xc"], conv[ref]["speed"], g)
    prref = _sample(conv[ref]["xc"], conv[ref]["p"], g)
    verr = [np.sqrt(np.mean((_sample(conv[N]["xc"], conv[N]["speed"], g) - spref)**2)) for N in coarse]
    perr = [np.sqrt(np.mean((_sample(conv[N]["xc"], conv[N]["p"], g) - prref)**2)) for N in coarse]
    keerr = [abs(conv[N]["KE"] - conv[ref]["KE"]) for N in coarse]
    print("VE disc-in-TG convergence (tau=%.2g):" % tau_c)
    print("  velocity order", [f"{o:.2f}" for o in _order(coarse, verr)])
    print("  pressure order", [f"{o:.2f}" for o in _order(coarse, perr)])
    print("  kinetic  order", [f"{o:.2f}" for o in _order(coarse, keerr)])

    # (b,c) tau sweep with histories at fixed N
    Nh, tend = 64, 1.2
    taus = [np.inf, 1.0, 0.25]
    runs = {tau: run_one(Nh, tend, tau=tau, history=True)["hist"] for tau in taus}
    base = run_one(Nh, tend, elastic_baseline=True, history=True)["hist"]
    # tau->inf recovery: max relative KE(t) difference between tau=inf and elastic baseline
    rec = np.max(np.abs(runs[np.inf]["KE"] - base["KE"])) / np.max(base["KE"])
    print(f"tau->inf vs elastic baseline: max |dKE|/KE = {rec:.2e}")

    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.7))
    labels = {np.inf: r"$\tau=\infty$", 1.0: r"$\tau=1$", 0.25: r"$\tau=0.25$"}
    cols = {np.inf: "C0", 1.0: "C1", 0.25: "C3"}

    ax[0].loglog(np.array([1/N for N in coarse]), verr, "o-", label="velocity")
    ax[0].loglog(np.array([1/N for N in coarse]), perr, "s-", label="pressure")
    ax[0].loglog(np.array([1/N for N in coarse]), keerr, "^-", label="kinetic energy")
    h = np.array([1/N for N in coarse])
    ax[0].loglog(h, verr[0]*(h/h[0])**2, "k--", lw=1, label="2nd order")
    ax[0].loglog(h, perr[0]*(h/h[0])**1, "k:", lw=1, label="1st order")
    ax[0].invert_xaxis(); ax[0].set_xlabel("$h=1/N$"); ax[0].set_ylabel("$L_2$ error vs finest")
    ax[0].set_title(r"(a) mesh convergence ($\tau=1$)")
    ax[0].legend(ncol=2, loc="upper left"); ax[0].grid(True, which="both", alpha=0.2)

    for tau in taus:
        ax[1].plot(runs[tau]["t"], runs[tau]["Eel"], "-", color=cols[tau], label=labels[tau])
    ax[1].plot(base["t"], base["Eel"], "k--", lw=1.4, label="elastic (ref. map)")
    ax[1].set_xlabel("time $t$"); ax[1].set_ylabel("stored elastic energy")
    ax[1].set_title("(b) elastic energy vs relaxation")
    ax[1].legend(loc="upper left")
    ax[1].text(0.97, 0.03, r"$\tau\!\to\!\infty$ recovers elastic dynamics"
               "\n" r"($\max|\Delta\mathrm{KE}|/\mathrm{KE}=%.1f\times10^{-4}$)" % (rec * 1e4),
               transform=ax[1].transAxes, ha="right", va="bottom", fontsize=9)

    for tau in taus:
        ax[2].plot(runs[tau]["t"], runs[tau]["detbe_max"], "-", color=cols[tau], label=labels[tau])
    ax[2].axhline(1.0, ls=":", color="0.6", lw=1)
    ax[2].set_xlabel("time $t$"); ax[2].set_ylabel(r"$\max\ \det b_e$ (solid)")
    axr = ax[2].twinx()
    for tau in taus:
        axr.plot(runs[tau]["t"], np.clip(runs[tau]["minJ"], -0.3, 2), "--",
                 color=cols[tau], lw=1.1, alpha=0.75)
    axr.axhline(0.0, ls=":", color="0.6", lw=0.8)
    axr.set_ylim(-0.35, 1.15); axr.set_ylabel(r"$\min J$ (dashed)")
    ax[2].set_title(r"(c) $\det b_e$ and $\min J$")
    ax[2].legend(loc="lower left")

    fig.tight_layout()
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data, exist_ok=True)
    out = os.path.join(data, "viscoelastic_disc_tg.png")
    fig.savefig(out); print("saved", out)
    return dict(verr=verr, perr=perr, keerr=keerr, coarse=coarse, recovery=rec)


if __name__ == "__main__":
    main()
