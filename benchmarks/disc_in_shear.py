"""Deformable viscoelastic-SOLID disc in plane shear -- a benchmark for viscoelastic FSI.

A soft disc is centred in a rectangular cell whose top and bottom walls translate in
opposite directions (u=+U top, u=-U bottom), giving a mean shear rate gdot=2U/Ly at the
centre. The disc deforms to a steady inclined ellipse; we report the Taylor deformation
parameter D=(a-b)/(a+b) (from the second-moment tensor of the solid) versus the elastic
capillary number Ca=mu_f*gdot/G (viscous stress / elastic modulus). This is the classic
deformable-particle-in-shear test (Gao & Hu; Rosti & Brandt), here for a finite-strain
viscoelastic solid carried by the reference map / log-conformation.

Anchoring:
  * small Ca: linear response D ~ Ca (verified slope), grid-converged  -> credibility of
    the elastic FSI coupling;
  * finite tau (SLS): relaxation lowers the transient effective stiffness and shifts D,
    with tau->inf recovering the elastic curve.

Constitution (``model``):  "elastic" sigma=mu_s dev(FF^T);  "sls" adds G dev(b_e).

Usage: python benchmarks/disc_in_shear.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyRMT.mac import (mac_grid, momentum_predictor, project, poisson_eigs_neumann,
                       divergence)
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map,
    rebuild_phi_from_reference_map, solid_cauchy_stress, smoothed_heaviside,
    grad_central_x_2nd, grad_central_y_2nd)
from pyRMT.viscoelastic import logconf_local_step_strang, sym_exp

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "shear")
os.makedirs(OUT, exist_ok=True)


def deformation(phi, Xc, Yc, dx, dy):
    """Taylor deformation D=(a-b)/(a+b) and orientation from the second-moment tensor
    of the solid region (phi<=0)."""
    m = phi <= 0
    if m.sum() < 4:
        return 0.0, 0.0
    x = Xc[m]; y = Yc[m]; cx = x.mean(); cy = y.mean()
    Ixx = np.mean((x - cx)**2); Iyy = np.mean((y - cy)**2); Ixy = np.mean((x - cx) * (y - cy))
    tr = 0.5 * (Ixx + Iyy); d = np.sqrt(max(0.25 * (Ixx - Iyy)**2 + Ixy**2, 0.0))
    l1, l2 = tr + d, tr - d                     # eigenvalues of the moment tensor
    a, b = np.sqrt(max(l1, 0)), np.sqrt(max(l2, 0))
    D = (a - b) / (a + b + 1e-30)
    th = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
    return float(D), float(th)


def run(N=48, Ca=0.1, model="elastic", tau=np.inf, G_ve=None, mu_f=0.1, U=0.5,
        R=0.15, rho=1.0, Lx=2.0, Ly=1.0, t_end=12.0, scheme="semilagrangian_cubic",
        w_cut_fac=4.0, history=False, return_fields=False):
    Nx, Ny = int(Lx * N), int(Ly * N)
    dx, dy = mac_grid(Nx, Ny, Lx, Ly)
    gdot = 2.0 * U / Ly
    mu_s = mu_f * gdot / Ca                      # so that Ca = mu_f*gdot/mu_s
    G_ve = mu_s if G_ve is None else G_ve
    xc = (np.arange(Nx) + 0.5) * dx; yc = (np.arange(Ny) + 0.5) * dy
    Xc, Yc = np.meshgrid(xc, yc)                 # (Ny,Nx)
    Xg, Yg = np.meshgrid(np.arange(Nx) * dx, np.arange(Ny) * dy)
    x0, y0 = 0.5 * Lx, 0.5 * Ly
    phi_init = lambda X, Y: np.sqrt((X - x0)**2 + (Y - y0)**2) - R
    phi = phi_init(Xc, Yc); sm = (phi <= 0).astype(float)
    X1, X2 = extrapolate_reference_map(Xc * sm, Yc * sm, phi, dx, dy, 3)
    p11 = np.zeros((Ny, Nx)); p12 = np.zeros((Ny, Nx)); p22 = np.zeros((Ny, Nx))
    xf = np.arange(Nx + 1) * dx; yf = np.arange(Ny + 1) * dy
    Xu, Yu = np.meshgrid(xf, yc); Xv, Yv = np.meshgrid(xc, yf)
    u = U * (2.0 * Yu / Ly - 1.0)                # linear shear initial guess
    v = np.zeros((Ny + 1, Nx))
    u[:, 0] = 0.0; u[:, -1] = 0.0               # no-slip side walls (normal vel)
    nu = mu_f / rho; cs = np.sqrt(mu_s / rho); w_t = 2.0 * dx
    eig = poisson_eigs_neumann(Nx, Ny, dx, dy)
    dt = min(0.25 * dx / (abs(U) + 1e-9), 0.2 * dx * dx / nu, 0.25 * dx / (cs + 1e-9))
    has_mx = model in ("maxwell", "sls")
    hist = {"t": [], "D": [], "th": []}
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
        eqxx, eqxy, eqyy, J = solid_cauchy_stress(X1, X2, dx, dy, mu_s, 0.0, phi)
        if has_mx:
            p11 = advect_reference_map(p11, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
            p12 = advect_reference_map(p12, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
            p22 = advect_reference_map(p22, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
            p11, p12 = extrapolate_reference_map(p11, p12, phi, dx, dy, 3)
            p22, _ = extrapolate_reference_map(p22, np.zeros((Ny, Nx)), phi, dx, dy, 3)
            L11 = grad_central_x_2nd(u_c, dx); L12 = grad_central_y_2nd(u_c, dy)
            L21 = grad_central_x_2nd(v_c, dx); L22 = grad_central_y_2nd(v_c, dy)
            p11, p12, p22 = logconf_local_step_strang(p11, p12, p22, L11, L12, L21, L22, tau, dt)
            be11, be12, be22 = sym_exp(p11, p12, p22); trbe = 0.5 * (be11 + be22)
            mxx = G_ve * (be11 - trbe); mxy = G_ve * be12; myy = G_ve * (be22 - trbe)
        if model == "elastic":
            sxx, sxy, syy = eqxx, eqxy, eqyy
        elif model == "maxwell":
            sxx, sxy, syy = mxx, mxy, myy
        else:
            sxx, sxy, syy = eqxx + mxx, eqxy + mxy, eqyy + myy
        H = smoothed_heaviside(phi, w_t)
        Sxx = (1 - H) * sxx; Sxy = (1 - H) * sxy; Syy = (1 - H) * syy
        divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dy)
        divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dy)
        fu = np.zeros((Ny, Nx + 1)); fu[:, 1:-1] = 0.5 * (divx[:, 1:] + divx[:, :-1])
        fv = np.zeros((Ny + 1, Nx)); fv[1:-1, :] = 0.5 * (divy[1:, :] + divy[:-1, :])
        us, vs = momentum_predictor(u, v, nu, dx, dy, dt, U, fu=fu, fv=fv, rho=rho, U_bot=-U)
        u, v, p = project(us, vs, dx, dy, dt, rho, eig)
        t += dt
        if not (np.all(np.isfinite(u)) and J.min() > 0):
            print(f"  [diverged/folded at t={t:.2f}, minJ={J.min():.2f}]"); break
        if history:
            D, th = deformation(phi, Xc, Yc, dx, dy)
            hist["t"].append(t); hist["D"].append(D); hist["th"].append(th)
    D, th = deformation(phi, Xc, Yc, dx, dy)
    out = dict(N=N, Ca=Ca, D=D, th=th, mu_s=mu_s, gdot=gdot, Wi=tau * gdot if np.isfinite(tau) else np.inf,
               divmax=float(np.max(np.abs(divergence(u, v, dx, dy)))), minJ=float(J.min()))
    if history:
        out["hist"] = {k: np.array(val) for k, val in hist.items()}
    if return_fields:
        u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
        out["fields"] = dict(phi=phi, speed=np.sqrt(u_c**2 + v_c**2), u=u_c, v=v_c,
                             Xc=Xc, Yc=Yc, dx=dx, dy=dy, Lx=Lx, Ly=Ly)
    return out


def main():
    import matplotlib; matplotlib.use("Agg")
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    TEND, SCH = 4.0, "semilagrangian"
    # (a) elastic D vs Ca (grid-converged); orientation theta
    Cas = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    el = [run(N=48, Ca=c, model="elastic", t_end=TEND, scheme=SCH) for c in Cas]
    D_el = [r["D"] for r in el]; th_el = [np.degrees(r["th"]) for r in el]
    print("elastic D vs Ca:", [(c, f"{d:.3f}") for c, d in zip(Cas, D_el)])
    # grid convergence at Ca=0.2
    conv = {Nn: run(N=Nn, Ca=0.2, model="elastic", t_end=TEND, scheme=SCH)["D"] for Nn in (32, 48, 64)}
    print("grid conv (Ca=0.2):", {k: f"{v:.4f}" for k, v in conv.items()})
    # (b) viscoelastic (SLS) at Ca=0.2: steady D vs Weissenberg number
    Ca_v = 0.2; gdot = 2 * 0.5 / 1.0
    Wis = [0.0, 0.25, 0.5, 1.0, 2.0]
    ve = []
    for Wi in Wis:
        tau = np.inf if Wi == 0 else Wi / gdot
        # SLS with G=mu_s(Ca_v); Wi=0 is the elastic limit (tau->inf handled by model)
        m = "elastic" if Wi == 0 else "sls"
        ve.append(run(N=48, Ca=Ca_v, model=m, tau=(np.inf if Wi == 0 else tau), t_end=TEND, scheme=SCH)["D"])
    print("SLS D vs Wi (Ca=0.2):", [(w, f"{d:.3f}") for w, d in zip(Wis, ve)])

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.2))
    ax[0].plot(Cas, D_el, "o-", color="C0", label="RMT (elastic disc)")
    ax[0].plot([0, max(Cas)], [0, max(Cas)], "k--", lw=1, label=r"linear limit $D=\mathrm{Ca}$")
    ax[0].set_xlabel(r"$\mathrm{Ca}=\mu_f\dot\gamma/G$"); ax[0].set_ylabel(r"deformation $D=(a-b)/(a+b)$")
    ax[0].set_title("(a) elastic disc in shear"); ax[0].legend(loc="upper left")
    axr = ax[0].twinx()
    axr.plot(Cas, th_el, "s:", color="C3", ms=4, alpha=0.8)
    axr.set_ylabel(r"orientation $\theta$ (deg)", color="C3"); axr.set_ylim(0, 50)
    axr.axhline(45, ls=":", color="C3", lw=0.7)

    ax[1].plot(Wis, ve, "^-", color="C1")
    ax[1].set_xlabel(r"$\mathrm{Wi}=\dot\gamma\,\tau$")
    ax[1].set_ylabel(r"steady $D$ at $\mathrm{Ca}=0.2$")
    ax[1].set_title("(b) viscoelastic (SLS) disc")
    ax[1].text(0.5, 0.06, r"$\mathrm{Wi}=0$: elastic limit", transform=ax[1].transAxes,
               ha="center", fontsize=9)

    fig.tight_layout()
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data, exist_ok=True)
    out = os.path.join(data, "disc_in_shear.png"); fig.savefig(out); print("saved", out)
    return dict(Cas=Cas, D_el=D_el, th_el=th_el, conv=conv, Wis=Wis, ve=ve)


if __name__ == "__main__":
    main()
