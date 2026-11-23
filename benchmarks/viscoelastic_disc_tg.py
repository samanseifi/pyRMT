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

from pyRMT.mac import (mac_grid, momentum_predictor_freeslip,
                       momentum_predictor_freeslip_imex, project,
                       poisson_eigs_neumann, divergence, cell_velocity_gradient,
                       solid_force_faces)
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map, advance_conformation,
    rebuild_phi_from_reference_map, solid_cauchy_stress, smoothed_heaviside,
    grad_central_x_2nd, grad_central_y_2nd)
from pyRMT.viscoelastic import logconf_local_step_strang, sym_exp
from pyRMT.output import compute_strain_energy

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "ve_disc_tg")
os.makedirs(OUT, exist_ok=True)


def _FFt_from_xi(X1, X2, dx, dy):
    """Left Cauchy-Green b=F F^T (F=(grad xi)^{-1}) and ln det b, from the reference map."""
    g11 = grad_central_x_2nd(X1, dx); g12 = grad_central_y_2nd(X1, dy)
    g21 = grad_central_x_2nd(X2, dx); g22 = grad_central_y_2nd(X2, dy)
    det = g11 * g22 - g12 * g21
    det = np.where(np.abs(det) < 1e-9, np.sign(det) * 1e-9 + (det == 0), det)
    F11 = g22 / det; F12 = -g12 / det; F21 = -g21 / det; F22 = g11 / det
    b11 = F11 * F11 + F12 * F12
    b12 = F11 * F21 + F12 * F22
    b22 = F21 * F21 + F22 * F22
    lndet = -2.0 * np.log(np.abs(det))
    return b11, b12, b22, lndet


def run_one(N, t_end, tau=np.inf, G_ve=0.1, R=0.15, x0=0.5, y0=0.7, mu_s=0.1,
            mu_f=0.01, U0=0.3, rho=1.0, scheme="semilagrangian_cubic", w_cut_fac=4.0,
            model="maxwell", history=False, snap_times=None, kw=1, integrator="explicit",
            trace_free_L=True, compact_force=True, band_stress=False, consistent_mask=True,
            dt_scale=1.0, band_width=None, band_extrap=None, material_band=False):
    """One disc-in-Taylor-Green FSI run. ``model`` selects the solid constitution:
        "elastic" -- reference-map neo-Hookean, sigma = mu_s dev(F F^T)  (Sec. 8.2)
        "maxwell" -- pure Maxwell branch,       sigma = G dev(b_e)       (fluid drop)
        "sls"     -- standard linear solid,     sigma = mu_s dev(F F^T) + G dev(b_e)
    The SLS ("sls") is the model defined in the paper (equilibrium branch + relaxing
    branch); "elastic" is the tau->inf reference and "maxwell" the fluid limit."""
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    xf = np.arange(N + 1) * dx
    Xu, Yu = np.meshgrid(xf, xc); Xv, Yv = np.meshgrid(xc, xf)

    k = kw * np.pi   # kw=1 single vortex (disc orbits); kw=2 four-cell (disc strains at centre)
    u = U0 * k * np.sin(k * Xu) * np.cos(k * Yu)
    v = -U0 * k * np.cos(k * Xv) * np.sin(k * Yv)
    u[:, 0] = 0.0; u[:, -1] = 0.0; v[0, :] = 0.0; v[-1, :] = 0.0

    phi_init = lambda X, Y: np.sqrt((X - x0)**2 + (Y - y0)**2) - R
    phi = phi_init(Xc, Yc); sm = (phi <= 0).astype(float)
    # blend half-width: 2 cells by default; band_width fixes it as a PHYSICAL length, so a
    # refinement study converges to one regularized problem. The extrapolation band must cover
    # phi < w_t (plus a stencil margin).
    w_t = 2.0 * dx if band_width is None else float(band_width)
    # band_extrap (idea 1, Richardson extrapolation in the band width): blend the stress with
    # c*(1-H_w) + (1-c)*(1-H_2w), c = 2 (cancels an O(w) error) or 4/3 (cancels O(w^2)).
    # The weight is signed; both widths must be covered by the extrapolation band.
    c_ex = {None: None, 1: 2.0, 2: 4.0 / 3.0}[band_extrap]
    w_max = w_t if c_ex is None else 2.0 * w_t
    NL = max(3, int(np.ceil(w_max / dx)) + 2)
    # material_band: the reference map and psi are transported (and psi stretched) EVERYWHERE,
    # so the band holds the labels / deformation of the actual neighbouring fluid parcels -- a
    # continuum-defined extension, continuous across the interface -- instead of a layer-by-layer
    # extrapolation that has no limit at a fixed physical band width. psi is reset to 0 far from
    # the solid (beyond the band), where its blend weight is zero anyway.
    if material_band:
        X1, X2 = Xc.copy(), Yc.copy()
    else:
        X1, X2 = extrapolate_reference_map(Xc * sm, Yc * sm, phi, dx, dy, NL)
    p11 = np.zeros((N, N)); p12 = np.zeros((N, N)); p22 = np.zeros((N, N))  # psi=log I=0
    nu = mu_f / rho; cs = np.sqrt(mu_s / rho)
    eig = poisson_eigs_neumann(N, N, dx, dy)
    imex = (integrator == "imex")
    _lim = [0.25 * dx / (U0 * k), 0.25 * dx / (cs + 1e-9)]
    if not imex:                       # explicit also bound by the viscous CFL
        _lim.append(0.2 * dx * dx / nu)
    dt = min(_lim) * dt_scale          # dt_scale < 1: refine the step independently of the grid
    p = np.zeros((N, N))

    area0 = float(np.sum(phi <= 0) * dx * dy)
    areas0 = float(np.sum(1.0 - smoothed_heaviside(phi, w_t)) * dx * dy)  # smooth (diffuse) area
    hist = {"t": [], "KE": [], "Eel": [], "Eel_s": [], "detbe_max": [], "minJ": [], "cx": [], "cy": [],
            "bevsF": [], "area": [], "areas": [], "gphi": []}
    has_maxwell = model in ("maxwell", "sls")
    snaps = []; snap_q = list(snap_times) if snap_times else []
    t = 0.0
    while t < t_end:
        if t + dt > t_end:
            dt = t_end - t
        u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
        phi = rebuild_phi_from_reference_map(X1, X2, phi_init); sm = (phi <= 0).astype(float)
        wc = w_cut_fac * dx
        if material_band:
            X1 = advect_reference_map(X1, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc)
            X2 = advect_reference_map(X2, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc)
        else:
            X1 = advect_reference_map(X1, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
            X2 = advect_reference_map(X2, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
            X1, X2 = extrapolate_reference_map(X1, X2, phi, dx, dy, NL)
        phi = rebuild_phi_from_reference_map(X1, X2, phi_init)
        # equilibrium (neo-Hookean) branch from the reference map -- used by "elastic"/"sls"
        # band_stress (bug 2 fix): evaluate across the blend band, not only where phi <= 0
        eqxx, eqxy, eqyy, J = solid_cauchy_stress(X1, X2, dx, dy, mu_s, 0.0, phi,
                                                  w_cut=(w_max if band_stress else 0.0))

        detbe_max = 1.0
        if has_maxwell:
            if material_band:    # transport everywhere, no cut / extrapolation
                p11 = advect_reference_map(p11, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc)
                p12 = advect_reference_map(p12, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc)
                p22 = advect_reference_map(p22, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc)
            elif consistent_mask:  # bug 4 fix: cut and extrapolate psi with the same (new) level set
                p11, p12, p22 = advance_conformation(p11, p12, p22, u_c, v_c, Xg, Yg, dt, dx, dy,
                                                     phi, scheme, wc, NL)
            else:                # former: start-of-step mask, new level set -> psi=0 strip
                p11 = advect_reference_map(p11, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
                p12 = advect_reference_map(p12, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
                p22 = advect_reference_map(p22, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
                p11, p12 = extrapolate_reference_map(p11, p12, phi, dx, dy, NL)
                p22, _ = extrapolate_reference_map(p22, np.zeros((N, N)), phi, dx, dy, NL)
            if trace_free_L:     # tr(L) = MAC divergence = 0 (bug 1 fix)
                L11, L12, L21, L22 = cell_velocity_gradient(u, v, dx, dy)
            else:                # former construction: trace is not zero (det b_e drifts)
                L11 = grad_central_x_2nd(u_c, dx); L12 = grad_central_y_2nd(u_c, dy)
                L21 = grad_central_x_2nd(v_c, dx); L22 = grad_central_y_2nd(v_c, dy)
            p11, p12, p22 = logconf_local_step_strang(p11, p12, p22, L11, L12, L21, L22, tau, dt)
            if material_band:
                far = phi > w_max + 4.0 * dx
                p11[far] = 0.0; p12[far] = 0.0; p22[far] = 0.0
            be11, be12, be22 = sym_exp(p11, p12, p22)
            trbe = 0.5 * (be11 + be22)
            mxx = G_ve * (be11 - trbe); mxy = G_ve * be12; myy = G_ve * (be22 - trbe)
            detbe = np.exp(p11 + p22)                     # det(exp psi)=exp(tr psi)
            detbe_max = float(np.max(detbe[phi <= 0])) if np.any(phi <= 0) else 1.0

        # assemble the solid Cauchy stress for the chosen model
        if model == "elastic":
            sxx, sxy, syy = eqxx, eqxy, eqyy
        elif model == "maxwell":
            sxx, sxy, syy = mxx, mxy, myy
        else:  # sls: equilibrium + relaxing branch
            sxx, sxy, syy = eqxx + mxx, eqxy + mxy, eqyy + myy

        H = smoothed_heaviside(phi, w_t)
        if c_ex is None:
            wgt = 1 - H
        else:
            wgt = c_ex * (1 - H) + (1 - c_ex) * (1 - smoothed_heaviside(phi, 2.0 * w_t))
        Sxx = wgt * sxx; Sxy = wgt * sxy; Syy = wgt * syy
        if compact_force:    # bug 3 fix: compact face divergence (isotropic part = MAC gradient)
            fu, fv = solid_force_faces(Sxx, Sxy, Syy, dx, dy)
        else:
            divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dy)
            divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dy)
            fu = np.zeros((N, N + 1)); fu[:, 1:-1] = 0.5 * (divx[:, 1:] + divx[:, :-1])
            fv = np.zeros((N + 1, N)); fv[1:-1, :] = 0.5 * (divy[1:, :] + divy[:-1, :])
        if imex:
            us, vs = momentum_predictor_freeslip_imex(u, v, nu, dx, dy, dt, fu=fu, fv=fv, rho=rho)
        else:
            us, vs = momentum_predictor_freeslip(u, v, nu, dx, dy, dt, fu=fu, fv=fv, rho=rho)
        u, v, p = project(us, vs, dx, dy, dt, rho, eig)
        t += dt

        while snap_q and t >= snap_q[0]:
            uc2 = 0.5 * (u[:, :-1] + u[:, 1:]); vc2 = 0.5 * (v[:-1, :] + v[1:, :])
            snaps.append(dict(t=t, phi=phi.copy(), speed=np.sqrt(uc2**2 + vc2**2),
                              uc=uc2.copy(), vc=vc2.copy(),
                              field=(np.exp(p11 + p22) if has_maxwell else J).copy(),
                              Xc=Xc, Yc=Yc)); snap_q.pop(0)

        if history:
            u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
            KE = 0.5 * np.sum(u_c**2 + v_c**2) * dx * dy
            msk = phi <= 0
            fb11, fb12, fb22, flndet = _FFt_from_xi(X1, X2, dx, dy)   # b = F F^T
            # stored elastic energy w = (G/2)(tr b - 2 - ln det b), same functional for
            # both strain descriptions; SLS = equilibrium (from F) + relaxing (from b_e)
            Eel = 0.0; bevsF = 0.0; Eel_s = 0.0
            wsm = 1.0 - smoothed_heaviside(phi, w_t)       # smooth solid weight (sub-cell)
            if model in ("elastic", "sls"):
                w_eq = 0.5 * mu_s * ((fb11 + fb22) - 2.0 - flndet)
                Eel += float(np.sum(np.where(msk, w_eq, 0.0)) * dx * dy)
                Eel_s += float(np.sum(wsm * w_eq) * dx * dy)
            if has_maxwell:
                be11, be12, be22 = sym_exp(p11, p12, p22)
                w_mx = 0.5 * G_ve * ((be11 + be22) - 2.0 - (p11 + p22))
                Eel += float(np.sum(np.where(msk, w_mx, 0.0)) * dx * dy)
                Eel_s += float(np.sum(wsm * w_mx) * dx * dy)
                # strain-description drift: ||exp(psi) - F F^T||_F / ||F F^T||_F over the
                # REGULARLY-deformed solid interior (0.5<J<2), so the metric measures the
                # genuine transport difference between the two representations of F F^T and
                # not the folding blow-up of F F^T (where b_e correctly stays bounded).
                num = (be11 - fb11)**2 + 2 * (be12 - fb12)**2 + (be22 - fb22)**2
                den = fb11**2 + 2 * fb12**2 + fb22**2 + 1e-30
                mi = (phi < -2.0 * dx) & (J > 0.5) & (J < 2.0)
                bevsF = float(np.sqrt(np.sum(num[mi]) / np.sum(den[mi]))) if mi.any() else 0.0
            # min J over the solid INTERIOR (exclude the diffuse blend band, where the
            # non-stress-bearing reference map is extrapolated and J is meaningless)
            mi = phi < -2.0 * dx
            minJ = float(J[mi].min()) if mi.any() else 1.0
            hist["t"].append(t); hist["KE"].append(KE); hist["Eel"].append(Eel); hist["Eel_s"].append(Eel_s)
            hist["detbe_max"].append(detbe_max); hist["minJ"].append(minJ)
            hist["bevsF"].append(bevsF); hist["area"].append(float(np.sum(msk) * dx * dy) / area0)
            hist["areas"].append(float(np.sum(1.0 - H) * dx * dy) / areas0)  # smooth area (no quant. jitter)
            # interface-band distortion: for the level set rebuilt from the reference map, |grad phi|
            # should stay ~1 (near-signed-distance). Measure the MEDIAN deviation from 1 in the band
            # |phi|<2dx -- robust to isolated kink cells (the diffuse-band diagnostic, referee C4).
            gpx = grad_central_x_2nd(phi, dx); gpy = grad_central_y_2nd(phi, dy)
            band = np.abs(phi) < 2.0 * dx
            hist["gphi"].append(float(np.median(np.abs(np.sqrt(gpx**2 + gpy**2)[band] - 1.0)))
                                if band.any() else 0.0)
            hist["cx"].append(float(Xc[msk].mean()) if msk.any() else np.nan)
            hist["cy"].append(float(Yc[msk].mean()) if msk.any() else np.nan)

    u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
    speed = np.sqrt(u_c**2 + v_c**2)
    KE = 0.5 * np.sum(u_c**2 + v_c**2) * dx * dy
    out = dict(N=N, dx=dx, xc=xc, speed=speed, p=p, KE=KE, phi=phi,
               divmax=float(np.max(np.abs(divergence(u, v, dx, dy)))))
    if history:
        out["hist"] = {k: np.array(val) for k, val in hist.items()}
    if snap_times:
        out["snaps"] = snaps
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

    # (a) mesh convergence of the SLS model at fixed finite tau
    NS = (32, 64, 128, 256); tau_c = 1.0; tconv = 0.6
    conv = {N: run_one(N, tconv, tau=tau_c, model="sls") for N in NS}
    g = conv[NS[-1]]["xc"]; ref = NS[-1]; coarse = list(NS[:-1])
    spref = _sample(conv[ref]["xc"], conv[ref]["speed"], g)
    prref = _sample(conv[ref]["xc"], conv[ref]["p"], g)
    verr = [np.sqrt(np.mean((_sample(conv[N]["xc"], conv[N]["speed"], g) - spref)**2)) for N in coarse]
    perr = [np.sqrt(np.mean((_sample(conv[N]["xc"], conv[N]["p"], g) - prref)**2)) for N in coarse]
    keerr = [abs(conv[N]["KE"] - conv[ref]["KE"]) for N in coarse]
    print("SLS disc-in-TG convergence (tau=%.2g):" % tau_c)
    print("  velocity order", [f"{o:.2f}" for o in _order(coarse, verr)])
    print("  pressure order", [f"{o:.2f}" for o in _order(coarse, perr)])

    # SLS histories at fixed N (the titled model: equilibrium + relaxing branch)
    Nh, tend = 64, 1.2
    taus = [np.inf, 1.0, 0.25]
    sls = {tau: run_one(Nh, tend, tau=tau, model="sls", history=True)["hist"] for tau in taus}
    # tau->inf recovery check + strain drift: pure Maxwell (G=mu_s) vs elastic reference
    mw_inf = run_one(Nh, tend, tau=np.inf, model="maxwell", G_ve=0.1, history=True)["hist"]
    base = run_one(Nh, tend, model="elastic", history=True)["hist"]
    rec = np.max(np.abs(mw_inf["KE"] - base["KE"])) / np.max(base["KE"])
    drift = mw_inf["bevsF"][-1]
    print(f"maxwell(tau=inf) vs elastic: max|dKE|/KE = {rec:.2e}, final ||exp psi - FF^T||/||FF^T|| = {drift:.2e}")

    labels = {np.inf: r"$\tau=\infty$", 1.0: r"$\tau=1$", 0.25: r"$\tau=0.25$"}
    cols = {np.inf: "C0", 1.0: "C1", 0.25: "C3"}
    h = np.array([1.0 / N for N in coarse])
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data, exist_ok=True)

    # ===== MAIN figure: the two physics panels =====
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.4))
    # (a) mesh convergence
    a = ax[0]
    a.loglog(h, verr, "o-", label="velocity"); a.loglog(h, perr, "s-", label="pressure")
    a.loglog(h, keerr, "^-", label="kinetic energy")
    a.loglog(h, verr[0]*(h/h[0])**2, "k--", lw=1, label="2nd order")
    a.loglog(h, perr[0]*(h/h[0])**1, "k:", lw=1, label="1st order")
    a.invert_xaxis(); a.set_xlabel("$h=1/N$"); a.set_ylabel("$L_2$ error vs finest")
    a.set_title(r"(a) SLS mesh convergence ($\tau=1$)")
    a.legend(ncol=2, loc="upper left", fontsize=9)
    # (b) stored elastic energy: SLS relaxing branch dissipates, equilibrium persists
    a = ax[1]
    for tau in taus:
        a.plot(sls[tau]["t"], sls[tau]["Eel"], "-", color=cols[tau], label="SLS "+labels[tau])
    a.plot(base["t"], base["Eel"], "k--", lw=1.3, label="elastic only")
    a.set_xlabel("time $t$"); a.set_ylabel("stored elastic energy")
    a.set_title("(b) SLS: relaxation vs shape memory"); a.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = os.path.join(data, "viscoelastic_disc_tg.png"); fig.savefig(out); print("saved", out)
    plt.close(fig)

    # ===== SUPPLEMENTARY figure: solver diagnostics =====
    figd, axd = plt.subplots(1, 2, figsize=(11.0, 4.4))
    # (a) det b_e bounded and solid area preserved (interface integrity)
    a = axd[0]
    for tau in taus:
        a.plot(sls[tau]["t"], sls[tau]["detbe_max"], "-", color=cols[tau], label=labels[tau])
    a.axhline(1.0, ls=":", color="0.6", lw=1)
    a.set_xlabel("time $t$"); a.set_ylabel(r"$\max\ \det b_e$ (solid)")
    ar = a.twinx()
    for tau in taus:
        ar.plot(sls[tau]["t"], sls[tau]["area"], "--", color=cols[tau], lw=1.1, alpha=0.8)
    ar.set_ylabel("solid area / initial (dashed)"); ar.set_ylim(0.8, 1.2)
    a.set_title(r"(a) $\det b_e$ and solid area"); a.legend(loc="upper left", fontsize=9)
    # (b) strain-description drift at tau=inf, where exp(psi) should equal F F^T
    a = axd[1]
    a.plot(mw_inf["t"], mw_inf["bevsF"], "-", color="C0", lw=2)
    a.set_xlabel("time $t$")
    a.set_ylabel(r"$\|\exp\psi - FF^\top\|/\|FF^\top\|$ (solid interior)")
    a.set_title(r"(b) strain-description drift ($\tau=\infty$)")
    a.text(0.5, 0.5, "transported $b_e$ vs reference-map $FF^\\top$:\n"
           "same tensor, two discretizations.\n"
           r"dynamics still agree to $%.1f\times10^{-4}$ (KE)." % (rec * 1e4),
           transform=a.transAxes, ha="center", va="center", fontsize=9,
           bbox=dict(boxstyle="round", fc="0.95", ec="0.8"))
    figd.tight_layout()
    outd = os.path.join(data, "viscoelastic_disc_tg_diag.png"); figd.savefig(outd); print("saved", outd)
    plt.close(figd)
    return dict(verr=verr, perr=perr, keerr=keerr, coarse=coarse, recovery=rec, drift=drift)


if __name__ == "__main__":
    main()
