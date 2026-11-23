"""Two soft discs colliding head-on with Rycroft's trace-free contact STRESS, in a
periodic box. Supports both constitutive regimes and both integrators:

  model      : "elastic" (neo-Hookean sigma = mu_s dev(F F^T))
               "sls"     (standard linear solid: + relaxing Maxwell branch G dev(b_e))
  integrator : "explicit" (forward Euler)  |  "imex" (implicit viscosity + elastic-wave
               stabilizer; contact/solid force in the implicit RHS)

The contact stress is added to the blended solid stress before taking its divergence, so
its momentum-conserving force survives the exact projection (a repulsive body force would
be nullified). Elastic discs rebound; the viscoelastic Maxwell branch dissipates part of
the collision energy, so they rebound more weakly.

Usage: python benchmarks/contact_collision.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import (mac_grid, momentum_predictor_periodic,
                       momentum_predictor_periodic_imex_elastic, project_per,
                       poisson_eigs_periodic, lap_eigs_periodic, divergence, contact_stress,
                       cell_velocity_gradient, solid_force_faces)
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map,
    rebuild_phi_from_reference_map, solid_cauchy_stress, smoothed_heaviside,
    grad_central_x_2nd, grad_central_y_2nd)
from pyRMT.viscoelastic import logconf_local_step_strang, sym_exp



# Bug 1 fix: build the strain-update velocity gradient so that tr(L) is the MAC divergence.
TRACE_FREE_L = True

# Bug fixes 2-3 (see tests/test_solid_force.py). Bug 4 does not occur here: psi is cut and
# extrapolated with the same (start-of-step) level set.
BAND_STRESS = False       # bug 2 fix: evaluate the reference-map stress across the blend band.
                          # Off: with the layer-by-layer extrapolation the band reference map is
                          # too noisy in deforming flows (J -> 0.12 in the lid case); it needs a
                          # well-defined band field (see material_band in viscoelastic_disc_tg).
COMPACT_FORCE = True      # compact face divergence of the stress (bug 3)

def _surface_gap(pa, pb, xc):
    """Distance between the two interfaces on the line of centres: right-most zero of phi_a
    and left-most zero of phi_b on the two rows adjacent to y = 0.5 (linear interpolation)."""
    j = pa.shape[0] // 2
    a = 0.5 * (pa[j - 1] + pa[j]); b = 0.5 * (pb[j - 1] + pb[j])
    ia = np.where((a[:-1] <= 0) & (a[1:] > 0))[0]; ib = np.where((b[:-1] > 0) & (b[1:] <= 0))[0]
    if len(ia) == 0 or len(ib) == 0:
        return np.nan
    i = ia[-1]; xa_s = xc[i] + (xc[i + 1] - xc[i]) * (0.0 - a[i]) / (a[i + 1] - a[i])
    i = ib[0];  xb_s = xc[i] + (xc[i + 1] - xc[i]) * (0.0 - b[i]) / (b[i + 1] - b[i])
    return xb_s - xa_s


def run(N=128, t_end=2.4, V0=0.3, eta=2.0, mu_s=0.2, G_ve=0.6, tau=0.4,
        model="elastic", integrator="explicit", mu_f=0.02, rho=1.0, R=0.12, sep=0.34,
        scheme="semilagrangian_cubic", history=True, dt=None, snap_times=None, stab=True,
        eps=None, eps_cells=None, band_layers=None, ordering="sequential", Rb=None, V0b=None):
    """Contact-model controls.

    eps         : PHYSICAL activation width of the contact stress (a length; default R/4). The
                  contact law is then the same on every grid, so a refinement study converges.
                  The result does depend on eps, which is a model parameter (report it).
    eps_cells   : if given, overrides eps with eps_cells*dx -- the ORIGINAL mesh-tied width. The
                  contact range then shrinks with the mesh and the rebound does not converge
                  (end gap 0.339/0.308/0.284 at N=96/128/192 for the convergence case). Kept
                  only to reproduce the old runs: eps_cells=3, band_layers=3, ordering="midpoint".
    band_layers : extrapolation layers of the reference map outside each solid. The contact stress
                  needs phi (and its gradient) out to phi = eps, so the band must reach at least
                  eps/dx + 2 cells; None -> max(3, ceil(eps/dx) + 2).
    ordering    : "sequential" (default) = transport the solids with u^n, then evaluate the force
                  from the updated maps (symplectic-Euler ordering, as in the other drivers);
                  "midpoint" = force from the start-of-step maps, then transport with
                  (u^n+u^{n+1})/2, which without the implicit elastic term is weakly
                  anti-dissipative: |lambda|^2 = 1 + (cs k dt)^2/2 per step.
    Rb, V0b     : radius and approach speed of disc B if different from disc A (R, V0); used for
                  the momentum-conservation check, where the total momentum is non-zero.

    History columns: t, ca, cb, centroid gap, KE, total x-momentum, total y-momentum,
    elastic energy, surface gap on the line of centres.
    """
    L = 1.0
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    xf = np.arange(N) * dx   # periodic staggered: u,v are (N,N)
    Xu, Yu = np.meshgrid(xf, xc); Xv, Yv = np.meshgrid(xc, np.arange(N) * dy)
    xa, xb, y0 = 0.5 - sep / 2, 0.5 + sep / 2, 0.5
    pia = lambda X, Y: np.sqrt((X - xa)**2 + (Y - y0)**2) - R
    Rb = R if Rb is None else Rb; V0b = V0 if V0b is None else V0b
    pib = lambda X, Y: np.sqrt((X - xb)**2 + (Y - y0)**2) - Rb
    pa = pia(Xc, Yc); pb = pib(Xc, Yc)
    ma = (pa <= 0).astype(float); mb = (pb <= 0).astype(float)
    sls = (model == "sls")
    if sls:
        pa11 = np.zeros((N, N)); pa12 = np.zeros((N, N)); pa22 = np.zeros((N, N))
        pb11 = np.zeros((N, N)); pb12 = np.zeros((N, N)); pb22 = np.zeros((N, N))
    w_t = 2.0 * dx; nu = mu_f / rho
    if eps_cells is not None:
        eps = float(eps_cells) * dx                    # original mesh-tied width
    elif eps is None:
        eps = 0.25 * R
    eps = float(eps)
    nl = max(3, int(np.ceil(eps / dx)) + 2) if band_layers is None else int(band_layers)
    X1a, X2a = extrapolate_reference_map(Xc * ma, Yc * ma, pa, dx, dy, nl)
    X1b, X2b = extrapolate_reference_map(Xc * mb, Yc * mb, pb, dx, dy, nl)
    Gtot = mu_s + (G_ve if sls else 0.0)      # total small-strain modulus per disc
    cs = np.sqrt(Gtot / rho)
    # head-on approach: A -> +x, B -> -x (zeroed in fluid)
    HaU = smoothed_heaviside(pia(Xu, Yu), w_t); HbU = smoothed_heaviside(pib(Xu, Yu), w_t)
    u = V0 * (1 - HaU) - V0b * (1 - HbU)
    v = np.zeros((N, N))
    peig = poisson_eigs_periodic(N, N, dx, dy)
    leig = lap_eigs_periodic(N, N, dx, dy)
    imex = (integrator == "imex")
    assert ordering in ("sequential", "midpoint")
    seq = (ordering == "sequential")
    dt_adv = 0.25 * dx / max(V0, V0b, 1e-9); dt_visc = 0.2 * dx * dx / nu
    dt_el = 0.25 * dx / (cs + 1e-9)
    if dt is None:
        dt = dt_adv if imex else min(dt_adv, dt_visc, dt_el)
    print(f"[contact] N={N} model={model} integrator={integrator} eta={eta} "
          f"mu_s={mu_s} G={G_ve if sls else 0} tau={tau if sls else '-'} dt={dt:.2e} "
          f"eps={eps:.4f} ({eps/dx:.2f} dx) band={nl} ordering={ordering}")

    def sigma_disc(X1, X2, phi, p11, p12, p22):
        """Solid Cauchy stress from the CURRENT state (no transport/stretch here), and the
        stored elastic energy of the body, W = (mu/2)(tr b - 2 - ln det b) per branch."""
        exx, exy, eyy, J = solid_cauchy_stress(X1, X2, dx, dy, mu_s, 0.0, phi,
                                               w_cut=(w_t if BAND_STRESS else 0.0))
        # tr b and det b from the reference-map gradient directly (independent of the stress form):
        # grad xi = G, F = G^{-1}, tr(F F^T) = |G|_F^2 / det(G)^2, det b = det(G)^-2
        g11 = (np.roll(X1, -1, 1) - np.roll(X1, 1, 1)) / (2 * dx); g12 = (np.roll(X1, -1, 0) - np.roll(X1, 1, 0)) / (2 * dy)
        g21 = (np.roll(X2, -1, 1) - np.roll(X2, 1, 1)) / (2 * dx); g22 = (np.roll(X2, -1, 0) - np.roll(X2, 1, 0)) / (2 * dy)
        dG = g11 * g22 - g12 * g21
        inside = (phi <= 0) & (dG > 0)
        trb = (g11**2 + g12**2 + g21**2 + g22**2)[inside] / dG[inside]**2
        W = 0.5 * mu_s * np.sum(trb - 2.0 + 2.0 * np.log(dG[inside])) * dx * dy
        if not sls:
            return exx, exy, eyy, W
        be11, be12, be22 = sym_exp(p11, p12, p22)
        trbe = 0.5 * (be11 + be22)
        W += 0.5 * G_ve * np.sum((be11 + be22 - 2.0 - (p11 + p22))[inside]) * dx * dy
        return exx + G_ve * (be11 - trbe), exy + G_ve * be12, eyy + G_ve * (be22 - trbe), W

    def advance_solid(uf, vf):
        """Transport the reference maps (and, for SLS, psi with its stretch/relax step)
        by the cell velocity (uc,vc). Explicit uses the start velocity; the IMEX
        (trapezoidal) coupling uses the implicit-midpoint velocity, which is lossless."""
        nonlocal X1a, X2a, X1b, X2b, pa11, pa12, pa22, pb11, pb12, pb22
        uc = 0.5 * (uf + np.roll(uf, -1, axis=1)); vc = 0.5 * (vf + np.roll(vf, -1, axis=0))
        pa_ = rebuild_phi_from_reference_map(X1a, X2a, pia); ma_ = (pa_ <= 0).astype(float)
        pb_ = rebuild_phi_from_reference_map(X1b, X2b, pib); mb_ = (pb_ <= 0).astype(float)
        X1a = advect_reference_map(X1a, uc, vc, Xg, Yg, dt, dx, dy, pa_, scheme, 0.0) * ma_
        X2a = advect_reference_map(X2a, uc, vc, Xg, Yg, dt, dx, dy, pa_, scheme, 0.0) * ma_
        X1b = advect_reference_map(X1b, uc, vc, Xg, Yg, dt, dx, dy, pb_, scheme, 0.0) * mb_
        X2b = advect_reference_map(X2b, uc, vc, Xg, Yg, dt, dx, dy, pb_, scheme, 0.0) * mb_
        X1a, X2a = extrapolate_reference_map(X1a, X2a, pa_, dx, dy, nl)
        X1b, X2b = extrapolate_reference_map(X1b, X2b, pb_, dx, dy, nl)
        if sls:
            pa11 = advect_reference_map(pa11, uc, vc, Xg, Yg, dt, dx, dy, pa_, scheme, 0.0) * ma_
            pa12 = advect_reference_map(pa12, uc, vc, Xg, Yg, dt, dx, dy, pa_, scheme, 0.0) * ma_
            pa22 = advect_reference_map(pa22, uc, vc, Xg, Yg, dt, dx, dy, pa_, scheme, 0.0) * ma_
            pb11 = advect_reference_map(pb11, uc, vc, Xg, Yg, dt, dx, dy, pb_, scheme, 0.0) * mb_
            pb12 = advect_reference_map(pb12, uc, vc, Xg, Yg, dt, dx, dy, pb_, scheme, 0.0) * mb_
            pb22 = advect_reference_map(pb22, uc, vc, Xg, Yg, dt, dx, dy, pb_, scheme, 0.0) * mb_
            if TRACE_FREE_L:     # tr(L) = MAC divergence = 0 (bug 1 fix); periodic box
                L11, L12, L21, L22 = cell_velocity_gradient(uf, vf, dx, dy, periodic=True)
            else:
                L11 = grad_central_x_2nd(uc, dx); L12 = grad_central_y_2nd(uc, dy)
                L21 = grad_central_x_2nd(vc, dx); L22 = grad_central_y_2nd(vc, dy)
            pa11, pa12, pa22 = logconf_local_step_strang(pa11, pa12, pa22, L11, L12, L21, L22, tau, dt)
            pb11, pb12, pb22 = logconf_local_step_strang(pb11, pb12, pb22, L11, L12, L21, L22, tau, dt)
            pa11, pa12 = extrapolate_reference_map(pa11, pa12, pa_, dx, dy, 3)
            pa22, _ = extrapolate_reference_map(pa22, np.zeros_like(pa22), pa_, dx, dy, 3)
            pb11, pb12 = extrapolate_reference_map(pb11, pb12, pb_, dx, dy, 3)
            pb22, _ = extrapolate_reference_map(pb22, np.zeros_like(pb22), pb_, dx, dy, 3)

    t = 0.0; step = 0; hist = []; snaps = []; snap_q = list(snap_times) if snap_times else []
    while t < t_end:
        step += 1
        if t + dt > t_end:
            dt = t_end - t
        u_c = 0.5 * (u + np.roll(u, -1, axis=1)); v_c = 0.5 * (v + np.roll(v, -1, axis=0))
        # sequential ordering: advance the solid state first, force from the updated maps
        if seq:
            advance_solid(u, v)
        pa = rebuild_phi_from_reference_map(X1a, X2a, pia)
        pb = rebuild_phi_from_reference_map(X1b, X2b, pib)

        if sls:
            sAxx, sAxy, sAyy, WA = sigma_disc(X1a, X2a, pa, pa11, pa12, pa22)
            sBxx, sBxy, sByy, WB = sigma_disc(X1b, X2b, pb, pb11, pb12, pb22)
        else:
            sAxx, sAxy, sAyy, WA = sigma_disc(X1a, X2a, pa, None, None, None)
            sBxx, sBxy, sByy, WB = sigma_disc(X1b, X2b, pb, None, None, None)
        Wel = WA + WB

        Ha = smoothed_heaviside(pa, w_t); Hb = smoothed_heaviside(pb, w_t)
        Sxx = (1 - Ha) * sAxx + (1 - Hb) * sBxx
        Sxy = (1 - Ha) * sAxy + (1 - Hb) * sBxy
        Syy = (1 - Ha) * sAyy + (1 - Hb) * sByy
        if eta > 0:
            txx, txy, tyy = contact_stress(pa, pb, eta, 2 * Gtot, eps, dx, dy)
            Sxx += txx; Sxy += txy; Syy += tyy
        if COMPACT_FORCE:
            fu, fv = solid_force_faces(Sxx, Sxy, Syy, dx, dy, periodic=True)
        else:
            divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dy)
            divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dy)
            # periodic x-face i is between cells i-1 and i; y-face j between cells j-1 and j
            fu = 0.5 * (divx + np.roll(divx, 1, axis=1))
            fv = 0.5 * (divy + np.roll(divy, 1, axis=0))

        u_old, v_old = u, v
        if imex:
            chi_c = np.clip((1 - Ha) + (1 - Hb), 0.0, 1.0)
            ustar, vstar = momentum_predictor_periodic_imex_elastic(
                u, v, nu, dx, dy, dt, leig, (cs * cs if stab else 0.0), chi_c, fu, fv, rho=rho)
        else:
            ustar, vstar = momentum_predictor_periodic(u, v, nu, dx, dy, dt, fu=fu, fv=fv, rho=rho)
        u, v, _ = project_per(ustar, vstar, dx, dy, dt, rho, peig)
        # trapezoidal (energy-conserving) coupling: advance the solid with the
        # implicit-midpoint velocity -- lossless, unlike advancing with the start velocity
        if not seq:
            um = 0.5 * (u_old + u); vm = 0.5 * (v_old + v)
            advance_solid(um, vm)
        t += dt
        # smooth (Heaviside-weighted) centroids -- the staircase mean Xc[phi<=0] jumps
        # by ~dx as cells flip in/out of the mask, which corrupts the gap-velocity (and
        # hence the restitution); the smeared solid weight (1-H) gives a sub-cell-smooth gap
        wa = 1 - smoothed_heaviside(pa, w_t); wb = 1 - smoothed_heaviside(pb, w_t)
        ca = (wa * Xc).sum() / wa.sum() if wa.sum() > 1e-12 else np.nan
        cb = (wb * Xc).sum() / wb.sum() if wb.sum() > 1e-12 else np.nan
        ke = 0.5 * rho * np.mean(u_c**2 + v_c**2)
        # total momentum of the periodic box (must stay constant: every force is a divergence)
        px = rho * u.sum() * dx * dy; py = rho * v.sum() * dx * dy
        hist.append((t, ca, cb, cb - ca, ke, px, py, Wel, _surface_gap(pa, pb, xc)))
        while snap_q and t >= snap_q[0]:
            ucp = 0.5 * (u + np.roll(u, -1, axis=1)); vcp = 0.5 * (v + np.roll(v, -1, axis=0))
            snaps.append(dict(t=t, pa=pa.copy(), pb=pb.copy(), uc=ucp, vc=vcp,
                              speed=np.sqrt(ucp**2 + vcp**2), Xc=Xc, Yc=Yc)); snap_q.pop(0)
        if step % 100 == 0 or t >= t_end:
            dmax = np.max(np.abs((np.roll(u, -1, 1) - u) / dx + (np.roll(v, -1, 0) - v) / dy))
            print(f"  step {step:5d} t={t:5.3f} gap={cb-ca:.3f} (2R={2*R:.3f}) "
                  f"KE={ke:.3e} max|div|={dmax:.1e}")
    if snap_times is not None:
        return np.array(hist), snaps
    return np.array(hist)


if __name__ == "__main__":
    run()
