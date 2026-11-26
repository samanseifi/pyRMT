"""RM-IIM stage 3, test 1: does the sharp interface stay clean as it crosses grid cells?

The original sharp RMT (Kamrin, Rycroft & Nave 2012) was abandoned partly because "shots of
pressure along the interface may erroneously appear when the interface crosses through grid cell
boundaries" (Valkov et al. 2015). Here a disc with a uniform internal stretch (so a non-zero,
deviatoric solid stress and both normal and tangential traction) is carried across the periodic
grid by a uniform background flow. The reference map is transported and extrapolated; the jumps
come from the solid stress (pyRMT.iim.traction_jumps). Diagnostics per step: the pressure at a
probe that the interface sweeps over, the perturbation kinetic energy, and max |u - U|.
"Noise" is the RMS of the second time difference of a series relative to its range: a clean
signal has small noise, cell-crossing spikes make it large.

  smeared : the current method (stress blended over 2 cells, compact force, plain projection)
  sharp   : traction imposed as jumps (pressure jump + viscous and divergence corrections)

Small and cheap (N <= 128, ~150 steps); run with 1 thread.
Usage: python benchmarks/iim_moving_disc.py
"""
import os, sys, io, contextlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyRMT.mac import (poisson_eigs_periodic, lap_eigs_periodic, solve_helmholtz_periodic, project_per,
                       momentum_predictor_periodic_imex, solid_force_faces)
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map, solid_cauchy_stress,
                             smoothed_heaviside)
from pyRMT.iim import traction_jumps, sharp_step_per, bulk_solid_force_per, second_order_jumps

U = (0.2, 0.08); NU, RHO, MU_S, STRETCH, R, C0, T_END = 0.05, 1.0, 0.5, 1.15, 0.15, (0.35, 0.45), 1.2
DT_FACTOR = 1.0                     # multiplies the step 0.25 h / |U|
BULK = True                         # include the bulk solid force chi div(sigma) in the sharp step
SECOND = True                       # second-order jumps: [p_n], [grad p] and second-derivative jumps
STRESS_CENTRAL = False              # interface-cell stress from central differences of the extended map
INCREMENTAL = True                  # incremental pressure form (keeps the intermediate velocity continuous)
SWEPT = True                        # move the stored pressure gradient across faces the interface swept
STRESS_EXT = "interior"             # "interior": stress from cells with a full solid stencil, extended
                                    # outward (second order at the interface) | "solid": all solid cells
BULK_MODE = "cells"                 # bulk force on faces with both cells inside | "face": face point inside


def run(N, method, fields=False, snap_times=()):
    h = 1.0 / N; xc = (np.arange(N) + 0.5) * h; X, Y = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * h, np.arange(N) * h)
    phi0 = lambda a, b: np.hypot(a - C0[0], b - C0[1]) - R
    # affine reference map: the disc is already stretched by STRETCH along x (area preserved)
    X1 = C0[0] + (X - C0[0]) / STRETCH; X2 = C0[1] + (Y - C0[1]) * STRETCH
    phi = phi0(X1, X2); m = (phi <= 0).astype(float)
    X1, X2 = extrapolate_reference_map(X1 * m, X2 * m, phi, h, h, 4)
    u = np.full((N, N), U[0]); v = np.full((N, N), U[1])
    leig = lap_eigs_periodic(N, N, h, h); peig = poisson_eigs_periodic(N, N, h, h)
    dt = DT_FACTOR * 0.25 * h / max(np.hypot(*U), 0.215); nstep = int(round(T_END / dt)); dt = T_END / nstep
    jp_, ip_ = int((C0[1]) / h), int((C0[0] + STRETCH * R + 0.03) / h)       # probe ahead of the disc
    hist = []; snaps = []; sq = list(snap_times)
    G = (np.zeros((N, N)), np.zeros((N, N)))       # jump-corrected pressure gradient (incremental form)
    for k in range(nstep):
        uc = 0.5 * (u + np.roll(u, -1, 1)); vc = 0.5 * (v + np.roll(v, -1, 0))
        phi = phi0(X1, X2); m = (phi <= 0).astype(float)
        X1 = advect_reference_map(X1, uc, vc, Xg, Yg, dt, h, h, phi, "semilagrangian_cubic", 0.0) * m
        X2 = advect_reference_map(X2, uc, vc, Xg, Yg, dt, h, h, phi, "semilagrangian_cubic", 0.0) * m
        X1, X2 = extrapolate_reference_map(X1, X2, phi, h, h, 4)
        phi = phi0(X1, X2)
        sxx, sxy, syy, J = solid_cauchy_stress(X1, X2, h, h, MU_S, 0.0, phi, isochoric=True, detg_clamp=3.0,
                                               w_cut=(1e-9 if STRESS_CENTRAL else 0.0))
        ua, va = momentum_predictor_periodic_imex(u, v, 0.0, h, h, dt, leig)     # explicit advection only
        if method == "smeared":
            H = smoothed_heaviside(phi, 2 * h)
            fu, fv = solid_force_faces((1 - H) * sxx, (1 - H) * sxy, (1 - H) * syy, h, h, periodic=True)
            us = solve_helmholtz_periodic(ua + dt * fu / RHO, leig, dt * NU)
            vs = solve_helmholtz_periodic(va + dt * fv / RHO, leig, dt * NU)
            u, v, p = project_per(us, vs, h, h, dt, RHO, peig)
            p = p                                                                # projection scalar = pressure
        else:
            # stress on the solid cells, extended 3 layers outward so it is known at the interface
            if STRESS_EXT == "interior":
                # interface cells use a one-sided (first-order) gradient of the map; keep only cells
                # whose 5-point stencil is inside the solid (central, second order) and extend those
                ins = phi <= 0
                good = ins & np.roll(ins, 1, 0) & np.roll(ins, -1, 0) & np.roll(ins, 1, 1) & np.roll(ins, -1, 1)
                pg = np.where(good, -1.0, 1.0)
                S1, S2 = extrapolate_reference_map(sxx * good, sxy * good, pg, h, h, 5)
                S3, _ = extrapolate_reference_map(syy * good, np.zeros_like(syy), pg, h, h, 5)
            else:
                S1, S2 = extrapolate_reference_map(sxx, sxy, phi, h, h, 3)
                S3, _ = extrapolate_reference_map(syy, np.zeros_like(syy), phi, h, h, 3)
            Jm = traction_jumps(phi, S1, S2, S3, RHO * NU)
            fb = bulk_solid_force_per(phi, S1, S2, S3, h, h, mode=BULK_MODE) if BULK else (None, None)
            J2 = None
            if SECOND:
                dsx = np.gradient(S1, h, axis=1) + np.gradient(S2, h, axis=0)
                dsy = np.gradient(S2, h, axis=1) + np.gradient(S3, h, axis=0)
                J2 = second_order_jumps(phi, Jm, dsx, dsy, mu=RHO * NU, h=h)
            if INCREMENTAL:
                if k == 0:      # initialise the pressure gradient: sweep the first step, discard the velocity
                    for _ in range(3):
                        _, _, _, G = sharp_step_per(ua, va, h, h, dt, NU, RHO, leig, peig, phi, Jm,
                                                    f_u=fb[0], f_v=fb[1], J2=J2, G_old=G)
                u, v, p, G = sharp_step_per(ua, va, h, h, dt, NU, RHO, leig, peig, phi, Jm,
                                            f_u=fb[0], f_v=fb[1], J2=J2, G_old=G)
                if not SWEPT:
                    G = G[:2]
            else:
                u, v, p = sharp_step_per(ua, va, h, h, dt, NU, RHO, leig, peig, phi, Jm, f_u=fb[0], f_v=fb[1], J2=J2)
        hist.append(((k + 1) * dt, p[jp_, ip_] - p.mean(), 0.5 * np.mean((u - U[0]) ** 2 + (v - U[1]) ** 2),
                     max(np.abs(u - U[0]).max(), np.abs(v - U[1]).max()), float(phi[jp_, ip_]),
                     elastic_energy(X1, X2, phi, h)))
        while sq and (k + 1) * dt >= sq[0] - 1e-12:
            ucs = 0.5 * (u + np.roll(u, -1, 1)); vcs = 0.5 * (v + np.roll(v, -1, 0))
            snaps.append(dict(t=sq.pop(0), uc=ucs - U[0], vc=vcs - U[1], p=p - p.mean(), phi=phi0(X1, X2)))
        if not np.all(np.isfinite(u)):
            break
    if snap_times:
        return np.array(hist), snaps
    if not fields:
        return np.array(hist)
    # end-of-run fields for convergence studies (cell-centred speed components, pressure, level set,
    # disc centroid with the smooth weight, stored elastic energy of the disc)
    uc = 0.5 * (u + np.roll(u, -1, 1)); vc = 0.5 * (v + np.roll(v, -1, 0))
    phi = phi0(X1, X2); w = 1 - smoothed_heaviside(phi, 2 * h)
    g11 = (np.roll(X1, -1, 1) - np.roll(X1, 1, 1)) / (2 * h); g12 = (np.roll(X1, -1, 0) - np.roll(X1, 1, 0)) / (2 * h)
    g21 = (np.roll(X2, -1, 1) - np.roll(X2, 1, 1)) / (2 * h); g22 = (np.roll(X2, -1, 0) - np.roll(X2, 1, 0)) / (2 * h)
    dG = g11 * g22 - g12 * g21; ins = (phi <= 0) & (dG > 0)
    trb = np.where(ins, (g11**2 + g12**2 + g21**2 + g22**2) / np.where(ins, dG, 1.0)**2, 2.0)
    Wsm = 0.5 * MU_S * np.sum(w * np.where(ins, trb - 2.0, 0.0)) * h * h
    return np.array(hist), dict(uc=uc - U[0], vc=vc - U[1], p=p - p.mean(), phi=phi,
                                cx=float((w * X).sum() / w.sum()), cy=float((w * Y).sum() / w.sum()), W=Wsm)


def elastic_energy(X1, X2, phi, h):
    """Stored elastic energy of the disc, W = (mu_s/2) int (tr b - 2), b = F F^T from the reference
    map, with the smooth solid weight 1 - H (sub-cell, the same functional for both methods)."""
    g11 = (np.roll(X1, -1, 1) - np.roll(X1, 1, 1)) / (2 * h); g12 = (np.roll(X1, -1, 0) - np.roll(X1, 1, 0)) / (2 * h)
    g21 = (np.roll(X2, -1, 1) - np.roll(X2, 1, 1)) / (2 * h); g22 = (np.roll(X2, -1, 0) - np.roll(X2, 1, 0)) / (2 * h)
    dG = g11 * g22 - g12 * g21; ok = (phi < 2 * h) & (dG > 0)
    trb = np.where(ok, (g11 ** 2 + g12 ** 2 + g21 ** 2 + g22 ** 2) / np.where(ok, dG, 1.0) ** 2, 2.0)
    w = 1 - smoothed_heaviside(phi, 2 * h)
    return float(0.5 * MU_S * np.sum(w * (trb - 2.0)) * h * h)


def noise(y):
    d2 = np.diff(y, 2); return float(np.sqrt(np.mean(d2 ** 2)) / (np.ptp(y) + 1e-300))


def main():
    out = {}
    for N in (64, 128):
        for meth in ("smeared", "sharp"):
            hh = run(N, meth); out[(N, meth)] = hh
            print(f"N={N:4d} {meth:7s} t_end {hh[-1,0]:.2f}  probe p: range {np.ptp(hh[:,1]):.3f} noise {noise(hh[:,1]):.2e}"
                  f" | perturbation KE: final {hh[-1,2]:.2e} noise {noise(hh[:,2]):.2e} | max|u-U| {hh[:,3].max():.2e}",
                  flush=True)
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "iim_moving_disc.npz"),
             **{f"{m}_{N}": v for (N, m), v in out.items()})


if __name__ == "__main__":
    main()
