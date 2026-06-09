"""Spatial convergence study for the soft disc in a Taylor-Green vortex
(Jain et al. 2019, Fig. 15).

Runs the disc-in-TG case to a fixed time t* with a FIXED time step (so the
temporal error is identical across grids and the spatial order is isolated),
then measures errors against the finest grid (reference) for:
    - velocity magnitude |u|     (L2 over the coarse grid points)
    - pressure p                 (L2)
    - reference map X1           (L2, restricted to the solid)
    - kinetic energy ke          (scalar)
    - strain energy se           (scalar)
The observed order is the slope of log(error) vs log(dx).

Usage:
    python benchmarks/convergence_taylor_green.py [scheme]
        scheme : 'semilagrangian' (default) | 'central2' | 'weno5'
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.interpolate import RegularGridInterpolator

from pyRMT.functions import (
    create_grid, apply_phi_BCs, extrapolate_transverse_layers_2field,
    advect_reference_map, rebuild_phi_from_reference_map, velocity_RK4,
    heaviside_smooth_alt, pressure_projection_amg, build_poisson_matrix,
    _precompute_poisson_eigenvalues,
)
from pyRMT.output import compute_kinetic_energy, compute_strain_energy
from benchmarks.common import (free_slip_box_bc, initialize_disc,
                               taylor_green_velocity, check_narrow_band, ensure_dir)


def simulate_tg(N, scheme, t_end=0.25, dt=1.0e-4, stress_band=False):
    """Run disc-in-TG to t_end with fixed dt; return final fields + energies."""
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    x0, y0, R = 0.5, 0.5, 0.2
    phi_init = lambda Xq, Yq: initialize_disc(Xq, Yq, x0, y0, R)
    phi = apply_phi_BCs(phi_init(X, Y))
    solid_mask = (phi <= 0).astype(float)

    mu_s, kappa, rho_s, eta_s = 1.0, 0.0, 1.0, 0.0
    mu_f, rho_f = 1.0e-3, 1.0
    w_t = 2.0 * dx
    num_layers = max(3, check_narrow_band(w_t, dx, 3))

    X1 = X * solid_mask
    X2 = Y * solid_mask
    X1, X2 = extrapolate_transverse_layers_2field(X1, X2, phi, dx, dy, w_t, num_layers)

    a, b = taylor_green_velocity(X, Y, U0=0.05)
    a, b = free_slip_box_bc(a, b)
    p = np.zeros((N, N))

    A = build_poisson_matrix(N, N, dx, dy)
    eig = _precompute_poisson_eigenvalues(N, N, dx, dy)
    ml = None

    nsteps = int(round(t_end / dt))
    for _ in range(nsteps):
        phi = rebuild_phi_from_reference_map(X1, X2, phi_init)
        solid_mask = (phi <= 0).astype(float)
        X1 = advect_reference_map(X1, a, b, X, Y, dt, dx, dy, phi, scheme, 0.0) * solid_mask
        X2 = advect_reference_map(X2, a, b, X, Y, dt, dx, dy, phi, scheme, 0.0) * solid_mask
        X1, X2 = extrapolate_transverse_layers_2field(X1, X2, phi, dx, dy, w_t, num_layers)
        phi = rebuild_phi_from_reference_map(X1, X2, phi_init)

        a_star, b_star, *_ , J = velocity_RK4(
            a, b, p, X1, X2, free_slip_box_bc, mu_s, kappa, eta_s, dx, dy, dt,
            rho_s, rho_f, phi, mu_f, w_t, gamma=0.0, stress_band=stress_band)
        H = heaviside_smooth_alt(phi, w_t)
        rho_local = (1 - H) * rho_s + H * rho_f
        a, b, p, A, ml = pressure_projection_amg(
            a_star, b_star, dx, dy, dt, rho_local, velocity_bc=free_slip_box_bc,
            A=A, ml=ml, p_prev=p, eigenvalues=eig, bc_type='neumann')

    ke = compute_kinetic_energy(a, b, rho_f, rho_s, phi, w_t, dx, dy)
    se = compute_strain_energy(X1, X2, phi, mu_s, dx, dy, kappa=kappa)
    return dict(N=N, dx=dx, X=X, Y=Y, a=a, b=b, p=p, X1=X1, X2=X2, phi=phi, ke=ke, se=se)


def _sample_ref_on(coarse, ref, key):
    """Interpolate reference field onto the coarse grid points."""
    xr = np.linspace(0, 1, ref['N']); yr = np.linspace(0, 1, ref['N'])
    f = RegularGridInterpolator((yr, xr), ref[key], bounds_error=False, fill_value=None)
    pts = np.column_stack([coarse['Y'].ravel(), coarse['X'].ravel()])
    return f(pts).reshape(coarse['X'].shape)


def l2(err, mask=None):
    if mask is not None:
        err = err[mask]
    return np.sqrt(np.mean(err ** 2))


def richardson_order(values):
    """Reference-free observed order from factor-2-spaced grids.
    values is a list of (N, Q) ordered coarse->fine. Returns a list of
    (N_triplet, p) for each consecutive triplet via
    p = log2(|Q_4h - Q_2h| / |Q_2h - Q_h|)."""
    out = []
    for i in range(len(values) - 2):
        (N0, q0), (N1, q1), (N2, q2) = values[i], values[i + 1], values[i + 2]
        d_coarse = q1 - q0
        d_fine = q2 - q1
        if abs(d_fine) > 0:
            out.append((N2, np.log(abs(d_coarse) / abs(d_fine)) / np.log(2.0)))
    return out


def run(scheme='semilagrangian', grids=(32, 64, 128, 256), N_ref=512, t_end=0.25, dt=1.0e-4,
        stress_band=False):
    print(f"[convergence-TG] scheme={scheme} grids={grids} ref={N_ref} t={t_end} dt={dt}")
    sols = {}
    for N in list(grids) + [N_ref]:
        sols[N] = simulate_tg(N, scheme, t_end, dt, stress_band=stress_band)
        s = sols[N]
        print(f"  N={N:4d}  dx={s['dx']:.5f}  ke={s['ke']:.6e}  se={s['se']:.6e}")

    ref = sols[N_ref]
    rows = []
    for N in grids:
        c = sols[N]
        umag_c = np.hypot(c['a'], c['b'])
        umag_r = np.hypot(_sample_ref_on(c, ref, 'a'), _sample_ref_on(c, ref, 'b'))
        p_r = _sample_ref_on(c, ref, 'p'); p_r -= p_r.mean()
        pc = c['p'] - c['p'].mean()
        X1_r = _sample_ref_on(c, ref, 'X1')
        solid = (c['phi'] <= 0)
        e_v = l2(umag_c - umag_r)
        e_p = l2(pc - p_r)
        e_x = l2(c['X1'] - X1_r, mask=solid)
        e_ke = abs(c['ke'] - ref['ke'])
        e_se = abs(c['se'] - ref['se'])
        rows.append((c['dx'], e_v, e_p, e_x, e_ke, e_se))
        print(f"  N={N:4d}  E_v={e_v:.3e}  E_p={e_p:.3e}  E_X1={e_x:.3e}  "
              f"E_ke={e_ke:.3e}  E_se={e_se:.3e}")

    # Reference-free Richardson order for the scalar energies (uses ALL grids,
    # incl. the reference, so it does not depend on a converged reference).
    ke_seq = [(N, sols[N]['ke']) for N in sorted(sols)]
    se_seq = [(N, sols[N]['se']) for N in sorted(sols)]
    print("  Richardson (reference-free) scalar orders:")
    for nm, seq in (("ke", ke_seq), ("se", se_seq)):
        for Ntrip, p in richardson_order(seq):
            print(f"    {nm} triplet ->N={Ntrip}: p = {p:.2f}")

    rows = np.array(rows)
    dxs = rows[:, 0]
    names = ["|u|", "p", "X1", "ke", "se"]
    print("  observed orders vs reference N=%d (slope of log E vs log dx):" % N_ref)
    orders = {}
    for k, nm in enumerate(names):
        E = rows[:, k + 1]
        good = E > 0
        order = np.polyfit(np.log(dxs[good]), np.log(E[good]), 1)[0] if good.sum() > 1 else np.nan
        orders[nm] = order
        print(f"    {nm:4s}: p = {order:.2f}")

    out_dir = ensure_dir(os.path.join("outputs", f"convergence_tg_{scheme}"))
    np.savetxt(os.path.join(out_dir, "errors.csv"), rows, delimiter=",",
               header="dx,E_v,E_p,E_X1,E_ke,E_se", comments="")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        for k, nm in enumerate(names):
            plt.loglog(dxs, rows[:, k + 1], "o-", label=f"{nm} (p={orders[nm]:.2f})")
        ref_line = rows[:, 1][0] * (dxs / dxs[0]) ** 2
        plt.loglog(dxs, ref_line, "k--", label="O(dx^2)")
        plt.xlabel("dx"); plt.ylabel("L2 / scalar error vs ref")
        plt.title(f"Disc-in-TG spatial convergence ({scheme}, ref N={N_ref})")
        plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
        fig = os.path.join(out_dir, "convergence.png")
        plt.savefig(fig, dpi=140); print(f"  saved {fig}")
    except Exception as e:
        print(f"  (plot skipped: {e})")
    return orders


if __name__ == "__main__":
    scheme = sys.argv[1] if len(sys.argv) > 1 else 'semilagrangian'
    run(scheme=scheme)
