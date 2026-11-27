"""Soft disc in the lid-driven cavity (Sugiyama et al. 2011): blended vs sharp interface.

Same case as the paper's lid example: R = 0.2 at (0.6, 0.5), mu_f = 0.01, rho = 1, lid U = 1,
elastic solid of shear modulus 0.1 carried by the log-conformation branch with tau = inf
(G = 0.1), no-slip walls. The blended runs use benchmarks/mac_soft_disc_lid.run (the paper's
driver); the sharp runs use pyRMT.iim.sharp_step_per(bc="lid").

Momentum advection is adaptive as in the paper's IMEX scheme: central while the advective CFL
and the cell-Reynolds bound hold, cubic semi-Lagrangian beyond. Time step: dt = fac * dt_ref,
dt_ref = min(0.3 dx / U, 0.2 dx^2 / nu) (the explicit step); fac = 4 is the paper's imex-sl run.

Usage: python benchmarks/iim_lid_disc.py <N> <fac> <method> [t_end=24]
Writes outputs/iim_lid/<method>_N<N>_f<fac>.npy: columns t, cx, cy (smooth-weight centroid),
min J, max J (solid interior).
"""
import os, sys, io, time, contextlib
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyRMT.mac import (poisson_eigs_neumann, cell_velocity_gradient, momentum_predictor_lid_imex,
                       momentum_predictor_lid_semilag)
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map, advance_conformation,
                             smoothed_heaviside)
from pyRMT.viscoelastic import logconf_local_step_strang, sym_exp
from pyRMT.iim import traction_jumps, sharp_step_per, bulk_solid_force_per, second_order_jumps

X0, Y0, R, MU_F, G, U_LID, RHO = 0.6, 0.5, 0.2, 0.01, 0.1, 1.0, 1.0
OUT = os.path.join(ROOT, "outputs", "iim_lid")
NL, SCHEME = 5, "semilagrangian_cubic"


def _nb3(ins):
    g = ins.copy()
    for sj in (-1, 0, 1):
        for si in (-1, 0, 1):
            g &= np.roll(np.roll(ins, sj, 0), si, 1)
    return g


def _extend(fields, keep, h):
    pg = np.where(keep, -1.0, 1.0); out = []
    for k in range(0, len(fields), 2):
        a = fields[k]; b = fields[k + 1] if k + 1 < len(fields) else np.zeros_like(a)
        ea, eb = extrapolate_reference_map(a * keep, b * keep, pg, h, h, NL); out += [ea, eb]
    return out[:len(fields)]


def advect_only(u, v, h, dt, nu):
    """explicit momentum advection with the paper's adaptive switch (no viscosity: it is implicit
    inside the sharp step)"""
    vmax = max(np.abs(u).max(), np.abs(v).max())
    if dt * vmax / h <= 0.9 and dt * vmax * vmax / (2 * nu) <= 1.0:
        return momentum_predictor_lid_imex(u, v, 0.0, h, h, dt, U_LID)
    return momentum_predictor_lid_semilag(u, v, 0.0, h, h, dt, U_LID)


def run_sharp(N, fac, t_end=24.0, log_every=2.0):
    h = 1.0 / N; xc = (np.arange(N) + 0.5) * h; Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * h, np.arange(N) * h)
    u = np.zeros((N, N + 1)); v = np.zeros((N + 1, N)); nu = MU_F / RHO
    phi0 = lambda a, b: np.hypot(a - X0, b - Y0) - R
    phi = phi0(Xc, Yc); sm = (phi <= 0).astype(float)
    X1, X2 = extrapolate_reference_map(Xc * sm, Yc * sm, phi, h, h, NL)
    p11 = np.zeros((N, N)); p12 = np.zeros((N, N)); p22 = np.zeros((N, N))
    eig = poisson_eigs_neumann(N, N, h, h)
    dt = fac * min(0.3 * h / U_LID, 0.2 * h * h / nu); n = int(np.ceil(t_end / dt - 1e-9)); dt = t_end / n
    Gp = (np.zeros((N, N)), np.zeros((N, N))); wc = 4.0 * h; traj = []; nxt = log_every
    for k in range(n):
        uc = 0.5 * (u[:, :-1] + u[:, 1:]); vc = 0.5 * (v[:-1, :] + v[1:, :])
        phi = phi0(X1, X2); sm = (phi <= 0).astype(float)
        X1 = advect_reference_map(X1, uc, vc, Xg, Yg, dt, h, h, phi, SCHEME, wc) * sm
        X2 = advect_reference_map(X2, uc, vc, Xg, Yg, dt, h, h, phi, SCHEME, wc) * sm
        X1, X2 = extrapolate_reference_map(X1, X2, phi, h, h, NL)
        phi = phi0(X1, X2); ins = phi <= 0
        p11, p12, p22 = advance_conformation(p11, p12, p22, uc, vc, Xg, Yg, dt, h, h, phi, SCHEME, wc, NL)
        L11, L12, L21, L22 = cell_velocity_gradient(u, v, h, h)
        p11, p12, p22 = logconf_local_step_strang(p11, p12, p22, L11, L12, L21, L22, np.inf, dt)
        p11, p12, p22 = _extend([p11, p12, p22], _nb3(ins), h)
        b11, b12, b22 = sym_exp(p11, p12, p22); tr = 0.5 * (b11 + b22); near = phi < (NL - 1) * h
        S1 = np.where(near, G * (b11 - tr), 0.0); S2 = np.where(near, G * b12, 0.0); S3 = np.where(near, G * (b22 - tr), 0.0)
        Jm = traction_jumps(phi, S1, S2, S3, RHO * nu)
        fb = bulk_solid_force_per(phi, S1, S2, S3, h, h, mode="face")
        dsx = np.gradient(S1, h, axis=1) + np.gradient(S2, h, axis=0)
        dsy = np.gradient(S2, h, axis=1) + np.gradient(S3, h, axis=0)
        J2 = second_order_jumps(phi, Jm, dsx, dsy, mu=RHO * nu, h=h)
        fb = (np.where(np.arange(N)[None, :] == 0, 0.0, fb[0]), np.where(np.arange(N)[:, None] == 0, 0.0, fb[1]))
        ua, va = advect_only(u, v, h, dt, nu)
        kw = dict(f_u=fb[0], f_v=fb[1], J2=J2, bc="lid", U_lid=U_LID, conserve=False)
        if k == 0:
            for _ in range(3):
                _, _, _, Gp = sharp_step_per(ua[:, :-1], va[:-1, :], h, h, dt, nu, RHO, None, eig, phi, Jm, G_old=Gp, **kw)
        un, vn, p, Gp = sharp_step_per(ua[:, :-1], va[:-1, :], h, h, dt, nu, RHO, None, eig, phi, Jm, G_old=Gp, **kw)
        u = np.zeros((N, N + 1)); v = np.zeros((N + 1, N)); u[:, :-1] = un; v[:-1, :] = vn
        u[:, 0] = 0.0; v[0, :] = 0.0
        if not np.all(np.isfinite(un)):
            print(f"  [sharp N={N} f={fac}: non-finite at t={(k + 1) * dt:.3f}]", flush=True); break
        w = 1 - smoothed_heaviside(phi, 2 * h)
        g11 = np.gradient(X1, h, axis=1); g12 = np.gradient(X1, h, axis=0)
        g21 = np.gradient(X2, h, axis=1); g22 = np.gradient(X2, h, axis=0)
        Jc = 1.0 / np.where(np.abs(g11 * g22 - g12 * g21) > 1e-12, g11 * g22 - g12 * g21, 1e-12)
        mi = phi < -2 * h
        traj.append(((k + 1) * dt, (w * Xc).sum() / w.sum(), (w * Yc).sum() / w.sum(),
                     Jc[mi].min() if mi.any() else 1.0, Jc[mi].max() if mi.any() else 1.0))
        if (k + 1) * dt >= nxt - 1e-12:
            print(f"  sharp N={N} f={fac} t={(k + 1) * dt:6.2f} centroid ({traj[-1][1]:.4f},{traj[-1][2]:.4f}) "
                  f"J in [{traj[-1][3]:.2f},{traj[-1][4]:.2f}]", flush=True); nxt += log_every
    return np.array(traj)


def run_blended(N, fac, t_end=24.0):
    from benchmarks.mac_soft_disc_lid import run
    h = 1.0 / N; dt = fac * min(0.3 * h, 0.2 * h * h / (MU_F / RHO))
    with contextlib.redirect_stdout(io.StringIO()):
        tr = run(N=N, t_end=t_end, dt=dt, integrator="imex-sl", scheme=SCHEME, mu_s=G, viscoelastic=True,
                 G_ve=G, tau=np.inf, write=False)
    return np.asarray(tr)


def cached(method, N, fac, t_end=24.0):
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f"{method}_N{N}_f{fac:g}_t{t_end:g}.npy")
    if os.path.isfile(fn):
        return np.load(fn)
    t0 = time.perf_counter()
    tr = (run_sharp if method == "sharp" else run_blended)(N, fac, t_end)
    print(f"  {method} N={N} fac={fac}: {len(tr)} steps, {time.perf_counter() - t0:.0f}s wall", flush=True)
    np.save(fn, tr); return tr


def sugiyama():
    return np.loadtxt(os.path.join(ROOT, "data", "Jain2019_fig17_sugiyama1024.csv"), delimiter=",", skiprows=1)[:, :2]


def to_ref(tr, pts):
    d = np.sqrt(((pts[:, None, :] - tr[None, :, 1:3]) ** 2).sum(-1)).min(1); return d.mean(), d.max()


if __name__ == "__main__":
    N, fac, meth = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    T = float(sys.argv[4]) if len(sys.argv) > 4 else 24.0
    tr = cached(meth, N, fac, T)
    print(f"{meth} N={N} fac={fac}: t_end {tr[-1, 0]:.2f}  vs Sugiyama mean %.2e max %.2e" % to_ref(tr, sugiyama())
          + f"  J in [{tr[:, 3].min():.2f},{tr[:, 4].max():.2f}]")
