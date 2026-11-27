"""RM-IIM on the paper's convergence case: a soft disc in a Taylor-Green vortex, free-slip box.

Same problem and parameters as the convergence study of the smeared method
(benchmarks/viscoelastic_disc_tg.py: R = 0.15 at (0.5, 0.7), mu_f = 0.01, U0 = 0.3, G = 0.1,
log-conformation solid with tau = inf, implicit viscosity, t_end = 0.6).

  smeared : the current method, run through viscoelastic_disc_tg.run_one
  sharp   : the solid traction imposed as interface jumps (pyRMT.iim.sharp_step_per, bc="freeslip")

Sharp-method details:
  - psi = log b_e is advanced by the transport + stretch step only on cells whose 3x3
    neighbourhood is inside the solid (their velocity-gradient stencil does not cross the
    interface, where grad u jumps); everywhere else within 5 layers it is the smooth extension
    of those values, rebuilt every step. The stress and the jump conditions use this field.
  - model "elastic" takes the stress from the reference map instead (cells with a full interior
    5-point stencil, extended the same way).

Convergence is measured without a reference solution, from differences between N and 2N
(fields restricted by 2x2 averaging). Runs are cached in outputs/iim_vortex/.
Usage: python benchmarks/iim_vortex_disc.py [N_max=128] [model=logconf] [dt_power=1] [tau=inf]
  dt_power 1: dt ~ h (the paper's step); 2: dt ~ h^2 (equal to the default step at N = 32)
"""
import os, sys, io, contextlib
import numpy as np
from scipy import ndimage as ndi
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyRMT.mac import (momentum_predictor_freeslip, poisson_eigs_neumann, cell_velocity_gradient)
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map, advance_conformation,
                             solid_cauchy_stress, smoothed_heaviside)
from pyRMT.viscoelastic import logconf_local_step_strang, sym_exp
from pyRMT.iim import traction_jumps, sharp_step_per, bulk_solid_force_per, second_order_jumps

P = dict(R=0.15, x0=0.5, y0=0.7, mu_f=0.01, U0=0.3, G=0.1, rho=1.0)
OUT = os.path.join(ROOT, "outputs", "iim_vortex")
NL, SCHEME = 5, "semilagrangian_cubic"


def _extend(fields, keep, h):
    """Smooth extension of cell fields from the cells in `keep` to NL layers around them."""
    pg = np.where(keep, -1.0, 1.0); out = []
    for k in range(0, len(fields), 2):
        a = fields[k]; b = fields[k + 1] if k + 1 < len(fields) else np.zeros_like(a)
        ea, eb = extrapolate_reference_map(a * keep, b * keep, pg, h, h, NL)
        out += [ea, eb]
    return out[:len(fields)]


def _nb(ins, corners):
    g = ins & np.roll(ins, 1, 0) & np.roll(ins, -1, 0) & np.roll(ins, 1, 1) & np.roll(ins, -1, 1)
    if corners:
        for sj in (1, -1):
            for si in (1, -1):
                g &= np.roll(np.roll(ins, sj, 0), si, 1)
    return g


def run_sharp(N, t_end, model="logconf", tau=np.inf, dt_scale=1.0, history=False):
    R, x0, y0, mu_f, U0, G, rho = (P[k] for k in ("R", "x0", "y0", "mu_f", "U0", "G", "rho"))
    h = 1.0 / N; xc = (np.arange(N) + 0.5) * h; Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * h, np.arange(N) * h)
    xf = np.arange(N + 1) * h; Xu, Yu = np.meshgrid(xf, xc); Xv, Yv = np.meshgrid(xc, xf)
    k = np.pi
    u = U0 * k * np.sin(k * Xu) * np.cos(k * Yu); v = -U0 * k * np.cos(k * Xv) * np.sin(k * Yv)
    u[:, 0] = u[:, -1] = 0.0; v[0, :] = v[-1, :] = 0.0
    phi0 = lambda a, b: np.hypot(a - x0, b - y0) - R
    phi = phi0(Xc, Yc); sm = (phi <= 0).astype(float)
    X1, X2 = extrapolate_reference_map(Xc * sm, Yc * sm, phi, h, h, NL)
    p11 = np.zeros((N, N)); p12 = np.zeros((N, N)); p22 = np.zeros((N, N))
    nu = mu_f / rho; eig = poisson_eigs_neumann(N, N, h, h)
    dt0 = min(0.25 * h / (U0 * k), 0.25 * h / np.sqrt(G / rho)) * dt_scale
    nstep = int(np.ceil(t_end / dt0 - 1e-9)); dt = t_end / nstep
    Gp = (np.zeros((N, N)), np.zeros((N, N))); wc = 4.0 * h
    hist = []
    for n in range(nstep):
        uc = 0.5 * (u[:, :-1] + u[:, 1:]); vc = 0.5 * (v[:-1, :] + v[1:, :])
        phi = phi0(X1, X2); sm = (phi <= 0).astype(float)
        X1 = advect_reference_map(X1, uc, vc, Xg, Yg, dt, h, h, phi, SCHEME, wc) * sm
        X2 = advect_reference_map(X2, uc, vc, Xg, Yg, dt, h, h, phi, SCHEME, wc) * sm
        X1, X2 = extrapolate_reference_map(X1, X2, phi, h, h, NL)
        phi = phi0(X1, X2); ins = phi <= 0
        if model == "elastic":
            sxx, sxy, syy, _ = solid_cauchy_stress(X1, X2, h, h, G, 0.0, phi, isochoric=True, detg_clamp=3.0)
            S1, S2, S3 = _extend([sxx, sxy, syy], _nb(ins, False), h)
        else:
            p11, p12, p22 = advance_conformation(p11, p12, p22, uc, vc, Xg, Yg, dt, h, h, phi, SCHEME, wc, NL)
            L11, L12, L21, L22 = cell_velocity_gradient(u, v, h, h)
            p11, p12, p22 = logconf_local_step_strang(p11, p12, p22, L11, L12, L21, L22, tau, dt)
            p11, p12, p22 = _extend([p11, p12, p22], _nb(ins, True), h)
            b11, b12, b22 = sym_exp(p11, p12, p22); tr = 0.5 * (b11 + b22)
            near = phi < (NL - 1) * h
            S1 = np.where(near, G * (b11 - tr), 0.0); S2 = np.where(near, G * b12, 0.0); S3 = np.where(near, G * (b22 - tr), 0.0)
        Jm = traction_jumps(phi, S1, S2, S3, rho * nu)
        fb = bulk_solid_force_per(phi, S1, S2, S3, h, h, mode="face")
        dsx = np.gradient(S1, h, axis=1) + np.gradient(S2, h, axis=0)
        dsy = np.gradient(S2, h, axis=1) + np.gradient(S3, h, axis=0)
        J2 = second_order_jumps(phi, Jm, dsx, dsy, mu=rho * nu, h=h)
        ua, va = momentum_predictor_freeslip(u, v, 0.0, h, h, dt)             # explicit advection only
        if n == 0:          # initialise the stored pressure gradient
            for _ in range(3):
                _, _, _, Gp = sharp_step_per(ua[:, :-1], va[:-1, :], h, h, dt, nu, rho, None, eig, phi, Jm,
                                             f_u=fb[0], f_v=fb[1], J2=J2, G_old=Gp, bc="freeslip")
        un, vn, p, Gp = sharp_step_per(ua[:, :-1], va[:-1, :], h, h, dt, nu, rho, None, eig, phi, Jm,
                                       f_u=fb[0], f_v=fb[1], J2=J2, G_old=Gp, bc="freeslip")
        u[:, :-1] = un; v[:-1, :] = vn
        if not np.all(np.isfinite(un)):
            raise FloatingPointError(f"sharp run diverged at step {n}")
        if history:
            hist.append(((n + 1) * dt,) + _energies(u, v, phi, X1, X2, p11, p12, p22, model, G, h))
    out = _pack(u, v, p, phi, X1, X2, p11, p12, p22, model, G, h, Xc, Yc)
    if history:
        out["hist"] = np.array(hist)
    return out


def _energies(u, v, phi, X1, X2, p11, p12, p22, model, G, h):
    uc = 0.5 * (u[:, :-1] + u[:, 1:]); vc = 0.5 * (v[:-1, :] + v[1:, :])
    KE = 0.5 * np.sum(uc ** 2 + vc ** 2) * h * h
    w = 1.0 - smoothed_heaviside(phi, 2 * h)
    if model == "elastic":
        g11 = np.gradient(X1, h, axis=1); g12 = np.gradient(X1, h, axis=0)
        g21 = np.gradient(X2, h, axis=1); g22 = np.gradient(X2, h, axis=0)
        det = g11 * g22 - g12 * g21; ok = (w > 0) & (det > 0.2)
        trb = np.where(ok, (g11 ** 2 + g12 ** 2 + g21 ** 2 + g22 ** 2) / np.where(ok, det, 1.0) ** 2, 2.0)
        W = 0.5 * G * np.sum(w * (trb - 2.0 + 2.0 * np.log(np.where(ok, det, 1.0)))) * h * h
    else:
        b11, b12, b22 = sym_exp(p11, p12, p22)
        W = 0.5 * G * np.sum(w * ((b11 + b22) - 2.0 - (p11 + p22))) * h * h
    return float(KE), float(W)


def _pack(u, v, p, phi, X1, X2, p11, p12, p22, model, G, h, Xc, Yc):
    uc = 0.5 * (u[:, :-1] + u[:, 1:]); vc = 0.5 * (v[:-1, :] + v[1:, :])
    KE, W = _energies(u, v, phi, X1, X2, p11, p12, p22, model, G, h)
    w = 1.0 - smoothed_heaviside(phi, 2 * h)
    return dict(speed=np.hypot(uc, vc), uc=uc, vc=vc, p=p - p.mean(), phi=phi, KE=KE, W=W,
                cx=float((w * Xc).sum() / w.sum()), cy=float((w * Yc).sum() / w.sum()))


def run_smeared(N, t_end, model="logconf", tau=np.inf, dt_scale=1.0, history=False):
    import viscoelastic_disc_tg as V
    with contextlib.redirect_stdout(io.StringIO()):
        r = V.run_one(N, t_end, model=("elastic" if model == "elastic" else "maxwell"), G_ve=P["G"], mu_s=P["G"],
                      tau=tau, integrator="imex", scheme=SCHEME, history=True, dt_scale=dt_scale,
                      snap_times=[t_end * (1 - 1e-9)],
                      R=P["R"], x0=P["x0"], y0=P["y0"], mu_f=P["mu_f"], U0=P["U0"], kw=1)
    h = 1.0 / N; xc = (np.arange(N) + 0.5) * h; Xc, Yc = np.meshgrid(xc, xc)
    w = 1.0 - smoothed_heaviside(r["phi"], 2 * h); H = r["hist"]
    sn = r["snaps"][-1]
    out = dict(speed=r["speed"], uc=sn["uc"], vc=sn["vc"], p=r["p"] - r["p"].mean(), phi=r["phi"],
               KE=float(r["KE"]), W=float(H["Eel_s"][-1]),
               cx=float((w * Xc).sum() / w.sum()), cy=float((w * Yc).sum() / w.sum()))
    if history:
        out["hist"] = np.stack([H["t"], H["KE"], H["Eel_s"]], axis=1)
    return out


def cached(method, N, t_end=0.6, model="logconf", dt_power=1, tau=np.inf):
    os.makedirs(OUT, exist_ok=True)
    tg = "" if np.isinf(tau) else f"_tau{tau:g}"
    fn = os.path.join(OUT, f"{method}_{model}{tg}_dt{dt_power}_t{t_end:g}_N{N}.npz")
    if os.path.isfile(fn):
        return dict(np.load(fn))
    sc = 1.0 if dt_power == 1 else 32.0 / N
    r = (run_sharp if method == "sharp" else run_smeared)(N, t_end, model=model, tau=tau, dt_scale=sc, history=True)
    np.savez(fn, **r); return r


restrict = lambda f: 0.25 * (f[0::2, 0::2] + f[1::2, 0::2] + f[0::2, 1::2] + f[1::2, 1::2])
rms = lambda a: float(np.sqrt(np.mean(a ** 2)))


def differences(fa, fb, N):
    s = fa["phi"] <= 0
    dist = np.where(s, ndi.distance_transform_edt(s), ndi.distance_transform_edt(~s)) / N
    mb = dist > 0.06
    # velocity COMPONENTS, not the speed: |u| has a cone point at every stagnation point, where
    # comparing magnitudes between grids gives an O(h) difference that is not an error of the flow
    dv = np.hypot(fa["uc"] - restrict(fb["uc"]), fa["vc"] - restrict(fb["vc"])); pa, pb = fa["p"], restrict(fb["p"])
    dp = (pa - pa[mb].mean()) - (pb - pb[mb].mean())
    return dict(vel_L2=rms(dv), vel_max=float(np.abs(dv).max()), vel_bulk=rms(dv[mb]), p_L2=rms(dp), p_bulk=rms(dp[mb]),
                KE=abs(float(fa["KE"]) - float(fb["KE"])), W=abs(float(fa["W"]) - float(fb["W"])),
                centroid=float(np.hypot(fa["cx"] - fb["cx"], fa["cy"] - fb["cy"])))


def main():
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    model = sys.argv[2] if len(sys.argv) > 2 else "logconf"
    dtp = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    tau = float(sys.argv[4]) if len(sys.argv) > 4 else np.inf
    NS = [n for n in (32, 64, 128, 256) if n <= nmax]
    for meth in ("smeared", "sharp"):
        F = {N: cached(meth, N, model=model, dt_power=dtp, tau=tau) for N in NS}
        D = [differences(F[a], F[2 * a], a) for a in NS[:-1]]
        print(f"{meth} ({model}, dt ~ h^{dtp}): differences " + ", ".join(f"{a}/{2*a}" for a in NS[:-1]))
        for q in D[0]:
            d = [x[q] for x in D]
            print(f"   {q:9s} " + "  ".join(f"{x:.2e}" for x in d) + "   orders "
                  + " ".join(f"{np.log2(d[i] / d[i + 1]):.2f}" for i in range(len(d) - 1)), flush=True)
        print(f"   KE(N) " + " ".join(f"{float(F[N]['KE']):.6f}" for N in NS) + " | W(N) " + " ".join(f"{float(F[N]['W']):.3e}" for N in NS))


if __name__ == "__main__":
    main()
