"""Demonstration of the fully monolithic Eulerian RMT-FSI scheme.

Companion to ``paper/monolithic_aml.tex``.  A neo-Hookean disc sits in a decaying
Taylor-Green vortex on a free-slip box; the monolithic solver
(:class:`pyRMT.monolithic.MonolithicRMT`) advances velocity, pressure and the
eliminated reference map in one Newton solve per step, and is compared against the
explicit RMT reference on the same operators and the same grid.

Three studies, matching the three claims of the draft:

  ``stability``  a stiff solid at increasing multiples of the elastic-wave CFL
                 dx/c_s -- where the explicit scheme blows up and the monolithic
                 one does not, with det(grad xi) bounded away from zero.
  ``temporal``   error vs dt against a fine-dt reference: theta=1 gives first
                 order, theta=1/2 second order.
  ``workprec``   wall-clock vs accuracy for both schemes (the net-speedup claim).

Usage:  python benchmarks/monolithic_disc_tg.py [stability|temporal|workprec|all]
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.monolithic import MonolithicRMT
from pyRMT.mac import (mac_grid, momentum_predictor_freeslip, project,
                       poisson_eigs_neumann, divergence)
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map,
                             solid_cauchy_stress, smoothed_heaviside,
                             grad_central_x_2nd, grad_central_y_2nd)

# ── the shared configuration (same as benchmarks/mac_disc_in_tg_convergence.py) ──
R_DISC, X0, Y0 = 0.15, 0.5, 0.7
U0, KW = 0.3, np.pi
MU_F, RHO = 0.01, 1.0
OUT = os.path.join("outputs", "monolithic_disc_tg")


def phi0(X, Y):
    return np.sqrt((X - X0) ** 2 + (Y - Y0) ** 2) - R_DISC


def _tg_velocity(N, dx):
    xf = np.arange(N + 1) * dx
    xc = (np.arange(N) + 0.5) * dx
    Xu, Yu = np.meshgrid(xf, xc)
    Xv, Yv = np.meshgrid(xc, xf)
    u = U0 * KW * np.sin(KW * Xu) * np.cos(KW * Yu)
    v = -U0 * KW * np.cos(KW * Xv) * np.sin(KW * Yv)
    u[:, 0] = u[:, -1] = 0.0
    v[0, :] = v[-1, :] = 0.0
    return u, v


def cfl_limits(N, mu_s):
    """The three explicit step restrictions the monolithic scheme removes."""
    dx = 1.0 / N
    return dict(advective=0.25 * dx / (U0 * KW),
                viscous=0.2 * dx * dx * RHO / MU_F,
                elastic=0.25 * dx / np.sqrt(mu_s / RHO))


# ── the monolithic run ──────────────────────────────────────────────────────
def run_monolithic(N, t_end, dt, mu_s, theta=1.0, verbose=False, n_outer=1, **kw):
    """mu_s = 0 runs the pure-fluid problem (no solid, no reference map), which is
    how the temporal-order study separates the fluid kernel from the RMT band."""
    solid = mu_s > 0.0
    s = MonolithicRMT(N, phi0 if solid else (lambda X, Y: np.full_like(X, 1e3)),
                      mu_s=mu_s, mu_f=MU_F, rho=RHO, theta=theta,
                      layers=5 if solid else 1, **kw)
    s.set_velocity(*_tg_velocity(N, s.dx))
    if solid:
        s.seed_reference_map()
    else:
        s.xi1 = np.zeros((N, N)); s.xi2 = np.zeros((N, N))
        s.phi = np.full((N, N), 1e3)
    t0 = time.time()
    nstep = 0
    while s.t < t_end - 1e-12:
        h = min(dt, t_end - s.t)
        st = s.step(h, n_outer=n_outer)
        nstep += 1
        if verbose:
            print(f"    t={s.t:.4f} newton={st['newton']} gmres={st['gmres']} "
                  f"minJ={st['min_det']:.4f} div={st['divmax']:.1e}")
    uc, vc = s.cell_velocity()
    return dict(scheme="monolithic", N=N, dt=dt, theta=theta,
                advection=kw.get("advection", "sl"),
                speed=np.sqrt(uc ** 2 + vc ** 2), u=uc, v=vc, p=s.p,
                KE=s.kinetic_energy(),
                Es=s.strain_energy() if solid else 0.0,
                minJ=min(x["min_det"] for x in s.stats),
                divmax=max(x["divmax"] for x in s.stats),
                newton=float(np.mean([x["newton"] for x in s.stats])),
                gmres=float(np.mean([x["gmres"] for x in s.stats])),
                wall=time.time() - t0, nstep=nstep, solver=s)


# ── the explicit reference (identical operators, so only TIME differs) ───────
def run_explicit(N, t_end, dt, mu_s, scheme="semilagrangian_cubic",
                 band=True, deviatoric=True, layers=5):
    """Explicit RMT reference.

    ``band``/``deviatoric``/``layers`` default to the SAME spatial operators the
    monolithic solver uses, so a comparison isolates the time discretization.
    The legacy alternative (``band=False``: stress only where phi<=0, with the
    one-sided stencils of ``solid_cauchy_stress``) is what the explicit drivers in
    this repo run by default -- but it is precisely the non-differentiable choice
    the monolithic scheme cannot make, and it also puts an O(dx) jump in the force
    at the interface, so leaving it on would confound the two effects."""
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    u, v = _tg_velocity(N, dx)
    phi = phi0(Xc, Yc)
    sm = (phi <= 0).astype(float)
    X1, X2 = extrapolate_reference_map(Xc * sm, Yc * sm, phi, dx, dy, layers)
    eig = poisson_eigs_neumann(N, N, dx, dy)
    w_t = 2.0 * dx
    w_cut = w_t if band else 0.0
    nu = MU_F / RHO
    t = 0.0
    t0 = time.time()
    nstep = 0
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        uc = 0.5 * (u[:, :-1] + u[:, 1:])
        vc = 0.5 * (v[:-1, :] + v[1:, :])
        phi = phi0(X1, X2)
        sm = (phi <= 0).astype(float)
        X1 = advect_reference_map(X1, uc, vc, Xg, Yg, h, dx, dy, phi, scheme, 4 * dx) * sm
        X2 = advect_reference_map(X2, uc, vc, Xg, Yg, h, dx, dy, phi, scheme, 4 * dx) * sm
        X1, X2 = extrapolate_reference_map(X1, X2, phi, dx, dy, layers)
        phi = phi0(X1, X2)
        sxx, sxy, syy, J = solid_cauchy_stress(X1, X2, dx, dy, mu_s, 0.0, phi,
                                               w_cut=w_cut)
        if deviatoric:
            tr_half = 0.5 * (sxx + syy)
            sxx = sxx - tr_half; syy = syy - tr_half
        H = smoothed_heaviside(phi, w_t)
        Sxx = (1 - H) * sxx; Sxy = (1 - H) * sxy; Syy = (1 - H) * syy
        divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dy)
        divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dy)
        fu = np.zeros((N, N + 1)); fu[:, 1:-1] = 0.5 * (divx[:, 1:] + divx[:, :-1])
        fv = np.zeros((N + 1, N)); fv[1:-1, :] = 0.5 * (divy[1:, :] + divy[:-1, :])
        us, vs = momentum_predictor_freeslip(u, v, nu, dx, dy, h, fu=fu, fv=fv, rho=RHO)
        u, v, p = project(us, vs, dx, dy, h, RHO, eig)
        t += h
        nstep += 1
        if not np.all(np.isfinite(u)):
            raise FloatingPointError(f"explicit scheme diverged at t={t:.4g}")
    from pyRMT.output import compute_strain_energy
    uc = 0.5 * (u[:, :-1] + u[:, 1:]); vc = 0.5 * (v[:-1, :] + v[1:, :])
    return dict(scheme="explicit", N=N, dt=dt,
                speed=np.sqrt(uc ** 2 + vc ** 2), u=uc, v=vc, p=p,
                KE=0.5 * RHO * np.sum(uc ** 2 + vc ** 2) * dx * dy,
                Es=compute_strain_energy(X1, X2, phi, mu_s, dx, dy, isochoric=True),
                minJ=float(np.min(J[phi <= 0])),
                divmax=float(np.max(np.abs(divergence(u, v, dx, dy)))),
                wall=time.time() - t0, nstep=nstep)


def _rel_err(a, b):
    return float(np.sqrt(np.mean((a["u"] - b["u"]) ** 2 + (a["v"] - b["v"]) ** 2))
                 / np.sqrt(np.mean(b["u"] ** 2 + b["v"] ** 2)))


# ── study 1: unconditional stability above the elastic-wave CFL ─────────────
def study_stability(N=48, mu_s=100.0, t_end=0.05):
    """Explicit vs monolithic at increasing multiples of the elastic-wave CFL.

    Both are measured against a well-resolved explicit run (dt = dt_elastic/8) on
    the SAME spatial operators, so what is being compared is purely the time
    discretization.
    """
    lim = cfl_limits(N, mu_s)
    print(f"\n{'='*78}\nSTABILITY -- stiff disc, N={N}, mu_s={mu_s} "
          f"(c_s={np.sqrt(mu_s/RHO):.1f})")
    print(f"  explicit limits: elastic {lim['elastic']:.3e}  advective {lim['advective']:.3e}"
          f"  viscous {lim['viscous']:.3e}  ->  dt_exp = {min(lim.values()):.3e}")
    ref = run_explicit(N, t_end, lim["elastic"] / 8.0, mu_s)
    print(f"  reference: explicit at dt_elastic/8 ({ref['nstep']} steps), "
          f"KE={ref['KE']:.5f}  min J={ref['minJ']:.4f}")
    print(f"{'-'*78}\n{'dt/dt_el':>9} | {'explicit':^30} | {'monolithic':^30}")
    print(f"{'':>9} | {'rel err':>9} {'KE':>9} {'minJ':>9} | {'rel err':>9} {'KE':>9} {'minJ':>9}")
    rows = []
    for m in (1, 2, 5, 10, 20, 40):
        dt = m * lim["elastic"]
        try:
            e = run_explicit(N, t_end, dt, mu_s)
            estr = f"{_rel_err(e, ref):>9.2e} {e['KE']:>9.4f} {e['minJ']:>9.4f}"
        except Exception as exc:
            e = None
            estr = f"{'DIVERGED':>29}"
        try:
            r = run_monolithic(N, t_end, dt, mu_s)
            mstr = f"{_rel_err(r, ref):>9.2e} {r['KE']:>9.4f} {r['minJ']:>9.4f}"
        except Exception as exc:
            r = None
            mstr = f"{'FAILED (' + type(exc).__name__ + ')':>29}"
        print(f"{m:>9d} | {estr} | {mstr}", flush=True)
        rows.append((m, dt, e, r))
    return rows


# ── study 2: temporal order ────────────────────────────────────────────────
def study_temporal(N=48, mu_s=1.0, t_end=0.12):
    """Order in dt, and where it is lost.

    The velocity-pressure kernel is second-order-capable: with theta=1/2 and the
    Eulerian (implicit central) advection of the draft's Eq. (8), and no solid
    present, the observed order is 2.  Reintroduce the solid and it falls back to
    1 -- the reference-map band (a masked, extrapolated field whose solid region is
    quantized to whole cells) is what caps it, not the time integrator.  The
    semi-Lagrangian momentum branch is first order for a separate reason: its
    pressure gradient acts at the arrival point rather than along the
    characteristic.
    """
    print(f"\n{'='*78}\nTEMPORAL ORDER -- N={N}, T={t_end}")
    lim = cfl_limits(N, mu_s)
    dt0 = min(lim.values())
    facs = [8, 4, 2, 1]
    out = {}
    cases = [("fluid only", 0.0, "eulerian"),
             (f"solid mu_s={mu_s:g}", mu_s, "eulerian"),
             (f"solid mu_s={mu_s:g}", mu_s, "sl")]
    for label, ms, adv in cases:
        for th in (1.0, 0.5):
            try:
                ref = run_monolithic(N, t_end, dt0 / 8.0, ms, theta=th, advection=adv)
                errs = [_rel_err(run_monolithic(N, t_end, dt0 * f, ms, theta=th,
                                                advection=adv), ref) for f in facs]
                ords = [np.log(errs[i] / errs[i + 1]) / np.log(2.0)
                        for i in range(len(errs) - 1)]
                print(f"  {label:>16}  {adv:>9}  theta={th}:  "
                      f"err={['%.2e' % e for e in errs]}  "
                      f"order={['%.2f' % o for o in ords]}", flush=True)
                out[(label, adv, th)] = (facs, errs, ords)
            except Exception as exc:
                print(f"  {label:>16}  {adv:>9}  theta={th}:  "
                      f"{type(exc).__name__}", flush=True)
    print("  (dt/dt_exp = " + str(facs) + ")")
    return out


# ── study 2b: spatial convergence ──────────────────────────────────────────
def study_spatial(mu_s=0.1, t_end=0.3, Ns=(32, 64, 128), Nref=192, dt_fac=0.5):
    """Grid convergence of the monolithic scheme against the explicit solver.

    Each run uses its OWN dt = dt_fac x the explicit limit at that N.  Note that the
    binding explicit limit here is the VISCOUS one, so dt ~ dx^2 and the step count
    grows quadratically -- which is what makes this study expensive, and is itself
    a restriction the monolithic scheme does not have (it is used here only to keep
    the explicit reference valid on the same steps).
    """
    print(f"\n{'='*78}\nSPATIAL CONVERGENCE -- mu_s={mu_s}, T={t_end}, ref N={Nref}")
    print(f"  each run at dt = {dt_fac} x its own explicit limit "
          f"(viscous-bound, so dt ~ dx^2)")
    ref = run_explicit(Nref, t_end, dt_fac * min(cfl_limits(Nref, mu_s).values()), mu_s)
    print(f"  reference: explicit N={Nref}, {ref['nstep']} steps, {ref['wall']:.0f}s")
    g = (np.arange(24) + 0.5) / 24
    sref = _sample(ref, g)
    rows = {}
    for name, runner in (("explicit", run_explicit), ("monolithic", run_monolithic)):
        errs = []
        for N in Ns:
            dt = dt_fac * min(cfl_limits(N, mu_s).values())
            r = runner(N, t_end, dt, mu_s)
            errs.append(float(np.sqrt(np.mean((_sample(r, g) - sref) ** 2))))
            print(f"  {name:>11}  N={N:4d}  dt={dt:.2e}  err={errs[-1]:.4e}  "
                  f"KE={r['KE']:.5f}  wall={r['wall']:.1f}s", flush=True)
        ords = [np.log(errs[i] / errs[i + 1]) / np.log(Ns[i + 1] / Ns[i])
                for i in range(len(errs) - 1)]
        print(f"  {name:>11}  order = {['%.2f' % o for o in ords]}", flush=True)
        rows[name] = (list(Ns), errs, ords)
    return rows


def _sample(r, g):
    """Bilinear-sample a run's speed field onto the common probe grid g."""
    xc = (np.arange(r["N"]) + 0.5) / r["N"]
    f = r["speed"]
    tmp = np.array([np.interp(g, xc, f[j, :]) for j in range(f.shape[0])])
    return np.array([np.interp(g, xc, tmp[:, k]) for k in range(tmp.shape[1])]).T


# ── study 3: work-precision ────────────────────────────────────────────────
def study_workprec(N=48, mu_s=100.0, t_end=0.05):
    print(f"\n{'='*74}\nWORK-PRECISION -- N={N}, mu_s={mu_s}, T={t_end}")
    lim = cfl_limits(N, mu_s)
    ref = run_monolithic(N, t_end, lim["elastic"] / 4.0, mu_s, theta=0.5)
    print(f"  {'scheme':>12} {'dt/dt_el':>9} {'steps':>6} {'wall(s)':>9} {'rel err':>10}")
    rows = []
    for m in (0.5, 1, 2):
        dt = m * lim["elastic"]
        try:
            e = run_explicit(N, t_end, dt, mu_s)
            rows.append(("explicit", m, e["nstep"], e["wall"], _rel_err(e, ref)))
        except Exception as exc:
            rows.append(("explicit", m, None, None, None))
    for m in (2, 5, 10, 20, 40):
        dt = m * lim["elastic"]
        try:
            r = run_monolithic(N, t_end, dt, mu_s)
            rows.append(("monolithic", m, r["nstep"], r["wall"], _rel_err(r, ref)))
        except Exception:
            rows.append(("monolithic", m, None, None, None))
    for nm, m, ns, w, er in rows:
        if ns is None:
            print(f"  {nm:>12} {m:>9g} {'--':>6} {'diverged':>9} {'--':>10}")
        else:
            print(f"  {nm:>12} {m:>9g} {ns:>6d} {w:>9.2f} {er:>10.3e}")
    return rows


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    if what in ("stability", "all"):
        study_stability()
    if what in ("temporal", "all"):
        study_temporal()
    if what in ("spatial", "all"):
        study_spatial()
    if what in ("workprec", "all"):
        study_workprec()
