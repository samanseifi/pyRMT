"""Soft disc in a lid-driven cavity: monolithic vs explicit RMT.

Companion to ``paper/monolithic_aml.tex``.  This is the second of the two standard
elastic RMT-FSI benchmarks (the first is ``monolithic_disc_tg.py``), and the one
that answers the efficiency question honestly, because the cavity is *flow*-driven:
at the soft modulus of the Sugiyama (2011) validation case the binding explicit
restriction is advection, not the elastic wave, so lifting the elastic CFL alone
buys nothing.  The speedup study therefore sweeps the modulus as well as dt.

Studies:
  ``validate``  centroid trajectory at the Sugiyama parameters, monolithic vs the
                explicit driver vs the published Sugiyama 1024^2 data.
  ``speedup``   wall-clock to a fixed end time at increasing dt, for the explicit
                scheme, the existing semi-implicit ``imex-elastic-sl`` integrator,
                and the monolithic solver, at a soft and a stiff modulus.

Usage: python benchmarks/monolithic_soft_disc_lid.py [validate|speedup|all]
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.monolithic import MonolithicRMT

# ── Sugiyama (2011) soft-disc-in-cavity configuration ───────────────────────
X0, Y0, R_DISC = 0.6, 0.5, 0.2
MU_F, RHO, U_LID = 0.01, 1.0, 1.0


def phi0(X, Y):
    return np.sqrt((X - X0) ** 2 + (Y - Y0) ** 2) - R_DISC


def cfl_limits(N, mu_s):
    dx = 1.0 / N
    return dict(advective=0.3 * dx / U_LID,
                viscous=0.2 * dx * dx * RHO / MU_F,
                elastic=0.3 * dx / np.sqrt(mu_s / RHO))


def run_monolithic(N, t_end, dt, mu_s, theta=1.0, isochoric=True, **kw):
    """``isochoric=True`` matches the explicit driver's ``isochoric`` stress, so an
    error column compares TIME discretizations rather than two different stresses.
    (The default non-deviatoric sigma = mu_s*b of the explicit driver is not usable
    as a reference for a stiff solid at all -- its isotropic part is not in the
    range of the discrete pressure gradient and injects energy.)"""
    s = MonolithicRMT(N, phi0, bc="lid", U_lid=U_LID, mu_s=mu_s, mu_f=MU_F,
                      rho=RHO, theta=theta, layers=5, isochoric=isochoric, **kw)
    s.set_velocity(np.zeros((N, N + 1)), np.zeros((N + 1, N)))
    s.seed_reference_map()
    xc = (np.arange(N) + 0.5) * s.dx
    Xc, Yc = np.meshgrid(xc, xc)
    traj = []
    t0 = time.time()
    while s.t < t_end - 1e-12:
        st = s.step(min(dt, t_end - s.t))
        m = s.phi <= 0
        if not m.any() or not np.all(np.isfinite(s.u)):
            raise FloatingPointError(f"monolithic lost the disc at t={s.t:.3f}")
        traj.append((s.t, Xc[m].mean(), Yc[m].mean(), st["min_det"]))
    return dict(scheme="monolithic", N=N, dt=dt, traj=np.array(traj),
                wall=time.time() - t0, nstep=len(traj),
                matvec=float(np.mean([x["gmres"] for x in s.stats])),
                newton=float(np.mean([x["newton"] for x in s.stats])),
                minJ=min(x["min_det"] for x in s.stats), solver=s)


def run_reference(N, t_end, dt, mu_s, integrator=None, isochoric=True):
    """The explicit / semi-implicit driver already in the repo."""
    from benchmarks.mac_soft_disc_lid import run as run_mac
    t0 = time.time()
    traj = run_mac(N=N, t_end=t_end, dt=dt, mu_s=mu_s, integrator=integrator,
                   isochoric=isochoric, write=False)
    wall = time.time() - t0
    traj = np.asarray(traj)
    if traj.size == 0 or traj[-1, 0] < 0.98 * t_end:
        raise FloatingPointError("explicit driver diverged before t_end")
    return dict(scheme=integrator or "explicit", N=N, dt=dt, traj=traj,
                wall=wall, nstep=len(traj), minJ=float(traj[:, 3].min()))


def _traj_err(a, b):
    """RMS centroid separation, a resampled onto b's times."""
    ta, tb = a["traj"][:, 0], b["traj"][:, 0]
    t = tb[tb <= min(ta[-1], tb[-1])]
    ax = np.interp(t, ta, a["traj"][:, 1]); ay = np.interp(t, ta, a["traj"][:, 2])
    bx = np.interp(t, tb, b["traj"][:, 1]); by = np.interp(t, tb, b["traj"][:, 2])
    return float(np.sqrt(np.mean((ax - bx) ** 2 + (ay - by) ** 2)))


# ── study 1: validation against Sugiyama ────────────────────────────────────
def study_validate(N=128, t_end=8.0, mu_s=0.1, out_root="outputs"):
    lim = cfl_limits(N, mu_s)
    dt_e = min(lim.values())
    print(f"\n{'='*78}\nLID CAVITY VALIDATION -- N={N}, mu_s={mu_s}, T={t_end}")
    print(f"  explicit limits: advective {lim['advective']:.2e}  elastic "
          f"{lim['elastic']:.2e}  viscous {lim['viscous']:.2e}  -> dt_exp={dt_e:.2e}")
    ref = run_reference(N, t_end, dt_e, mu_s)
    print(f"  explicit   : {ref['nstep']:5d} steps  {ref['wall']:7.1f}s  "
          f"end=({ref['traj'][-1,1]:.4f},{ref['traj'][-1,2]:.4f})  minJ={ref['minJ']:.3f}")
    rows = [ref]
    for m in (1, 4):
        try:
            r = run_monolithic(N, t_end, dt_e * m, mu_s)
            print(f"  monolithic x{m}: {r['nstep']:5d} steps  {r['wall']:7.1f}s  "
                  f"end=({r['traj'][-1,1]:.4f},{r['traj'][-1,2]:.4f})  "
                  f"minJ={r['minJ']:.3f}  vs explicit RMS={_traj_err(r, ref):.2e}", flush=True)
            rows.append(r)
        except Exception as e:
            print(f"  monolithic x{m}: FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
    _plot(rows, N, out_root)
    return rows


def _plot(rows, N, out_root):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (plot skipped: {e})"); return
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    for nm, fn in (("Sugiyama $1024^2$", "Sugiyama_1024x1024.csv"),
                   ("Kolahdouz 2023", "Kolahduz_2023.csv")):
        pth = os.path.join(data, fn)
        if os.path.isfile(pth):
            d = np.loadtxt(pth, delimiter=",")
            ax.plot(d[:, 0], d[:, 1], "o", ms=3.5, mfc="none", label=nm)
    for r in rows:
        lbl = r["scheme"] + (f" (dt={r['dt']:.1e})" if r["scheme"] == "monolithic" else "")
        ax.plot(r["traj"][:, 1], r["traj"][:, 2], "-", lw=1.8, label=lbl)
    ax.set_xlabel("centroid $x$"); ax.set_ylabel("centroid $y$")
    ax.set_title(f"Soft disc in a lid-driven cavity ($N={N}$)")
    ax.legend(fontsize=8); ax.axis("equal"); fig.tight_layout()
    out = os.path.join(out_root, "monolithic_soft_disc_lid")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "centroid_compare.png"), dpi=150)
    print(f"  saved {out}/centroid_compare.png")


# ── study 2: is there a speedup? ────────────────────────────────────────────
def study_speedup(N=64, t_end=2.0, moduli=(0.1, 10.0, 100.0)):
    """Wall-clock to a fixed end time, at the largest dt each scheme survives.

    The comparison includes ``imex-elastic-sl`` -- the repo's existing
    semi-implicit integrator, which already lifts every explicit CFL by operator
    splitting -- because beating plain forward Euler is not the interesting bar.
    """
    for mu_s in moduli:
        lim = cfl_limits(N, mu_s)
        dt_e = min(lim.values())
        binding = min(lim, key=lim.get)
        print(f"\n{'='*78}\nSPEEDUP -- N={N}, mu_s={mu_s} (c_s={np.sqrt(mu_s/RHO):.2f}), T={t_end}")
        print(f"  limits: advective {lim['advective']:.2e}  elastic {lim['elastic']:.2e}"
              f"  viscous {lim['viscous']:.2e}   binding: {binding}")
        try:
            ref = run_reference(N, t_end, dt_e / 2, mu_s)
        except Exception as e:
            print(f"  reference FAILED: {e}"); continue
        print(f"  reference: explicit @ dt_exp/2, {ref['nstep']} steps, {ref['wall']:.1f}s")
        print(f"  {'scheme':>20} {'dt/dt_exp':>10} {'steps':>6} {'wall(s)':>9} "
              f"{'traj RMS':>9} {'matvec':>7} {'speedup':>8}")
        base = None
        for m in (1, 2, 4):
            try:
                r = run_reference(N, t_end, dt_e * m, mu_s)
                if base is None:
                    base = r["wall"]
                print(f"  {'explicit':>20} {m:>10g} {r['nstep']:>6d} {r['wall']:>9.2f} "
                      f"{_traj_err(r, ref):>9.2e} {'-':>7} {'1.00x' if m==1 else '':>8}", flush=True)
            except Exception:
                print(f"  {'explicit':>20} {m:>10g} {'-':>6} {'DIVERGED':>9}", flush=True)
        for m in (2, 4, 8, 16):
            for nm, integ in (("imex-elastic-sl", "imex-elastic-sl"), ("monolithic", None)):
                try:
                    if integ:
                        r = run_reference(N, t_end, dt_e * m, mu_s, integrator=integ)
                        mv = "-"
                    else:
                        r = run_monolithic(N, t_end, dt_e * m, mu_s)
                        mv = f"{r['matvec']:.0f}"
                    sp = f"{base / r['wall']:.2f}x" if base else "-"
                    print(f"  {nm:>20} {m:>10g} {r['nstep']:>6d} {r['wall']:>9.2f} "
                          f"{_traj_err(r, ref):>9.2e} {mv:>7} {sp:>8}", flush=True)
                except Exception as e:
                    print(f"  {nm:>20} {m:>10g} {'-':>6} {'FAILED':>9} "
                          f"{type(e).__name__}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("validate", "all"):
        study_validate()
    if what in ("speedup", "all"):
        study_speedup()
