"""Numbers quoted in the paper text that no figure script prints.

Usage: python benchmarks/reruns/paper_numbers.py <what>
  energy          Taylor-Green energy benchmark (N = 128): first KE dip, SE peak, period, aspect
                  ratio, max det b_e, b_e vs F F^T
  detbe           SLS disc in the four-cell vortex: max det b_e for tau = 1, 0.25
  work_precision  soft disc in the lid cavity to t = 1.5, N = 64, 128: centroid error against an
                  explicit run at half its step, and wall-clock (indicative only)
  bilinear        lid cavity with bilinear transport to t = 16 (against the bicubic run)
  momentum        collision of two unequal discs: drift of the total momentum
  summary         everything that can be read off the saved histories (lid, temporal, SLS, drift,
                  contact, shear)
Results are saved in outputs/$RERUN_DIR/paper_numbers/ and printed.
"""
import os, sys, io, time, contextlib
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
OUT = os.path.join(ROOT, "outputs", os.environ.get("RERUN_DIR", "rerun_2026-11"), "paper_numbers")
OVL = os.path.join(ROOT, "outputs", "overlay")
os.makedirs(OUT, exist_ok=True)
quiet = lambda: contextlib.redirect_stdout(io.StringIO())


def _sugiyama():
    return np.loadtxt(os.path.join(ROOT, "data", "Jain2019_fig17_sugiyama1024.csv"), delimiter=",", skiprows=1)[:, :2]


def _to_ref(tr, pts, tmax=None):
    """mean / max distance from each reference point to the nearest trajectory sample"""
    xy = tr[:, 1:3] if tmax is None else tr[tr[:, 0] <= tmax, 1:3]
    d = np.sqrt(((pts[:, None, :] - xy[None, :, :]) ** 2).sum(-1)).min(1)
    return d.mean(), d.max()


def _matched(a, b):
    """time-matched centroid deviation of a from b: mean, max, time of the max"""
    t = a[:, 0]; d = np.hypot(a[:, 1] - np.interp(t, b[:, 0], b[:, 1]), a[:, 2] - np.interp(t, b[:, 0], b[:, 2]))
    return d.mean(), d.max(), t[d.argmax()]


def energy():
    from benchmarks.viscoelastic_disc_tg import run_one
    from scipy.signal import argrelextrema
    ts = list(np.round(np.arange(0.02, 1.0001, 0.02), 3))
    with quiet():
        r = run_one(128, 1.0, model="maxwell", G_ve=1.0, tau=np.inf, integrator="imex", scheme="semilagrangian_cubic",
                    R=0.2, x0=0.5, y0=0.5, mu_s=1.0, mu_f=1e-3, U0=0.05, kw=2, history=True, snap_times=ts)
    h = r["hist"]; t, KE, SE = h["t"], h["KE"], h["Eel"]
    ar = []
    for s in r["snaps"]:
        m = s["phi"] <= 0; ev = np.linalg.eigvalsh(np.cov(np.vstack([s["Xc"][m], s["Yc"][m]])))
        ar.append((s["t"], np.sqrt(ev[1] / ev[0])))
    ar = np.array(ar)
    np.savez(os.path.join(OUT, "tg_energy.npz"), t=t, KE=KE, SE=SE, SEs=h["Eel_s"], bevsF=h["bevsF"], detbe=h["detbe_max"], ar=ar)
    imn = argrelextrema(KE, np.less, order=8)[0]; imx = argrelextrema(SE, np.greater, order=8)[0]
    print(f"energy benchmark N=128: KE(0+) {KE[0]:.5f} (analytic {(0.05 * 2 * np.pi) ** 2 / 4:.5f})")
    print("  KE minima (t, KE):", [(round(float(t[i]), 3), round(float(KE[i]), 5)) for i in imn])
    print("  SE maxima (t, SE):", [(round(float(t[i]), 3), round(float(SE[i]), 5)) for i in imx])
    print("  periods from KE minima:", np.round(np.diff(t[imn]), 3), " from SE maxima:", np.round(np.diff(t[imx]), 3))
    print(f"  max ||b_e - F F^T||/||F F^T|| {h['bevsF'].max():.4f}   max det b_e {h['detbe_max'].max():.4f}"
          f"   peak aspect ratio {ar[:, 1].max():.3f} at t={ar[ar[:, 1].argmax(), 0]:.2f}", flush=True)


def detbe():
    from benchmarks.viscoelastic_disc_tg import run_one
    for tau in (1.0, 0.25):
        with quiet():
            h = run_one(N=128, t_end=2.0, history=True, kw=2, x0=0.5, y0=0.5, R=0.2, U0=0.2, integrator="imex",
                        model="sls", tau=tau, mu_s=0.5, G_ve=1.5)["hist"]
        print(f"SLS in the four-cell vortex, tau={tau}: max det b_e {h['detbe_max'].max():.3f}  peak stored energy"
              f" {h['Eel'].max():.4f}  min J {h['minJ'].min():.3f}", flush=True)


def work_precision():
    import benchmarks.work_precision as W
    with quiet():
        dxr = 1 / 128; dt_ref = min(0.3 * dxr, 0.2 * dxr * dxr / 0.01)
        ref, _ = W._traj(N=128, dt=dt_ref / 2, integrator="explicit")       # reference: half the explicit step
        res = {N: W.collect(N, ref) for N in (64, 128)}
    for N in res:
        wex = res[N][0][3]
        print(f"N={N}: error = max centroid deviation over t<=1.5 from the N=128 explicit run at half its step")
        for nm, fac, e, wt in res[N]:
            print(f"  {nm:9s} step x{fac:<3} err {e:.2e}  wall {wt:.1f}s  speedup {wex / wt:.2f}", flush=True)
    np.save(os.path.join(OUT, "work_precision.npy"), np.array([(N, fac, e, wt) for N in res for _, fac, e, wt in res[N]]))


def bilinear():
    from benchmarks.mac_soft_disc_lid import run
    N = 128; dx = 1 / N; dt_big = 4 * min(0.3 * dx, 0.2 * dx * dx / 0.01)
    with quiet():
        tr = np.asarray(run(N=N, t_end=16.0, dt=dt_big, integrator="imex-sl", scheme="semilagrangian", mu_s=0.1,
                            viscoelastic=True, G_ve=0.1, tau=np.inf, write=False))
    np.save(os.path.join(OUT, "lid_bilinear.npy"), tr)
    cub = np.load(os.path.join(OVL, "lid_stab_tauinf.npz"))["sls"]; sug = _sugiyama()
    for nm, t in (("bilinear", tr), ("bicubic", cub)):
        xy = t[t[:, 0] <= 16.0, 1:3]
        d = np.sqrt(((xy[:, None, :] - sug[None, :, :]) ** 2).sum(-1)).min(1)       # trajectory -> reference points
        print(f"{nm}: t<=16 (reached t={t[-1, 0]:.1f}), mean distance from the trajectory to the Sugiyama points {d.mean():.2e}")
    n = min(len(tr), len(cub)); d = np.hypot(tr[:n, 1] - cub[:n, 1], tr[:n, 2] - cub[:n, 2])
    print(f"bilinear vs bicubic, time-matched: mean {d.mean():.2e} max {d.max():.2e} at t={tr[d.argmax(), 0]:.1f}", flush=True)


def momentum():
    from benchmarks.contact_collision import run
    for N in (96, 128):
        with quiet():
            h = np.asarray(run(N=N, t_end=1.6, model="elastic", mu_s=0.8, V0=0.9, V0b=0.5, Rb=0.09, mu_f=0.005, eta=2.0,
                               sep=0.30, R=0.12, integrator="imex", stab=False, eps=1 / 32, ordering="sequential"))
        p0 = np.hypot(h[0, 5], h[0, 6])
        print(f"unequal discs N={N}: |p| {p0:.4e}  max relative drift {np.hypot(h[:, 5] - h[0, 5], h[:, 6] - h[0, 6]).max() / p0:.2e}"
              f"  min gap {h[:, 3].min():.4f}", flush=True)
        np.save(os.path.join(OUT, f"unequal_{N}.npy"), h)


def summary():
    sug = _sugiyama()
    T = np.load(os.path.join(OVL, "soft_disc_traj.npz")); S = np.load(os.path.join(OVL, "lid_stab_tauinf.npz"))
    print("LID (N=128)")
    for k in T.files:
        print(f"  reference-map solid, {k:8s}: to Sugiyama mean %.2e max %.2e" % _to_ref(T[k], sug),
              f" t_end {T[k][-1, 0]:.2f}  J in [{T[k][:, 3].min():.2f}, {T[k][:, 4].max():.2f}]")
    print("  log-conformation tau=inf (imex-sl, 4x): to Sugiyama mean %.2e max %.2e" % _to_ref(S["sls"], sug), f" t_end {S['sls'][-1, 0]:.2f}")
    print("  imex vs explicit, time-matched: mean %.2e max %.2e at t=%.1f" % _matched(T["imex"], T["explicit"]))
    print("  imex-sl (4x) vs explicit:       mean %.2e max %.2e at t=%.1f" % _matched(T["imex-sl"], T["explicit"]))
    print("  log-conformation vs reference-map (both imex-sl): mean %.2e max %.2e at t=%.1f" % _matched(S["sls"], S["neo"]))
    d = np.load(os.path.join(OVL, "temporal_conv.npz"))
    print("TEMPORAL errs", d["errs"], "pairwise orders", np.round(np.log2(d["errs"][1:] / d["errs"][:-1]), 2))
    s = np.load(os.path.join(OVL, "sls_energy.npz")); En = np.interp(s["t"], s["t_neo"], s["E_neo"])
    print("SLS peaks: tau=1e3 %.4f  tau=1 %.4f  tau=0.25 %.4f  neo-Hookean %.4f" % (s["E_inf"].max(), s["E_tau1"].max(), s["E_tau025"].max(), s["E_neo"].max()),
          "| elastic limit vs neo: peak %.1f%%, history %.1f%% of the peak" % (100 * abs(s["E_inf"].max() / s["E_neo"].max() - 1), 100 * np.abs(s["E_inf"] - En).max() / s["E_inf"].max()),
          "| at t=2 (fraction of peak):", " ".join(f"{np.interp(2.0, s['t'], s[k]) / s[k].max():.1e}" for k in ("E_inf", "E_tau1", "E_tau025")))
    g = np.load(os.path.join(OVL, "drift_diag.npz"))
    for N in (64, 128):
        print(f"DRIFT N={N}: smooth area max drift {100 * (1 - g[f'areas_{N}'].min()):.2f}% final {100 * (1 - g[f'areas_{N}'][-1]):.2f}%"
              f"  max det b_e {g[f'detbe_max_{N}'].max():.3f}  min J {g[f'minJ_{N}'].min():.3f}  max median ||grad phi|-1| {g[f'gphi_{N}'].max():.3f}")
    c = np.load(os.path.join(OVL, "contact_conv.npz"))
    print("CONTACT (V0 = 0.9)")
    for k in c.files:
        h = c[k]; i = h[:, 3].argmin(); E = h[:, 4] + h[:, 7]
        print(f"  {k:13s} min gap {h[:, 3].min():.4f}  end gap {h[-1, 3]:.4f}  min surface gap {h[:, 8].min():.4f}"
              f"  (KE+W) at max compression {100 * E[i] / h[0, 4]:.1f}% of KE0")
    ref = c["fixed_192"]
    for k in ("fixed_96", "fixed_128", "meshtied_96", "meshtied_128"):
        r2 = c["meshtied_192"] if k.startswith("mesh") else ref; h = c[k]
        print(f"  gap RMS {k} vs N=192: {np.sqrt(np.mean((h[:, 3] - np.interp(h[:, 0], r2[:, 0], r2[:, 3])) ** 2)):.2e}")
    q = np.load(os.path.join(OVL, "contact_collision.npz"))
    for k in q.files:
        if q[k].ndim == 2:
            print(f"  figure run {k}: min gap {q[k][:, 3].min():.4f}  end gap {q[k][-1, 3]:.4f}  min surface gap {q[k][:, 8].min():.4f}")


if __name__ == "__main__":
    {"energy": energy, "detbe": detbe, "work_precision": work_precision, "bilinear": bilinear,
     "momentum": momentum, "summary": summary}[sys.argv[1]]()
