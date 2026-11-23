"""Spatial convergence of the disc-in-Taylor-Green case with the current solver (trace-free strain
velocity gradient, bug 1 fix), measured reference-free from successive-grid differences on the
same grids as the paper's study. Fields are sampled on the N=256 cell centres; the bulk mask is
"farther than 0.06 from the N=256 interface". Resumable; writes
outputs/rerun_2026-11/conv_fix1.npz. Compare with outputs/overlay/bulk_conv.npz (old solver).

Usage: python benchmarks/reruns/convergence_after_fix.py [tag]
  tag "fix1"   : bug 1 fix only (trace-free strain velocity gradient)
  tag "fix123" : bugs 1-4: also the compact face force (bug 3), the band-evaluated equilibrium
                 stress (bug 2; inert here, the case has no equilibrium branch) and the consistent
                 conformation mask (bug 4). The tag name predates bug 4.
  tag "band16" : bugs 1-4 with the blend half-width FIXED at 1/16 (a physical length), N=32-256:
                 convergence to one regularized problem, separating the discretization error
                 from the O(width) modelling error of the diffuse interface.
  tag "band16mat" / "mat" : bugs 1 and 3 fixed and the MATERIAL band (reference map and psi
                 transported everywhere instead of cut and extrapolated), at a fixed band width
                 1/16 / at the default 2-cell width.
  tag "matex1" / "matex2" : material band at the default width, with the band-width
                 extrapolation c=2 / c=4/3 (idea 1).
  tag "final"  : the current driver defaults (bug fixes 1, 3, 4 on; band stress off), N=32-256.
  tag "dtsq"   : bug 1 fix only, but dt proportional to dx^2 (equal to the default step at N=32),
                 on N = 32, 64, 128: separates the temporal error from the spatial one.
"""
import os, sys, time, io, contextlib
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from benchmarks.viscoelastic_disc_tg import run_one
from benchmarks.mac_disc_in_tg_convergence import _sample

TAG = sys.argv[1] if len(sys.argv) > 1 else "fix1"
FIXES = dict(compact_force=True, band_stress=True, consistent_mask=True) if TAG in ("fix123", "band16") else {}
if TAG == "band16":
    FIXES["band_width"] = 1.0 / 16.0
if TAG in ("band16mat", "mat", "matex1", "matex2"):   # material band: transport everywhere
    FIXES = dict(compact_force=True, material_band=True)
    if TAG == "matex1":
        FIXES["band_extrap"] = 1                  # 2 f(w) - f(2w): cancels an O(w) error
    if TAG == "matex2":
        FIXES["band_extrap"] = 2                  # (4 f(w) - f(2w))/3: cancels O(w^2)
    if TAG == "band16mat":
        FIXES["band_width"] = 1.0 / 16.0
NPZ = os.path.join(ROOT, "outputs", "rerun_2026-11", f"conv_{TAG}.npz")
SOFT = dict(R=0.15, x0=0.5, y0=0.7, mu_f=0.01, U0=0.3, kw=1)
NS = [256, 32, 48, 64, 96, 128]          # the finest first: it defines the sampling grid
PAIRS = [(32, 64), (48, 96), (64, 128), (128, 256)]
if TAG == "dtsq":
    NS = [128, 32, 64]; PAIRS = [(32, 64), (64, 128)]


def compute():
    d = dict(np.load(NPZ)) if os.path.isfile(NPZ) else {}
    for N in NS:
        if f"sp_{N}" in d:
            continue
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_one(N, 0.6, model="maxwell", G_ve=0.1, tau=np.inf, integrator="imex",
                        scheme="semilagrangian_cubic", mu_s=0.1, history=True, **SOFT, **FIXES,
                        dt_scale=(32.0 / N if TAG == "dtsq" else 1.0))
        if N == NS[0]:
            d["g"] = r["xc"]; d["phig"] = r["phi"]
        d[f"sp_{N}"] = _sample(r["xc"], r["speed"], d["g"]); d[f"pr_{N}"] = _sample(r["xc"], r["p"], d["g"])
        d[f"KE_{N}"] = float(r["KE"]); d[f"SE_{N}"] = float(r["hist"]["Eel"][-1])
        d[f"SEs_{N}"] = float(r["hist"]["Eel_s"][-1])
        d[f"detbe_{N}"] = float(np.max(r["hist"]["detbe_max"]))
        np.savez(NPZ, **d)
        print(time.strftime("%H:%M"), f"conv {TAG} N={N} done: KE={d[f'KE_{N}']:.8f} SE={d[f'SE_{N}']:.6e} "
              f"max det b_e={d[f'detbe_{N}']:.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)
    return d


def orders(d):
    from scipy import ndimage as ndi
    solid = d["phig"] <= 0; n = d["phig"].shape[0]
    dist = np.where(solid, ndi.distance_transform_edt(solid), ndi.distance_transform_edt(~solid)) / n
    mb = dist > 0.06
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    out = {}
    for a, b in PAIRS:
        if f"sp_{a}" not in d or f"sp_{b}" not in d:
            continue
        dv = d[f"sp_{a}"] - d[f"sp_{b}"]; dp = d[f"pr_{a}"] - d[f"pr_{b}"]
        out[(a, b)] = dict(full_v=rms(dv), bulk_v=rms(dv[mb]), full_p=rms(dp), bulk_p=rms(dp[mb]),
                           dKE=d[f"KE_{a}"] - d[f"KE_{b}"], dSE=d[f"SE_{a}"] - d[f"SE_{b}"],
                           dSEs=(d[f"SEs_{a}"] - d[f"SEs_{b}"]) if f"SEs_{a}" in d else np.nan)
    keys = sorted(out)
    for q in ("full_v", "bulk_v", "full_p", "bulk_p", "dKE", "dSE", "dSEs"):
        txt = []
        for (a, b) in keys:
            for (c, e) in keys:
                if c == 2 * a and e == 2 * b:
                    txt.append(f"{a}/{b}->{c}/{e}: {np.log2(abs(out[(a, b)][q]) / abs(out[(c, e)][q])):.2f}")
        print(f"  {q:7s} " + "  ".join(txt), flush=True)
    return out


if __name__ == "__main__":
    orders(compute())
