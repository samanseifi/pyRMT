"""Grid convergence of the two-disc collision, and its dependence on the contact width.

Two elastic discs collide head-on and rebound (configuration modelled on Jain, Kamrin & Mani
2019, their solid-solid collision). The contact stress is the trace-free solid stress of Rycroft
et al. (2020); its activation width eps is a PHYSICAL length here.

What this script shows
  (a) eps tied to the mesh (eps = 3 dx, the original setting): the contact law changes with the
      grid and the rebound does not converge (end gap 0.339 / 0.308 / 0.284 at N = 96/128/192).
  (b) eps fixed (EPS = 1/32): the same collision converges under refinement.
  (c) eps is a model parameter: at fixed resolution the rebound weakens as eps is reduced.

Results are cached in outputs/overlay/contact_conv.npz (keys fixed_<N>, meshtied_<N>, eps_<tag>);
only missing runs are computed. `--plot-only` never runs anything. Writes
paper/figures/contact_convergence.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.contact_collision import run
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "outputs", "overlay", "contact_conv.npz")
NS = [96, 128, 192]
EPS = 1.0 / 32.0                                   # physical contact width (= R/3.84)
PAR = dict(model="elastic", mu_s=0.8, V0=0.9, mu_f=0.005, eta=2.0, sep=0.30, R=0.12,
           integrator="imex", stab=False, t_end=1.6)
FIXED = dict(eps=EPS, ordering="sequential")
MESHTIED = dict(eps_cells=3, band_layers=3, ordering="midpoint")       # the original settings
EPS_SWEEP = {"1/32": 1.0 / 32.0, "3/128": 3.0 / 128.0, "1/64": 1.0 / 64.0}   # at N = 192


def _ekey(tag):
    return "eps_" + tag.replace("/", "o")          # npz keys must not contain "/"


def _cases():
    c = {f"fixed_{N}": dict(N=N, **FIXED) for N in NS}
    c.update({f"meshtied_{N}": dict(N=N, **MESHTIED) for N in NS})
    c.update({_ekey(tag): dict(N=NS[-1], eps=e, ordering="sequential")
              for tag, e in EPS_SWEEP.items() if e != EPS})
    return c


def load():
    return dict(np.load(NPZ)) if os.path.isfile(NPZ) else {}


def compute(only=None):
    d = load()
    for key, kw in _cases().items():
        if key in d or (only is not None and key not in only):
            continue
        d[key] = np.asarray(run(**PAR, **kw)); np.savez(NPZ, **d)
        print(f"  {key}: done", flush=True)
    return d


def report(d):
    R2 = 2 * PAR["R"]
    for fam in ("meshtied", "fixed"):
        keys = [f"{fam}_{N}" for N in NS if f"{fam}_{N}" in d]
        if not keys:
            continue
        print(f"  {fam}: " + "  ".join(
            f"N={k.split('_')[1]} min {np.nanmin(d[k][:, 3]):.4f} end {d[k][-1, 3]:.4f}" for k in keys))
        ref = d[keys[-1]]
        for k in keys[:-1]:
            g = np.interp(ref[:, 0], d[k][:, 0], d[k][:, 3])
            print(f"     gap RMS({k} vs {keys[-1]}) = {np.sqrt(np.mean((g - ref[:, 3])**2)):.2e}")
        print(f"     centroids ever cross: {any(bool(np.any(d[k][:, 1] >= d[k][:, 2])) for k in keys)}"
              f"   (2R = {R2:.3f})")
    for tag in EPS_SWEEP:
        k = f"fixed_{NS[-1]}" if EPS_SWEEP[tag] == EPS else _ekey(tag)
        if k in d:
            print(f"  eps = {tag}: end gap {d[k][-1, 3]:.4f}  min gap {np.nanmin(d[k][:, 3]):.4f}")


def main(recompute=True):
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks")); from _plotstyle import apply as _s; _s()
    import matplotlib.pyplot as plt
    d = compute() if recompute else load()
    report(d)
    cols = {96: "blue", 128: "darkgreen", 192: "red"}
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for a, fam, ttl in ((ax[0], "meshtied", r"(a) width tied to the mesh, $\epsilon=3\Delta x$"),
                        (ax[1], "fixed", r"(b) fixed width, $\epsilon=1/32$")):
        for N in NS:
            if f"{fam}_{N}" in d:
                h = d[f"{fam}_{N}"]; a.plot(h[:, 0], h[:, 3], "-", color=cols[N], lw=1.8, label=f"$N={N}$")
        a.set_title(ttl); a.legend(loc="lower right")
    for (tag, e), c in zip(EPS_SWEEP.items(), ("red", "darkorange", "purple")):
        k = f"fixed_{NS[-1]}" if e == EPS else _ekey(tag)
        if k in d:
            ax[2].plot(d[k][:, 0], d[k][:, 3], "-", color=c, lw=1.8, label=rf"$\epsilon={tag}$")
    ax[2].set_title(rf"(c) dependence on $\epsilon$ ($N={NS[-1]}$)"); ax[2].legend(loc="lower right")
    for a in ax:
        a.axhline(2 * PAR["R"], color="0.6", lw=0.9, ls=":"); a.set_xlabel("time $t$")
    ax[0].set_ylabel("centroid gap $c_b-c_a$")
    ax[0].text(0.02, 2 * PAR["R"], " $2R$", color="0.4", fontsize=9, va="bottom")
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "contact_convergence.png")
    fig.savefig(out, dpi=200); print("saved", out)


if __name__ == "__main__":
    main(recompute="--plot-only" not in sys.argv)
