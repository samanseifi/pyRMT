"""FSI convergence of the soft disc in a Taylor-Green vortex (the *coupled* case, not the
pure fluid). Measures L2 convergence of velocity, pressure, and kinetic + strain (total)
energy vs the finest grid, showing that the diffuse interface caps the FSI fields BELOW the
second order the pure-fluid solver attains. Writes outputs/overlay/disc_tg_convergence.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmarks._plotstyle import apply as _apply_style; _apply_style()
from benchmarks.mac_disc_in_tg_convergence import run_one, _sample

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "overlay")
os.makedirs(OUT, exist_ok=True)


def run(NS=(32, 48, 64, 96, 128), t_end=0.4, scheme="semilagrangian"):
    data = {N: run_one(N, t_end, scheme=scheme) for N in NS}
    for N in NS:
        print(f"  N={N:4d} KE={data[N]['KE']:.5e} Estrain={data[N]['Estrain']:.5e} "
              f"div={data[N]['divmax']:.1e}")
    ref = NS[-1]; coarse = list(NS[:-1]); h = np.array([1.0 / N for N in coarse])
    g = data[ref]['xc']
    spref = _sample(data[ref]['xc'], data[ref]['speed'], g)
    prref = _sample(data[ref]['xc'], data[ref]['p'], g)
    verr = [np.sqrt(np.mean((_sample(data[N]['xc'], data[N]['speed'], g) - spref) ** 2)) for N in coarse]
    perr = [np.sqrt(np.mean((_sample(data[N]['xc'], data[N]['p'], g) - prref) ** 2)) for N in coarse]
    keref = data[ref]['KE']; eref = data[ref]['Estrain']; tref = keref + eref
    keerr = [abs(data[N]['KE'] - keref) for N in coarse]
    terr = [abs((data[N]['KE'] + data[N]['Estrain']) - tref) for N in coarse]

    def order(e):
        e = np.array(e); return np.log(e[:-1] / e[1:]) / np.log(np.array(coarse[1:]) / np.array(coarse[:-1]))
    print(f"\nFSI disc-in-TG convergence (ref N={ref}):")
    for nm, e in (("velocity", verr), ("pressure", perr), ("kinetic E", keerr), ("total E", terr)):
        print(f"  {nm:10s} L2 errs {[f'{x:.2e}' for x in e]}  orders {[f'{o:.2f}' for o in order(e)]}")

    plt.figure(figsize=(6.4, 4.8))
    for nm, e, m in (("velocity", verr, "o"), ("pressure", perr, "s"),
                     ("kinetic energy", keerr, "^"), ("total energy", terr, "D")):
        plt.loglog(h, e, m + "-", lw=1.6, ms=6, label=nm)
    plt.loglog(h, verr[0] * (h / h[0]) ** 2, "k--", lw=1, label="2nd order")
    plt.loglog(h, verr[0] * (h / h[0]) ** 1, "k:", lw=1, label="1st order")
    plt.gca().invert_xaxis(); plt.xlabel("$h=1/N$"); plt.ylabel("$L_2$ error vs finest grid")
    plt.legend(frameon=False, ncol=2, loc="upper left"); plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    p = os.path.join(OUT, "disc_tg_convergence.png"); plt.savefig(p); plt.close()
    print(f"  saved {p}")


if __name__ == "__main__":
    run()
