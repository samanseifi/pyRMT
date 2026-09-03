"""GATE TEST: does the tau->inf log-conformation elastic limit reproduce the neo-Hookean
disc-in-Taylor-Green energy benchmark (Fig. 3, vs Jain/Zhao)?

Decision experiment for unifying the constitutive model on the log-conformation SLS with
elasticity as the tau->inf limit. Compares, at the Jain parameters (N=128):
  neo        -- reference-map neo-Hookean, sigma = mu_s dev(F F^T)   (the current Fig.3 curve)
  tauinf     -- pure log-conformation branch at tau=inf, G_ve=mu_s   (cubic transport)
  tauinf_bl  -- same, bilinear transport (expected to artificially relax -- illustrative)
against the digitized Jain/Zhao kinetic-energy references. If neo and tauinf(cubic) coincide
and track the references, the replacement is sound.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.mac_disc_in_tg_convergence import run_one as run_neo
from benchmarks.viscoelastic_disc_tg import run_one as run_ve
from benchmarks.disc_tg_fsi_convergence import JAIN, _load_jain_methods

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JP = dict(R=0.2, x0=0.5, y0=0.5, mu_s=1.0, mu_f=1e-3, U0=0.05, kw=2)   # Jain params


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    N, T = 128, 1.0
    print("neo (reference-map FFt, bilinear)...", flush=True)
    hneo = run_neo(N, T, integrator="imex", **JAIN)["hist"]
    print("tauinf (log-conf, cubic)...", flush=True)
    hcub = run_ve(N, T, model="maxwell", G_ve=1.0, tau=np.inf, integrator="imex",
                  scheme="semilagrangian_cubic", history=True, **JP)["hist"]
    print("tauinf (log-conf, bilinear)...", flush=True)
    hbl = run_ve(N, T, model="maxwell", G_ve=1.0, tau=np.inf, integrator="imex",
                 scheme="semilagrangian", history=True, **JP)["hist"]
    np.savez(os.path.join(ROOT, "outputs", "overlay", "tg_tauinf_gate.npz"),
             neo_t=hneo["t"], neo_ke=hneo["ke"], neo_se=hneo["se"],
             cub_t=hcub["t"], cub_ke=hcub["KE"], cub_se=hcub["Eel"],
             bl_t=hbl["t"], bl_ke=hbl["KE"], bl_se=hbl["Eel"])

    methods = _load_jain_methods()
    ddir = os.path.join(ROOT, "benchmarks", "data")
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for m, c in (("Jain", "black"), ("Zhao", "blue")):
        p = os.path.join(ddir, f"jain2019_fig14b_{m}_ke.csv")
        if os.path.isfile(p):
            d = np.loadtxt(p, delimiter=",", skiprows=1)
            ax[0].plot(d[:, 0], d[:, 1], color=c, lw=1.2, ls="--", label=f"{m} et al.")
        if methods and f"{m}_se" in methods:
            t, y = methods[f"{m}_se"]; ax[1].plot(t, y, color=c, lw=1.2, ls="--", label=f"{m} et al.")
    ax[0].plot(hneo["t"], hneo["ke"], "-", color="tab:red", lw=2.0, label="neo-Hookean")
    ax[0].plot(hcub["t"], hcub["KE"], "--", color="tab:green", lw=1.8, label=r"$\tau\!\to\!\infty$ log-conf (cubic)")
    ax[0].plot(hbl["t"], hbl["KE"], ":", color="tab:orange", lw=1.6, label=r"$\tau\!\to\!\infty$ (bilinear)")
    ax[1].plot(hneo["t"], hneo["se"], "-", color="tab:red", lw=2.0, label="neo-Hookean")
    ax[1].plot(hcub["t"], hcub["Eel"], "--", color="tab:green", lw=1.8, label=r"$\tau\!\to\!\infty$ (cubic)")
    ax[1].plot(hbl["t"], hbl["Eel"], ":", color="tab:orange", lw=1.6, label=r"$\tau\!\to\!\infty$ (bilinear)")
    ax[0].set_title("(a) kinetic energy"); ax[0].set_xlabel("time $t$"); ax[0].set_xlim(0, 1)
    ax[1].set_title("(b) strain energy"); ax[1].set_xlabel("time $t$"); ax[1].set_xlim(0, 1)
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(ROOT, "outputs", "overlay", "tg_tauinf_gate.png")
    fig.savefig(out, dpi=170); print("saved", out)

    def rms(a, b):  # over the shorter overlap
        n = min(len(a), len(b)); return float(np.sqrt(np.mean((np.array(a[:n]) - np.array(b[:n]))**2)))
    print(f"\nKE  rms(neo, cubic)={rms(hneo['ke'], hcub['KE']):.3e}   rms(neo, bilinear)={rms(hneo['ke'], hbl['KE']):.3e}")
    print(f"SE  rms(neo, cubic)={rms(hneo['se'], hcub['Eel']):.3e}   rms(neo, bilinear)={rms(hneo['se'], hbl['Eel']):.3e}")
    print(f"final SE: neo={hneo['se'][-1]:.4e}  cubic={hcub['Eel'][-1]:.4e}  bilinear={hbl['Eel'][-1]:.4e}")


if __name__ == "__main__":
    main()
