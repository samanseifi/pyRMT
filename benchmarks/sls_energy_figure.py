"""Fig.: stored elastic energy of the standard-linear-solid (SLS) disc in a Taylor-Green
vortex, and the tau->inf validation. The SLS has an equilibrium neo-Hookean branch (mu_s)
plus a relaxing Maxwell branch (G, log-conformation). At tau->inf the relaxing branch stops
relaxing, so the SLS must reduce to a pure neo-Hookean solid of the *total* modulus mu_s+G.
We verify this: SLS at large tau (mu_s=G=1) overlays a pure neo-Hookean run at mu_s=2.
Finite tau (=1, 0.25) dissipates the relaxing branch.

Writes paper/figures/sls_energy.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.viscoelastic_disc_tg import run_one

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, T = 128, 2.0
# four-cell Taylor-Green (kw=2), disc at the extensional centre -- Jain's disc-in-TG setup,
# where the solid genuinely stretches (a single vortex would only rotate it).
FLOW = dict(kw=2, x0=0.5, y0=0.5, R=0.2, U0=0.2, integrator="imex")


def eel(**kw):
    h = run_one(N=N, t_end=T, history=True, **FLOW, **kw)["hist"]
    return np.asarray(h["t"]), np.asarray(h["Eel"])


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    # relaxing branch dominant (mu_s=0.5, G=1.5) so relaxation is prominent; total
    # modulus mu_s+G=2 is unchanged, so tau->inf still equals a neo-Hookean mu_s=2 (validation).
    MU, G = 0.5, 1.5
    t_sls_inf, E_sls_inf = eel(model="sls", tau=1e3, mu_s=MU, G_ve=G)
    t_neo,     E_neo     = eel(model="elastic", mu_s=2.0)
    t_tau1,  E_tau1  = eel(model="sls", tau=1.0,  mu_s=MU, G_ve=G)
    t_tau025,E_tau025 = eel(model="sls", tau=0.25, mu_s=MU, G_ve=G)
    np.savez(os.path.join(ROOT, "outputs", "overlay", "sls_energy.npz"),
             t=t_sls_inf, E_inf=E_sls_inf, t_neo=t_neo, E_neo=E_neo,
             E_tau1=E_tau1, E_tau025=E_tau025)

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.plot(t_sls_inf, E_sls_inf, "-", color="k", lw=2.0,
            label=r"$\tau\!\to\!\infty$ (elastic limit)")
    ax.plot(t_tau1, E_tau1, "-", color="blue", lw=1.7, label=r"SLS $\tau=1$")
    ax.plot(t_tau025, E_tau025, "-", color="red", lw=1.7, label=r"SLS $\tau=0.25$")
    ax.set_xlabel("time $t$"); ax.set_ylabel("stored elastic energy $E_{\\mathrm{el}}$")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "sls_energy.png")
    fig.savefig(out, dpi=200); print("saved", out)
    # report the tau->inf match quantitatively
    Ei = np.interp(t_neo, t_sls_inf, E_sls_inf)
    denom = np.max(np.abs(E_neo)) + 1e-30
    print(f"tau->inf vs neo-Hookean: max|dE|/max|E| = {np.max(np.abs(Ei-E_neo))/denom:.2e}")


if __name__ == "__main__":
    main()
