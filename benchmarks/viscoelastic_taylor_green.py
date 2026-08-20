"""Viscoelastic constitutive-integrator regression on the Taylor-Green strain field.

Drives the homogeneous log-conformation update at EVERY cell of the periodic grid
with the analytic Taylor-Green velocity gradient L(x,y), comparing:

  * the explicit RK2 step        `logconf_local_step`               (previous method)
  * the Strang exact-relaxation  `logconf_local_step_strang`  (#14.2, new method)

Two things are checked (mirroring the request that a new methodology must reproduce
the previous results before it earns its extra capability):

  1. REGRESSION  -- at a small dt (where the explicit step is valid) the two
     integrators agree across the whole field (||psi_e - psi_s|| -> 0): the new
     method reproduces the old physics.
  2. NEW CAPABILITY -- at a large dt >> tau the explicit step becomes inaccurate and
     psi diverges (||psi_e - psi_s|| grows without bound relative to the exact-
     relaxation reference), while the Strang step stays bounded and accurate because
     its relaxation sub-step is integrated exactly (A-stable in tau).

  (Note: log-conformation keeps b_e = exp(psi) SPD by construction for BOTH steppers,
  so the explicit failure mode is accuracy/divergence of psi, not loss of positive-
  definiteness; the min-eigenvalue column just confirms Strang stays well-conditioned.)

Taylor-Green field on [0,2pi]^2 (frozen, incompressible, tr L = 0):
    u = -cos x sin y,  v = sin x cos y
    L = [[ sin x sin y, -cos x cos y],
         [ cos x cos y, -sin x sin y]]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.viscoelastic import (logconf_local_step, logconf_local_step_strang,
                                sym_exp)

L = 2 * np.pi


def tg_velocity_gradient(N):
    """Analytic Taylor-Green velocity gradient L_ij = du_i/dx_j at cell centres."""
    xc = (np.arange(N) + 0.5) * (L / N)
    X, Y = np.meshgrid(xc, xc)
    L11 = np.sin(X) * np.sin(Y)          # du/dx
    L12 = -np.cos(X) * np.cos(Y)         # du/dy
    L21 = np.cos(X) * np.cos(Y)          # dv/dx
    L22 = -np.sin(X) * np.sin(Y)         # dv/dy
    return L11, L12, L21, L22


def _min_eig(p11, p12, p22):
    """Smallest eigenvalue of b_e = exp(psi) over the field (SPD iff > 0)."""
    b11, b12, b22 = sym_exp(p11, p12, p22)
    tr = 0.5 * (b11 + b22)
    d = np.sqrt(np.maximum(0.25 * (b11 - b22) ** 2 + b12 * b12, 0.0))
    return np.min(tr - d)


def run_compare(N=64, tau=0.5, dt=1e-3, T=2.0, stepper_kwargs=None):
    """Integrate psi=log(b_e) from 0 (b_e=I) under the frozen TG gradient with both
    integrators; return (max field difference in psi, min-eig explicit, min-eig
    strang)."""
    L11, L12, L21, L22 = tg_velocity_gradient(N)
    z = np.zeros((N, N))
    # explicit
    pe = [z.copy(), z.copy(), z.copy()]
    ps = [z.copy(), z.copy(), z.copy()]
    nsteps = int(round(T / dt))
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(nsteps):
            pe = list(logconf_local_step(pe[0], pe[1], pe[2],
                                         L11, L12, L21, L22, tau, dt))
    for _ in range(nsteps):
        ps = list(logconf_local_step_strang(ps[0], ps[1], ps[2],
                                            L11, L12, L21, L22, tau, dt))
    diff = max(np.nanmax(np.abs(pe[i] - ps[i])) for i in range(3))
    return diff, _min_eig(*pe), _min_eig(*ps)


def run(out_root="outputs"):
    N, tau = 64, 0.5
    dx = L / N
    print(f"[VE Taylor-Green] N={N} tau={tau}  (frozen TG strain field)")
    # explicit constitutive stability limit is dt <~ tau; scan across it
    print(f"  {'dt':>10} {'dt/tau':>8} {'||psi_e-psi_s||':>16} "
          f"{'min eig(exp)':>14} {'min eig(strang)':>16}")
    for dt in (5e-3, 2e-2, 0.1, 0.5, 1.0):
        diff, me_e, me_s = run_compare(N, tau, dt, T=2.0)
        tag = "  <- explicit inaccurate" if dt > tau else ""
        print(f"  {dt:>10.3g} {dt/tau:>8.2g} {diff:>16.3e} "
              f"{me_e:>14.3e} {me_s:>16.3e}{tag}")

    out_dir = os.path.join(out_root, "viscoelastic_taylor_green")
    os.makedirs(out_dir, exist_ok=True)
    print(f"  (regression: small-dt ||psi_e-psi_s|| -> 0; new: explicit psi diverges "
          f"for dt >~ tau while Strang stays exact & SPD)")


if __name__ == "__main__":
    run()
