"""Finite-strain viscoelastic constitutive update for the reference-map solver.

Carries an elastic left-Cauchy-Green (Finger) tensor ``b_e`` evolved by the
upper-convected Maxwell (UCM) model with relaxation time ``tau``:

    D b_e / Dt - L b_e - b_e L^T = -(1/tau) (b_e - I),     L = grad(u)

and the elastic (Maxwell-branch) Cauchy stress ``sigma = G * b_e`` (the trace is
carried by the pressure, so using ``b_e`` vs ``dev(b_e)`` only shifts pressure).
A standard-linear-solid (Zener) material adds a non-relaxing equilibrium branch
``G_inf * (F F^T)`` from the reference map so the body still returns to shape.

Limits / checks (verified in benchmarks/viscoelastic_relaxation_test.py):
  * step shear strain gamma, then hold:  sigma_xy(t) = G gamma exp(-t/tau)
  * steady shear rate gdot:               sigma_xy = G tau gdot,
                                          N1 = sigma_xx - sigma_yy = 2 G tau^2 gdot^2
  * tau -> infinity:                      D b_e/Dt = L b_e + b_e L^T  (pure elastic)

b_e and L are passed as separate symmetric / full 2x2 components so the same
routines work on scalars (0-D constitutive test) and on (Ny,Nx) field arrays
(full solver).  Components:  b_e = [[b11, b12],[b12, b22]],
L = [[L11, L12],[L21, L22]]  with  L_ij = d u_i / d x_j.
"""
import numpy as np


def ucm_local_rhs(b11, b12, b22, L11, L12, L21, L22, tau):
    """Right-hand side of the *homogeneous* (no-advection) UCM evolution
    d b_e/dt = L b_e + b_e L^T - (1/tau)(b_e - I), returned per component.

    The advective term -(u.grad)b_e is handled separately by the solver's
    reference-map advection, so this is the part that is local to a material
    point (and is all that a 0-D constitutive test needs)."""
    # D = L b + b L^T  (symmetric)
    D11 = 2.0 * (L11 * b11 + L12 * b12)
    D12 = (L11 * b12 + L12 * b22) + (L21 * b11 + L22 * b12)
    D22 = 2.0 * (L21 * b12 + L22 * b22)
    inv_tau = 0.0 if tau == np.inf else 1.0 / tau
    r11 = D11 - inv_tau * (b11 - 1.0)
    r12 = D12 - inv_tau * b12
    r22 = D22 - inv_tau * (b22 - 1.0)
    return r11, r12, r22


def ucm_local_step(b11, b12, b22, L11, L12, L21, L22, tau, dt):
    """Advance b_e one step under the homogeneous UCM evolution with explicit
    RK2 (Heun).  Inputs may be scalars or equally-shaped arrays."""
    k1 = ucm_local_rhs(b11, b12, b22, L11, L12, L21, L22, tau)
    p11 = b11 + dt * k1[0]; p12 = b12 + dt * k1[1]; p22 = b22 + dt * k1[2]
    k2 = ucm_local_rhs(p11, p12, p22, L11, L12, L21, L22, tau)
    b11 = b11 + 0.5 * dt * (k1[0] + k2[0])
    b12 = b12 + 0.5 * dt * (k1[1] + k2[1])
    b22 = b22 + 0.5 * dt * (k1[2] + k2[2])
    return b11, b12, b22


# ── log-conformation (Fattal & Kupferman 2004) ──────────────────────────────
# Evolve psi = log(b_e) instead of b_e, so b_e = exp(psi) is SPD by construction
# (even diffusive advection cannot make it indefinite or blow up). Required for
# the grid solver: advecting b_e directly is either too diffusive (artificial
# relaxation) or, if non-dissipative, loses positive-definiteness and explodes.

def _sym_eig2(a, b, c):
    """Eigen-decomposition of the symmetric 2x2 field [[a,b],[b,c]]: eigenvalues
    l1>=l2 and rotation angle theta (eigenvector e1 = (cos,sin) for l1)."""
    tr = 0.5 * (a + c)
    d = np.sqrt(np.maximum(0.25 * (a - c)**2 + b * b, 0.0))
    return tr + d, tr - d, 0.5 * np.arctan2(2.0 * b, a - c)


def sym_log(a, b, c):
    """Matrix log of symmetric 2x2 SPD field, returned as components."""
    l1, l2, th = _sym_eig2(a, b, c)
    f1 = np.log(np.maximum(l1, 1e-300)); f2 = np.log(np.maximum(l2, 1e-300))
    cs, sn = np.cos(th), np.sin(th)
    return (cs*cs*f1 + sn*sn*f2, cs*sn*(f1 - f2), sn*sn*f1 + cs*cs*f2)


def sym_exp(a, b, c):
    """Matrix exp of symmetric 2x2 field, returned as components (always SPD)."""
    l1, l2, th = _sym_eig2(a, b, c)
    f1 = np.exp(l1); f2 = np.exp(l2)
    cs, sn = np.cos(th), np.sin(th)
    return (cs*cs*f1 + sn*sn*f2, cs*sn*(f1 - f2), sn*sn*f1 + cs*cs*f2)


def logconf_local_rhs(p11, p12, p22, L11, L12, L21, L22, tau):
    """RHS of the homogeneous log-conformation evolution of psi=log(b_e):
    Dpsi/Dt = (Omega psi - psi Omega) + 2B + (1/tau)(exp(-psi) - I),
    with B (extension) and Omega (rotation) from the velocity gradient L
    decomposed in the eigenframe of psi (Fattal-Kupferman). 2x2 closed form."""
    l1, l2, th = _sym_eig2(p11, p12, p22)            # eigenvalues/frame of psi
    cs, sn = np.cos(th), np.sin(th)
    # M = R^T L R  in the eigenframe (R = [[cs,-sn],[sn,cs]])
    a11 = cs*L11 + sn*L21; a12 = cs*L12 + sn*L22
    a21 = -sn*L11 + cs*L21; a22 = -sn*L12 + cs*L22
    M11 = a11*cs + a12*sn; M12 = -a11*sn + a12*cs
    M21 = a21*cs + a22*sn; M22 = -a21*sn + a22*cs
    lam1 = np.exp(l1); lam2 = np.exp(l2)             # eigenvalues of b_e
    # eigenframe off-diagonal of Dpsi: ratio*(lam2 M12 + lam1 M21), where
    # ratio = (l2-l1)/(lam2-lam1) -> exp(-l1) in the degenerate limit l1=l2.
    # (At c=I this gives M12+M21, recovering Dpsi=2 sym(L); for l1!=l2 it is the
    # eigenvector-rotation rate -- both captured by one smooth expression.)
    denom = lam2 - lam1
    deg = np.abs(denom) < 1e-10
    ratio = np.where(deg, np.exp(-l1), (l2 - l1) / np.where(deg, 1.0, denom))
    inv_tau = 0.0 if tau == np.inf else 1.0/tau
    # eigenframe RHS: diagonal = 2*extension + relaxation; off-diagonal as above
    E11 = 2.0*M11 + inv_tau*(np.exp(-l1) - 1.0)
    E22 = 2.0*M22 + inv_tau*(np.exp(-l2) - 1.0)
    E12 = ratio * (lam2*M12 + lam1*M21)
    # rotate E back to lab frame: R E R^T
    r11 = cs*cs*E11 - 2.0*cs*sn*E12 + sn*sn*E22
    r12 = cs*sn*(E11 - E22) + (cs*cs - sn*sn)*E12
    r22 = sn*sn*E11 + 2.0*cs*sn*E12 + cs*cs*E22
    return r11, r12, r22


def logconf_local_step(p11, p12, p22, L11, L12, L21, L22, tau, dt):
    """Advance psi=log(b_e) one homogeneous step (RK2). Reconstruct b_e via sym_exp."""
    k1 = logconf_local_rhs(p11, p12, p22, L11, L12, L21, L22, tau)
    q1 = p11 + dt*k1[0]; q2 = p12 + dt*k1[1]; q3 = p22 + dt*k1[2]
    k2 = logconf_local_rhs(q1, q2, q3, L11, L12, L21, L22, tau)
    return (p11 + 0.5*dt*(k1[0]+k2[0]), p12 + 0.5*dt*(k1[1]+k2[1]),
            p22 + 0.5*dt*(k1[2]+k2[2]))


def viscoelastic_stress(b11, b12, b22, G, b_ref=None, G_inf=0.0):
    """Cauchy stress components from the elastic Finger tensor.

    Maxwell branch:           sigma = G b_e.
    Standard-linear-solid:    sigma = G b_e + G_inf (F F^T), pass the equilibrium
                              branch b_ref = (bxx, bxy, byy) from the reference map.
    """
    sxx = G * b11; sxy = G * b12; syy = G * b22
    if b_ref is not None and G_inf != 0.0:
        sxx = sxx + G_inf * b_ref[0]
        sxy = sxy + G_inf * b_ref[1]
        syy = syy + G_inf * b_ref[2]
    return sxx, sxy, syy
