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
