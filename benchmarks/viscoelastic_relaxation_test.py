"""Verify the finite-strain viscoelastic constitutive update (upper-convected
Maxwell, pyRMT/viscoelastic.py) against analytical responses, with NO solver/FSI
-- a 0-D constitutive test that isolates the model.

Test 1 -- stress relaxation after a step shear strain gamma (then hold):
    b_e(0) = F F^T,  F = [[1, gamma],[0,1]];  L = 0 for t>0
    => b_e(t) = I + (b_e(0)-I) e^{-t/tau}
    => sigma_xy(t) = G (b_e)_xy = G gamma e^{-t/tau}        (pure exponential)
  This exercises the RELAXATION term.

Test 2 -- steady simple shear at rate gdot (UCM viscometric functions):
    L = [[0, gdot],[0,0]];  integrate b_e from I to steady state
    => (b_e)_xy = tau gdot,  (b_e)_xx = 1 + 2 tau^2 gdot^2,  (b_e)_yy = 1
    => sigma_xy = G tau gdot,   N1 = sigma_xx - sigma_yy = 2 G tau^2 gdot^2
  This exercises the UPPER-CONVECTED (L b + b L^T) terms.

Test 3 -- tau -> infinity recovers pure elasticity: under a constant L the UCM
  derivative is zero for b_e = F F^T with dF/dt = L F, so b_e tracks F F^T exactly.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.viscoelastic import ucm_local_step


def test_step_relaxation(G=1.0, tau=2.0, gamma=0.5, t_end=10.0, dt=2e-4):
    # step strain: b_e(0) = F F^T, F = [[1,gamma],[0,1]]
    b11, b12, b22 = 1.0 + gamma**2, gamma, 1.0
    n = int(round(t_end / dt))
    err = 0.0
    for k in range(1, n + 1):
        b11, b12, b22 = ucm_local_step(b11, b12, b22, 0., 0., 0., 0., tau, dt)
        t = k * dt
        exact = gamma * np.exp(-t / tau)            # (b_e)_xy analytical
        err = max(err, abs(b12 - exact))
    sxy_final = G * b12
    print(f"  [relaxation] gamma={gamma} tau={tau}: max|b_xy - gamma*exp(-t/tau)| = {err:.2e}")
    return err


def test_steady_shear(G=1.0, tau=2.0, gdot=0.3, t_end=80.0, dt=2e-4):
    b11, b12, b22 = 1.0, 0.0, 1.0                   # start undeformed
    L11, L12, L21, L22 = 0.0, gdot, 0.0, 0.0        # simple shear u=(gdot*y, 0)
    n = int(round(t_end / dt))
    for _ in range(n):
        b11, b12, b22 = ucm_local_step(b11, b12, b22, L11, L12, L21, L22, tau, dt)
    bxy_ex = tau * gdot
    bxx_ex = 1.0 + 2.0 * tau**2 * gdot**2
    byy_ex = 1.0
    sxy = G * b12; N1 = G * (b11 - b22)
    sxy_ex = G * tau * gdot; N1_ex = 2.0 * G * tau**2 * gdot**2
    e_bxy = abs(b12 - bxy_ex); e_bxx = abs(b11 - bxx_ex); e_byy = abs(b22 - byy_ex)
    print(f"  [steady shear] gdot={gdot} tau={tau}:")
    print(f"     b_xy={b12:.6f} (exact {bxy_ex:.6f}, err {e_bxy:.2e})")
    print(f"     b_xx={b11:.6f} (exact {bxx_ex:.6f}, err {e_bxx:.2e})")
    print(f"     b_yy={b22:.6f} (exact {byy_ex:.6f}, err {e_byy:.2e})")
    print(f"     sigma_xy={sxy:.6f} (exact {sxy_ex:.6f});  N1={N1:.6f} (exact {N1_ex:.6f})")
    return max(e_bxy, e_bxx, e_byy)


def test_elastic_limit(gdot=0.4, t_end=2.0, dt=2e-4):
    # tau -> inf: b_e should track F F^T exactly under constant L
    b11, b12, b22 = 1.0, 0.0, 1.0
    F11, F12, F21, F22 = 1.0, 0.0, 0.0, 1.0
    L11, L12, L21, L22 = 0.0, gdot, 0.0, 0.0
    n = int(round(t_end / dt))
    err = 0.0
    for _ in range(n):
        b11, b12, b22 = ucm_local_step(b11, b12, b22, L11, L12, L21, L22, np.inf, dt)
        # F evolves by dF/dt = L F (same RK2 for a fair comparison)
        def fr(a11, a12, a21, a22):
            return (L11*a11 + L12*a21, L11*a12 + L12*a22,
                    L21*a11 + L22*a21, L21*a12 + L22*a22)
        k1 = fr(F11, F12, F21, F22)
        g11, g12, g21, g22 = (F11+dt*k1[0], F12+dt*k1[1], F21+dt*k1[2], F22+dt*k1[3])
        k2 = fr(g11, g12, g21, g22)
        F11 += 0.5*dt*(k1[0]+k2[0]); F12 += 0.5*dt*(k1[1]+k2[1])
        F21 += 0.5*dt*(k1[2]+k2[2]); F22 += 0.5*dt*(k1[3]+k2[3])
        bb11 = F11*F11 + F12*F12; bb12 = F11*F21 + F12*F22; bb22 = F21*F21 + F22*F22
        err = max(err, abs(b11-bb11), abs(b12-bb12), abs(b22-bb22))
    print(f"  [elastic limit tau=inf] max|b_e - F F^T| = {err:.2e}")
    return err


if __name__ == "__main__":
    print("[viscoelastic] 0-D constitutive verification (upper-convected Maxwell)")
    e1 = test_step_relaxation()
    e2 = test_steady_shear()
    e3 = test_elastic_limit()
    ok = (e1 < 1e-3) and (e2 < 1e-3) and (e3 < 1e-6)
    print(f"[viscoelastic] {'PASS' if ok else 'FAIL'} (relax {e1:.1e}, steady {e2:.1e}, elastic {e3:.1e})")
    sys.exit(0 if ok else 1)
