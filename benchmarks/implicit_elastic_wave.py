"""Stage-2 core: energy-conserving IMPLICIT elastic-wave integrator (de-risk).

Model problem: the linear elastic shear wave  d_tt = cs^2 nabla^2 d,  u = d_t, on a
periodic box -- the wave that sets the RMT elastic CFL dt < dx/cs. Standing-wave exact
solution  d = A sin(k x) cos(w t),  w = cs k  (divergence-free, incompressible-consistent).

Three integrators, all one transform (Helmholtz) solve per step:

  explicit  symplectic Euler                       -> CFL-limited (dt < dx/cs)
  stage1    explicit force + dt^2 cs^2 stabilizer  -> unconditionally stable, |lambda|
                                                      = 1/sqrt(1+a): DAMPED
  stage2    trapezoidal (implicit-midpoint) of the (u,d) system
                                                    -> unconditionally stable, |lambda|=1:
                                                      UNDAMPED (energy-conserving)

This isolates the stage-2 improvement over stage-1 (removing the numerical damping via a
consistent, symmetric implicit coupling) before wiring it into the nonlinear FSI loop.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import lap_eigs_periodic, solve_helmholtz_periodic, _lap_per

L = 2 * np.pi


def _grid(N):
    dx = L / N
    x = (np.arange(N)) * dx
    X, Y = np.meshgrid(x, x)
    return dx, X, Y


def exact(X, k, A, cs, t):
    w = cs * k
    d = A * np.sin(k * X) * np.cos(w * t)
    u = -A * w * np.sin(k * X) * np.sin(w * t)
    return d, u


def energy(u, d, cs, dx):
    """Total (kinetic + elastic) energy density, mean over the box."""
    gx = (np.roll(d, -1, 1) - np.roll(d, 1, 1)) / (2 * dx)
    gy = (np.roll(d, -1, 0) - np.roll(d, 1, 0)) / (2 * dx)
    return 0.5 * np.mean(u * u) + 0.5 * cs * cs * np.mean(gx * gx + gy * gy)


def step_explicit(u, d, cs, dx, dt, lap_eig):
    u_new = u + dt * cs * cs * _lap_per(d, dx, dx)     # symplectic Euler
    d_new = d + dt * u_new
    return u_new, d_new


def step_stage1(u, d, cs, dx, dt, lap_eig):
    # (I - dt^2 cs^2 Lap) u_new = u + dt cs^2 Lap(d)     [force inside, stabilizer]
    rhs = u + dt * cs * cs * _lap_per(d, dx, dx)
    u_new = solve_helmholtz_periodic(rhs, lap_eig, dt * dt * cs * cs)
    d_new = d + dt * u_new
    return u_new, d_new


def step_stage2(u, d, cs, dx, dt, lap_eig):
    # trapezoidal (implicit-midpoint) of  u_t = cs^2 Lap d,  d_t = u :
    #   (I - (dt^2/4) cs^2 Lap) u_new = u + dt cs^2 Lap(d) + (dt^2/4) cs^2 Lap(u)
    #   d_new = d + (dt/2)(u + u_new)
    c = 0.25 * dt * dt * cs * cs
    rhs = u + dt * cs * cs * _lap_per(d, dx, dx) + c * _lap_per(u, dx, dx)
    u_new = solve_helmholtz_periodic(rhs, lap_eig, c)
    d_new = d + 0.5 * dt * (u + u_new)
    return u_new, d_new


STEPPERS = {"explicit": step_explicit, "stage1": step_stage1, "stage2": step_stage2}


def run_one(scheme, N=64, k=2, A=1.0, cs=1.0, dt_fac=5.0, n_periods=5.0):
    dx, X, Y = _grid(N)
    dt_cfl = dx / cs
    dt = dt_fac * dt_cfl
    w = cs * k
    T = n_periods * (2 * np.pi / w)
    nsteps = max(1, int(round(T / dt)))
    dt = T / nsteps
    lap_eig = lap_eigs_periodic(N, N, dx, dx)
    d, u = exact(X, k, A, cs, 0.0)
    E0 = energy(u, d, cs, dx)
    step = STEPPERS[scheme]
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(nsteps):
            u, d = step(u, d, cs, dx, dt, lap_eig)
    de, ue = exact(X, k, A, cs, nsteps * dt)
    E = energy(u, d, cs, dx)
    err = np.sqrt(np.mean((d - de) ** 2)) if np.all(np.isfinite(d)) else np.inf
    edrift = abs(E - E0) / E0 if np.isfinite(E) else np.inf
    return dict(dt=dt, dt_cfl=dt_cfl, dt_fac=dt / dt_cfl, err=err, E0=E0, E=E, edrift=edrift)


def run():
    print("Elastic standing shear wave (cs=1, k=2): integrator comparison")
    print(f"  {'scheme':8s} {'dt/dt_cfl':>10s} {'L2 err(d)':>12s} {'energy drift':>14s}")
    for fac in (0.5, 5.0):
        print(f"  -- dt = {fac}x the explicit CFL --")
        for scheme in ("explicit", "stage1", "stage2"):
            r = run_one(scheme, dt_fac=fac, n_periods=5.0)
            print(f"  {scheme:8s} {r['dt_fac']:10.2f} {r['err']:12.3e} {r['edrift']:14.3e}")


if __name__ == "__main__":
    run()
