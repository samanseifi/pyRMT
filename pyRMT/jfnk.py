"""Jacobian-free Newton-Krylov (JFNK) monolithic time step -- stage 1 (incompressible
fluid), the proof-of-route for the fully-monolithic RMT (backlog #14.4, stage 2d).

Backward-Euler incompressible Navier-Stokes on the periodic staggered grid, solved as ONE
coupled system for X = (u, v, p) rather than the fractional-step split:

    R_u = (u - u^n)/dt + (u.grad)u + (1/rho) dp/dx - nu lap(u) = 0
    R_v = (v - v^n)/dt + (u.grad)v + (1/rho) dp/dy - nu lap(v) = 0
    R_p = div(u, v)                                            = 0

Newton with a Jacobian-free Krylov solve (J w ~ (R(X+eps w) - R(X))/eps, GMRES). The
preconditioner is PHYSICS-BASED: one step of the existing operator-split IMEX solver
(implicit-viscosity Helmholtz + pressure projection) approximately inverts the coupled
system (Knoll & Keyes 2004). This is the pattern the full (u,p,xi,psi) monolithic solver
will reuse -- the split solver becomes the preconditioner and Newton removes the split
error.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from pyRMT.mac import (lap_eigs_periodic, poisson_eigs_periodic, project_per,
                       _v_at_u_per, _u_at_v_per)


def _adv_u_per(u, v, dx, dy):
    dudx = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx)
    dudy = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2 * dy)
    return u * dudx + _v_at_u_per(v) * dudy


def _adv_v_per(u, v, dx, dy):
    dvdx = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2 * dx)
    dvdy = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dy)
    return _u_at_v_per(u) * dvdx + v * dvdy


def _lap_per(q, dx, dy):
    return ((np.roll(q, -1, 1) - 2 * q + np.roll(q, 1, 1)) / dx**2
            + (np.roll(q, -1, 0) - 2 * q + np.roll(q, 1, 0)) / dy**2)


def _gradx_per(p, dx):   # d/dx to the same (periodic) staggered u location
    return (p - np.roll(p, 1, 1)) / dx


def _grady_per(p, dy):
    return (p - np.roll(p, 1, 0)) / dy


def _div_per(u, v, dx, dy):
    return (np.roll(u, -1, 1) - u) / dx + (np.roll(v, -1, 0) - v) / dy


def _pack(u, v, p):
    return np.concatenate([u.ravel(), v.ravel(), p.ravel()])


def _unpack(X, N):
    n = N * N
    return X[:n].reshape(N, N), X[n:2 * n].reshape(N, N), X[2 * n:].reshape(N, N)


def residual(X, un, vn, nu, dx, dy, dt, rho, N):
    u, v, p = _unpack(X, N)
    Ru = (u - un) / dt + _adv_u_per(u, v, dx, dy) + _gradx_per(p, dx) / rho - nu * _lap_per(u, dx, dy)
    Rv = (v - vn) / dt + _adv_v_per(u, v, dx, dy) + _grady_per(p, dy) / rho - nu * _lap_per(v, dx, dy)
    Rp = _div_per(u, v, dx, dy)
    Rp = Rp - Rp.mean()                       # pressure defined up to a constant
    return _pack(Ru, Rv, Rp)


def _precond(r, nu, dx, dy, dt, rho, N, lap_eig, peig):
    """Physics-based preconditioner: approximately invert the coupled residual by one
    implicit-viscosity solve for the velocity plus a pressure projection."""
    ru, rv, rp = _unpack(r, N)
    # velocity: (I/dt - nu lap) du = ru  -> FFT Helmholtz with coef = nu*dt on (dt*ru)
    du = np.real(np.fft.ifft2(np.fft.fft2(dt * ru) / (1.0 - nu * dt * lap_eig)))
    dv = np.real(np.fft.ifft2(np.fft.fft2(dt * rv) / (1.0 - nu * dt * lap_eig)))
    # pressure: lap(dp) = rp  (project the divergence residual)
    dphat = np.fft.fft2(rp) / peig; dphat[0, 0] = 0.0
    dp = np.real(np.fft.ifft2(dphat))
    return _pack(du, dv, dp)


def step(un, vn, nu, dx, dy, dt, rho=1.0, newton_tol=1e-8, max_newton=20,
         gmres_tol=1e-3, info=None):
    """Advance one backward-Euler step by JFNK. Returns (u, v, p). If `info` is a dict it
    receives 'newton_iters', 'gmres_iters', 'res0', 'res'."""
    N = un.shape[0]
    lap_eig = lap_eigs_periodic(N, N, dx, dy)
    peig = poisson_eigs_periodic(N, N, dx, dy)
    X = _pack(un.copy(), vn.copy(), np.zeros((N, N)))
    R = residual(X, un, vn, nu, dx, dy, dt, rho, N)
    res0 = np.linalg.norm(R); newton_iters = 0; gmres_total = 0
    for _ in range(max_newton):
        nrm = np.linalg.norm(R)
        if nrm <= newton_tol * max(res0, 1.0):
            break
        eps = 1e-7 * (np.linalg.norm(X) / max(np.linalg.norm(R), 1e-30) + 1.0)

        def matvec(w):
            return (residual(X + eps * w, un, vn, nu, dx, dy, dt, rho, N) - R) / eps

        J = LinearOperator((X.size, X.size), matvec=matvec)
        M = LinearOperator((X.size, X.size),
                           matvec=lambda r: _precond(r, nu, dx, dy, dt, rho, N, lap_eig, peig))
        gi = [0]
        dX, _ = gmres(J, -R, M=M, rtol=gmres_tol, atol=0.0, maxiter=200,
                      callback=lambda rk: gi.__setitem__(0, gi[0] + 1))
        X = X + dX
        R = residual(X, un, vn, nu, dx, dy, dt, rho, N)
        newton_iters += 1; gmres_total += gi[0]
    u, v, p = _unpack(X, N)
    if info is not None:
        info.update(newton_iters=newton_iters, gmres_iters=gmres_total,
                    res0=res0, res=np.linalg.norm(R))
    return u, v, p
