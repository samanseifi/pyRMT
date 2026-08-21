"""Staggered (Marker-and-Cell) operators and projection.

Layout for Nx x Ny cells on [0,Lx] x [0,Ly], dx=Lx/Nx, dy=Ly/Ny:

    p : (Ny, Nx)      cell centres,  xc_i=(i+0.5)dx,  yc_j=(j+0.5)dy
    u : (Ny, Nx+1)    x-faces,       x_i = i dx,      yc_j
    v : (Ny+1, Nx)    y-faces,       xc_i,            y_j = j dy

u[:,0]/u[:,Nx] and v[0,:]/v[Ny,:] are the domain-boundary (wall) faces.

The discrete divergence (cell centres), pressure gradient (to faces) and Laplacian
(div of grad) are consistent by construction, so the projection is exact: the
projected velocity is divergence-free to machine precision (no checkerboard, no
Rhie-Chow needed).
"""

import numpy as np
from scipy.fft import dctn, idctn
from scipy.sparse.linalg import cg, LinearOperator


def mac_grid(Nx, Ny, Lx=1.0, Ly=1.0):
    return Lx / Nx, Ly / Ny


# ── Time-integration strategy selector ───────────────────────────────────────
# One knob for the whole MAC/RMT program. Each level makes one more stiff term
# implicit (the strategy is a stack, not a menu of unrelated options):
#
#   explicit      forward-Euler everything; dt limited by viscous, relaxation and
#                 the elastic wave (backlog baseline).
#   imex          implicit (backward-Euler) viscosity via the transform/CG Helmholtz
#                 (#14.1) + exact/exponential relaxation where a relaxation time
#                 exists (#14.2). Lifts the viscous & dt<~tau CFLs.
#   imex-elastic  additionally the linearly-implicit elastic-wave stabilizer (#14.4
#                 stage 1). Lifts the elastic-wave CFL dt<dx/cs. Viscoelastic driver
#                 only (needs the solid-stress force path).
#
# resolve_integrator() maps the unified `integrator=` name to internal (imex,
# elastic) booleans, and stays backward-compatible with the old per-driver flags
# (implicit_visc / imex / elastic_imex) when integrator is left None.

INTEGRATORS = ("explicit", "imex", "imex-elastic")


def resolve_integrator(integrator=None, imex=False, elastic_imex=False,
                       implicit_visc=False, supports_elastic=True):
    """Return (canonical_name, use_imex, use_elastic).

    integrator : one of INTEGRATORS (aliases accepted), or None to fall back to the
                 legacy booleans. supports_elastic=False makes a driver reject the
                 elastic strategy with a clear error instead of silently ignoring it.
    """
    if integrator is None:
        if elastic_imex:
            name = "imex-elastic"
        elif imex or implicit_visc:
            name = "imex"
        else:
            name = "explicit"
    else:
        key = str(integrator).lower().replace("_", "-").strip()
        aliases = {
            "explicit": "explicit", "exp": "explicit", "forward-euler": "explicit",
            "imex": "imex", "imex-visc": "imex", "implicit-viscosity": "imex",
            "semi-implicit": "imex",
            "imex-elastic": "imex-elastic", "elastic": "imex-elastic",
            "implicit-elastic": "imex-elastic",
        }
        if key not in aliases:
            raise ValueError(
                f"unknown integrator {integrator!r}; choose from {INTEGRATORS}")
        name = aliases[key]
    if name == "imex-elastic" and not supports_elastic:
        raise ValueError(
            "integrator='imex-elastic' is not available for this driver "
            "(no solid-stress force path); use 'explicit' or 'imex'")
    return name, name in ("imex", "imex-elastic"), name == "imex-elastic"


def divergence(u, v, dx, dy):
    """Cell-centred divergence of a staggered velocity. u:(Ny,Nx+1), v:(Ny+1,Nx)
    -> (Ny, Nx)."""
    return (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy


def gradient_p_u(p, dx):
    """d p/dx at u-faces. p:(Ny,Nx) -> (Ny,Nx+1). Wall faces (i=0,Nx) are 0
    (no pressure correction of the zero wall-normal velocity)."""
    Ny, Nx = p.shape
    g = np.zeros((Ny, Nx + 1))
    g[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dx
    return g


def gradient_p_v(p, dy):
    """d p/dy at v-faces. p:(Ny,Nx) -> (Ny+1,Nx). Wall faces (j=0,Ny) are 0."""
    Ny, Nx = p.shape
    g = np.zeros((Ny + 1, Nx))
    g[1:-1, :] = (p[1:, :] - p[:-1, :]) / dy
    return g


def poisson_eigs_neumann(Nx, Ny, dx, dy):
    """Eigenvalues of the cell-centred Neumann Laplacian (dp/dn=0 at the walls),
    diagonalised by DCT-II.  lambda_k = -2(1-cos(pi k/N))/h^2.  The (0,0) constant
    mode is pinned to 1 (handled in the solve)."""
    kx = np.arange(Nx)
    ky = np.arange(Ny)
    lx = -2.0 * (1.0 - np.cos(np.pi * kx / Nx)) / dx**2
    ly = -2.0 * (1.0 - np.cos(np.pi * ky / Ny)) / dy**2
    eig = lx[np.newaxis, :] + ly[:, np.newaxis]
    eig = eig.copy()
    eig[0, 0] = 1.0
    return eig


def solve_poisson_neumann(rhs, eig):
    """Solve lap(p) = rhs with homogeneous-Neumann BC via DCT-II (mean removed)."""
    rhat = dctn(rhs, type=2, norm='ortho')
    phat = rhat / eig
    phat[0, 0] = 0.0
    return idctn(phat, type=2, norm='ortho')


def project(u_star, v_star, dx, dy, dt, rho, eig):
    """Project a staggered velocity onto the divergence-free space.

    Solves lap(phi) = (rho/dt) div(u*) and returns
    (u, v, p) with u = u* - (dt/rho) grad(phi), divergence-free to machine
    precision.  `rho` is a scalar (constant density).
    """
    div = divergence(u_star, v_star, dx, dy)
    rhs = (rho / dt) * div
    rhs = rhs - rhs.mean()                      # enforce solvability (zero mean)
    phi = solve_poisson_neumann(rhs, eig)
    u = u_star - (dt / rho) * gradient_p_u(phi, dx)
    v = v_star - (dt / rho) * gradient_p_v(phi, dy)
    return u, v, phi


# ── Lid-driven cavity: momentum (advection + diffusion) with ghost-cell BCs ───
# Walls: no-slip. Top lid moves at U_lid (tangential to the x-walls... at the top
# y-wall). Normal velocity at every wall is zero (u[:,0]=u[:,-1]=0, v[0,:]=v[-1,:]=0);
# tangential no-slip / lid is imposed via reflected ghost rows/cols.

def _u_ghost_y(u, U_lid):
    """Pad u (Ny,Nx+1) with one ghost row top & bottom enforcing tangential BC:
    bottom wall u=0 -> ghost=-u[0]; top lid u=U_lid -> ghost=2*U_lid-u[-1]."""
    Ny, Nxp1 = u.shape
    up = np.empty((Ny + 2, Nxp1))
    up[1:-1, :] = u
    up[0, :] = -u[0, :]                 # bottom no-slip
    up[-1, :] = 2.0 * U_lid - u[-1, :]  # top lid
    return up


def _v_ghost_x(v):
    """Pad v (Ny+1,Nx) with one ghost col left & right enforcing no-slip (v=0)."""
    Nyp1, Nx = v.shape
    vp = np.empty((Nyp1, Nx + 2))
    vp[:, 1:-1] = v
    vp[:, 0] = -v[:, 0]
    vp[:, -1] = -v[:, -1]
    return vp


def _v_at_u(v):
    """Interpolate v (Ny+1,Nx) to the interior u-faces (Ny, Nx-1)."""
    # u-face (i, j) sits between v[j,i-1],v[j,i],v[j+1,i-1],v[j+1,i]
    return 0.25 * (v[:-1, :-1] + v[:-1, 1:] + v[1:, :-1] + v[1:, 1:])


def _u_at_v(u):
    """Interpolate u (Ny,Nx+1) to the interior v-faces (Ny-1, Nx)."""
    return 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])


def interfacial_force_faces(kappa, H, gamma, dx, dy):
    """Continuum-surface-force at faces: f = -gamma * kappa * grad(H), with kappa
    (cell centres) interpolated to faces and grad(H) the SAME compact face gradient
    as the pressure gradient -> balanced-force by construction. Returns (fu, fv)
    on the u-faces (Ny,Nx+1) and v-faces (Ny+1,Nx); wall faces are 0."""
    Ny, Nx = H.shape
    fu = np.zeros((Ny, Nx + 1))
    fv = np.zeros((Ny + 1, Nx))
    # u-faces (interior i=1..Nx-1): kappa interp in x, grad H compact in x
    ku = 0.5 * (kappa[:, 1:] + kappa[:, :-1])
    fu[:, 1:-1] = -gamma * ku * (H[:, 1:] - H[:, :-1]) / dx
    # v-faces (interior j=1..Ny-1)
    kv = 0.5 * (kappa[1:, :] + kappa[:-1, :])
    fv[1:-1, :] = -gamma * kv * (H[1:, :] - H[:-1, :]) / dy
    return fu, fv


def momentum_predictor(u, v, nu, dx, dy, dt, U_lid, fu=None, fv=None, rho=1.0):
    """One explicit predictor step (central advection + diffusion) for the
    lid-driven cavity, plus optional face body forces fu (Ny,Nx+1)/fv (Ny+1,Nx)
    (e.g. surface tension) added as fu/rho. Returns u*, v* with wall faces zeroed."""
    Ny, Nxp1 = u.shape
    Nx = Nxp1 - 1
    up = _u_ghost_y(u, U_lid)            # (Ny+2, Nx+1)
    vp = _v_ghost_x(v)                   # (Ny+1, Nx+2)

    # --- u-momentum at interior u-faces i=1..Nx-1 ---
    uc = u[:, 1:-1]                                  # (Ny, Nx-1)
    dudx = (u[:, 2:] - u[:, :-2]) / (2 * dx)         # (Ny, Nx-1)
    dudy = (up[2:, 1:-1] - up[:-2, 1:-1]) / (2 * dy) # (Ny, Nx-1)
    lapu = ((u[:, 2:] - 2 * uc + u[:, :-2]) / dx**2
            + (up[2:, 1:-1] - 2 * up[1:-1, 1:-1] + up[:-2, 1:-1]) / dy**2)
    v_u = _v_at_u(v)                                 # (Ny, Nx-1)
    rhs_u = -(uc * dudx + v_u * dudy) + nu * lapu
    if fu is not None:
        rhs_u = rhs_u + fu[:, 1:-1] / rho
    ustar = u.copy()
    ustar[:, 1:-1] = uc + dt * rhs_u
    ustar[:, 0] = 0.0; ustar[:, -1] = 0.0

    # --- v-momentum at interior v-faces j=1..Ny-1 ---
    vc = v[1:-1, :]                                  # (Ny-1, Nx)
    dvdy = (v[2:, :] - v[:-2, :]) / (2 * dy)
    dvdx = (vp[1:-1, 2:] - vp[1:-1, :-2]) / (2 * dx)
    lapv = ((vp[1:-1, 2:] - 2 * vp[1:-1, 1:-1] + vp[1:-1, :-2]) / dx**2
            + (v[2:, :] - 2 * vc + v[:-2, :]) / dy**2)
    u_v = _u_at_v(u)                                 # (Ny-1, Nx)
    rhs_v = -(u_v * dvdx + vc * dvdy) + nu * lapv
    if fv is not None:
        rhs_v = rhs_v + fv[1:-1, :] / rho
    vstar = v.copy()
    vstar[1:-1, :] = vc + dt * rhs_v
    vstar[0, :] = 0.0; vstar[-1, :] = 0.0
    return ustar, vstar


# ── IMEX wall/lid implicit viscosity (matrix-free CG Helmholtz) — backlog #14.1 ──
# The lid-cavity velocity BCs are Dirichlet (no-slip walls + moving lid), so the
# viscous Helmholtz operator (I - dt*nu*Laplacian) does not diagonalise under the
# DCT used for the (Neumann) pressure. Instead we solve it matrix-free with CG,
# reusing the SAME ghost-cell viscous stencil as the explicit predictor -- this
# guarantees the implicit solve is consistent with the validated explicit operator
# and generalises to variable (one-fluid) viscosity. SPD -> CG converges quickly.

def _lap_u_lid_hom(u, dx, dy):
    """Homogeneous-BC Laplacian of u (Ny,Nx+1) on the interior faces (Ny,Nx-1):
    no-slip walls u=0 at i=0,Nx and reflect ghosts (U_lid=0) top & bottom."""
    Ny, Nxp1 = u.shape
    up = np.empty((Ny + 2, Nxp1)); up[1:-1] = u; up[0] = -u[0]; up[-1] = -u[-1]
    uc = u[:, 1:-1]
    return ((u[:, 2:] - 2 * uc + u[:, :-2]) / dx**2
            + (up[2:, 1:-1] - 2 * up[1:-1, 1:-1] + up[:-2, 1:-1]) / dy**2)


def _lap_v_lid_hom(v, dx, dy):
    """Homogeneous-BC Laplacian of v (Ny+1,Nx) on the interior faces (Ny-1,Nx):
    no-slip v=0 at j=0,Ny and reflect ghosts left & right."""
    Nyp1, Nx = v.shape
    vp = np.empty((Nyp1, Nx + 2)); vp[:, 1:-1] = v; vp[:, 0] = -v[:, 0]; vp[:, -1] = -v[:, -1]
    vc = v[1:-1, :]
    return ((vp[1:-1, 2:] - 2 * vp[1:-1, 1:-1] + vp[1:-1, :-2]) / dx**2
            + (v[2:, :] - 2 * vc + v[:-2, :]) / dy**2)


def _cg_helmholtz(rhs_int, apply_lap_hom, embed, coef, rtol=1e-10, maxiter=500):
    """Solve (I - coef*Lap_hom) x = rhs_int on interior DOFs by CG. apply_lap_hom
    takes the full (BC-embedded) field and returns the interior Laplacian; embed
    places an interior vector into a full zero-padded field."""
    shp = rhs_int.shape

    def matvec(xflat):
        x = xflat.reshape(shp)
        return (x - coef * apply_lap_hom(embed(x))).ravel()

    A = LinearOperator((rhs_int.size, rhs_int.size), matvec=matvec)
    sol, info = cg(A, rhs_int.ravel(), x0=rhs_int.ravel(), rtol=rtol, atol=0.0, maxiter=maxiter)
    return sol.reshape(shp)


def _dst_helmholtz_eigs(shp, dx, dy):
    """Eigenvalues of the homogeneous-Dirichlet Laplacian on `shp` cell-centred DOFs,
    diagonalized by the DST-II. lambda_k = -2(1-cos(pi (k+1)/N))/h^2."""
    Ny, Nx = shp
    lx = -2.0 * (1.0 - np.cos(np.pi * (np.arange(Nx) + 1) / Nx)) / dx**2
    ly = -2.0 * (1.0 - np.cos(np.pi * (np.arange(Ny) + 1) / Ny)) / dy**2
    return ly[:, None] + lx[None, :]


def _pcg_helmholtz(rhs_int, apply_lap_hom, embed, coef, dx, dy,
                   rtol=1e-8, maxiter=500, count=None):
    """Preconditioned CG for (I - coef*Lap_hom) x = rhs_int. The constant-coefficient
    operator is ill-conditioned at high wavenumber for large `coef`; a DST spectral
    preconditioner M^{-1} = (I - coef*Lap_Dirichlet)^{-1} collapses the spectrum, cutting
    iterations from O(10-100) to a handful (the fix for the net-speedup problem). The
    preconditioner need not be exact -- CG converges on the true wall operator regardless.
    If `count` is a list, the iteration count is appended to it."""
    from scipy.fft import dstn, idstn
    shp = rhs_int.shape
    lap_eig = _dst_helmholtz_eigs(shp, dx, dy)
    denom = 1.0 - coef * lap_eig                       # >= 1, safe to divide

    def matvec(xflat):
        x = xflat.reshape(shp)
        return (x - coef * apply_lap_hom(embed(x))).ravel()

    def prec(rflat):
        r = rflat.reshape(shp)
        return idstn(dstn(r, type=2, norm='ortho') / denom, type=2, norm='ortho').ravel()

    A = LinearOperator((rhs_int.size, rhs_int.size), matvec=matvec)
    M = LinearOperator((rhs_int.size, rhs_int.size), matvec=prec)
    its = [0]
    cb = (lambda xk: its.__setitem__(0, its[0] + 1))
    sol, info = cg(A, rhs_int.ravel(), x0=rhs_int.ravel(), rtol=rtol, atol=0.0,
                   maxiter=maxiter, M=M, callback=cb)
    if count is not None:
        count.append(its[0])
    return sol.reshape(shp)


def momentum_predictor_lid_imex(u, v, nu, dx, dy, dt, U_lid, fu=None, fv=None,
                                rho=1.0, rtol=1e-8, cs2=0.0):
    """IMEX predictor for the lid-driven cavity: explicit central advection + face
    forces, implicit (backward-Euler) viscous diffusion solved by preconditioned CG
    (DST spectral preconditioner -> a handful of iterations, independent of N). Same
    advection and BCs as momentum_predictor, but the viscous CFL dt < dx^2/(4 nu) is
    removed.

    cs2 > 0 additionally activates the linearly-implicit TRAPEZOIDAL elastic-wave
    stabilizer (#14.4 stage 2b): the implicit operator gains a constant
    (dt^2/4) cs^2 Lap term and the RHS the matching (dt^2/4) cs^2 Lap(u^n) term, so the
    elastic force (passed in fu/fv, already blended by the solid fraction) is advanced
    by the energy-conserving implicit-midpoint rule rather than the dissipative stage-1
    stabilizer. The constant-coefficient form keeps the fast PCG solve; the O(dt^2)
    elastic term in the fluid is a harmless, consistent perturbation. Returns u*, v*."""
    Ny, Nxp1 = u.shape; Nx = Nxp1 - 1
    up = _u_ghost_y(u, U_lid); vp = _v_ghost_x(v)
    c_el = 0.25 * dt * dt * cs2
    coef = dt * nu + c_el

    # --- u: explicit advection + force (+ trapezoidal elastic RHS) -> PCG solve ---
    uc = u[:, 1:-1]
    dudx = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    dudy = (up[2:, 1:-1] - up[:-2, 1:-1]) / (2 * dy)
    v_u = _v_at_u(v)
    rhs_u = uc + dt * (-(uc * dudx + v_u * dudy))
    if fu is not None:
        rhs_u = rhs_u + dt * fu[:, 1:-1] / rho
    if c_el > 0.0:
        rhs_u = rhs_u + c_el * _lap_u_lid_hom(u, dx, dy)   # trapezoidal term (u^n)
    rhs_u[-1, :] += coef * (2.0 * U_lid / dy**2)           # lid inhomogeneity -> RHS
    embed_u = lambda x: np.pad(x, ((0, 0), (1, 1)))        # walls (i=0,Nx) = 0
    sol_u = _pcg_helmholtz(rhs_u, lambda w: _lap_u_lid_hom(w, dx, dy), embed_u,
                           coef, dx, dy, rtol)
    ustar = u.copy(); ustar[:, 1:-1] = sol_u; ustar[:, 0] = 0.0; ustar[:, -1] = 0.0

    # --- v: explicit advection + force (+ trapezoidal elastic RHS) -> PCG solve ---
    vc = v[1:-1, :]
    dvdy = (v[2:, :] - v[:-2, :]) / (2 * dy)
    dvdx = (vp[1:-1, 2:] - vp[1:-1, :-2]) / (2 * dx)
    u_v = _u_at_v(u)
    rhs_v = vc + dt * (-(u_v * dvdx + vc * dvdy))
    if fv is not None:
        rhs_v = rhs_v + dt * fv[1:-1, :] / rho
    if c_el > 0.0:
        rhs_v = rhs_v + c_el * _lap_v_lid_hom(v, dx, dy)
    embed_v = lambda x: np.pad(x, ((1, 1), (0, 0)))        # walls (j=0,Ny) = 0
    sol_v = _pcg_helmholtz(rhs_v, lambda w: _lap_v_lid_hom(w, dx, dy), embed_v,
                           coef, dx, dy, rtol)
    vstar = v.copy(); vstar[1:-1, :] = sol_v; vstar[0, :] = 0.0; vstar[-1, :] = 0.0
    return ustar, vstar


# ── Periodic MAC (for Taylor-Green convergence) ──────────────────────────────
# Periodic layout (Nx x Ny): u,v both (Ny,Nx); u at x-faces (i dx,(j+.5)dy),
# v at y-faces ((i+.5)dx, j dy), p at centres. All operators use np.roll.

def divergence_per(u, v, dx, dy):
    return (np.roll(u, -1, 1) - u) / dx + (np.roll(v, -1, 0) - v) / dy


def grad_p_u_per(p, dx):
    return (p - np.roll(p, 1, 1)) / dx           # d/dx to x-face


def grad_p_v_per(p, dy):
    return (p - np.roll(p, 1, 0)) / dy           # d/dy to y-face


def poisson_eigs_periodic(Nx, Ny, dx, dy):
    kx = np.arange(Nx); ky = np.arange(Ny)
    lx = -4.0 * np.sin(np.pi * kx / Nx) ** 2 / dx**2
    ly = -4.0 * np.sin(np.pi * ky / Ny) ** 2 / dy**2
    eig = lx[np.newaxis, :] + ly[:, np.newaxis]
    eig = eig.copy(); eig[0, 0] = 1.0
    return eig


def solve_poisson_periodic(rhs, eig):
    rhat = np.fft.fft2(rhs)
    phat = rhat / eig; phat[0, 0] = 0.0
    return np.real(np.fft.ifft2(phat))


def project_per(u_star, v_star, dx, dy, dt, rho, eig):
    div = divergence_per(u_star, v_star, dx, dy)
    phi = solve_poisson_periodic((rho / dt) * div, eig)
    return (u_star - (dt / rho) * grad_p_u_per(phi, dx),
            v_star - (dt / rho) * grad_p_v_per(phi, dy), phi)


def _v_at_u_per(v):
    return 0.25 * (v + np.roll(v, 1, 1) + np.roll(v, -1, 0)
                   + np.roll(np.roll(v, -1, 0), 1, 1))


def _u_at_v_per(u):
    return 0.25 * (u + np.roll(u, -1, 1) + np.roll(u, 1, 0)
                   + np.roll(np.roll(u, 1, 0), -1, 1))


# ── IMEX: implicit (backward-Euler) viscous diffusion — backlog #14.1 ─────────
# The explicit predictors are limited by the parabolic stability bound dt < dx^2/(4 nu).
# Treating the viscous term implicitly removes it: the Helmholtz operator
# (I - dt*nu*Laplacian) diagonalises under the SAME FFT used for the pressure Poisson
# solve, so one extra transform-solve per component per step buys an unconditionally-
# stable diffusion. Advection stays explicit (this is the de-risking IMEX step; the
# semi-Lagrangian FSI advection is already unconditionally stable).

def lap_eigs_periodic(Nx, Ny, dx, dy):
    """Eigenvalues of the periodic discrete Laplacian (the np.roll 3-point stencil),
    diagonalised by the FFT.  lambda_k = -4 sin^2(pi k/N)/h^2 <= 0.  NOT pinned: the
    k=0 mode is the true eigenvalue 0, and the Helmholtz symbol (1 - coef*lambda) = 1
    there, so it is invertible without a pin (unlike the pure-Poisson solve)."""
    kx = np.arange(Nx); ky = np.arange(Ny)
    lx = -4.0 * np.sin(np.pi * kx / Nx) ** 2 / dx**2
    ly = -4.0 * np.sin(np.pi * ky / Ny) ** 2 / dy**2
    return lx[np.newaxis, :] + ly[:, np.newaxis]


def solve_helmholtz_periodic(rhs, lap_eig, coef):
    """Solve (I - coef*Laplacian) x = rhs on the periodic grid via FFT.
    coef = dt*nu >= 0, so the symbol (1 - coef*lap_eig) >= 1 -> unconditionally
    invertible and unconditionally stable (backward Euler on the diffusion term)."""
    rhat = np.fft.fft2(rhs)
    xhat = rhat / (1.0 - coef * lap_eig)
    return np.real(np.fft.ifft2(xhat))


def momentum_predictor_periodic_imex(u, v, nu, dx, dy, dt, lap_eig):
    """IMEX predictor on the periodic staggered grid: explicit central advection,
    implicit (backward-Euler) viscous diffusion.  Solves

        (I - dt*nu*Laplacian) u* = u - dt*(u.grad)u

    per velocity component, removing the viscous CFL (dt < dx^2/4nu).  dt is then
    limited only by advection.  `lap_eig` from lap_eigs_periodic(Nx,Ny,dx,dy)."""
    # explicit advection (same central stencil as the explicit predictor)
    dudx = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx)
    dudy = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2 * dy)
    adv_u = u * dudx + _v_at_u_per(v) * dudy

    dvdx = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2 * dx)
    dvdy = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dy)
    adv_v = _u_at_v_per(u) * dvdx + v * dvdy

    ustar = solve_helmholtz_periodic(u - dt * adv_u, lap_eig, dt * nu)
    vstar = solve_helmholtz_periodic(v - dt * adv_v, lap_eig, dt * nu)
    return ustar, vstar


# ── Linearly-implicit elastic stabilizer (periodic) — backlog #14.4, stage 1 ──
# Lifts the elastic-wave CFL dt < dx/cs (cs = sqrt(mu_s/rho)) that survives the IMEX
# viscous+relaxation lift. The nonlinear elastic force f = div(sigma_el) is kept
# EXPLICIT (physics), but a linearly-implicit O(dt^2) wave operator dt^2 cs^2 chi Lap
# is added implicitly (chi = solid indicator). A 1D von-Neumann analysis of the
# linear elastic wave (u_t = cs^2 d_xx, d_t = u) gives amplification |lambda| =
# 1/sqrt(1+a), a = dt^2 cs^2 k^2 -> UNCONDITIONALLY STABLE (mildly damped) PROVIDED
# the force enters the implicit RHS (adding it after the solve is unstable at large dt).
# Frozen-coefficient FFT split: a constant c_el = dt^2 cs^2 is treated implicitly
# everywhere and removed from the fluid by an explicit defect, so the solve stays one
# FFT divide (same transform as pressure/viscosity).

def _lap_per(q, dx, dy):
    return ((np.roll(q, -1, 1) - 2 * q + np.roll(q, 1, 1)) / dx**2
            + (np.roll(q, -1, 0) - 2 * q + np.roll(q, 1, 0)) / dy**2)


def momentum_predictor_periodic_imex_elastic(u, v, nu, dx, dy, dt, lap_eig, cs2,
                                             chi_c, fu, fv, rho=1.0):
    """IMEX predictor (periodic) with implicit viscosity AND a linearly-implicit
    elastic-wave stabilizer. The elastic (and any other) face force fu/fv enters the
    implicit RHS -- required for stability. chi_c is the cell-centred solid indicator
    (1 in solid); cs2 = mu_s/rho (or G/rho). Returns u*, v*."""
    dudx = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx)
    dudy = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2 * dy)
    adv_u = u * dudx + _v_at_u_per(v) * dudy
    dvdx = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2 * dx)
    dvdy = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dy)
    adv_v = _u_at_v_per(u) * dvdx + v * dvdy

    c_el = dt * dt * cs2
    chi_u = 0.5 * (chi_c + np.roll(chi_c, 1, 1))       # solid indicator on u-faces
    chi_v = 0.5 * (chi_c + np.roll(chi_c, 1, 0))       # ... on v-faces
    # RHS: advection + explicit force (INSIDE the solve) + defect that removes the
    # constant elastic stabilizer from the fluid (where chi=0 -> (1-chi)=1).
    rhs_u = u - dt * adv_u + dt * fu / rho + c_el * (1.0 - chi_u) * _lap_per(u, dx, dy)
    rhs_v = v - dt * adv_v + dt * fv / rho + c_el * (1.0 - chi_v) * _lap_per(v, dx, dy)
    coef = dt * nu + c_el
    denom = 1.0 - coef * lap_eig
    ustar = np.real(np.fft.ifft2(np.fft.fft2(rhs_u) / denom))
    vstar = np.real(np.fft.ifft2(np.fft.fft2(rhs_v) / denom))
    return ustar, vstar


def momentum_predictor_periodic(u, v, nu, dx, dy, dt):
    """Forward-Euler predictor (central advection + diffusion) on the periodic
    staggered grid. Returns u*, v*."""
    dudx = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx)
    dudy = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2 * dy)
    lapu = ((np.roll(u, -1, 1) - 2 * u + np.roll(u, 1, 1)) / dx**2
            + (np.roll(u, -1, 0) - 2 * u + np.roll(u, 1, 0)) / dy**2)
    ru = -(u * dudx + _v_at_u_per(v) * dudy) + nu * lapu

    dvdx = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2 * dx)
    dvdy = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dy)
    lapv = ((np.roll(v, -1, 1) - 2 * v + np.roll(v, 1, 1)) / dx**2
            + (np.roll(v, -1, 0) - 2 * v + np.roll(v, 1, 0)) / dy**2)
    rv = -(_u_at_v_per(u) * dvdx + v * dvdy) + nu * lapv
    return u + dt * ru, v + dt * rv


# ── Free-slip box momentum (for the disc-in-Taylor-Green benchmark) ──────────
# Normal velocity zero at walls; tangential free (zero normal-gradient ghosts).

def momentum_predictor_freeslip(u, v, nu, dx, dy, dt, fu=None, fv=None, rho=1.0):
    """Explicit predictor (central advection + diffusion) on a free-slip box,
    plus optional face body forces fu/fv. Returns u*, v* with wall (normal)
    faces zeroed."""
    Ny, Nxp1 = u.shape
    # free-slip tangential ghosts: mirror (+interior)
    up = np.empty((Ny + 2, Nxp1)); up[1:-1] = u; up[0] = u[0]; up[-1] = u[-1]
    Nyp1, Nx = v.shape
    vp = np.empty((Nyp1, Nx + 2)); vp[:, 1:-1] = v; vp[:, 0] = v[:, 0]; vp[:, -1] = v[:, -1]

    uc = u[:, 1:-1]
    dudx = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    dudy = (up[2:, 1:-1] - up[:-2, 1:-1]) / (2 * dy)
    lapu = ((u[:, 2:] - 2 * uc + u[:, :-2]) / dx**2
            + (up[2:, 1:-1] - 2 * up[1:-1, 1:-1] + up[:-2, 1:-1]) / dy**2)
    v_u = _v_at_u(v)
    rhs_u = -(uc * dudx + v_u * dudy) + nu * lapu
    if fu is not None:
        rhs_u = rhs_u + fu[:, 1:-1] / rho
    ustar = u.copy(); ustar[:, 1:-1] = uc + dt * rhs_u
    ustar[:, 0] = 0.0; ustar[:, -1] = 0.0

    vc = v[1:-1, :]
    dvdy = (v[2:, :] - v[:-2, :]) / (2 * dy)
    dvdx = (vp[1:-1, 2:] - vp[1:-1, :-2]) / (2 * dx)
    lapv = ((vp[1:-1, 2:] - 2 * vp[1:-1, 1:-1] + vp[1:-1, :-2]) / dx**2
            + (v[2:, :] - 2 * vc + v[:-2, :]) / dy**2)
    u_v = _u_at_v(u)
    rhs_v = -(u_v * dvdx + vc * dvdy) + nu * lapv
    if fv is not None:
        rhs_v = rhs_v + fv[1:-1, :] / rho
    vstar = v.copy(); vstar[1:-1, :] = vc + dt * rhs_v
    vstar[0, :] = 0.0; vstar[-1, :] = 0.0
    return ustar, vstar


# ── Conservative reference-map advection with the divergence-free face velocity ──
# Jain 2019 Eq. 26: d(xi)/dt + H div(u xi) = 0. Using the MAC FACE velocity (which
# is discretely divergence-free) makes div(u xi) = u.grad(xi) exactly -- no spurious
# xi*div(u) source (the cell-interpolated velocity is NOT divergence-free and folds
# the map). H gates the update to the solid; the band is filled by extrapolation.

def _xi_flux_div_faces(xi, u, v, dx, dy):
    """div(u xi) at cell centres using face velocities u:(Ny,Nx+1), v:(Ny+1,Nx)
    and xi:(Ny,Nx) interpolated to the faces (edge values at domain walls)."""
    Ny, Nx = xi.shape
    xi_uf = np.empty((Ny, Nx + 1))
    xi_uf[:, 1:-1] = 0.5 * (xi[:, :-1] + xi[:, 1:])
    xi_uf[:, 0] = xi[:, 0]; xi_uf[:, -1] = xi[:, -1]
    fx = u * xi_uf
    xi_vf = np.empty((Ny + 1, Nx))
    xi_vf[1:-1, :] = 0.5 * (xi[:-1, :] + xi[1:, :])
    xi_vf[0, :] = xi[0, :]; xi_vf[-1, :] = xi[-1, :]
    fy = v * xi_vf
    return (fx[:, 1:] - fx[:, :-1]) / dx + (fy[1:, :] - fy[:-1, :]) / dy


def advect_xi_conservative(xi, u, v, dx, dy, dt, phi, w_cut=0.0):
    """SSP-RK3 conservative advection of a cell-centred reference-map component
    with the divergence-free MAC face velocity, gated to phi<=w_cut (Jain Eq. 26)."""
    H = (phi <= w_cut).astype(float)
    def rhs(q):
        return -H * _xi_flux_div_faces(q, u, v, dx, dy)
    q1 = xi + dt * rhs(xi)
    q2 = 0.75 * xi + 0.25 * (q1 + dt * rhs(q1))
    return (1.0 / 3.0) * xi + (2.0 / 3.0) * (q2 + dt * rhs(q2))


# ── Solid-solid contact STRESS (Rycroft et al. 2018, arXiv 1810.03015 Eq.4.10-4.12) ──
# Unlike a repulsive BODY force (which is curl-free -> nullified by the projection),
# contact is added as a trace-free STRESS tensor; its divergence is a momentum-
# conserving force that survives the exact projection.

def contact_stress(phi_a, phi_b, eta, Gsum, eps, dx, dy):
    """Trace-free contact stress for a pair of solids whose blur zones overlap.

      f(phi)  = 1/2 (1 - phi/eps) for phi < eps, else 0   (contact intensity)
      n       = grad(phi_a - phi_b)/|grad(phi_a - phi_b)|  (pair normal)
      tau_col = -eta * min{f_a,f_b} * Gsum * (n⊗n - 1/2 I)   (2D, trace-free)

    Returns the cell-centred components (txx, txy, tyy), to be ADDED to the solid
    stress before taking its divergence. Gsum = G_a + G_b, eps ~ contact width.
    """
    fa = np.where(phi_a < eps, 0.5 * (1.0 - phi_a / eps), 0.0)
    fb = np.where(phi_b < eps, 0.5 * (1.0 - phi_b / eps), 0.0)
    fc = np.minimum(fa, fb)                       # active only where both overlap
    d = phi_a - phi_b
    dpx = np.zeros_like(d); dpy = np.zeros_like(d)
    dpx[:, 1:-1] = (d[:, 2:] - d[:, :-2]) / (2 * dx)
    dpy[1:-1, :] = (d[2:, :] - d[:-2, :]) / (2 * dy)
    mag = np.sqrt(dpx * dpx + dpy * dpy) + 1e-12
    nx = dpx / mag; ny = dpy / mag
    s = -eta * fc * Gsum
    return s * (nx * nx - 0.5), s * (nx * ny), s * (ny * ny - 0.5)
