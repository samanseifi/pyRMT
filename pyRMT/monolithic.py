"""Fully monolithic Eulerian RMT-FSI: one velocity-pressure Newton solve per step.

This is backlog #14 stage 2d, and the numerical realization of
``paper/monolithic_aml.tex``.  The whole timestep is posed as a single nonlinear
system in the end-of-step velocity and pressure ``(u, v, p)``; the reference map
is *eliminated* rather than carried as an unknown.

The device is an **implicit semi-Lagrangian** transport of the reference map:

    xi(x) = xihat^n( x - dt * u^{n+1}(x) )                              (Eq. 5)

with ``xihat^n`` a *C1* (unclamped Catmull-Rom bicubic) interpolant of xi^n.
Tracing along the unknown end-of-step velocity removes the advective CFL, and the
C1 interpolant makes xi -- hence F = (grad xi)^-1, the neo-Hookean stress, and the
interface phi = phi0(xi) -- a differentiable function of the velocity alone.  What
remains is the incompressible momentum/continuity residual, solved by an inexact
Jacobian-free Newton-Krylov iteration.  Every stiff term (elastic wave, viscous
diffusion, advection) sits at time n+1, so the step carries no CFL restriction.

Three implementation points, matching Sec. 5 of the draft:

1. *Differentiable extrapolation.*  ``functions.extrapolate_reference_map`` grows
   layers by boolean flags and does a weighted least-squares plane fit per cell;
   as written it is neither differentiable nor deterministic (``known_flag`` is
   both read and written inside a ``prange`` loop).  Here the flags and the fit
   weights are FROZEN at the start of the step (from phi^n) into an explicit
   linear "plan"; applying the plan is a fixed sparse linear map on xi, so the
   band values are as smooth in u as the interior ones.  Layers are also frozen
   per layer, which makes the result deterministic.

2. *Positivity of F.*  det(grad xi) > 0 is enforced by a line search that caps the
   Newton step so the minimum of det(grad xi) over the solid stays above a fixed
   fraction of its current value.

3. *Preconditioning.*  Both supported boundary conditions are exactly separable.
   A velocity component is Dirichlet across the two walls it is normal to (interior
   faces -> DST-I); along the other two the free-slip box mirrors (Neumann ->
   DCT-II) while the lid cavity reflects with a sign flip (cell-centred Dirichlet ->
   DST-II).  So (rho/dt - mu_f lap) diagonalizes under DST-I x DCT-II for the
   free-slip box and DST-I x DST-II for the cavity, machine-exactly in both cases.
   Together with the DCT pressure Poisson solve of the underlying MAC solver this
   gives an exact projection preconditioner at the cost of three fast transforms per
   Krylov iteration, plus the Cahouet-Chabard identity term on the Schur complement.

Time discretization is a theta-scheme: theta=1 is the backward-Euler system of the
draft; theta=0.5 is the trapezoidal/midpoint variant.

MEASURED BEHAVIOUR (benchmarks/monolithic_disc_tg.py, disc in a Taylor-Green
vortex, N=48, mu_s=100 so the elastic wave is the binding CFL):

* Differentiability.  Newton converges quadratically (|R| 2.3e-1 -> 4.1e-5 ->
  8.0e-13) and the closed-form tangent of Sec. 4 agrees with a central finite
  difference to round-off.  This is the claim the whole design serves.
* Stability.  The scheme runs at 40x the elastic-wave CFL with min det(grad xi) =
  1.001 and 0.66% velocity error against a resolved explicit reference, where the
  explicit scheme diverges at 10x and produces det = -128 (a folded map) at 40x.
* Order in dt.  FIRST order, for both theta.  The velocity-pressure kernel itself
  is second-order-capable -- with theta=1/2 and Eulerian advection and no solid
  present the measured order is 2.00 at machine-level error -- but the reference-map
  band (a masked, extrapolated field on a cell-quantized solid region) holds the
  coupled scheme to first order.  An outer interface sweep (``n_outer``) buys a
  factor ~2.5 in the constant, not the order.  The draft's second-order claim is
  therefore NOT supported as written.
* Cost.  There IS a net speedup, but only in the resolved / viscous-limited regime.
  Which explicit restriction binds decides everything, and the two routes behave
  OPPOSITELY:
    - elastic & advective limits both scale as dx, so dt_mono/dt_exp ~ c_s/U is
      RESOLUTION-INDEPENDENT; worse, chasing stiffness raises mu_s*dt, which is
      exactly what degrades the conditioning of A, so the larger step is partly
      eaten by extra Krylov iterations.  No crossover in stiffness up to mu_s=1e5.
    - the viscous limit scales as dx^2, so refining makes it bind (N>75 for the
      benchmark parameters) and the gain grows ~N while leaving mu_s*dt alone, so
      the iteration count stays bounded.  THIS is where the method pays.
  MEASURED (disc-in-TG, mu_s=0.1, T=0.1, explicit and monolithic timed in one
  process so the ratio is machine-independent):
        N     binds   explicit        monolithic       speedup   rel err
        32    adv     13 st / 0.46 s  13 st / 10 mv    0.27x     1.8e-3
        64    adv     25 st / 1.70 s  25 st /  9 mv    0.29x     9.3e-4
        128   VISC    82 st / 26.2 s  49 st /  7 mv    1.11x     4.6e-4
        192   VISC   185 st / 130.7s  73 st /  6 mv    1.77x     3.1e-4
        256   VISC   328 st / 387.5s  97 st /  6 mv    2.40x     2.3e-4
  The step gain tracks the predicted nu/(U*dx) to two digits, and the matvec count
  FALLS with resolution (10 -> 6 over an eightfold refinement), so the
  preconditioner is at least mesh-independent.  Not saturating at N=256.
  Caveat: the comparison is against a FULLY EXPLICIT solver (as all published RMT
  solvers are).  A fractional-step scheme with implicit viscosity would also lift
  the viscous limit; against that the advantage would have to come from the elastic
  wave, which is still preconditioner-limited (see below).

* Preconditioner.  ``pc_mode='sparse'`` assembles and factors the
  variable-coefficient velocity block: linearizing the elastic force for a
  near-undeformed map gives d(sigma_s) ~ 2*mu_s*dt*dev(D(du)), i.e. an extra
  viscosity mu_s*dt inside the solid.  MEASURED (N=48, mu_s=100): matvecs at 40x
  the elastic CFL 2211 -> 522, Newton 8.0 -> 2.7, wall 49.4 s -> 13.7 s, cost per
  unit simulated time flat in dt, identical errors.  Below mu_s*dt ~ mu_f the
  sparse block needs MORE iterations than the exact transform solve (16 vs 11) and
  is also dearer per application, so ``pc_mode='auto'`` (default) switches on that
  criterion.  What the assembled K still omits: the deviatoric tensor structure
  (the u-v coupling of dev D), the wider bicubic-plus-extrapolation stencil, and
  the interface term -- that is what would extend the speedup above from the
  viscous regime to the stiff one.

* Constant-factor work on the inner loop was tried and MEASURED, and is not the
  answer.  Fusing the stress, its tangent and the force assembly into numba
  kernels, precomputing the constant-offset staggered stencils, and sharing the
  kinematics between residual and tangent cut the residual 2.8x (6.7 -> 2.4 ms) and
  the Jacobian action 1.7x (4.1 -> 2.4 ms), yet the STEP cost fell only ~1.15x --
  better than ninety per cent of a step is inside the Krylov solve.  Also rejected
  on measurement: the SIMPLE gradient correction (`pc_simple`, halves the
  transforms but nearly triples the Krylov count at large dt) and an extrapolated
  initial guess (`extrapolate_guess`, two extra residual evaluations, no reduction
  in the Newton count).
  What the scheme DOES buy at large dt is accuracy: at 16x the explicit step on the
  cavity its centroid error is 1.9e-3 against 1.2e-2 for `imex-elastic-sl` -- the
  operator-splitting error is what the monolithic solve removes.
"""

import numpy as np
from numba import njit
from scipy.fft import dctn, idctn, dstn, idstn
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pyRMT.mac import (divergence, gradient_p_u, gradient_p_v,
                       poisson_eigs_neumann, solve_poisson_neumann)


# =====================================================================
# 1.  C1 (unclamped Catmull-Rom) bicubic interpolation
# =====================================================================
# pyRMT.interpolators.bicubic_interpolate clamps the result to the 4x4 stencil
# min/max.  That limiter is what keeps the EXPLICIT semi-Lagrangian step
# monotone, but min/max is not differentiable, so it would leave the monolithic
# residual non-smooth exactly at the interface where the elastic force lives.
# The monolithic solve therefore uses the plain (unlimited) Catmull-Rom kernel,
# which is C1 everywhere, and relies on the positivity line search (Sec. 3
# above) rather than a limiter for robustness.

@njit(inline='always')
def _cr(v0, v1, v2, v3, t):
    """Catmull-Rom value at fractional position t in [0,1] between v1 and v2."""
    a0 = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3
    a1 = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3
    a2 = -0.5 * v0 + 0.5 * v2
    return ((a0 * t + a1) * t + a2) * t + v1


@njit(inline='always')
def _cr_dt(v0, v1, v2, v3, t):
    """d/dt of _cr."""
    a0 = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3
    a1 = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3
    a2 = -0.5 * v0 + 0.5 * v2
    return (3.0 * a0 * t + 2.0 * a1) * t + a2


@njit(cache=True)
def bicubic_c1(f, xq, yq, dx, dy, x0, y0):
    """Unclamped Catmull-Rom bicubic interpolation of ``f`` at (xq, yq).

    ``f`` lives on a uniform grid whose (0,0) sample is at physical (x0, y0) --
    (0, 0) for cell corners, (dx/2, dy/2) for cell centres, (0, dy/2) for
    x-faces, and so on.  Stencil indices are clamped to the array (a constant
    extension outside the domain); the *result* is never clamped, so the map is
    C1 in the query point.
    """
    Ny, Nx = f.shape
    out = np.empty(xq.shape)
    for j in range(xq.shape[0]):
        for i in range(xq.shape[1]):
            xi = (xq[j, i] - x0) / dx
            yi = (yq[j, i] - y0) / dy
            if not (np.isfinite(xi) and np.isfinite(yi)):
                out[j, i] = np.nan
                continue
            if xi < 0.0:
                xi = 0.0
            elif xi > Nx - 1.0:
                xi = Nx - 1.0
            if yi < 0.0:
                yi = 0.0
            elif yi > Ny - 1.0:
                yi = Ny - 1.0
            ix = int(np.floor(xi))
            iy = int(np.floor(yi))
            if ix > Nx - 2:
                ix = Nx - 2
            if iy > Ny - 2:
                iy = Ny - 2
            if ix < 0:
                ix = 0
            if iy < 0:
                iy = 0
            fx = xi - ix
            fy = yi - iy
            r0 = 0.0; r1 = 0.0; r2 = 0.0; r3 = 0.0
            for m in range(4):
                jg = iy - 1 + m
                if jg < 0:
                    jg = 0
                elif jg > Ny - 1:
                    jg = Ny - 1
                i0 = ix - 1
                i1 = ix
                i2 = ix + 1
                i3 = ix + 2
                if i0 < 0: i0 = 0
                if i1 < 0: i1 = 0
                if i2 > Nx - 1: i2 = Nx - 1
                if i3 > Nx - 1: i3 = Nx - 1
                rv = _cr(f[jg, i0], f[jg, i1], f[jg, i2], f[jg, i3], fx)
                if m == 0: r0 = rv
                elif m == 1: r1 = rv
                elif m == 2: r2 = rv
                else: r3 = rv
            out[j, i] = _cr(r0, r1, r2, r3, fy)
    return out


@njit(cache=True)
def bicubic_c1_grad(f, xq, yq, dx, dy, x0, y0):
    """Value and physical-space gradient of :func:`bicubic_c1`.

    Returns ``(val, dfdx, dfdy)``.  The gradient is what appears in the transport
    tangent of the draft, Eq. (9):  d(xi) = -dt * grad(xihat^n) . d(u).  It exists
    (and is continuous) precisely because the Catmull-Rom kernel is C1 -- with the
    bilinear interpolant it would jump at every cell edge.
    """
    Ny, Nx = f.shape
    val = np.empty(xq.shape)
    gx = np.empty(xq.shape)
    gy = np.empty(xq.shape)
    for j in range(xq.shape[0]):
        for i in range(xq.shape[1]):
            xi = (xq[j, i] - x0) / dx
            yi = (yq[j, i] - y0) / dy
            if not (np.isfinite(xi) and np.isfinite(yi)):
                val[j, i] = np.nan; gx[j, i] = np.nan; gy[j, i] = np.nan
                continue
            # Outside the grid the interpolant is extended by a constant, so the
            # gradient vanishes -- but only along the clamped AXIS.  Zeroing both
            # components when just one coordinate is out of range (which happens
            # routinely for departure points near a wall) would make the tangent
            # inconsistent with the residual and cost Newton its quadratic rate.
            edge_x = False
            edge_y = False
            if xi < 0.0:
                xi = 0.0; edge_x = True
            elif xi > Nx - 1.0:
                xi = Nx - 1.0; edge_x = True
            if yi < 0.0:
                yi = 0.0; edge_y = True
            elif yi > Ny - 1.0:
                yi = Ny - 1.0; edge_y = True
            ix = int(np.floor(xi))
            iy = int(np.floor(yi))
            if ix > Nx - 2: ix = Nx - 2
            if iy > Ny - 2: iy = Ny - 2
            if ix < 0: ix = 0
            if iy < 0: iy = 0
            fx = xi - ix
            fy = yi - iy
            rv = np.empty(4)
            rd = np.empty(4)
            for m in range(4):
                jg = iy - 1 + m
                if jg < 0: jg = 0
                elif jg > Ny - 1: jg = Ny - 1
                i0 = ix - 1; i1 = ix; i2 = ix + 1; i3 = ix + 2
                if i0 < 0: i0 = 0
                if i1 < 0: i1 = 0
                if i2 > Nx - 1: i2 = Nx - 1
                if i3 > Nx - 1: i3 = Nx - 1
                a = f[jg, i0]; b = f[jg, i1]; c = f[jg, i2]; d = f[jg, i3]
                rv[m] = _cr(a, b, c, d, fx)
                rd[m] = _cr_dt(a, b, c, d, fx)
            val[j, i] = _cr(rv[0], rv[1], rv[2], rv[3], fy)
            gx[j, i] = 0.0 if edge_x else _cr(rd[0], rd[1], rd[2], rd[3], fy) / dx
            gy[j, i] = 0.0 if edge_y else _cr_dt(rv[0], rv[1], rv[2], rv[3], fy) / dy
    return val, gx, gy


# =====================================================================
# 2.  Reference-map extrapolation as a FROZEN LINEAR PLAN
# =====================================================================
# ``functions.extrapolate_reference_map`` grows layers of "known" cells outward
# from the solid and fills each new cell with a distance-weighted least-squares
# plane fit through its known neighbours.  The layer flags depend on phi (hence on
# u), and the fit is written with a boolean mask, so as an operator on xi it is
# neither differentiable nor -- because ``known_flag`` is read and written inside
# the same ``prange`` loop -- deterministic.
#
# For a monolithic solve we only need the extrapolation to be smooth in xi *within
# one Newton solve*.  Freezing the flags at phi^n does exactly that: with the
# geometry fixed, the plane fit is a fixed linear functional of the neighbouring
# xi values,
#
#     xi(x0) = sum_n c_n xi(x_n),    c_n = w_n * ( [1, x0, y0] . Aw^-1 [1, x_n, y_n] ),
#
# so the whole extrapolation is one fixed sparse matrix.  We store it as a "plan"
# -- targets in layer order plus CSR-style (src, coef) -- and apply it by a single
# sequential sweep, which reproduces the layered algorithm exactly (later layers
# read earlier layers' freshly written values).  Layer membership is frozen at the
# START of each layer, which removes the race and makes the result deterministic.

_MAXSRC = 81          # 9x9 gather window
_LSQ_TOL = 1e-10


@njit(cache=True)
def _build_plan(phi, dx, dy, max_layers):
    """Return (tgt, ptr, src, coef, known) for the frozen extrapolation."""
    Ny, Nx = phi.shape
    known = np.zeros((Ny, Nx), dtype=np.bool_)
    for j in range(Ny):
        for i in range(Nx):
            known[j, i] = phi[j, i] < 0.0

    cap = Ny * Nx
    tgt = np.empty(cap, dtype=np.int64)
    ptr = np.empty(cap + 1, dtype=np.int64)
    src = np.empty(cap * _MAXSRC, dtype=np.int64)
    coef = np.empty(cap * _MAXSRC, dtype=np.float64)
    ptr[0] = 0
    nt = 0
    ns = 0

    r2 = (4.0 * np.sqrt(dx * dx + dy * dy)) ** 2

    for _layer in range(max_layers):
        # --- frontier: unknown cells touching a known cell (flags frozen here) ---
        target = np.zeros((Ny, Nx), dtype=np.bool_)
        any_t = False
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                if known[j, i]:
                    continue
                hit = False
                for dj in range(-1, 2):
                    for di in range(-1, 2):
                        if known[j + dj, i + di]:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    target[j, i] = True
                    any_t = True
        if not any_t:
            break

        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                if not target[j, i]:
                    continue
                x0 = dx * i
                y0 = dy * j
                # gather known neighbours inside the radius
                gi = np.empty(_MAXSRC, dtype=np.int64)
                gw = np.empty(_MAXSRC)
                gx = np.empty(_MAXSRC)
                gy = np.empty(_MAXSRC)
                cnt = 0
                jlo = j - 4 if j - 4 > 0 else 0
                jhi = j + 5 if j + 5 < Ny else Ny
                ilo = i - 4 if i - 4 > 0 else 0
                ihi = i + 5 if i + 5 < Nx else Nx
                for jj in range(jlo, jhi):
                    for ii in range(ilo, ihi):
                        if not known[jj, ii]:
                            continue
                        xs = dx * ii
                        ys = dy * jj
                        d2 = (xs - x0) ** 2 + (ys - y0) ** 2
                        if d2 <= r2:
                            gi[cnt] = jj * Nx + ii
                            gx[cnt] = xs
                            gy[cnt] = ys
                            gw[cnt] = np.exp(-d2 / r2)
                            cnt += 1
                if cnt < 3:
                    continue

                # weighted normal equations  Aw q = [1, x0, y0]
                Aw = np.zeros((3, 3))
                for n in range(cnt):
                    w = gw[n]
                    a1 = gx[n]
                    a2 = gy[n]
                    Aw[0, 0] += w
                    Aw[0, 1] += w * a1
                    Aw[0, 2] += w * a2
                    Aw[1, 1] += w * a1 * a1
                    Aw[1, 2] += w * a1 * a2
                    Aw[2, 2] += w * a2 * a2
                Aw[1, 0] = Aw[0, 1]
                Aw[2, 0] = Aw[0, 2]
                Aw[2, 1] = Aw[1, 2]

                det = (Aw[0, 0] * (Aw[1, 1] * Aw[2, 2] - Aw[1, 2] * Aw[2, 1])
                       - Aw[0, 1] * (Aw[1, 0] * Aw[2, 2] - Aw[1, 2] * Aw[2, 0])
                       + Aw[0, 2] * (Aw[1, 0] * Aw[2, 1] - Aw[1, 1] * Aw[2, 0]))
                if abs(det) < _LSQ_TOL:
                    continue

                # q = Aw^-1 [1, x0, y0]  (Aw is SPD; Cramer is fine at 3x3)
                rhs = np.empty(3)
                rhs[0] = 1.0; rhs[1] = x0; rhs[2] = y0
                q = np.empty(3)
                for k in range(3):
                    M = Aw.copy()
                    M[0, k] = rhs[0]; M[1, k] = rhs[1]; M[2, k] = rhs[2]
                    dk = (M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
                          - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
                          + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]))
                    q[k] = dk / det

                tgt[nt] = j * Nx + i
                for n in range(cnt):
                    src[ns] = gi[n]
                    coef[ns] = gw[n] * (q[0] + q[1] * gx[n] + q[2] * gy[n])
                    ns += 1
                nt += 1
                ptr[nt] = ns

        # promote this layer's targets to known (AFTER the whole layer)
        for k in range(nt):
            known[tgt[k] // Nx, tgt[k] % Nx] = True

    return tgt[:nt].copy(), ptr[:nt + 1].copy(), src[:ns].copy(), coef[:ns].copy(), known


@njit(cache=True)
def _apply_plan(q, solid, tgt, ptr, src, coef):
    """Apply a frozen extrapolation plan to one cell-centred field."""
    Ny, Nx = q.shape
    out = np.zeros(Ny * Nx)
    qf = q.ravel()
    sf = solid.ravel()
    for k in range(Ny * Nx):
        if sf[k]:
            out[k] = qf[k]
    for k in range(tgt.shape[0]):
        acc = 0.0
        for m in range(ptr[k], ptr[k + 1]):
            acc += coef[m] * out[src[m]]
        out[tgt[k]] = acc
    return out.reshape(Ny, Nx)


class ExtrapPlan:
    """Frozen, linear reference-map extrapolation (see module docstring, point 1).

    Built once per timestep from phi^n; applied inside every residual evaluation,
    where it acts as a fixed sparse matrix on xi -- differentiable by construction.

    Attributes
    ----------
    known : (Ny,Nx) bool
        Cells with a valid xi after extrapolation (solid + band).
    valid : (Ny,Nx) bool
        ``known`` eroded by one cell -- where the CENTRAL gradient stencil of
        ``grad xi`` sees only known neighbours, i.e. where the stress is meaningful.
    """

    __slots__ = ("tgt", "ptr", "src", "coef", "known", "solid", "valid")

    def __init__(self, phi, dx, dy, layers):
        self.tgt, self.ptr, self.src, self.coef, self.known = _build_plan(
            np.ascontiguousarray(phi), dx, dy, layers)
        self.solid = phi < 0.0
        k = self.known
        v = np.zeros_like(k)
        v[1:-1, 1:-1] = (k[1:-1, 1:-1] & k[:-2, 1:-1] & k[2:, 1:-1]
                         & k[1:-1, :-2] & k[1:-1, 2:])
        self.valid = v

    def apply(self, xi1, xi2):
        return (_apply_plan(xi1, self.solid, self.tgt, self.ptr, self.src, self.coef),
                _apply_plan(xi2, self.solid, self.tgt, self.ptr, self.src, self.coef))


# =====================================================================
# 3.  Free-slip velocity Helmholtz: exact spectral solve
# =====================================================================
# On the free-slip box the wall-normal velocity vanishes at the two walls it is
# normal to (Dirichlet) and mirrors across the other two (Neumann).  For u this is
# Dirichlet in x on the interior faces i=1..Nx-1 and Neumann in y on the cell-centre
# rows, so (a*I - b*lap) diagonalises under DST-I (x) x DCT-II (y); v is the
# transpose.  Both are exact direct solves -- no CG, no preconditioner-inside-a-
# preconditioner -- which is what makes the projection preconditioner of the
# monolithic Newton solve cost only three transforms per Krylov iteration.

def _fs_eigs_u(Nx, Ny, dx, dy):
    kx = np.arange(1, Nx)                                   # Dirichlet modes
    lx = -2.0 * (1.0 - np.cos(np.pi * kx / Nx)) / dx ** 2
    my = np.arange(Ny)                                      # Neumann modes
    ly = -2.0 * (1.0 - np.cos(np.pi * my / Ny)) / dy ** 2
    return lx[np.newaxis, :] + ly[:, np.newaxis]            # (Ny, Nx-1)


def _fs_eigs_v(Nx, Ny, dx, dy):
    ky = np.arange(1, Ny)
    ly = -2.0 * (1.0 - np.cos(np.pi * ky / Ny)) / dy ** 2
    mx = np.arange(Nx)
    lx = -2.0 * (1.0 - np.cos(np.pi * mx / Nx)) / dx ** 2
    return lx[np.newaxis, :] + ly[:, np.newaxis]            # (Ny-1, Nx)


def _solve_helm_u(rhs, lap_eig, a, b):
    """Solve (a - b*lap) x = rhs on the interior u-faces (Ny, Nx-1)."""
    r = dstn(dctn(rhs, type=2, axes=0, norm='ortho'), type=1, axes=1, norm='ortho')
    r = r / (a - b * lap_eig)
    return idctn(dstn(r, type=1, axes=1, norm='ortho'), type=2, axes=0, norm='ortho')


def _solve_helm_v(rhs, lap_eig, a, b):
    """Solve (a - b*lap) x = rhs on the interior v-faces (Ny-1, Nx)."""
    r = dstn(dctn(rhs, type=2, axes=1, norm='ortho'), type=1, axes=0, norm='ortho')
    r = r / (a - b * lap_eig)
    return idctn(dstn(r, type=1, axes=0, norm='ortho'), type=2, axes=1, norm='ortho')


def _lap_u_fs(u, dx, dy):
    """Free-slip Laplacian of u at the interior faces -> (Ny, Nx-1)."""
    up = np.empty((u.shape[0] + 2, u.shape[1]))
    up[1:-1] = u; up[0] = u[0]; up[-1] = u[-1]
    return ((u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / dx ** 2
            + (up[2:, 1:-1] - 2.0 * up[1:-1, 1:-1] + up[:-2, 1:-1]) / dy ** 2)


# ── Precomputed Catmull-Rom stencils for FIXED query points ──────────────────
# Most interpolations in the scheme are between staggered grids at points that
# never move -- faces to cell centres, and so on -- yet each call recomputed the
# index arithmetic and the cubic basis per point.  Profiling showed these to be the
# single largest cost of a Newton step.  The stencil is therefore built once, by
# running exactly the index logic of :func:`bicubic_c1`, and stored as sixteen
# (index, weight) pairs per output point; applying it is then a pure gather.
# Duplicate indices from edge clamping simply accumulate, which reproduces the
# clamped stencil identically -- important, because the tangent must agree with the
# residual to the last bit.

@njit(inline='always')
def _cr_basis(t, w):
    t2 = t * t
    t3 = t2 * t
    w[0] = -0.5 * t3 + t2 - 0.5 * t
    w[1] = 1.5 * t3 - 2.5 * t2 + 1.0
    w[2] = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w[3] = 0.5 * t3 - 0.5 * t2


@njit(cache=True)
def _build_stencil(Ny, Nx, xq, yq, dx, dy, x0, y0):
    """(idx, wgt) of the Catmull-Rom stencil of a source grid (Ny,Nx) at fixed
    query points.  Mirrors :func:`bicubic_c1` exactly, clamping included."""
    M = xq.size
    xf = xq.ravel()
    yf = yq.ravel()
    idx = np.zeros((M, 16), dtype=np.int64)
    wgt = np.zeros((M, 16))
    wx = np.zeros(4)
    wy = np.zeros(4)
    for k in range(M):
        xi = (xf[k] - x0) / dx
        yi = (yf[k] - y0) / dy
        if xi < 0.0:
            xi = 0.0
        elif xi > Nx - 1.0:
            xi = Nx - 1.0
        if yi < 0.0:
            yi = 0.0
        elif yi > Ny - 1.0:
            yi = Ny - 1.0
        ix = int(np.floor(xi))
        iy = int(np.floor(yi))
        if ix > Nx - 2: ix = Nx - 2
        if iy > Ny - 2: iy = Ny - 2
        if ix < 0: ix = 0
        if iy < 0: iy = 0
        _cr_basis(xi - ix, wx)
        _cr_basis(yi - iy, wy)
        for n in range(4):
            jg = iy - 1 + n
            if jg < 0: jg = 0
            elif jg > Ny - 1: jg = Ny - 1
            for m in range(4):
                ig = ix - 1 + m
                if ig < 0: ig = 0
                elif ig > Nx - 1: ig = Nx - 1
                idx[k, n * 4 + m] = jg * Nx + ig
                wgt[k, n * 4 + m] = wy[n] * wx[m]
    return idx, wgt


@njit(cache=True)
def _apply_stencil(f, idx, wgt, ny, nx):
    ff = f.ravel()
    M = idx.shape[0]
    out = np.empty(M)
    for k in range(M):
        acc = 0.0
        for m in range(16):
            acc += wgt[k, m] * ff[idx[k, m]]
        out[k] = acc
    return out.reshape(ny, nx)


class FixedInterp:
    """Catmull-Rom interpolation from one staggered grid onto a fixed point set."""

    __slots__ = ("idx", "wgt", "ny", "nx")

    def __init__(self, shape, xq, yq, dx, dy, x0, y0):
        self.idx, self.wgt = _build_stencil(shape[0], shape[1],
                                            np.ascontiguousarray(xq),
                                            np.ascontiguousarray(yq),
                                            dx, dy, x0, y0)
        self.ny, self.nx = xq.shape

    def apply(self, f):
        return _apply_stencil(np.ascontiguousarray(f), self.idx, self.wgt,
                              self.ny, self.nx)


class ShiftInterp:
    """Catmull-Rom interpolation with a CONSTANT fractional offset.

    Every staggered-grid transfer used here (faces to cell centres and the like)
    lands on the source grid at the same fractional position for every output
    point, so the sixteen weights are the same everywhere.  Replicate-padding the
    source by two cells turns the stencil into sixteen contiguous slices, which
    streams instead of gathering -- roughly an order of magnitude faster than the
    per-point form, and the dominant cost of a Newton step before this change.

    The constructor checks itself against :func:`bicubic_c1` on a random field and
    sets ``self.exact``; the caller falls back to the general path if the offset was
    not in fact constant (which happens if a query set is not grid-aligned).
    """

    __slots__ = ("wx", "wy", "i0", "j0", "ny", "nx", "exact")

    def __init__(self, shape, xq, yq, dx, dy, x0, y0):
        Ny, Nx = shape
        self.ny, self.nx = xq.shape
        xi = (xq - x0) / dx
        yi = (yq - y0) / dy
        ix = np.clip(np.floor(np.clip(xi, 0.0, Nx - 1.0)).astype(int), 0, Nx - 2)
        iy = np.clip(np.floor(np.clip(yi, 0.0, Ny - 1.0)).astype(int), 0, Ny - 2)
        fx = np.clip(xi, 0.0, Nx - 1.0) - ix
        fy = np.clip(yi, 0.0, Ny - 1.0) - iy
        # the offset must be constant, and the base index a pure shift of the output index
        oi = ix - np.arange(self.nx)[np.newaxis, :]
        oj = iy - np.arange(self.ny)[:, np.newaxis]
        self.wx = np.zeros(4); self.wy = np.zeros(4)
        self.i0 = self.j0 = 0
        self.exact = (np.ptp(fx) < 1e-13 and np.ptp(fy) < 1e-13
                      and np.ptp(oi) == 0 and np.ptp(oj) == 0)
        if not self.exact:
            return
        t = float(fx.flat[0]); self.wx[:] = (
            -0.5*t**3 + t**2 - 0.5*t, 1.5*t**3 - 2.5*t**2 + 1.0,
             -1.5*t**3 + 2.0*t**2 + 0.5*t, 0.5*t**3 - 0.5*t**2)
        t = float(fy.flat[0]); self.wy[:] = (
            -0.5*t**3 + t**2 - 0.5*t, 1.5*t**3 - 2.5*t**2 + 1.0,
             -1.5*t**3 + 2.0*t**2 + 0.5*t, 0.5*t**3 - 0.5*t**2)
        self.i0 = int(oi.flat[0]) + 1        # +1 for the two-cell replicate pad, -1 stencil start
        self.j0 = int(oj.flat[0]) + 1
        if (self.i0 < 0 or self.j0 < 0
                or self.i0 + 3 + self.nx > Nx + 4 or self.j0 + 3 + self.ny > Ny + 4):
            self.exact = False
            return
        rs = np.random.RandomState(0)
        f = rs.randn(Ny, Nx)
        if np.max(np.abs(self.apply(f) - bicubic_c1(f, xq, yq, dx, dy, x0, y0))) > 1e-12:
            self.exact = False

    def apply(self, f):
        fp = np.pad(f, 2, mode='edge')
        out = np.zeros((self.ny, self.nx))
        for n in range(4):
            wy = self.wy[n]
            if wy == 0.0:
                continue
            j = self.j0 + n
            for m in range(4):
                w = wy * self.wx[m]
                if w == 0.0:
                    continue
                i = self.i0 + m
                # in-place axpy: 16 temporaries of the output size would otherwise
                # be allocated per call, and this is called once per Krylov iteration
                np.add(out, w * fp[j:j + self.ny, i:i + self.nx], out)
        return out


# ── Lid-driven cavity (no-slip walls + moving lid) ───────────────────────────
# The lid cavity is separable too, with a different pair of transforms.  A
# wall-normal velocity component is Dirichlet across the two walls it is normal to
# (interior faces -> DST-I, exactly as for free slip); the difference is the OTHER
# direction, where free slip mirrors (Neumann, DCT-II) but no-slip reflects with a
# sign flip, ghost = -u, which is the cell-centred Dirichlet Laplacian -- diagonal
# under DST-II.  So u uses DST-I(x) x DST-II(y) and v the transpose, and the exact
# projection preconditioner carries over unchanged.

def _u_ghost_lid(u, U_lid, U_bot=0.0):
    """Pad u with the tangential ghost rows of the lid cavity -> (Ny+2, Nx+1)."""
    up = np.empty((u.shape[0] + 2, u.shape[1]))
    up[1:-1] = u
    up[0] = 2.0 * U_bot - u[0]
    up[-1] = 2.0 * U_lid - u[-1]
    return up


def _v_ghost_lid(v):
    """Pad v with the no-slip ghost columns -> (Ny+1, Nx+2)."""
    vp = np.empty((v.shape[0], v.shape[1] + 2))
    vp[:, 1:-1] = v
    vp[:, 0] = -v[:, 0]
    vp[:, -1] = -v[:, -1]
    return vp


def _lap_u_lid(u, dx, dy, U_lid=0.0, U_bot=0.0):
    """Laplacian of u at the interior u-faces, lid cavity.  U_lid/U_bot=0 gives the
    HOMOGENEOUS operator that the DST-I x DST-II transform diagonalizes."""
    up = _u_ghost_lid(u, U_lid, U_bot)
    return ((u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / dx ** 2
            + (up[2:, 1:-1] - 2.0 * up[1:-1, 1:-1] + up[:-2, 1:-1]) / dy ** 2)


def _lap_v_lid(v, dx, dy):
    """Laplacian of v at the interior v-faces, lid cavity."""
    vp = _v_ghost_lid(v)
    return ((vp[1:-1, 2:] - 2.0 * vp[1:-1, 1:-1] + vp[1:-1, :-2]) / dx ** 2
            + (v[2:, :] - 2.0 * v[1:-1, :] + v[:-2, :]) / dy ** 2)


def _adv_u_lid(u, v, dx, dy, U_lid=0.0, U_bot=0.0):
    up = _u_ghost_lid(u, U_lid, U_bot)
    dudx = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)
    dudy = (up[2:, 1:-1] - up[:-2, 1:-1]) / (2.0 * dy)
    return u[:, 1:-1] * dudx + _v_at_u_fs(v) * dudy


def _adv_v_lid(u, v, dx, dy):
    vp = _v_ghost_lid(v)
    dvdy = (v[2:, :] - v[:-2, :]) / (2.0 * dy)
    dvdx = (vp[1:-1, 2:] - vp[1:-1, :-2]) / (2.0 * dx)
    return _u_at_v_fs(u) * dvdx + v[1:-1, :] * dvdy


def _lid_eigs_u(Nx, Ny, dx, dy):
    kx = np.arange(1, Nx)                                   # DST-I (Dirichlet faces)
    lx = -2.0 * (1.0 - np.cos(np.pi * kx / Nx)) / dx ** 2
    my = np.arange(1, Ny + 1)                               # DST-II (Dirichlet centres)
    ly = -2.0 * (1.0 - np.cos(np.pi * my / Ny)) / dy ** 2
    return lx[np.newaxis, :] + ly[:, np.newaxis]            # (Ny, Nx-1)


def _lid_eigs_v(Nx, Ny, dx, dy):
    ky = np.arange(1, Ny)
    ly = -2.0 * (1.0 - np.cos(np.pi * ky / Ny)) / dy ** 2
    mx = np.arange(1, Nx + 1)
    lx = -2.0 * (1.0 - np.cos(np.pi * mx / Nx)) / dx ** 2
    return lx[np.newaxis, :] + ly[:, np.newaxis]            # (Ny-1, Nx)


def _solve_helm_u_lid(rhs, lap_eig, a, b):
    r = dstn(dstn(rhs, type=2, axes=0, norm='ortho'), type=1, axes=1, norm='ortho')
    r = r / (a - b * lap_eig)
    return idstn(dstn(r, type=1, axes=1, norm='ortho'), type=2, axes=0, norm='ortho')


def _solve_helm_v_lid(rhs, lap_eig, a, b):
    r = dstn(dstn(rhs, type=2, axes=1, norm='ortho'), type=1, axes=0, norm='ortho')
    r = r / (a - b * lap_eig)
    return idstn(dstn(r, type=1, axes=0, norm='ortho'), type=2, axes=1, norm='ortho')


def _fgmres(matvec, b, precond, rtol=1e-3, restart=40, maxcycles=8):
    """Flexible GMRES with Givens-rotation least squares.

    Standard GMRES fixes the preconditioner as a linear operator; ours may be a
    CG sweep on the variable-coefficient velocity block, which is only
    approximately linear when run to a loose tolerance.  FGMRES stores the
    PRECONDITIONED basis vectors, so a preconditioner that varies from iteration
    to iteration is admissible -- which lets any inner solve stop early instead of
    grinding to round-off just to stay linear.

    The Hessenberg least-squares problem is updated by Givens rotations (O(j) per
    iteration); re-solving it from scratch each iteration costs O(j^3) and was
    measurably comparable to the matvec at restart=50.

    Returns ``(x, n_matvecs)``.
    """
    from scipy.linalg import solve_triangular
    n = b.size
    x = np.zeros(n)
    bnorm = np.linalg.norm(b)
    if bnorm == 0.0:
        return x, 0
    tol = rtol * bnorm
    nmv = 0
    for cycle in range(maxcycles):
        if cycle:
            r = b - matvec(x)
            nmv += 1
        else:
            r = b.copy()
        beta = np.linalg.norm(r)
        if beta <= tol:
            break
        m = restart
        # V/Z grow on demand.  Preallocating the full restart basis costs
        # (restart+1)*n doubles per cycle -- ~10 MB at restart=50 on a 64^2 grid --
        # when a typical cycle converges in well under ten iterations, and that
        # allocation dominated the solve.
        V = [r / beta]
        Z = []
        H = np.zeros((m + 1, m))
        cs = np.zeros(m)
        sn = np.zeros(m)
        g = np.zeros(m + 1)
        g[0] = beta
        err = beta
        j = -1
        for j in range(m):
            Z.append(precond(V[j]))
            w = matvec(Z[j])
            nmv += 1
            for i in range(j + 1):                       # modified Gram-Schmidt
                H[i, j] = V[i] @ w
                w -= H[i, j] * V[i]
            H[j + 1, j] = np.linalg.norm(w)
            V.append(w / H[j + 1, j] if H[j + 1, j] > 1e-300 else np.zeros(n))
            for i in range(j):                           # old rotations
                t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t
            d = np.hypot(H[j, j], H[j + 1, j])
            if d == 0.0:
                cs[j], sn[j] = 1.0, 0.0
            else:
                cs[j], sn[j] = H[j, j] / d, H[j + 1, j] / d
            H[j, j] = d
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            err = abs(g[j + 1])
            if err <= tol or d == 0.0:
                break
        k = j + 1
        if k:
            y = solve_triangular(H[:k, :k], g[:k], lower=False)
            for i in range(k):
                x = x + y[i] * Z[i]
        if err <= tol:
            break
    return x, nmv


def _v_at_u_fs(v):
    """v interpolated to the interior u-faces -> (Ny, Nx-1)."""
    return 0.25 * (v[:-1, :-1] + v[:-1, 1:] + v[1:, :-1] + v[1:, 1:])


def _u_at_v_fs(u):
    """u interpolated to the interior v-faces -> (Ny-1, Nx)."""
    return 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])


def _adv_u_fs(u, v, dx, dy):
    """(u.grad)u at the interior u-faces, free-slip box.  Same central stencil as
    ``mac.momentum_predictor_freeslip``, so the Eulerian branch of the monolithic
    residual is term-for-term the implicit counterpart of the explicit driver."""
    up = np.empty((u.shape[0] + 2, u.shape[1]))
    up[1:-1] = u; up[0] = u[0]; up[-1] = u[-1]
    dudx = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)
    dudy = (up[2:, 1:-1] - up[:-2, 1:-1]) / (2.0 * dy)
    return u[:, 1:-1] * dudx + _v_at_u_fs(v) * dudy


def _adv_v_fs(u, v, dx, dy):
    """(u.grad)v at the interior v-faces, free-slip box."""
    vp = np.empty((v.shape[0], v.shape[1] + 2))
    vp[:, 1:-1] = v; vp[:, 0] = v[:, 0]; vp[:, -1] = v[:, -1]
    dvdy = (v[2:, :] - v[:-2, :]) / (2.0 * dy)
    dvdx = (vp[1:-1, 2:] - vp[1:-1, :-2]) / (2.0 * dx)
    return _u_at_v_fs(u) * dvdx + v[1:-1, :] * dvdy


def _mu_at_u_rows(muc, lid=False):
    """Cell-centred mu averaged to the y-midpoints between u-face rows -> (Ny+1, Nx-1).

    On a free-slip wall the mirror ghost makes the transverse flux vanish, so rows 0
    and Ny are irrelevant and set to zero.  On a no-slip wall the ghost is -u, the
    flux does not vanish, and the wall value of mu is needed."""
    Ny, Nx = muc.shape
    m = np.zeros((Ny + 1, Nx - 1))
    m[1:-1, :] = 0.25 * (muc[:-1, :-1] + muc[:-1, 1:] + muc[1:, :-1] + muc[1:, 1:])
    if lid:
        m[0, :] = 0.5 * (muc[0, :-1] + muc[0, 1:])
        m[-1, :] = 0.5 * (muc[-1, :-1] + muc[-1, 1:])
    return m


def _mu_at_v_cols(muc, lid=False):
    """Cell-centred mu averaged to the x-midpoints between v-face columns -> (Ny-1, Nx+1)."""
    Ny, Nx = muc.shape
    m = np.zeros((Ny - 1, Nx + 1))
    m[:, 1:-1] = 0.25 * (muc[:-1, :-1] + muc[:-1, 1:] + muc[1:, :-1] + muc[1:, 1:])
    if lid:
        m[:, 0] = 0.5 * (muc[:-1, 0] + muc[1:, 0])
        m[:, -1] = 0.5 * (muc[:-1, -1] + muc[1:, -1])
    return m


def _apply_helm_u_var(x, a, muc, muy, dx, dy):
    """(a - div(mu grad)) on the interior u-faces (Ny, Nx-1).  SPD."""
    Ny = x.shape[0]
    xe = np.zeros((Ny, x.shape[1] + 2))          # Dirichlet across the x-walls
    xe[:, 1:-1] = x
    fx = muc * (xe[:, 1:] - xe[:, :-1]) / dx      # (Ny, Nx) fluxes at cell centres
    tx = (fx[:, 1:] - fx[:, :-1]) / dx
    dyj = np.zeros((Ny + 1, x.shape[1]))          # mirror (zero flux) at the y-walls
    dyj[1:-1, :] = (x[1:, :] - x[:-1, :]) / dy
    fy = muy * dyj
    ty = (fy[1:, :] - fy[:-1, :]) / dy
    return a * x - (tx + ty)


def _apply_helm_v_var(y, a, muc, mux, dx, dy):
    """(a - div(mu grad)) on the interior v-faces (Ny-1, Nx).  SPD."""
    Nx = y.shape[1]
    ye = np.zeros((y.shape[0] + 2, Nx))
    ye[1:-1, :] = y
    fy = muc * (ye[1:, :] - ye[:-1, :]) / dy
    ty = (fy[1:, :] - fy[:-1, :]) / dy
    dxi = np.zeros((y.shape[0], Nx + 1))
    dxi[:, 1:-1] = (y[:, 1:] - y[:, :-1]) / dx
    fx = mux * dxi
    tx = (fx[:, 1:] - fx[:, :-1]) / dx
    return a * y - (tx + ty)


def _assemble_helm_u(a, muc, muy, dx, dy, lid=False):
    """Sparse matrix form of :func:`_apply_helm_u_var` on the interior u-faces.

    Same stencil, assembled instead of applied, so it can be factored once per step
    and used as an EXACT variable-coefficient velocity block in the preconditioner.
    ``lid`` switches the transverse walls from mirror (zero flux) to the
    cell-centred Dirichlet ghost -u, which contributes 2*mu at the wall row.
    """
    Ny, Nxm1 = muy.shape[0] - 1, muy.shape[1]
    Nx = Nxm1 + 1
    n = Ny * Nxm1
    rows = []; cols = []; vals = []

    def idx(j, m):
        return j * Nxm1 + m

    for j in range(Ny):
        for m in range(Nxm1):
            k = idx(j, m)
            diag = a
            # --- x direction: Dirichlet at both walls (u = 0 there) ---
            cR = muc[j, m + 1] / dx ** 2
            cL = muc[j, m] / dx ** 2
            diag += cR + cL
            if m + 1 <= Nxm1 - 1:
                rows.append(k); cols.append(idx(j, m + 1)); vals.append(-cR)
            if m - 1 >= 0:
                rows.append(k); cols.append(idx(j, m - 1)); vals.append(-cL)
            # --- y direction ---
            cU = muy[j + 1, m] / dy ** 2
            cD = muy[j, m] / dy ** 2
            if j + 1 <= Ny - 1:
                diag += cU
                rows.append(k); cols.append(idx(j + 1, m)); vals.append(-cU)
            elif lid:
                diag += 2.0 * cU
            if j - 1 >= 0:
                diag += cD
                rows.append(k); cols.append(idx(j - 1, m)); vals.append(-cD)
            elif lid:
                diag += 2.0 * cD
            rows.append(k); cols.append(k); vals.append(diag)
    return sp.csc_matrix((vals, (rows, cols)), shape=(n, n))


def _assemble_helm_v(a, muc, mux, dx, dy, lid=False):
    """Sparse form of :func:`_apply_helm_v_var` on the interior v-faces."""
    Nym1, Nx = mux.shape[0], mux.shape[1] - 1
    n = Nym1 * Nx
    rows = []; cols = []; vals = []

    def idx(r, i):
        return r * Nx + i

    for r in range(Nym1):
        for i in range(Nx):
            k = idx(r, i)
            diag = a
            # --- y direction: Dirichlet at both walls (v = 0 there) ---
            cU = muc[r + 1, i] / dy ** 2
            cD = muc[r, i] / dy ** 2
            diag += cU + cD
            if r + 1 <= Nym1 - 1:
                rows.append(k); cols.append(idx(r + 1, i)); vals.append(-cU)
            if r - 1 >= 0:
                rows.append(k); cols.append(idx(r - 1, i)); vals.append(-cD)
            # --- x direction ---
            cR = mux[r, i + 1] / dx ** 2
            cL = mux[r, i] / dx ** 2
            if i + 1 <= Nx - 1:
                diag += cR
                rows.append(k); cols.append(idx(r, i + 1)); vals.append(-cR)
            elif lid:
                diag += 2.0 * cR
            if i - 1 >= 0:
                diag += cL
                rows.append(k); cols.append(idx(r, i - 1)); vals.append(-cL)
            elif lid:
                diag += 2.0 * cL
            rows.append(k); cols.append(k); vals.append(diag)
    return sp.csc_matrix((vals, (rows, cols)), shape=(n, n))


def _pcg(apply_A, apply_M, rhs, rtol=1e-6, maxit=60):
    """Preconditioned CG for the SPD variable-coefficient Helmholtz blocks.

    Each iteration is a handful of array operations -- orders of magnitude cheaper
    than an outer GMRES iteration, which costs a full nonlinear residual
    evaluation.  Moving the localized elastic stiffness from the outer Krylov
    space into this inner solve is what makes the monolithic step affordable.
    """
    x = np.zeros_like(rhs)
    r = rhs.copy()
    nb = np.sqrt(np.sum(rhs * rhs))
    if nb == 0.0:
        return x, 0
    z = apply_M(r)
    pdir = z.copy()
    rz = np.sum(r * z)
    for k in range(maxit):
        Ap = apply_A(pdir)
        denom = np.sum(pdir * Ap)
        if denom == 0.0:
            break
        al = rz / denom
        x += al * pdir
        r -= al * Ap
        if np.sqrt(np.sum(r * r)) <= rtol * nb:
            return x, k + 1
        z = apply_M(r)
        rz_new = np.sum(r * z)
        pdir = z + (rz_new / rz) * pdir
        rz = rz_new
    return x, maxit


def _lap_v_fs(v, dx, dy):
    """Free-slip Laplacian of v at the interior faces -> (Ny-1, Nx)."""
    vp = np.empty((v.shape[0], v.shape[1] + 2))
    vp[:, 1:-1] = v; vp[:, 0] = v[:, 0]; vp[:, -1] = v[:, -1]
    return ((vp[1:-1, 2:] - 2.0 * vp[1:-1, 1:-1] + vp[1:-1, :-2]) / dx ** 2
            + (v[2:, :] - 2.0 * v[1:-1, :] + v[:-2, :]) / dy ** 2)


# =====================================================================
# 4.  The monolithic solver
# =====================================================================

DET_FLOOR = 1e-8          # keeps 1/det finite inside a rejected trial step


def _grad_cc(q, dx, dy):
    """Cell-centred gradient (central interior, one-sided at the domain edge)."""
    qx = np.empty_like(q)
    qy = np.empty_like(q)
    qx[:, 1:-1] = (q[:, 2:] - q[:, :-2]) / (2.0 * dx)
    qx[:, 0] = (q[:, 1] - q[:, 0]) / dx
    qx[:, -1] = (q[:, -1] - q[:, -2]) / dx
    qy[1:-1, :] = (q[2:, :] - q[:-2, :]) / (2.0 * dy)
    qy[0, :] = (q[1, :] - q[0, :]) / dy
    qy[-1, :] = (q[-1, :] - q[-2, :]) / dy
    return qx, qy


# ── Fused kernels for the stress, its tangent, and the force assembly ────────
# These replace ~40 numpy temporaries per call.  At N=64 the array work is only a
# few thousand elements per operation, so numpy's per-operation overhead (dispatch
# plus a fresh 32 kB temporary each time) dominated the real arithmetic by roughly
# an order of magnitude -- and the Jacobian action is applied once per Krylov
# iteration, so it sets the cost of the whole solve.

@njit(inline='always')
def _dfdx(f, j, i, Nx, dx):
    """2nd-order d/dx, matching utils.grad_central_x_2nd exactly (incl. edges)."""
    if i == 0:
        return (-3.0 * f[j, 0] + 4.0 * f[j, 1] - f[j, 2]) / (2.0 * dx)
    if i == Nx - 1:
        return (3.0 * f[j, Nx - 1] - 4.0 * f[j, Nx - 2] + f[j, Nx - 3]) / (2.0 * dx)
    return (f[j, i + 1] - f[j, i - 1]) / (2.0 * dx)


@njit(inline='always')
def _dfdy(f, j, i, Ny, dy):
    if j == 0:
        return (-3.0 * f[0, i] + 4.0 * f[1, i] - f[2, i]) / (2.0 * dy)
    if j == Ny - 1:
        return (3.0 * f[Ny - 1, i] - 4.0 * f[Ny - 2, i] + f[Ny - 3, i]) / (2.0 * dy)
    return (f[j + 1, i] - f[j - 1, i]) / (2.0 * dy)


@njit(cache=True)
def _gradcc_nb(q, dx, dy):
    """Cell-centred gradient: central inside, one-sided at the domain edge
    (matching _grad_cc)."""
    Ny, Nx = q.shape
    qx = np.empty((Ny, Nx))
    qy = np.empty((Ny, Nx))
    for j in range(Ny):
        for i in range(Nx):
            if i == 0:
                qx[j, i] = (q[j, 1] - q[j, 0]) / dx
            elif i == Nx - 1:
                qx[j, i] = (q[j, Nx - 1] - q[j, Nx - 2]) / dx
            else:
                qx[j, i] = (q[j, i + 1] - q[j, i - 1]) / (2.0 * dx)
            if j == 0:
                qy[j, i] = (q[1, i] - q[0, i]) / dy
            elif j == Ny - 1:
                qy[j, i] = (q[Ny - 1, i] - q[Ny - 2, i]) / dy
            else:
                qy[j, i] = (q[j + 1, i] - q[j - 1, i]) / (2.0 * dy)
    return qx, qy


@njit(cache=True)
def _stress_state(xi1, xi2, dx, dy, mu, isochoric, det_floor):
    """One pass over the grid: grad(xi) -> det, F, b and the neo-Hookean stress.

    Returns everything the residual AND the tangent need, so the two share a single
    traversal instead of recomputing the kinematics.
    """
    Ny, Nx = xi1.shape
    g11, g12 = _gradcc_nb(xi1, dx, dy)
    g21, g22 = _gradcc_nb(xi2, dx, dy)
    det = np.empty((Ny, Nx)); safe = np.empty((Ny, Nx))
    f11 = np.empty((Ny, Nx)); f12 = np.empty((Ny, Nx))
    f21 = np.empty((Ny, Nx)); f22 = np.empty((Ny, Nx))
    b11 = np.empty((Ny, Nx)); b12 = np.empty((Ny, Nx)); b22 = np.empty((Ny, Nx))
    sxx = np.empty((Ny, Nx)); sxy = np.empty((Ny, Nx)); syy = np.empty((Ny, Nx))
    for j in range(Ny):
        for i in range(Nx):
            a = g11[j, i]; b = g12[j, i]; c = g21[j, i]; d = g22[j, i]
            dd = a * d - b * c
            det[j, i] = dd
            sv = dd if abs(dd) >= det_floor else det_floor
            safe[j, i] = sv
            p11 = d / sv; p12 = -b / sv; p21 = -c / sv; p22 = a / sv
            f11[j, i] = p11; f12[j, i] = p12; f21[j, i] = p21; f22[j, i] = p22
            q11 = p11 * p11 + p12 * p12
            q12 = p11 * p21 + p12 * p22
            q22 = p21 * p21 + p22 * p22
            b11[j, i] = q11; b12[j, i] = q12; b22[j, i] = q22
            ht = 0.5 * (q11 + q22)
            jm2 = sv * sv if isochoric else 1.0
            sxx[j, i] = mu * jm2 * (q11 - ht)
            sxy[j, i] = mu * jm2 * q12
            syy[j, i] = mu * jm2 * (q22 - ht)
    return (det, safe, g11, g12, g21, g22, f11, f12, f21, f22,
            b11, b12, b22, sxx, sxy, syy)


@njit(cache=True)
def _tangent_stress(d1, d2, f11, f12, f21, f22, g11, g12, g21, g22, safe,
                    b11, b12, b22, sxx, sxy, syy, H, Hp, p1, p2, valid,
                    dx, dy, mu, isochoric):
    """delta[(1-H) sigma_s], masked to the valid band, from a reference-map
    perturbation.  Fuses the material tangent (draft Eq. 10) with the interface
    term dH = H'(phi) grad(phi0).d(xi)."""
    Ny, Nx = d1.shape
    e11, e12 = _gradcc_nb(d1, dx, dy)
    e21, e22 = _gradcc_nb(d2, dx, dy)
    Sxx = np.zeros((Ny, Nx)); Sxy = np.zeros((Ny, Nx)); Syy = np.zeros((Ny, Nx))
    for j in range(Ny):
        for i in range(Nx):
            if not valid[j, i]:
                continue
            a11 = f11[j, i]; a12 = f12[j, i]; a21 = f21[j, i]; a22 = f22[j, i]
            k11 = e11[j, i]; k12 = e12[j, i]; k21 = e21[j, i]; k22 = e22[j, i]
            m11 = k11 * a11 + k12 * a21
            m12 = k11 * a12 + k12 * a22
            m21 = k21 * a11 + k22 * a21
            m22 = k21 * a12 + k22 * a22
            dF11 = -(a11 * m11 + a12 * m21)
            dF12 = -(a11 * m12 + a12 * m22)
            dF21 = -(a21 * m11 + a22 * m21)
            dF22 = -(a21 * m12 + a22 * m22)
            q11 = 2.0 * (dF11 * a11 + dF12 * a12)
            q12 = dF11 * a21 + dF12 * a22 + a11 * dF21 + a12 * dF22
            q22 = 2.0 * (dF21 * a21 + dF22 * a22)
            half = 0.5 * (q11 + q22)
            if isochoric:
                sv = safe[j, i]
                ddet = (k11 * g22[j, i] + g11[j, i] * k22
                        - k12 * g21[j, i] - g12[j, i] * k21)
                kk = 2.0 * sv * ddet
                hb = 0.5 * (b11[j, i] + b22[j, i])
                dsxx = mu * (kk * (b11[j, i] - hb) + sv * sv * (q11 - half))
                dsxy = mu * (kk * b12[j, i] + sv * sv * q12)
                dsyy = mu * (kk * (b22[j, i] - hb) + sv * sv * (q22 - half))
            else:
                dsxx = mu * (q11 - half)
                dsxy = mu * q12
                dsyy = mu * (q22 - half)
            dH = Hp[j, i] * (p1[j, i] * d1[j, i] + p2[j, i] * d2[j, i])
            w = 1.0 - H[j, i]
            Sxx[j, i] = w * dsxx - dH * sxx[j, i]
            Sxy[j, i] = w * dsxy - dH * sxy[j, i]
            Syy[j, i] = w * dsyy - dH * syy[j, i]
    return Sxx, Sxy, Syy


@njit(cache=True)
def _div_stress_faces(Sxx, Sxy, Syy, dx, dy):
    """div of a symmetric cell-centred tensor, interpolated to the MAC faces.

    Identical stencils to grad_central_{x,y}_2nd followed by face averaging, so
    this is bit-for-bit the force assembly of the explicit driver."""
    Ny, Nx = Sxx.shape
    divx = np.empty((Ny, Nx))
    divy = np.empty((Ny, Nx))
    for j in range(Ny):
        for i in range(Nx):
            divx[j, i] = _dfdx(Sxx, j, i, Nx, dx) + _dfdy(Sxy, j, i, Ny, dy)
            divy[j, i] = _dfdx(Sxy, j, i, Nx, dx) + _dfdy(Syy, j, i, Ny, dy)
    fu = np.zeros((Ny, Nx + 1))
    fv = np.zeros((Ny + 1, Nx))
    for j in range(Ny):
        for i in range(1, Nx):
            fu[j, i] = 0.5 * (divx[j, i] + divx[j, i - 1])
    for j in range(1, Ny):
        for i in range(Nx):
            fv[j, i] = 0.5 * (divy[j, i] + divy[j - 1, i])
    return fu, fv


@njit(cache=True)
def _weight_stress(sxx, sxy, syy, H, valid):
    Ny, Nx = sxx.shape
    Axx = np.zeros((Ny, Nx)); Axy = np.zeros((Ny, Nx)); Ayy = np.zeros((Ny, Nx))
    for j in range(Ny):
        for i in range(Nx):
            if valid[j, i]:
                w = 1.0 - H[j, i]
                Axx[j, i] = w * sxx[j, i]
                Axy[j, i] = w * sxy[j, i]
                Ayy[j, i] = w * syy[j, i]
    return Axx, Axy, Ayy


class NewtonDivergence(RuntimeError):
    """Raised when the monolithic Newton solve cannot make progress."""


class MonolithicRMT:
    """Fully monolithic Eulerian RMT-FSI on a free-slip box.

    One step solves a single nonlinear system in ``(u, v, p)^{n+1}``; the reference
    map is eliminated by the implicit semi-Lagrangian transport (draft Eq. 5), so
    the elastic wave, the viscous diffusion and the advection are ALL at time n+1
    and no explicit CFL restriction survives.

    Parameters
    ----------
    N : int                      cells per side (square grid on [0,Lx]x[0,Ly])
    bc : {'freeslip', 'lid'}     'freeslip' is the disc-in-Taylor-Green box (mirror
                                 tangential walls); 'lid' is the driven cavity --
                                 no-slip walls with the top one moving at U_lid.
                                 Both are exactly separable, so the projection
                                 preconditioner is a direct transform solve either
                                 way (DST-I x DCT-II vs DST-I x DST-II).
    phi0 : callable(X1, X2)      undeformed level set, evaluated on the reference
                                 map -- this is what makes the interface a function
                                 of the unknown velocity.
    mu_s, mu_f, rho              shear modulus, dynamic fluid viscosity, density
    theta : float                1.0 = backward Euler (draft Eq. 8, first order);
                                 0.5 = trapezoidal along the characteristic, which
                                 is second order in dt.
    bt_order : int or None       semi-Lagrangian backtrace order (default: 1 for
                                 theta=1, 2 for theta=0.5).
    layers : int                 extrapolation band width in cells.  Must comfortably
                                 exceed w_t/dx + 2 so the stress is resolved wherever
                                 (1-H) > 0.
    w_t_fac : float              blend half-width w_t = w_t_fac * dx.
    advection : {'sl', 'eulerian'}
                                 momentum advection.  'sl' shares the implicit
                                 semi-Lagrangian departure point with the reference
                                 map; 'eulerian' is the implicit central form
                                 rho (u.grad)u of the draft's Eq. (8).  The
                                 reference map is transported semi-Lagrangian
                                 either way -- that is the point of the scheme.
                                 MEASURED: with 'sl' the overall order in dt is
                                 capped at 1 even for theta=1/2, because the
                                 pressure gradient is applied at the arrival point
                                 rather than along the characteristic; 'eulerian'
                                 with theta=1/2 is genuinely second order.
    isochoric : bool             use the J^-2-corrected deviatoric neo-Hookean stress.
    normalize_phi : bool         feed the Heaviside phi/|grad phi| (see the explicit
                                 drivers) for a consistent blend-band width.
    """

    def __init__(self, N, phi0, *, mu_s, mu_f, rho=1.0, Lx=1.0, Ly=1.0,
                 bc="freeslip", U_lid=1.0,
                 theta=1.0, bt_order=None, layers=5, w_t_fac=2.0,
                 isochoric=False, normalize_phi=False, advection="sl",
                 jacobian="analytic",
                 pc_mode="transform", pc_elastic=None, pc_schur=True,
                 pc_simple=False, pc_rtol=1e-2, pc_maxit=30):
        self.N = N
        self.Nx = self.Ny = N
        self.dx = Lx / N
        self.dy = Ly / N
        self.phi0 = phi0
        self.bc = str(bc)
        if self.bc not in ("freeslip", "lid"):
            raise ValueError("bc must be 'freeslip' or 'lid'")
        self.U_lid = float(U_lid)
        self.mu_s = float(mu_s)
        self.mu_f = float(mu_f)
        self.rho = float(rho)
        self.theta = float(theta)
        self.bt_order = int(bt_order) if bt_order is not None else (1 if theta == 1.0 else 2)
        self.layers = int(layers)
        self.w_t = w_t_fac * self.dx
        self.isochoric = bool(isochoric)
        self.normalize_phi = bool(normalize_phi)
        self.advection = str(advection)
        if self.advection not in ("sl", "eulerian"):
            raise ValueError("advection must be 'sl' or 'eulerian'")
        self.jacobian = str(jacobian)
        self.pc_mode = str(pc_mode)
        if self.pc_mode not in ("transform", "sparse", "auto"):
            raise ValueError("pc_mode must be 'transform', 'sparse' or 'auto'")
        # 'sparse' exists precisely to put the elastic stiffness into the velocity
        # block, so it defaults the elastic augmentation on.
        if pc_elastic is None:
            pc_elastic = 0.0 if self.pc_mode == "transform" else 1.0
        self.pc_elastic = float(pc_elastic)
        self.pc_schur = bool(pc_schur)
        self.pc_simple = bool(pc_simple)
        self.pc_rtol = float(pc_rtol)
        self.pc_maxit = int(pc_maxit)

        dx, dy = self.dx, self.dy
        xc = (np.arange(N) + 0.5) * dx
        yc = (np.arange(N) + 0.5) * dy
        self.Xc, self.Yc = np.meshgrid(xc, yc)
        # interior face coordinates (where the momentum residual lives)
        self.Xu, self.Yu = np.meshgrid(np.arange(1, N) * dx, yc)
        self.Xv, self.Yv = np.meshgrid(xc, np.arange(1, N) * dy)

        self.eig_p = poisson_eigs_neumann(N, N, dx, dy)
        if self.bc == "lid":
            self.eig_u = _lid_eigs_u(N, N, dx, dy)
            self.eig_v = _lid_eigs_v(N, N, dx, dy)
        else:
            self.eig_u = _fs_eigs_u(N, N, dx, dy)
            self.eig_v = _fs_eigs_v(N, N, dx, dy)

        # Precomputed constant-offset stencils for the staggered transfers that
        # occur at fixed points (see ShiftInterp).  Built once; they depend only on
        # the grid and the boundary condition.
        self._q = {"cc": (self.Xc, self.Yc), "uf": (self.Xu, self.Yu),
                   "vf": (self.Xv, self.Yv)}
        self._S = {}
        for tag, (xq, yq) in self._q.items():
            self._S[("u", tag)] = ShiftInterp((N + 2, N + 1), xq, yq, dx, dy,
                                              0.0, -0.5 * dy)
            self._S[("v", tag)] = ShiftInterp((N + 1, N + 2), xq, yq, dx, dy,
                                              -0.5 * dx, 0.0)

        self.u = np.zeros((N, N + 1))
        self.v = np.zeros((N + 1, N))
        self.p = np.zeros((N, N))
        self.xi1 = np.zeros((N, N))
        self.xi2 = np.zeros((N, N))
        self.phi = np.zeros((N, N))
        self.t = 0.0
        self.stats = []

    # ------------------------------------------------- BC-dispatched operators
    # Everything that touches a wall goes through these four; the rest of the
    # scheme (transport, stress, pressure, packing) is boundary-condition-agnostic.

    def _lap_u(self, u, homogeneous=False):
        if self.bc == "lid":
            return _lap_u_lid(u, self.dx, self.dy,
                              0.0 if homogeneous else self.U_lid)
        return _lap_u_fs(u, self.dx, self.dy)

    def _lap_v(self, v):
        if self.bc == "lid":
            return _lap_v_lid(v, self.dx, self.dy)
        return _lap_v_fs(v, self.dx, self.dy)

    def _adv_u(self, u, v):
        if self.bc == "lid":
            return _adv_u_lid(u, v, self.dx, self.dy, self.U_lid)
        return _adv_u_fs(u, v, self.dx, self.dy)

    def _adv_v(self, u, v):
        if self.bc == "lid":
            return _adv_v_lid(u, v, self.dx, self.dy)
        return _adv_v_fs(u, v, self.dx, self.dy)

    def _helm_u_solve(self, rhs, a, b):
        if self.bc == "lid":
            return _solve_helm_u_lid(rhs, self.eig_u, a, b)
        return _solve_helm_u(rhs, self.eig_u, a, b)

    def _helm_v_solve(self, rhs, a, b):
        if self.bc == "lid":
            return _solve_helm_v_lid(rhs, self.eig_v, a, b)
        return _solve_helm_v(rhs, self.eig_v, a, b)

    # ---------------------------------------------------------------- state
    def set_velocity(self, u, v):
        self.u = np.array(u, dtype=float)
        self.v = np.array(v, dtype=float)
        self.u[:, 0] = 0.0; self.u[:, -1] = 0.0
        self.v[0, :] = 0.0; self.v[-1, :] = 0.0

    def seed_reference_map(self):
        """xi = x inside the solid, extrapolated into the band (undeformed state)."""
        phi = self.phi0(self.Xc, self.Yc)
        plan = ExtrapPlan(phi, self.dx, self.dy, self.layers)
        sm = plan.solid
        self.xi1, self.xi2 = plan.apply(self.Xc * sm, self.Yc * sm)
        self.phi = self.phi0(self.xi1, self.xi2)
        self._plan = plan

    # ------------------------------------------------------- sampling helpers
    def _u_field(self, u, homogeneous=False):
        """(array, x0, y0) to interpolate a u-component from.

        BOTH boundary conditions interpolate from a ghosted array: the lid cavity
        needs it so a departure point leaving through the lid reads the driven
        value rather than the constant extension of the last interior row, and the
        free-slip box gets its mirror ghost, which is the correct extension there.
        Ghosting both also makes every staggered transfer a pure constant-offset
        stencil (see :class:`ShiftInterp`), which is what makes them fast.

        ``homogeneous`` drops the lid's inhomogeneous part (2*U_lid) from the ghost
        row.  A PERTURBATION du must be padded with -du, not 2*U_lid-du: the lid is
        a constant in the residual and so has zero derivative.  Getting this wrong
        left the tangent inconsistent by ~7e-3 and cost Newton its quadratic rate.
        """
        if self.bc == "lid":
            U = 0.0 if homogeneous else self.U_lid
            return _u_ghost_lid(u, U), 0.0, -0.5 * self.dy
        up = np.empty((u.shape[0] + 2, u.shape[1]))
        up[1:-1] = u; up[0] = u[0]; up[-1] = u[-1]      # free-slip mirror
        return up, 0.0, -0.5 * self.dy

    def _v_field(self, v):
        if self.bc == "lid":
            return _v_ghost_lid(v), -0.5 * self.dx, 0.0
        vp = np.empty((v.shape[0], v.shape[1] + 2))
        vp[:, 1:-1] = v; vp[:, 0] = v[:, 0]; vp[:, -1] = v[:, -1]
        return vp, -0.5 * self.dx, 0.0

    def _su(self, tag, u, homogeneous=False):
        """u-component at one of the fixed point sets, via the precomputed stencil."""
        f, x0, y0 = self._u_field(u, homogeneous)
        S = self._S[("u", tag)]
        if S.exact:
            return S.apply(f)
        xq, yq = self._q[tag]
        return bicubic_c1(f, xq, yq, self.dx, self.dy, x0, y0)

    def _sv(self, tag, v):
        f, x0, y0 = self._v_field(v)
        S = self._S[("v", tag)]
        if S.exact:
            return S.apply(f)
        xq, yq = self._q[tag]
        return bicubic_c1(f, xq, yq, self.dx, self.dy, x0, y0)

    def _sample_u(self, u, xq, yq, homogeneous=False):
        f, x0, y0 = self._u_field(u, homogeneous)
        return bicubic_c1(f, xq, yq, self.dx, self.dy, x0, y0)

    def _sample_v(self, v, xq, yq):
        f, x0, y0 = self._v_field(v)
        return bicubic_c1(f, xq, yq, self.dx, self.dy, x0, y0)

    def _sample_c(self, q, xq, yq):
        return bicubic_c1(q, xq, yq, self.dx, self.dy, 0.5 * self.dx, 0.5 * self.dy)

    def _backtrace(self, u, v, X, Y, tag=None):
        """Departure points of the implicit semi-Lagrangian step (draft Eq. 5).

        Order 1 traces along the END-OF-STEP velocity (backward Euler); order 2
        takes a midpoint step along the time-centred velocity, which is what the
        theta=1/2 scheme needs to actually be second order.
        """
        dt = self._dt
        if self.bt_order == 1:
            if tag is not None:
                uu = self._su(tag, u); vv = self._sv(tag, v)
            else:
                uu = self._sample_u(u, X, Y); vv = self._sample_v(v, X, Y)
            return X - dt * uu, Y - dt * vv
        ub = 0.5 * (u + self._un)
        vb = 0.5 * (v + self._vn)
        if tag is not None:
            uu = self._su(tag, ub); vv = self._sv(tag, vb)
        else:
            uu = self._sample_u(ub, X, Y); vv = self._sample_v(vb, X, Y)
        Xm = X - 0.5 * dt * uu
        Ym = Y - 0.5 * dt * vv
        uu = self._sample_u(ub, Xm, Ym)
        vv = self._sample_v(vb, Xm, Ym)
        return X - dt * uu, Y - dt * vv

    # ------------------------------------------------------------- kinematics
    def transport(self, u, v):
        """xi^{n+1} as an explicit, differentiable function of the velocity.

        This is the heart of the scheme: the value at a cell centre is the OLD
        reference map, read through the C1 bicubic interpolant, at the departure
        point traced back along the UNKNOWN end-of-step velocity, followed by the
        frozen (hence linear) band extrapolation.
        """
        Xd, Yd = self._backtrace(u, v, self.Xc, self.Yc, 'cc')
        a = self._sample_c(self._xi1n, Xd, Yd)
        b = self._sample_c(self._xi2n, Xd, Yd)
        return self._plan.apply(a * self._plan.solid, b * self._plan.solid)

    def _stress(self, xi1, xi2):
        """Deviatoric neo-Hookean Cauchy stress and det(grad xi) (draft Eq. 2)."""
        st = _stress_state(xi1, xi2, self.dx, self.dy, self.mu_s,
                           self.isochoric, DET_FLOOR)
        return st[13], st[14], st[15], st[0]

    def elastic_force(self, xi1, xi2, state=None):
        """Face-centred div[(1-H) sigma_s], with H = H(phi0(xi)) (draft Eq. 3).

        Uses exactly the operators of the explicit driver
        (``benchmarks/mac_disc_in_tg_convergence.py``) so the two schemes can be
        compared term by term.  The band mask is the frozen ``plan.valid``; the
        smooth part of the interface dependence, H(phi0(xi(u))), differentiates
        through, so no interface lagging is needed inside the Newton solve.

        ``state`` is the tuple from :func:`_stress_state`; it is returned so the
        tangent can reuse the kinematics this call already computed rather than
        traversing the grid a second time.
        """
        from pyRMT.functions import smoothed_heaviside
        phi = self.phi0(xi1, xi2)
        if self.normalize_phi:
            gx, gy = _grad_cc(phi, self.dx, self.dy)
            phi = phi / (np.sqrt(gx * gx + gy * gy) + 1e-12)
        H = smoothed_heaviside(phi, self.w_t)
        if state is None:
            state = _stress_state(xi1, xi2, self.dx, self.dy, self.mu_s,
                                  self.isochoric, DET_FLOOR)
        Axx, Axy, Ayy = _weight_stress(state[13], state[14], state[15],
                                       H, self._plan.valid)
        fu, fv = _div_stress_faces(Axx, Axy, Ayy, self.dx, self.dy)
        return fu, fv, state[0], phi, H, state

    # ------------------------------------------------------------ pack/unpack
    def _unpack(self, z):
        N = self.N
        nu = N * (N + 1)
        u = np.zeros((N, N + 1))
        v = np.zeros((N + 1, N))
        u[:, 1:-1] = z[:N * (N - 1)].reshape(N, N - 1)
        off = N * (N - 1)
        v[1:-1, :] = z[off:off + N * (N - 1)].reshape(N - 1, N)
        off += N * (N - 1)
        p = z[off:off + N * N].reshape(N, N)
        return u, v, p

    def _pack(self, u, v, p):
        return np.concatenate([u[:, 1:-1].ravel(), v[1:-1, :].ravel(), p.ravel()])

    # --------------------------------------------------------------- residual
    def residual(self, z, want_extra=False):
        """The monolithic residual (draft Eq. 8), rows scaled to velocity units.

        R_u : rho (u - u^n o X_d)/dt + grad p + theta*G(u) + (1-theta)*G^n o X_d
              with G = -mu_f lap u - div[(1-H) sigma_s(xi(u))]
        R_p : (rho dx/dt) div u          (scaled so both blocks share a norm)

        The pressure gradient is NOT theta-weighted.  In an incompressible flow p
        is a Lagrange multiplier for div u = 0 at the level the momentum balance is
        written -- effectively t^{n+1/2} for theta=1/2 -- not a quantity carried at
        integer time levels.  Blending grad p^{n+1} with grad p^n mixes two
        different multipliers and reinstates an O(dt) error, which was measured:
        it held theta=1/2 at first order until the pressure was pulled out of the
        theta-average.

        Every term on the u-row is evaluated at time n+1 -- the elastic force
        through the eliminated reference map, the diffusion, and the advection
        through the implicit departure point -- so nothing here restricts dt.
        """
        u, v, p = self._unpack(z)
        dt, rho, mu_f = self._dt, self.rho, self.mu_f
        th = self.theta

        xi1, xi2 = self.transport(u, v)
        fu, fv, detG, phi, H, state = self.elastic_force(xi1, xi2)

        Gu = -mu_f * self._lap_u(u) - fu[:, 1:-1]
        Gv = -mu_f * self._lap_v(v) - fv[1:-1, :]

        if self.advection == "eulerian":
            Gu = Gu + rho * self._adv_u(u, v)
            Gv = Gv + rho * self._adv_v(u, v)
            Ru = rho * (u[:, 1:-1] - self._un[:, 1:-1]) / dt
            Rv = rho * (v[1:-1, :] - self._vn[1:-1, :]) / dt
        else:
            Xd, Yd = self._backtrace(u, v, self.Xu, self.Yu, 'uf')
            Xdv, Ydv = self._backtrace(u, v, self.Xv, self.Yv, 'vf')
            Ru = rho * (u[:, 1:-1] - self._sample_u(self._un, Xd, Yd)) / dt
            Rv = rho * (v[1:-1, :] - self._sample_v(self._vn, Xdv, Ydv)) / dt

        Ru = Ru + gradient_p_u(p, self.dx)[:, 1:-1] + th * Gu
        Rv = Rv + gradient_p_v(p, self.dy)[1:-1, :] + th * Gv
        if th != 1.0:
            if self.advection == "eulerian":
                Ru = Ru + (1.0 - th) * self._Gun[:, 1:-1]
                Rv = Rv + (1.0 - th) * self._Gvn[1:-1, :]
            else:
                # old-time force carried along the SAME characteristic
                Ru = Ru + (1.0 - th) * self._sample_u(self._Gun, Xd, Yd)
                Rv = Rv + (1.0 - th) * self._sample_v(self._Gvn, Xdv, Ydv)

        div = divergence(u, v, self.dx, self.dy)
        Rp = self._pscale * (div - div.mean())

        R = np.concatenate([Ru.ravel(), Rv.ravel(), Rp.ravel()])
        if want_extra:
            return R, dict(xi1=xi1, xi2=xi2, detG=detG, phi=phi, div=div,
                           H=H, state=state)
        return R

    def _old_time_force(self):
        """G^n = -mu_f lap u^n - f_el^n, on the FULL face arrays (no pressure).

        Needed only for theta != 1.  It is sampled at the departure points, so it
        must be defined on the whole face grid (wall faces included); the wall
        values are extended from the first interior face.
        """
        fu, fv = self.elastic_force(self._xi1n, self._xi2n)[:2]
        Gu = np.zeros_like(self.u)
        Gv = np.zeros_like(self.v)
        Gu[:, 1:-1] = -self.mu_f * self._lap_u(self._un) - fu[:, 1:-1]
        Gv[1:-1, :] = -self.mu_f * self._lap_v(self._vn) - fv[1:-1, :]
        if self.advection == "eulerian":
            Gu[:, 1:-1] += self.rho * self._adv_u(self._un, self._vn)
            Gv[1:-1, :] += self.rho * self._adv_v(self._un, self._vn)
        Gu[:, 0] = Gu[:, 1]; Gu[:, -1] = Gu[:, -2]
        Gv[0, :] = Gv[1, :]; Gv[-1, :] = Gv[-2, :]
        return Gu, Gv

    # --------------------------------------------------------- preconditioner
    def _precondition(self, r):
        """Exact-projection block preconditioner (draft Sec. 5, point 3).

        Approximate block factorization of  J = [[A, G], [D, 0]]  with
        A = rho/dt - theta*mu_f*lap (solved EXACTLY by the free-slip transforms)
        and the Schur complement S = -D A^-1 G ~= -(dt/rho) lap_cc (the DCT
        pressure Poisson solve of the underlying MAC projection).  Three fast
        transforms per Krylov iteration; the elastic stiffness is left to Krylov.
        """
        N = self.N
        n = N * (N - 1)
        ru = r[:n].reshape(N, N - 1)
        rv = r[n:2 * n].reshape(N - 1, N)
        rp = r[2 * n:].reshape(N, N) / self._pscale

        a = self.rho / self._dt
        yu = self._helm_u(ru, a)
        yv = self._helm_v(rv, a)

        Yu = np.zeros((N, N + 1)); Yu[:, 1:-1] = yu
        Yv = np.zeros((N + 1, N)); Yv[1:-1, :] = yv
        s = divergence(Yu, Yv, self.dx, self.dy) - rp
        s = s - s.mean()
        # Schur complement S = -D A^-1 G.  The plain projection uses
        # S^-1 ~ -(rho/dt) lap^-1, which is only right when A ~ (rho/dt) I.  At
        # large dt the viscous (and, with pc_elastic, elastic) part of A dominates
        # at high wavenumber and the approximation collapses -- exactly where the
        # Krylov count was measured to blow up.  The Cahouet-Chabard form adds the
        # high-wavenumber limit as an identity term,
        #     S^-1 ~ -(rho/dt) lap^-1 + mu_eff I,
        # which is exact in both asymptotic regimes and costs one extra axpy.
        dp = -(self.rho / self._dt) * solve_poisson_neumann(s, self.eig_p)
        if self.pc_schur:
            dp = dp + self._pc_mu_schur * s
        dp = dp - dp.mean()

        gu = gradient_p_u(dp, self.dx)[:, 1:-1]
        gv = gradient_p_v(dp, self.dy)[1:-1, :]
        if self.pc_simple:
            # SIMPLE: approximate A^-1 by (dt/rho) I in the gradient correction
            # only.  Halves the transforms per preconditioner application
            # (18 -> 10) but weakens the preconditioner, and MEASURED to be a bad
            # trade at large dt -- exactly where the viscous/elastic content of A
            # matters most, the Krylov count nearly tripled (54 -> 156 matvecs at
            # 16x the explicit step).  Off by default; worth ~1.2x at small dt.
            du = yu - gu / a
            dv = yv - gv / a
        else:
            du = yu - self._helm_u(gu, a)
            dv = yv - self._helm_v(gv, a)
        return np.concatenate([du.ravel(), dv.ravel(), dp.ravel()])

    def _helm_u(self, rhs, a):
        """Velocity-block solve of the preconditioner.

        Either a direct solve of the assembled variable-coefficient block (which
        contains the elastic stiffness) or, when that is not worth its
        factorization, the exact constant-coefficient transform solve.
        """
        if self._lu_u is not None:
            return self._lu_u.solve(np.ascontiguousarray(rhs).ravel()).reshape(rhs.shape)
        return self._helm_u_solve(rhs, a, self.theta * self.mu_f)

    def _helm_v(self, rhs, a):
        if self._lu_v is not None:
            return self._lu_v.solve(np.ascontiguousarray(rhs).ravel()).reshape(rhs.shape)
        return self._helm_v_solve(rhs, a, self.theta * self.mu_f)

    # ------------------------------------------------------------- the step
    def step(self, dt, *, n_outer=1, newton_tol=1e-6, newton_maxit=20,
             gmres_maxit=6, eta_max=1e-2, eta_min=1e-5, det_frac=0.5,
             ls_min=1.0 / 64.0, extrapolate_guess=False, verbose=False):
        """Advance one fully monolithic step.

        Inexact Newton with an Eisenstat-Walker-style forcing term; the Jacobian
        acts matrix-free (a directional finite difference of the residual, which is
        legitimate precisely because the C1 transport made the residual smooth),
        preconditioned by the exact projection above.  ``gmres_maxit`` counts
        FGMRES RESTART CYCLES (restart=50), so it caps the work per Newton step
        at ~50*gmres_maxit residual evaluations.

        ``newton_tol`` is RELATIVE to the initial residual, which is dominated by
        rho*(u^{n+1}-u^n)/dt -- i.e. by the whole physics of the step.  1e-6 of
        that is already several orders below the O(dt) time-discretization error;
        chasing 1e-8 roughly quintupled the Krylov work for no change in the
        answer, because Eisenstat-Walker then drives the final linear solve to a
        forcing term the preconditioner cannot reach cheaply.

        The line search is the positivity safeguard of the draft, Sec. 5 point 2:
        a step is accepted only if it reduces the residual AND leaves
        min det(grad xi) positive and above ``det_frac`` of its current value, so F
        stays finite at every iterate and every dt.
        """
        self._dt = float(dt)
        self._un = self.u.copy()
        self._vn = self.v.copy()
        self._pn = self.p.copy()
        self._xi1n = self.xi1.copy()
        self._xi2n = self.xi2.copy()
        self._pscale = self.rho * self.dx / dt
        # Velocity-block coefficient of the preconditioner.  Linearizing the
        # elastic force through the transport tangent (draft Eqs. 9-10) gives, for
        # a near-undeformed map, d(div sigma_s)/du ~ -mu_s*dt*lap inside the solid;
        # folding that into the (constant-coefficient, hence transformable)
        # Helmholtz block as an extra viscosity mu_s*dt is what keeps the Krylov
        # spectrum clustered when the elastic wave is the stiff term.  It only
        # affects the CONVERGENCE RATE -- the converged step is unchanged.
        self._pcg_its = 0
        # Interface/band geometry is frozen at phi^n for the whole solve (the
        # SMOOTH part of the interface dependence, H(phi0(xi(u))), is still
        # differentiated through -- only the discrete extrapolation stencil lags).
        self._plan = ExtrapPlan(self.phi, self.dx, self.dy, self.layers)
        self._build_pc_coeffs(dt, self.phi)
        if self.theta != 1.0:
            self._Gun, self._Gvn = self._old_time_force()

        # Optional linear extrapolation from the last two steps as the initial
        # guess.  Off by default: MEASURED to cost two extra residual evaluations
        # per step without reducing the Newton count, which already sits at two.
        z = self._pack(self.u, self.v, self.p)
        prev = getattr(self, "_zprev", None)
        if extrapolate_guess and prev is not None \
                and abs(self._dtprev - dt) < 1e-14 * max(dt, 1.0):
            zx = 2.0 * z - prev
            try:
                if np.linalg.norm(self.residual(zx)) < np.linalg.norm(self.residual(z)):
                    z = zx
            except FloatingPointError:
                pass
        self._zprev = self._pack(self.u, self.v, self.p)
        self._dtprev = dt
        it = n_gm = 0
        scale = None
        for sweep in range(max(1, n_outer)):
            if sweep:
                # Outer fixed point on the INTERFACE (draft Sec. 5, point 1).  The
                # extrapolation stencil and band mask are frozen at a level-set --
                # phi^n on the first sweep -- which lags the true end-of-step
                # geometry by O(dt).  Re-freezing at the geometry just computed and
                # re-solving removes that lag; with theta=1/2 it is what stands
                # between first and second order once the fluid part is exact.
                self._refreeze(extra["phi"], dt)
            z, R, extra, it_s, gm_s = self._newton(
                z, newton_tol, newton_maxit, gmres_maxit, eta_max, eta_min,
                det_frac, ls_min, verbose, ref_scale=scale)
            if sweep == 0:
                scale = self._res0
            it += it_s
            n_gm += gm_s

        u, v, p = self._unpack(z)
        self.u, self.v = u, v
        self.p = p - p.mean()
        self.xi1, self.xi2 = extra["xi1"], extra["xi2"]
        self.phi = extra["phi"]
        self.t += dt
        # Incompressibility here is enforced by the NEWTON SOLVE, not by an exact
        # projection, so max|div u| tracks the Newton tolerance rather than machine
        # precision.  ``div_rel`` normalizes it by the discrete velocity scale
        # |u|_max/dx, which is the meaningful measure.
        dmax = float(np.max(np.abs(extra["div"])))
        uscale = max(float(np.max(np.abs(u))), float(np.max(np.abs(v))), 1e-30) / self.dx
        st = dict(t=self.t, dt=dt, newton=it, gmres=n_gm, pcg=self._pcg_its,
                  outer=max(1, n_outer),
                  res0=self._res0, res=self._res, min_det=self._min_det(extra["detG"]),
                  divmax=dmax, div_rel=dmax / uscale)
        self.stats.append(st)
        return st

    def _refreeze(self, phi, dt):
        """Rebuild the frozen extrapolation plan and preconditioner coefficients
        at a new level set (used by the outer interface sweep)."""
        self._plan = ExtrapPlan(phi, self.dx, self.dy, self.layers)
        self._build_pc_coeffs(dt, phi)

    def _build_pc_coeffs(self, dt, phi):
        """Velocity-block coefficients of the preconditioner, rebuilt each step.

        Linearizing the elastic force through the transport tangent gives, for a
        near-undeformed map, d(sigma_s) ~ 2*mu*dt*dev(D(du)), i.e. the elastic block
        behaves like an extra viscosity mu*dt concentrated in the solid.  In
        ``pc_mode='sparse'`` that variable coefficient is assembled and factored
        directly, so the preconditioner finally contains the elastic stiffness --
        which the constant-coefficient transform solve cannot represent.
        """
        chi = np.zeros((self.N, self.N))
        if self.pc_elastic > 0.0:
            from pyRMT.functions import smoothed_heaviside
            chi = (1.0 - smoothed_heaviside(phi, self.w_t)) * self._plan.valid
        self._muc = self.theta * (self.mu_f + self.pc_elastic * self.mu_s * dt * chi)
        lid = (self.bc == "lid")
        self._muy = _mu_at_u_rows(self._muc, lid)
        self._mux = _mu_at_v_cols(self._muc, lid)
        self._pc_bar = self.theta * self.mu_f
        # Cahouet-Chabard identity term.  It must carry the SAME spatially varying
        # viscosity as the velocity block; using a single scalar (the max) is what
        # made the earlier elastic-augmented attempt lose more than it gained.
        self._pc_mu_schur = self._muc if self.pc_elastic > 0.0 else self.theta * self.mu_f
        self._lu_u = self._lu_v = None
        # 'auto': the sparse block is worth its factorization exactly when the
        # elastic viscosity mu_s*dt exceeds the fluid one -- below that the
        # constant-coefficient transform solve is both exact and cheaper (measured:
        # sparse needs MORE Krylov iterations at small dt, 16 vs 11, and fewer by
        # 4.2x at forty times the elastic CFL).
        use_sparse = (self.pc_mode == "sparse"
                      or (self.pc_mode == "auto"
                          and self.pc_elastic * self.mu_s * dt > self.mu_f))
        if not use_sparse:
            self._pc_mu_schur = self.theta * self.mu_f
            self._muc = np.full((self.N, self.N), self.theta * self.mu_f)
            self.__dict__["_pc_elastic_active"] = 0.0
        if use_sparse:
            a = self.rho / dt
            self._lu_u = spla.splu(_assemble_helm_u(a, self._muc, self._muy,
                                                    self.dx, self.dy, lid))
            self._lu_v = spla.splu(_assemble_helm_v(a, self._muc, self._mux,
                                                    self.dx, self.dy, lid))

    def _newton(self, z, newton_tol, newton_maxit, gmres_maxit, eta_max, eta_min,
                det_frac, ls_min, verbose, ref_scale=None):
        """One inexact Newton-Krylov solve of the monolithic residual at fixed
        (frozen) interface geometry.  Returns (z, R, extra, n_newton, n_matvec).

        ``ref_scale`` fixes what the relative tolerance is measured against.  On an
        outer interface sweep the solve restarts from an already-converged iterate,
        so its own initial residual is tiny; scaling to THAT would demand a
        meaningless 1e-6 of 1e-6.  The scale therefore stays the one from the first
        sweep -- the physical size of the step.
        """
        R, extra = self.residual(z, want_extra=True)
        nR0 = np.linalg.norm(R)
        scale = nR0 if ref_scale is None else max(ref_scale, nR0)
        nR = nR0
        det_ref = self._min_det(extra["detG"])
        n_gm = 0
        eta = eta_max
        it = 0
        for it in range(newton_maxit):
            if nR <= newton_tol * max(scale, 1e-30) or nR <= 1e-13:
                break
            jv = self._jacobian(z, R, extra)
            dz, nmv = _fgmres(jv, -R, self._precondition, rtol=eta,
                              restart=50, maxcycles=gmres_maxit)
            n_gm += nmv
            if not np.all(np.isfinite(dz)):
                raise NewtonDivergence("the Krylov solve returned a non-finite step")

            alpha = 1.0
            accepted = False
            while alpha >= ls_min:
                zt = z + alpha * dz
                try:
                    Rt, ext = self.residual(zt, want_extra=True)
                except FloatingPointError:
                    alpha *= 0.5
                    continue
                nRt = np.linalg.norm(Rt)
                dmin = self._min_det(ext["detG"])
                # Positivity safeguard.  Once the iterate is admissible
                # (det > 0) the step may not lose more than a fixed fraction of
                # the margin; if the INITIAL iterate is already folded (which a
                # very large dt can do to the u^n warm start) we only insist the
                # step does not make it worse, so Newton can climb back out.
                ok_det = (dmin > det_frac * det_ref) if det_ref > 0.0 else (dmin > det_ref)
                if np.isfinite(nRt) and nRt < (1.0 - 1e-4 * alpha) * nR and ok_det:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                if nR < 1e-6 * max(scale, 1e-30):
                    break               # good enough; stalling on round-off
                raise NewtonDivergence(
                    f"line search failed at Newton iteration {it} "
                    f"(|R|={nR:.3e}, scale={scale:.3e}, min det={det_ref:.3e})")

            ratio = nRt / nR
            z, R, extra, nR = zt, Rt, ext, nRt
            det_ref = dmin           # the safeguard tracks the accepted iterate
            eta = min(eta_max, max(eta_min, 0.9 * ratio * ratio))
            if verbose:
                print(f"    newton {it}: |R|={nR:.3e}  alpha={alpha:.3g}  "
                      f"matvec={nmv}  eta={eta:.2e}")
        self._res0 = nR0
        self._res = nR
        return z, R, extra, it, n_gm

    def _min_det(self, detG):
        """min det(grad xi) over the SOLID (the quantity the line search guards)."""
        m = self._plan.solid
        return float(detG[m].min()) if m.any() else 1.0

    # ------------------------------------------------ the consistent tangent
    # Draft Sec. 4.  Chaining the TRANSPORT tangent (Eq. 9) into the MATERIAL
    # tangent (Eq. 10) gives the exact derivative of the elastic force with
    # respect to the velocity, so the Jacobian action costs a handful of array
    # operations instead of a whole extra residual evaluation.  Every step is
    # linear in the perturbation, which is the concrete pay-off of having made the
    # transport C1 and the extrapolation a frozen linear map.

    def _prepare_tangent(self, z, extra=None):
        """Freeze the quantities the tangent is built from at the current iterate.

        ``extra`` is the dict returned by :meth:`residual` at the SAME iterate; when
        supplied, the transport and the stress kinematics are taken from it instead
        of being recomputed, which removes a duplicate grid traversal per Newton
        iteration.
        """
        from pyRMT.functions import smoothed_heaviside
        u, v, p = self._unpack(z)
        T = {}
        # transport tangent (Eq. 9): grad of the OLD reference map at the
        # departure points -- closed form because the interpolant is C1
        Xd, Yd = self._backtrace(u, v, self.Xc, self.Yc, 'cc')
        _, T["g1x"], T["g1y"] = bicubic_c1_grad(self._xi1n, Xd, Yd, self.dx, self.dy,
                                                0.5 * self.dx, 0.5 * self.dy)
        _, T["g2x"], T["g2y"] = bicubic_c1_grad(self._xi2n, Xd, Yd, self.dx, self.dy,
                                                0.5 * self.dx, 0.5 * self.dy)
        # advection tangent: grad of the old velocity at the face departure points
        Xdu, Ydu = self._backtrace(u, v, self.Xu, self.Yu, 'uf')
        fu_, x0u, y0u = self._u_field(self._un)
        _, T["uxn"], T["uyn"] = bicubic_c1_grad(fu_, Xdu, Ydu, self.dx, self.dy, x0u, y0u)
        Xdv, Ydv = self._backtrace(u, v, self.Xv, self.Yv, 'vf')
        fv_, x0v, y0v = self._v_field(self._vn)
        _, T["vxn"], T["vyn"] = bicubic_c1_grad(fv_, Xdv, Ydv, self.dx, self.dy, x0v, y0v)

        if extra is None:
            xi1, xi2 = self.transport(u, v)
            state = _stress_state(xi1, xi2, self.dx, self.dy, self.mu_s,
                                  self.isochoric, DET_FLOOR)
            phi = self.phi0(xi1, xi2)
            H = smoothed_heaviside(phi, self.w_t)
        else:
            xi1, xi2 = extra["xi1"], extra["xi2"]
            state, phi, H = extra["state"], extra["phi"], extra["H"]
        T["state"] = state
        T["H"] = H
        w = self.w_t
        T["Hp"] = np.where(np.abs(phi) < w,
                           (1.0 + np.cos(np.pi * phi / w)) / (2.0 * w), 0.0)
        # d(phi0)/d(xi) by a central difference in REFERENCE space (phi0 is an
        # analytic shape function, so this is exact to round-off and costs four
        # evaluations per Newton iteration)
        h = 1e-6
        T["p1"] = (self.phi0(xi1 + h, xi2) - self.phi0(xi1 - h, xi2)) / (2 * h)
        T["p2"] = (self.phi0(xi1, xi2 + h) - self.phi0(xi1, xi2 - h)) / (2 * h)
        self._T = T

    def _jacobian_action(self, w):
        """J w, assembled from the closed-form tangent (draft Eqs. 9-10)."""
        T = self._T
        st = T["state"]
        dt, rho, th = self._dt, self.rho, self.theta
        du, dv, dp = self._unpack(w)

        # --- Eq. 9: the perturbed reference map -------------------------------
        duc = self._su('cc', du, homogeneous=True)
        dvc = self._sv('cc', dv)
        d1 = -dt * (T["g1x"] * duc + T["g1y"] * dvc)
        d2 = -dt * (T["g2x"] * duc + T["g2y"] * dvc)
        d1, d2 = self._plan.apply(d1 * self._plan.solid, d2 * self._plan.solid)

        # --- Eq. 10 + the solution-dependent interface, in one traversal ------
        Sxx, Sxy, Syy = _tangent_stress(
            d1, d2, st[6], st[7], st[8], st[9],          # F
            st[2], st[3], st[4], st[5], st[1],           # grad xi, det
            st[10], st[11], st[12],                      # b
            st[13], st[14], st[15],                      # sigma_s
            T["H"], T["Hp"], T["p1"], T["p2"], self._plan.valid,
            self.dx, self.dy, self.mu_s, self.isochoric)
        dfu_f, dfv_f = _div_stress_faces(Sxx, Sxy, Syy, self.dx, self.dy)
        dfu = dfu_f[:, 1:-1]
        dfv = dfv_f[1:-1, :]

        # --- the implicit advection tangent -----------------------------------
        # X_d = x - dt u(x), so d(X_d) = -dt d(u) and d(u^n o X_d) = grad(u^n).d(X_d)
        dXu = -dt * du[:, 1:-1]
        dYu = -dt * self._sv('uf', dv)
        dudep = T["uxn"] * dXu + T["uyn"] * dYu
        dXv = -dt * self._su('vf', du, homogeneous=True)
        dYv = -dt * dv[1:-1, :]
        dvdep = T["vxn"] * dXv + T["vyn"] * dYv

        Ru = (rho * (du[:, 1:-1] - dudep) / dt
              + gradient_p_u(dp, self.dx)[:, 1:-1]
              + th * (-self.mu_f * self._lap_u(du, homogeneous=True) - dfu))
        Rv = (rho * (dv[1:-1, :] - dvdep) / dt
              + gradient_p_v(dp, self.dy)[1:-1, :]
              + th * (-self.mu_f * self._lap_v(dv) - dfv))
        d = divergence(du, dv, self.dx, self.dy)
        Rp = self._pscale * (d - d.mean())
        return np.concatenate([Ru.ravel(), Rv.ravel(), Rp.ravel()])

    def _use_analytic(self):
        """The closed-form tangent covers the scheme as the draft states it
        (backward-Euler backtrace, unnormalized level set); anything else falls
        back to the finite-difference Jacobian action."""
        return (self.jacobian == "analytic" and self.bt_order == 1
                and self.theta == 1.0 and not self.normalize_phi
                and self.advection == "sl")

    def _jacobian(self, z, R, extra=None):
        if self._use_analytic():
            self._prepare_tangent(z, extra)
            return self._jacobian_action
        return self._make_jv(z, R)

    def _make_jv(self, z, R):
        """Matrix-free Jacobian action by a scaled directional difference.

        The fallback when the closed-form tangent does not apply.  It is
        legitimate at all only because the C1 transport made the residual smooth.
        """
        nz = np.linalg.norm(z)
        base = np.sqrt(np.finfo(float).eps) * (1.0 + nz)

        def jv(w):
            nw = np.linalg.norm(w)
            if nw == 0.0:
                return np.zeros_like(w)
            eps = base / nw
            return (self.residual(z + eps * w) - R) / eps

        return jv

    # ------------------------------------------------------------ diagnostics
    def kinetic_energy(self):
        uc = 0.5 * (self.u[:, :-1] + self.u[:, 1:])
        vc = 0.5 * (self.v[:-1, :] + self.v[1:, :])
        return 0.5 * self.rho * np.sum(uc ** 2 + vc ** 2) * self.dx * self.dy

    def strain_energy(self, isochoric=True):
        from pyRMT.output import compute_strain_energy
        return compute_strain_energy(self.xi1, self.xi2, self.phi, self.mu_s,
                                     self.dx, self.dy, isochoric=isochoric)

    def cell_velocity(self):
        return (0.5 * (self.u[:, :-1] + self.u[:, 1:]),
                0.5 * (self.v[:-1, :] + self.v[1:, :]))


# =====================================================================
# 5.  Transport tangent (draft Eqs. 9-10) -- used for verification
# =====================================================================

def transport_tangent(solver, u, v, du, dv):
    """d(xi) for a velocity perturbation, from the closed-form transport tangent.

    Draft Eq. (9):  d(xi) = -dt (grad xihat^n) . d(u), evaluated at the departure
    point.  This exists only because the interpolant is C1; comparing it with a
    finite difference of :meth:`MonolithicRMT.transport` is the sharpest test that
    the monolithic residual really is differentiable (see tests/test_monolithic.py).
    Valid for the order-1 (backward-Euler) backtrace.
    """
    dt = solver._dt
    Xd, Yd = solver._backtrace(u, v, solver.Xc, solver.Yc)
    _, g1x, g1y = bicubic_c1_grad(solver._xi1n, Xd, Yd, solver.dx, solver.dy,
                                  0.5 * solver.dx, 0.5 * solver.dy)
    _, g2x, g2y = bicubic_c1_grad(solver._xi2n, Xd, Yd, solver.dx, solver.dy,
                                  0.5 * solver.dx, 0.5 * solver.dy)
    duc = solver._sample_u(du, solver.Xc, solver.Yc)
    dvc = solver._sample_v(dv, solver.Xc, solver.Yc)
    d1 = -dt * (g1x * duc + g1y * dvc)
    d2 = -dt * (g2x * duc + g2y * dvc)
    return solver._plan.apply(d1 * solver._plan.solid, d2 * solver._plan.solid)
