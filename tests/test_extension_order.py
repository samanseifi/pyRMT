"""The band extension must be at least second order, or the transported interface lags.

Every cell the solid moves into takes its value from the extension and keeps it as material
data. A plane fit leaves an O(dx^2) error per newly covered layer, which accumulates to an O(dx)
lag of the advancing interface; the quadratic fit removes it."""
import numpy as np
from scipy.integrate import solve_ivp
import pyRMT.functions as F
from pyRMT.functions import extrapolate_reference_map, advect_reference_map


def test_quadratic_extension_is_exact_for_quadratics():
    N = 32; h = 1.0 / N; x = (np.arange(N) + 0.5) * h; X, Y = np.meshgrid(x, x)
    phi = np.hypot(X - 0.5, Y - 0.5) - 0.2; m = phi <= 0
    f = 1 + X + 2 * Y + 3 * X * X - X * Y + 0.5 * Y * Y
    band = (phi > 0) & (phi < 3 * h)
    q, _ = extrapolate_reference_map(f * m, X * m, phi, h, h, 4, order=2)
    p, _ = extrapolate_reference_map(f * m, X * m, phi, h, h, 4, order=1)
    assert np.abs(q - f)[band].max() < 1e-8
    assert np.abs(p - f)[band].max() > 1e-3          # the plane fit is not


def _centroid_error(order, N=32, T=0.3):
    """Disc transported in a frozen Taylor-Green vortex with the drivers' procedure
    (advect, keep the old solid cells, extend); centroid error in cells against the exact one."""
    U0, k, R, x0, y0 = 0.3, np.pi, 0.15, 0.5, 0.7
    uf = lambda x, y: U0 * k * np.sin(k * x) * np.cos(k * y)
    vf = lambda x, y: -U0 * k * np.cos(k * x) * np.sin(k * y)
    th = np.linspace(0, 2 * np.pi, 2001)[:-1]; n = th.size
    sol = solve_ivp(lambda t, z: np.concatenate([uf(z[:n], z[n:]), vf(z[:n], z[n:])]), (0, T),
                    np.concatenate([x0 + R * np.cos(th), y0 + R * np.sin(th)]), rtol=1e-10, atol=1e-12)
    Xb, Yb = sol.y[:n, -1], sol.y[n:, -1]; Xn, Yn = np.roll(Xb, -1), np.roll(Yb, -1)
    cr = Xb * Yn - Xn * Yb; A = cr.sum() / 2
    cex, cey = ((Xb + Xn) * cr).sum() / (6 * A), ((Yb + Yn) * cr).sum() / (6 * A)
    h = 1.0 / N; xc = (np.arange(N) + 0.5) * h; Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * h, np.arange(N) * h)
    uc, vc = uf(Xc, Yc), vf(Xc, Yc); phi0 = lambda a, b: np.hypot(a - x0, b - y0) - R
    phi = phi0(Xc, Yc); sm = (phi <= 0).astype(float)
    X1, X2 = extrapolate_reference_map(Xc * sm, Yc * sm, phi, h, h, 5, order=order)
    nst = int(np.ceil(T / (0.25 * h / (U0 * k)))); dt = T / nst
    for _ in range(nst):
        phi = phi0(X1, X2); sm = (phi <= 0).astype(float)
        X1 = advect_reference_map(X1, uc, vc, Xg, Yg, dt, h, h, phi, "semilagrangian_cubic", 4 * h) * sm
        X2 = advect_reference_map(X2, uc, vc, Xg, Yg, dt, h, h, phi, "semilagrangian_cubic", 4 * h) * sm
        X1, X2 = extrapolate_reference_map(X1, X2, phi, h, h, 5, order=order)
    w = np.clip(0.5 - phi0(X1, X2) / h, 0, 1)
    return np.hypot((w * Xc).sum() / w.sum() - cex, (w * Yc).sum() / w.sum() - cey) / h


def test_transported_interface_does_not_lag():
    e2, e1 = _centroid_error(2), _centroid_error(1)
    assert e2 < 0.02, e2                 # quadratic extension: a few thousandths of a cell
    assert e1 > 3 * e2, (e1, e2)         # plane fit: about 0.1 cell behind at this time


def test_default_is_quadratic():
    assert F.EXTRAP_ORDER == 2
