"""Finite-difference operators: accuracy on manufactured smooth fields."""
import numpy as np
import pytest
from pyRMT.utils import grad_central_x_2nd, grad_central_y_2nd, lap_2nd


def _grid(N):
    x = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, x)
    return X, Y, x[1] - x[0]


@pytest.mark.parametrize("N", [33, 65])
def test_grad_x_interior_exact_on_linear(N):
    X, Y, h = _grid(N)
    f = 3.0 * X + 2.0 * Y
    dfdx = grad_central_x_2nd(f, h)
    assert np.allclose(dfdx[1:-1, 1:-1], 3.0, atol=1e-10)


@pytest.mark.parametrize("N", [33, 65])
def test_grad_y_interior_exact_on_linear(N):
    X, Y, h = _grid(N)
    f = 3.0 * X + 2.0 * Y
    dfdy = grad_central_y_2nd(f, h)
    assert np.allclose(dfdy[1:-1, 1:-1], 2.0, atol=1e-10)


def test_grad_second_order():
    """Interior gradient error should drop ~4x when h halves (2nd order)."""
    errs = []
    for N in (33, 65):
        X, Y, h = _grid(N)
        f = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        exact = 2 * np.pi * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        num = grad_central_x_2nd(f, h)
        errs.append(np.max(np.abs((num - exact)[2:-2, 2:-2])))
    order = np.log(errs[0] / errs[1]) / np.log(2)
    assert order > 1.8


def test_laplacian_zero_on_harmonic_quadratic():
    """lap(x^2 - y^2) = 0; 2nd-order central differences are exact for quadratics."""
    X, Y, h = _grid(65)
    f = X**2 - Y**2
    lap = lap_2nd(f, h, h)
    assert np.max(np.abs(lap[1:-1, 1:-1])) < 1e-9


def test_laplacian_second_order_on_sine():
    """Laplacian error of sin(pi x) sin(pi y) drops ~4x as h halves."""
    errs = []
    for N in (33, 65):
        X, Y, h = _grid(N)
        f = np.sin(np.pi * X) * np.sin(np.pi * Y)
        exact = -2.0 * np.pi**2 * f
        lap = lap_2nd(f, h, h)
        errs.append(np.max(np.abs((lap - exact)[2:-2, 2:-2])))
    order = np.log(errs[0] / errs[1]) / np.log(2)
    assert order > 1.8
