"""RM-IIM test 1: static isotropic stress in a disc (exact answer).

A disc of radius R carries a uniform isotropic solid stress s*I in quiescent fluid. The exact
solution is u = 0 and a pressure that jumps by exactly s at the interface: q = s inside, 0 outside
(plus a constant). The interface is a reference-map level set phi = phi0(xi) with a non-trivial
(rotated, translated) reference map, so the test goes through the same geometry path as the solver.

  smeared : the solid stress blended by (1 - H) over 2 cells, compact face force, plain projection
            (the current method)
  ghost   : no force; the jump [q] = -s imposed by RM-IIM ghost-fluid corrections

Usage: python benchmarks/iim_static_jump.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyRMT.mac import mac_grid, project, poisson_eigs_neumann, solid_force_faces
from pyRMT.functions import smoothed_heaviside
from pyRMT.iim import project_with_jump

S, R, C = 0.7, 0.23, (0.47, 0.52)


def setup(N):
    dx, dy = mac_grid(N, N); xc = (np.arange(N) + 0.5) * dx; X, Y = np.meshgrid(xc, xc)
    a = 0.4                                                      # rigid motion of the reference map
    X1 = C[0] + np.cos(a) * (X - C[0]) + np.sin(a) * (Y - C[1])
    X2 = C[1] - np.sin(a) * (X - C[0]) + np.cos(a) * (Y - C[1])
    phi = np.hypot(X1 - C[0], X2 - C[1]) - R                       # phi = phi0(xi)
    exact = np.where(phi <= 0, S, 0.0)
    return dx, dy, phi, exact


def errors(q, exact, phi, dx):
    e = (q - q.mean()) - (exact - exact.mean())
    bulk = np.abs(phi) > 0.06
    return np.sqrt(np.mean(e ** 2)), np.sqrt(np.mean(e[bulk] ** 2)), np.abs(e).max()


def run(N, method, dt=1e-2):
    dx, dy, phi, exact = setup(N); eig = poisson_eigs_neumann(N, N, dx, dy)
    if method == "smeared":
        w = (1 - smoothed_heaviside(phi, 2 * dx)) * S
        fu, fv = solid_force_faces(w, np.zeros_like(w), w, dx, dy)
        u, v, q = project(dt * fu, dt * fv, dx, dy, dt, 1.0, eig)
    else:
        u, v, q = project_with_jump(np.zeros((N, N + 1)), np.zeros((N + 1, N)), dx, dy, dt, 1.0,
                                    eig, phi, np.full((N, N), -S))
    return errors(q, exact, phi, dx) + (max(np.abs(u).max(), np.abs(v).max()),)


def main():
    Ns = (32, 64, 128, 256)
    for method in ("smeared", "ghost"):
        res = [run(N, method) for N in Ns]
        print(f"{method}:")
        for i, N in enumerate(Ns):
            o = "" if i == 0 else "  orders " + " ".join(
                f"{np.log2(res[i-1][k] / res[i][k]):5.2f}" if res[i][k] > 1e-13 else "   --" for k in range(3))
            print(f"  N={N:4d}  q L2 {res[i][0]:.2e}  q bulk {res[i][1]:.2e}  q max {res[i][2]:.2e}"
                  f"  max|u| {res[i][3]:.1e}{o}")


if __name__ == "__main__":
    main()
