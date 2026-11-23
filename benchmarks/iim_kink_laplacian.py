"""RM-IIM stage 2 building block: the 5-point Laplacian of a field with a derivative jump.

u = r^2 inside the circle r = R and u = R^2 + c (r^2 - R^2) outside: continuous, with a jump
2(c-1) x_i in du/dx_i. Exact Laplacian: 4 inside, 4c outside. The standard 5-point stencil is
O(1/h) wrong next to the interface; with the immersed-interface correction the local error there
is O(h), and the solution of a Poisson/Helmholtz problem with it is second order.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyRMT.iim import laplacian_jump_correction
from pyRMT.mac import poisson_eigs_neumann, solve_poisson_neumann

R, CEN, c = 0.27, (0.48, 0.53), 3.0
SECOND = "--first-only" not in sys.argv     # include the second-derivative jumps


def fields(N):
    h = 1.0 / N; x = (np.arange(N) + 0.5) * h; X, Y = np.meshgrid(x, x)
    r2 = (X - CEN[0]) ** 2 + (Y - CEN[1]) ** 2; phi = np.sqrt(r2) - R
    u = np.where(phi <= 0, r2, R ** 2 + c * (r2 - R ** 2))
    lap = np.where(phi <= 0, 4.0, 4.0 * c)
    # jump (outside minus inside) of du/dx and du/dy, written on both sides: 2(c-1)(x - x0)
    gx = 2 * (c - 1) * (X - CEN[0]); gy = 2 * (c - 1) * (Y - CEN[1])
    g2 = np.full_like(X, 2 * (c - 1))          # jump of d2u/dx2 and d2u/dy2
    return h, phi, u, lap, gx, gy, g2


def lap5(u, h):
    L = np.full_like(u, np.nan)
    L[1:-1, 1:-1] = (u[1:-1, 2:] + u[1:-1, :-2] + u[2:, 1:-1] + u[:-2, 1:-1] - 4 * u[1:-1, 1:-1]) / h ** 2
    return L


def main():
    print("local truncation error at the points next to the interface:")
    prev = None
    for N in (32, 64, 128, 256, 512):
        h, phi, u, lap, gx, gy, g2 = fields(N)
        band = np.zeros_like(phi, bool); band[1:-1, 1:-1] = True
        near = band & (np.abs(phi) < 1.5 * h); far = band & (np.abs(phi) > 3 * h)
        L = lap5(u, h); Lc = L + laplacian_jump_correction(phi, gx, gy, h, h, gxx=(g2 if SECOND else None), gyy=(g2 if SECOND else None))
        e_raw, e_cor, e_far = np.abs(L - lap)[near].max(), np.abs(Lc - lap)[near].max(), np.abs(L - lap)[far].max()
        o = "" if prev is None else "  orders raw %.2f corrected %.2f" % (np.log2(prev[0] / e_raw), np.log2(prev[1] / e_cor))
        print(f"  N={N:4d}  raw {e_raw:.2e}  corrected {e_cor:.2e}  (away from interface {e_far:.1e}){o}")
        prev = (e_raw, e_cor)

    print("Poisson solve with the kinked solution (Neumann data from the exact u):")
    prev = None
    for N in (32, 64, 128, 256, 512):
        h, phi, u, lap, gx, gy, g2 = fields(N); eig = poisson_eigs_neumann(N, N, h, h)
        # boundary: ghost values from the exact solution -> move them to the right-hand side
        x = (np.arange(-1, N + 1) + 0.5) * h; X, Y = np.meshgrid(x, x)
        r2 = (X - CEN[0]) ** 2 + (Y - CEN[1]) ** 2
        ue = np.where(np.sqrt(r2) <= R, r2, R ** 2 + c * (r2 - R ** 2))
        bc = np.zeros((N, N))
        bc[:, 0] += (ue[1:-1, 0] - ue[1:-1, 1]) / h ** 2; bc[:, -1] += (ue[1:-1, -1] - ue[1:-1, -2]) / h ** 2
        bc[0, :] += (ue[0, 1:-1] - ue[1, 1:-1]) / h ** 2; bc[-1, :] += (ue[-1, 1:-1] - ue[-2, 1:-1]) / h ** 2
        out = []
        for corr in (False, True):
            rhs = lap - bc - (laplacian_jump_correction(phi, gx, gy, h, h, gxx=(g2 if SECOND else None), gyy=(g2 if SECOND else None)) if corr else 0.0)
            s = solve_poisson_neumann(rhs, eig)
            e = (s - s.mean()) - (u - u.mean()); out.append((np.abs(e).max(), np.sqrt(np.mean(e ** 2))))
        o = "" if prev is None else "  orders raw %.2f/%.2f corrected %.2f/%.2f" % (
            np.log2(prev[0][0] / out[0][0]), np.log2(prev[0][1] / out[0][1]),
            np.log2(prev[1][0] / out[1][0]), np.log2(prev[1][1] / out[1][1]))
        print(f"  N={N:4d}  raw max/L2 {out[0][0]:.2e}/{out[0][1]:.2e}   corrected {out[1][0]:.2e}/{out[1][1]:.2e}{o}")
        prev = out


if __name__ == "__main__":
    main()
