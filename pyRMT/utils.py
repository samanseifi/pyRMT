from numba import njit
import numpy as np

@njit
def grad_central_x_2nd(f, dx):
    df_dx = np.zeros_like(f)
    # Interior 2nd order central
    df_dx[:, 1:-1] = (f[:, 2:] - f[:, 0:-2]) / (2 * dx)

    # Left boundary (i=0)
    df_dx[:, 0] = (-3*f[:, 0] + 4*f[:, 1] - f[:, 2]) / (2*dx)
    # Right boundary (i=Nx-1)
    df_dx[:, -1] = (3*f[:, -1] - 4*f[:, -2] + f[:, -3]) / (2*dx)
    return df_dx

@njit
def grad_central_y_2nd(f, dy):
    df_dy = np.zeros_like(f)
    df_dy[1:-1, :] = (f[2:, :] - f[0:-2, :]) / (2 * dy)

    # Bottom boundary (j=0)
    df_dy[0, :] = (-3*f[0, :] + 4*f[1, :] - f[2, :]) / (2*dy)
    # Top boundary (j=Ny-1)
    df_dy[-1, :] = (3*f[-1, :] - 4*f[-2, :] + f[-3, :]) / (2*dy)
    return df_dy

@njit
def diff_upwind_3rd(f, u, h, axis):
    """
    3rd Order Upwind Biased Finite Difference with 1st-order fallback at boundaries.
    axis=0 for y, axis=1 for x
    """
    df = np.zeros_like(f)
    Ny, Nx = f.shape

    if axis == 1: # X-deriv
        for j in range(Ny):
            # Boundary cells: 1st-order upwind fallback
            for i in (0, 1, Nx-2, Nx-1):
                if i < 0 or i >= Nx:
                    continue
                vel = u[j, i]
                if vel > 0 and i > 0:
                    df[j, i] = (f[j, i] - f[j, i-1]) / h
                elif vel <= 0 and i < Nx - 1:
                    df[j, i] = (f[j, i+1] - f[j, i]) / h
                elif i > 0:
                    df[j, i] = (f[j, i] - f[j, i-1]) / h
                elif i < Nx - 1:
                    df[j, i] = (f[j, i+1] - f[j, i]) / h
            # Interior cells: 3rd-order upwind
            for i in range(2, Nx-2):
                vel = u[j, i]
                if vel > 0:
                    df[j, i] = (2*f[j, i+1] + 3*f[j, i] - 6*f[j, i-1] + f[j, i-2]) / (6*h)
                else:
                    df[j, i] = (-f[j, i+2] + 6*f[j, i+1] - 3*f[j, i] - 2*f[j, i-1]) / (6*h)
    else: # Y-deriv
        for i in range(Nx):
            # Boundary cells: 1st-order upwind fallback
            for j in (0, 1, Ny-2, Ny-1):
                if j < 0 or j >= Ny:
                    continue
                vel = u[j, i]
                if vel > 0 and j > 0:
                    df[j, i] = (f[j, i] - f[j-1, i]) / h
                elif vel <= 0 and j < Ny - 1:
                    df[j, i] = (f[j+1, i] - f[j, i]) / h
                elif j > 0:
                    df[j, i] = (f[j, i] - f[j-1, i]) / h
                elif j < Ny - 1:
                    df[j, i] = (f[j+1, i] - f[j, i]) / h
            # Interior cells: 3rd-order upwind
            for j in range(2, Ny-2):
                vel = u[j, i]
                if vel > 0:
                    df[j, i] = (2*f[j+1, i] + 3*f[j, i] - 6*f[j-1, i] + f[j-2, i]) / (6*h)
                else:
                    df[j, i] = (-f[j+2, i] + 6*f[j+1, i] - 3*f[j, i] - 2*f[j-1, i]) / (6*h)
    return df

@njit
def lap_2nd(f, dx, dy):
    lap = np.zeros_like(f)
    # Second derivative in x (interior: central difference)
    lap[:, 1:-1] += (f[:, 2:] - 2*f[:, 1:-1] + f[:, 0:-2]) / dx**2
    # Boundaries (2nd-order one-sided)
    lap[:, 0] += (2*f[:, 0] - 5*f[:, 1] + 4*f[:, 2] - f[:, 3]) / dx**2
    lap[:, -1] += (2*f[:, -1] - 5*f[:, -2] + 4*f[:, -3] - f[:, -4]) / dx**2

    # Second derivative in y (interior: central difference)
    lap[1:-1, :] += (f[2:, :] - 2*f[1:-1, :] + f[0:-2, :]) / dy**2
    # Boundaries (2nd-order one-sided)
    lap[0, :] += (2*f[0, :] - 5*f[1, :] + 4*f[2, :] - f[3, :]) / dy**2
    lap[-1, :] += (2*f[-1, :] - 5*f[-2, :] + 4*f[-3, :] - f[-4, :]) / dy**2

    return lap


@njit(inline='always')
def fast_solve_3x3(A, b):
    """
    Explicitly solves Ax = b for a 3x3 matrix using Cramer's Rule.
    Designed for high-performance inlining within Numba loops.
    """
    # Compute the determinant of A
    detA = (A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) -
            A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) +
            A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]))

    # Check for singularity (using a small epsilon)
    if abs(detA) < 1e-15:
        return np.zeros(3)

    inv_det = 1.0 / detA

    # Solve for x, y, z using Cramer's Rule
    # x = det(A with col 0 replaced by b) / detA
    x = (b[0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) -
         A[0, 1] * (b[1] * A[2, 2] - A[1, 2] * b[2]) +
         A[0, 2] * (b[1] * A[2, 1] - A[1, 1] * b[2])) * inv_det

    # y = det(A with col 1 replaced by b) / detA
    y = (A[0, 0] * (b[1] * A[2, 2] - A[1, 2] * b[2]) -
         b[0] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) +
         A[0, 2] * (A[1, 0] * b[2] - b[1] * A[2, 0])) * inv_det

    # z = det(A with col 2 replaced by b) / detA
    z = (A[0, 0] * (A[1, 1] * b[2] - b[1] * A[2, 1]) -
         A[0, 1] * (A[1, 0] * b[2] - b[1] * A[2, 0]) +
         b[0] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])) * inv_det

    return np.array([x, y, z])