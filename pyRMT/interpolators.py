from numba import njit, prange
import numpy as np

@njit(cache=True)
def bilinear_interpolate(u, xq, yq, dx, dy, Nx, Ny):
    """
    Fast bilinear interpolator for 2D grid data.

    Parameters:
        u  : 2D array of shape (Ny, Nx)
        xq : query x positions (same shape as u)
        yq : query y positions (same shape as u)
        dx, dy : grid spacing
        Nx, Ny : number of grid points in x and y

    Returns:
        Interpolated values at query points
    """
    out = np.zeros_like(xq)
    for j in prange(xq.shape[0]):
        for i in range(xq.shape[1]):
            x = xq[j, i] / dx
            y = yq[j, i] / dy

            # Guard against non-finite query coordinates: int(floor(NaN/Inf)) is
            # undefined and indexes out of bounds (segfault). Fail gracefully so
            # an upstream instability surfaces as NaN rather than a crash.
            if not (np.isfinite(x) and np.isfinite(y)):
                out[j, i] = np.nan
                continue

            # Clamp the float coordinate into range BEFORE the int cast. A huge
            # (but finite) value would overflow float->int and index out of
            # bounds; clamping first keeps the cast well-defined.
            if x < 0.0:
                x = 0.0
            elif x > Nx - 1.0:
                x = Nx - 1.0
            if y < 0.0:
                y = 0.0
            elif y > Ny - 1.0:
                y = Ny - 1.0

            ix = int(np.floor(x))
            iy = int(np.floor(y))
            if ix >= Nx - 1:
                ix = Nx - 2
            if iy >= Ny - 1:
                iy = Ny - 2

            fx = x - ix
            fy = y - iy

            v00 = u[iy,   ix  ]
            v10 = u[iy,   ix+1]
            v01 = u[iy+1, ix  ]
            v11 = u[iy+1, ix+1]

            out[j, i] = (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 + \
                        (1 - fx) * fy * v01 + fx * fy * v11

    return out

@njit(parallel=True, cache=True)
def bicubic_interpolate(u, xq, yq, dx, dy, Nx, Ny):
    """
    Bicubic interpolation for Semi-Lagrangian Advection.
    Reduces numerical diffusion significantly compared to bilinear.
    """
    out = np.zeros_like(xq)
    
    for j in prange(xq.shape[0]):
        for i in range(xq.shape[1]):
            # Normalized coordinates
            x_idx = xq[j, i] / dx
            y_idx = yq[j, i] / dy

            # Guard against non-finite query coordinates (see bilinear_interpolate).
            if not (np.isfinite(x_idx) and np.isfinite(y_idx)):
                out[j, i] = np.nan
                continue

            # Clamp float coords before the int cast (avoid float->int overflow on
            # huge finite values); per-point stencil indices are clamped below.
            if x_idx < 0.0:
                x_idx = 0.0
            elif x_idx > Nx - 1.0:
                x_idx = Nx - 1.0
            if y_idx < 0.0:
                y_idx = 0.0
            elif y_idx > Ny - 1.0:
                y_idx = Ny - 1.0

            # Integer part (bottom-left corner of the central cell)
            ix = int(np.floor(x_idx))
            iy = int(np.floor(y_idx))
            
            # Fractional part
            fx = x_idx - ix
            fy = y_idx - iy
            
            # We need a 4x4 stencil: indices from ix-1 to ix+2
            # Handle boundaries by clamping
            row_vals = np.zeros(4)
            local_min = 1e18
            local_max = -1e18

            for m in range(4): # Loop over y-rows (local index 0..3)
                # Global y index: iy - 1 + m
                y_global = iy - 1 + m

                # Clamp y
                if y_global < 0: y_global = 0
                if y_global >= Ny: y_global = Ny - 1

                # Gather the 4 x-values for this row
                col_vals = np.zeros(4)
                for n in range(4): # Loop over x-cols (local index 0..3)
                    x_global = ix - 1 + n

                    # Clamp x
                    if x_global < 0: x_global = 0
                    if x_global >= Nx: x_global = Nx - 1

                    v = u[y_global, x_global]
                    col_vals[n] = v
                    if v < local_min: local_min = v
                    if v > local_max: local_max = v

                # Interpolate this row in x-direction
                row_vals[m] = cubic_convolution(col_vals[0], col_vals[1], col_vals[2], col_vals[3], fx)

            # Final interpolate in y-direction
            result = cubic_convolution(row_vals[0], row_vals[1], row_vals[2], row_vals[3], fy)
            # Clamp to stencil bounds to suppress Catmull-Rom overshoots near
            # steep gradients (e.g. reference-map jump at solid/fluid interface)
            if result < local_min: result = local_min
            if result > local_max: result = local_max
            out[j, i] = result
            
    return out

@njit(cache=True)
def cubic_convolution(v0, v1, v2, v3, x):
    """
    Catmull-Rom Cubic Spline interpolation.
    v0, v1, v2, v3 are values at x=-1, 0, 1, 2.
    x is the fractional distance between v1 and v2 (0 <= x <= 1).
    """
    a0 = -0.5*v0 + 1.5*v1 - 1.5*v2 + 0.5*v3
    a1 = v0 - 2.5*v1 + 2.0*v2 - 0.5*v3
    a2 = -0.5*v0 + 0.5*v2
    a3 = v1
    return a0*x**3 + a1*x**2 + a2*x + a3