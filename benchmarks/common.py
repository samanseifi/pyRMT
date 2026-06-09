"""Shared utilities for the pyRMT benchmark drivers.

Centralises boundary conditions, narrow-band consistency checks, and the
post-processing / comparison helpers so the individual benchmark drivers stay
small and consistent with one another.

Boundary-condition / pressure-solver pairing (kept consistent on purpose):

  * No-slip walls (lid-driven cavity, soft disc in lid-driven)
        -> velocity BC `no_slip_lid_bc`
        -> pressure BC 'neumann'  (DCT direct solve)

  * Free-slip impermeable box (disc in Taylor-Green)
        The TG streamfunction sin(2*pi*x) sin(2*pi*y) has ZERO normal velocity
        on every wall, so the consistent partner is free-slip no-penetration
        walls + Neumann pressure (NOT periodic).
        -> velocity BC `free_slip_box_bc`
        -> pressure BC 'neumann'
"""

import os
import numpy as np


# ── Boundary conditions ──────────────────────────────────────────────────────

def no_slip_lid_bc(u, v, lid_speed=1.0):
    """Lid-driven cavity: no-slip on left/right/bottom, moving lid on top."""
    u = u.copy(); v = v.copy()
    u[:, 0] = 0.0;  v[:, 0] = 0.0      # left
    u[:, -1] = 0.0; v[:, -1] = 0.0     # right
    u[0, :] = 0.0;  v[0, :] = 0.0      # bottom
    u[-1, :] = lid_speed; v[-1, :] = 0.0   # top lid
    # corners pinned to zero (consistent with stationary walls)
    u[0, 0] = u[0, -1] = u[-1, 0] = u[-1, -1] = 0.0
    v[0, 0] = v[0, -1] = v[-1, 0] = v[-1, -1] = 0.0
    return u, v


def free_slip_box_bc(u, v):
    """Free-slip impermeable walls: zero normal velocity, zero-gradient
    tangential.  Consistent with Neumann pressure for the Taylor-Green box."""
    u = u.copy(); v = v.copy()
    # x-walls: no penetration (u=0), free tangential (copy v from interior)
    u[:, 0] = 0.0;  u[:, -1] = 0.0
    v[:, 0] = v[:, 1]; v[:, -1] = v[:, -2]
    # y-walls: no penetration (v=0), free tangential (copy u from interior)
    v[0, :] = 0.0;  v[-1, :] = 0.0
    u[0, :] = u[1, :]; u[-1, :] = u[-2, :]
    return u, v


# ── Field initialisers ───────────────────────────────────────────────────────

def initialize_disc(X, Y, x0, y0, R):
    """Signed-distance level set for a disc of radius R centred at (x0, y0)."""
    return np.sqrt((X - x0) ** 2 + (Y - y0) ** 2) - R


def taylor_green_velocity(X, Y, U0=1.0):
    """Taylor-Green vortex velocity, u = U0 k sin(kx) cos(ky), k = 2*pi."""
    k = 2.0 * np.pi
    u = U0 * k * np.sin(k * X) * np.cos(k * Y)
    v = -U0 * k * np.cos(k * X) * np.sin(k * Y)
    return u, v


# ── Narrow-band consistency ──────────────────────────────────────────────────

def required_extrapolation_layers(w_t, dx):
    """Minimum extrapolation layers so the solid stress is defined everywhere
    the stress-blend weight (1-H) is non-zero, i.e. out to phi = w_t.

    The smoothed Heaviside has (1-H) > 0 for phi < w_t, so we need at least
    ceil(w_t/dx) cells plus one buffer cell for the central stencil.
    """
    return int(np.ceil(w_t / dx)) + 1


def check_narrow_band(w_t, dx, num_layers):
    """Validate the narrow-band / transition-width coupling; raise if the
    extrapolation band is too thin to cover the stress-blend region."""
    need = required_extrapolation_layers(w_t, dx)
    if num_layers < need:
        raise ValueError(
            "Narrow-band inconsistency: w_t=%.4g (=%0.2f dx) needs >= %d "
            "extrapolation layers but only %d requested. The solid stress would "
            "be truncated inside the (1-H)>0 blend region."
            % (w_t, w_t / dx, need, num_layers)
        )
    return need


# ── Post-processing / comparison ─────────────────────────────────────────────

def extract_centerlines(a, b, X, Y):
    """Return (y, u_centerline) along the vertical center line x=0.5 and
    (x, v_centerline) along the horizontal center line y=0.5.  Assumes a
    uniform grid that contains the center node (odd N)."""
    Ny, Nx = a.shape
    j_mid = Ny // 2
    i_mid = Nx // 2
    y = Y[:, i_mid]
    u_line = a[:, i_mid]
    x = X[j_mid, :]
    v_line = b[j_mid, :]
    return y, u_line, x, v_line


def disc_centroid(phi, X, Y):
    """Area-weighted centroid of the solid region (phi <= 0)."""
    mask = (phi <= 0.0)
    if not np.any(mask):
        return np.nan, np.nan
    return X[mask].mean(), Y[mask].mean()


def load_xy_csv(path, has_header=False):
    """Load a 2-column CSV (x,y). Returns (x, y) arrays."""
    skip = 1 if has_header else 0
    data = np.loadtxt(path, delimiter=',', skiprows=skip)
    return data[:, 0], data[:, 1]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
