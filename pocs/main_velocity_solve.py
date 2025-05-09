import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ------------------------------
# Grid and physical parameters
# ------------------------------
Nx, Ny = 100, 100
Lx, Ly = 1.0, 1.0
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

rho = 1.0
nu = 0.01
dt = 0.001
nt = 10000  # total time steps
beta = 1.5  # under-relaxation for Rhie–Chow

# ------------------------------
# Initialization
# ------------------------------
u = np.zeros((Ny, Nx))
v = np.zeros((Ny, Nx))
p = np.zeros((Ny, Nx))

# Top lid velocity
u[0, :] = 1.0

# ------------------------------
# Boundary conditions
# ------------------------------
def apply_bcs(u, v):
    u[0, :] = 0.0
    u[-1, :] = 1.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    v[0, :] = 0.0
    v[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    return u, v

# ------------------------------
# RHS for pressure Poisson
# ------------------------------
def build_rhs(u, v):
    dudx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
    dvdy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
    rhs = np.zeros_like(u)
    rhs[1:-1, 1:-1] = rho * (dudx + dvdy) / dt
    return rhs

# ------------------------------
# Pressure Poisson solver
# ------------------------------
def pressure_poisson(p, rhs):
    for _ in range(100):
        p[1:-1, 1:-1] = (
            (p[1:-1, 2:] + p[1:-1, :-2]) * dy**2 +
            (p[2:, 1:-1] + p[:-2, 1:-1]) * dx**2 -
            rhs[1:-1, 1:-1] * dx**2 * dy**2
        ) / (2 * (dx**2 + dy**2))

        # Boundary conditions for pressure
        p[:, -1] = p[:, -2]
        p[:, 0] = p[:, 1]
        p[0, :] = p[1, :]
        p[-1, :] = 0
    return p

# ------------------------------
# Rhie–Chow velocity correction
# ------------------------------
def update_velocities_rc(u, v, p, u_star, v_star):
    u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - beta * dt / rho * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - beta * dt / rho * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)
    return u, v

# ------------------------------
# Advection + Diffusion predictor
# ------------------------------
def compute_predictor(u, v):
    un = u.copy()
    vn = v.copy()

    # Apply Gaussian filter (optional)
    # un = gaussian_filter(un, sigma=0.5)
    # vn = gaussian_filter(vn, sigma=0.5)

    u_star = un.copy()
    v_star = vn.copy()

    # u momentum equation
    u_star[1:-1, 1:-1] += dt * (
        - un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2]) / dx
        - vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[:-2, 1:-1]) / dy
        + nu * (
            (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) / dx**2 +
            (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) / dy**2
        )
    )

    # v momentum equation
    v_star[1:-1, 1:-1] += dt * (
        - un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) / dx
        - vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) / dy
        + nu * (
            (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) / dx**2 +
            (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1]) / dy**2
        )
    )

    return u_star, v_star

# ------------------------------
# Main time-stepping loop
# ------------------------------
for n in range(nt):
    u, v = apply_bcs(u, v)
    u_star, v_star = compute_predictor(u, v)
    rhs = build_rhs(u_star, v_star)
    p = pressure_poisson(p, rhs)
    u, v = update_velocities_rc(u, v, p, u_star, v_star)
    u, v = apply_bcs(u, v)

# ------------------------------
# Post-process and visualize
# ------------------------------
plt.figure()
plt.streamplot(X, Y, u, v, density=1.0)
plt.title("Lid-Driven Cavity (Collocated Grid with Advection + Rhie-Chow)")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect('equal')
plt.show()

# ------------------------------
# Benchmark plot vs Ghia et al.
# ------------------------------
u_center_x = u[:, Nx // 2]
y = np.linspace(0, 1, Ny)

u_ghia_data = np.loadtxt("plot_u_y_Ghia100.csv", delimiter=",", skiprows=1)
y_ghia = u_ghia_data[:, 0]
u_ghia = u_ghia_data[:, 1]

plt.figure()
plt.plot(u_center_x, y, label="Simulation")
plt.plot(u_ghia, y_ghia, "o", label="Ghia et al. (1982)")
plt.xlabel("u (horizontal velocity)")
plt.ylabel("y")
plt.title("u vs y at x=0.5")
plt.grid(True)
plt.legend()
plt.gca().invert_yaxis()
plt.show()
