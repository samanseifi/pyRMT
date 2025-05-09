import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pyamg
from scipy.sparse import diags, kron, identity, csr_matrix

import numpy as np

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
mu = 0.01
nu = mu / rho
dt = 0.001
nt = 1000
beta = 1.5  # Rhie–Chow under-relaxation

# ------------------------------
# Initialization
# ------------------------------
u = np.zeros((Ny, Nx))
v = np.zeros((Ny, Nx))
p = np.zeros((Ny, Nx))

# ------------------------------
# Boundary Conditions
# ------------------------------
def apply_bcs(u, v):
    u[0, :] = 0.0       # bottom
    u[-1, :] = 1.0      # top (lid)
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 0.0
    return u, v

# ------------------------------
# Advection Term: -(u·∇)u
# ------------------------------
def advection_term(u, v, dx, dy):
    u_adv = - u * (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx) \
            - v * (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dy)
    
    v_adv = - u * (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dx) \
            - v * (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)
    return u_adv, v_adv

# ------------------------------
# Diffusion Term: ν ∇²u
# ------------------------------
def diffusion_term(u, dx, dy, nu):
    lap_u = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dx**2 \
          + (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dy**2
    return nu * lap_u

# ------------------------------
# Compute Predictor: u* = uⁿ + dt * (Advection + Diffusion)
# ------------------------------
def compute_predictor(u, v):
    u_adv, v_adv = advection_term(u, v, dx, dy)
    u_diff = diffusion_term(u, dx, dy, nu)
    v_diff = diffusion_term(v, dx, dy, nu)
    u_star = u + dt * (u_adv + u_diff)
    v_star = v + dt * (v_adv + v_diff)
    return u_star, v_star

# ------------------------------
# RHS of Pressure Poisson: ∇·u*
# ------------------------------
def build_rhs(u_star, v_star):
    dudx = (np.roll(u_star, -1, axis=1) - np.roll(u_star, 1, axis=1)) / (2 * dx)
    dvdy = (np.roll(v_star, -1, axis=0) - np.roll(v_star, 1, axis=0)) / (2 * dy)
    rhs = rho * (dudx + dvdy) / dt
    return rhs

# ------------------------------
# Solve Pressure Poisson Equation
# ------------------------------
def pressure_poisson(p, rhs):
    for _ in range(100):
        p[1:-1, 1:-1] = (
            (p[1:-1, 2:] + p[1:-1, :-2]) * dy**2 +
            (p[2:, 1:-1] + p[:-2, 1:-1]) * dx**2 -
            rhs[1:-1, 1:-1] * dx**2 * dy**2
        ) / (2 * (dx**2 + dy**2))

        # Neumann BCs
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[-1, :] = 0.0
    return p

# ------------------------------
# Pressure Gradient Correction
# ------------------------------
def pressure_gradient_correction(u_star, v_star, p):
    dpdx = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * dx)
    dpdy = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2 * dy)

    u = u_star - beta * dt / rho * dpdx
    v = v_star - beta * dt / rho * dpdy
    return u, v

# ------------------------------
# Time Integration Loop
# ------------------------------
for n in range(nt):
    u, v = apply_bcs(u, v)
    u_star, v_star = compute_predictor(u, v)
    rhs = build_rhs(u_star, v_star)
    p = pressure_poisson(p, rhs)
    u, v = pressure_gradient_correction(u_star, v_star, p)
    u, v = apply_bcs(u, v)



# ------------------------------
# Compute fluid stress (for FSI)
# ------------------------------

# ------------------------------
# Plot streamlines
# ------------------------------
plt.figure()
plt.streamplot(X, Y, u, v, density=1.0)
plt.title("Lid-Driven Cavity (Improved Collocated Solver)")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect('equal')
plt.show()


plt.figure()
plt.contourf(X, Y, u)
plt.title("Lid-Driven Cavity (Improved Collocated Solver)")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect('equal')
plt.show()

# ------------------------------
# Ghia Comparison
# ------------------------------
try:
    u_center_x = u[:, Nx // 2]
    y_plot = np.linspace(0, 1, Ny)
    ghia = np.loadtxt("data/plot_u_y_Ghia100.csv", delimiter=",", skiprows=1)
    y_ghia, u_ghia = ghia[:, 0], ghia[:, 1]

    plt.figure()
    plt.plot(u_center_x, y_plot, label="Simulation")
    plt.plot(u_ghia, y_ghia, "o", label="Ghia et al. (1982)")
    plt.xlabel("u (horizontal velocity)")
    plt.ylabel("y")
    plt.title("u vs y at x=0.5")
    plt.grid(True)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()
except:
    print("Benchmark data not found. Skipping Ghia plot.")
