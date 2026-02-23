import os
import h5py
import numpy as np
import csv

def compute_kinetic_energy(a, b, rho_f, rho_s, phi, w_t, dx, dy):
    """
    Compute kinetic energy: KE = ∫ (0.5 * rho * u_i * u_i) dA

    Parameters:
    -----------
    a, b : ndarray
        Velocity components (u, v)
    rho_f, rho_s : float
        Fluid and solid densities
    phi : ndarray
        Level set function
    w_t : float
        Smoothing width for Heaviside function
    dx, dy : float
        Grid spacing

    Returns:
    --------
    float : Total kinetic energy
    """
    from pyRMT.functions import heaviside_smooth_alt

    # Compute local density using smooth Heaviside
    H = heaviside_smooth_alt(phi, w_t)
    rho_local = (1 - H) * rho_s + H * rho_f

    # Kinetic energy density: 0.5 * rho * |u|^2
    ke_density = 0.5 * rho_local * (a**2 + b**2)

    # Integrate over domain
    ke_total = np.sum(ke_density) * dx * dy

    return ke_total

def compute_strain_energy(X1, X2, phi, mu_s, dx, dy, kappa=0.0):
    """
    Compute elastic strain energy: SE = ∫ μ_s * (tr(F^T F) - 2) dA

    For Neo-Hookean solid: W = (μ_s/2) * (I_1 - 2)
    where I_1 = tr(F^T F) = tr(C) with C = F^T F (right Cauchy-Green tensor)

    Parameters:
    -----------
    X1, X2 : ndarray
        Reference map components
    phi : ndarray
        Level set function (solid where phi <= 0)
    mu_s : float
        Solid shear modulus
    dx, dy : float
        Grid spacing

    Returns:
    --------
    float : Total strain energy
    """
    from pyRMT.functions import grad_central_x_2nd, grad_central_y_2nd

    # Compute gradients of reference maps: G = ∂X/∂x
    pad_width = 4
    X1_padded = np.pad(X1, pad_width, mode='edge')
    X2_padded = np.pad(X2, pad_width, mode='edge')

    dX1_dx = grad_central_x_2nd(X1_padded, dx)
    dX1_dy = grad_central_y_2nd(X1_padded, dy)
    dX2_dx = grad_central_x_2nd(X2_padded, dx)
    dX2_dy = grad_central_y_2nd(X2_padded, dy)

    # Remove padding
    dX1_dx = dX1_dx[pad_width:-pad_width, pad_width:-pad_width]
    dX1_dy = dX1_dy[pad_width:-pad_width, pad_width:-pad_width]
    dX2_dx = dX2_dx[pad_width:-pad_width, pad_width:-pad_width]
    dX2_dy = dX2_dy[pad_width:-pad_width, pad_width:-pad_width]

    # Solid mask
    solid_mask = (phi <= 0.0)

    # Initialize strain energy density
    se_density = np.zeros_like(phi)

    if np.any(solid_mask):
        # G = ∂X/∂x
        G11 = dX1_dx
        G12 = dX1_dy
        G21 = dX2_dx
        G22 = dX2_dy

        # det(G) and inverse to get F = G^{-1}
        detG = G11 * G22 - G12 * G21
        good = (np.abs(detG) > 1e-10) & solid_mask

        if np.any(good):
            # F = G^{-1}
            F11 = np.zeros_like(G11)
            F12 = np.zeros_like(G12)
            F21 = np.zeros_like(G21)
            F22 = np.zeros_like(G22)

            F11[good] =  G22[good] / detG[good]
            F12[good] = -G12[good] / detG[good]
            F21[good] = -G21[good] / detG[good]
            F22[good] =  G11[good] / detG[good]

            # C = F^T F (right Cauchy-Green tensor)
            C11 = F11**2 + F21**2
            C22 = F12**2 + F22**2
            # C12 = F11*F12 + F21*F22  # Off-diagonal (not needed for trace)

            # I_1 = tr(C) = C11 + C22
            I1 = C11 + C22

            # Full compressible Neo-Hookean energy density:
            # W = (μ/2)(I₁ - 2 - 2·ln J) + (κ/2)(J-1)²
            # Consistent with σ = (μ/J)(B-I) + κ(J-1)I
            J_vals = np.ones_like(detG)
            J_vals[good] = 1.0 / detG[good]
            J_safe = np.abs(J_vals)  # guard against clamped negative detG
            se_density[good] = (0.5 * mu_s * (I1[good] - 2.0 - 2.0 * np.log(J_safe[good]))
                                + 0.5 * kappa * (J_vals[good] - 1.0)**2)

    # Integrate over solid domain
    se_total = np.sum(se_density) * dx * dy

    return se_total

def compute_viscous_dissipation(a, b, mu_f, phi, w_t, dx, dy, eta_s=0.0):
    """
    Compute viscous dissipation rate: ε = ∫ μ_f * (∂u_i/∂x_j) * (∂u_i/∂x_j) dA

    For Newtonian fluid: ε = ∫ μ_f * D_ij * D_ij dA
    where D_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i) is the rate-of-deformation tensor

    Full expression: ε = 2 * μ_f * ∫ (D_ij * D_ij) dA
                       = 2 * μ_f * ∫ (D_xx^2 + D_yy^2 + 2*D_xy^2) dA

    Parameters:
    -----------
    a, b : ndarray
        Velocity components (u, v)
    mu_f : float
        Fluid dynamic viscosity
    phi : ndarray
        Level set function
    w_t : float
        Smoothing width for Heaviside function
    dx, dy : float
        Grid spacing
    eta_s : float, optional
        Solid viscosity (Kelvin-Voigt damping)

    Returns:
    --------
    float : Viscous dissipation rate (power)
    """
    from pyRMT.functions import grad_central_x_2nd, grad_central_y_2nd, heaviside_smooth_alt

    # Velocity gradients
    du_dx = grad_central_x_2nd(a, dx)
    dv_dy = grad_central_y_2nd(b, dy)
    du_dy = grad_central_y_2nd(a, dy)
    dv_dx = grad_central_x_2nd(b, dx)

    # Rate-of-deformation tensor components
    D_xx = du_dx
    D_yy = dv_dy
    D_xy = 0.5 * (du_dy + dv_dx)

    # Dissipation function: Φ = 2 * μ * (D_ij * D_ij)
    # In 2D: Φ = 2 * μ * (D_xx^2 + D_yy^2 + 2*D_xy^2)

    # Compute Heaviside for blending
    H = heaviside_smooth_alt(phi, w_t)

    # Local viscosity (fluid viscosity in fluid, solid viscosity in solid if eta_s > 0)
    mu_local = H * mu_f + (1 - H) * eta_s

    # Dissipation density
    dissipation_density = 2.0 * mu_local * (D_xx**2 + D_yy**2 + 2.0 * D_xy**2)

    # Integrate over domain
    dissipation_rate = np.sum(dissipation_density) * dx * dy

    return dissipation_rate

def divergence_2d_interior(u, v, dx, dy, pad=3):
    """
    Computes divergence but ignores the boundary layers (pad)
    to avoid reporting singularity errors at the lid corners.
    """
    divU = np.zeros_like(u)
    
    # Only compute for the interior, excluding 'pad' cells from edges
    # Standard central difference
    divU[pad:-pad, pad:-pad] = (
        (u[pad:-pad, 2+pad-1 : -pad+1] - u[pad:-pad, pad-1 : -2-pad+1]) / (2*dx) +
        (v[2+pad-1 : -pad+1, pad:-pad] - v[pad-1 : -2-pad+1, pad:-pad]) / (2*dy)
    )
    
    # Extract only the interior for statistics
    interior_div = divU[pad:-pad, pad:-pad]
    return divU, interior_div

def output_simulation_data(dx, dy, phi, solid_mask, X1, X2, a, b, p, vis_output_freq,
                          directory_name, step, dt, sigma_sxx, sigma_sxy, sigma_syy, J,
                          mu_s=0.0, mu_f=0.0, rho_s=1.0, rho_f=1.0, w_t=None, eta_s=0.0,
                          kappa=0.0, time=0.0, integrated_dissipation=0.0):
    """
    Output simulation data including energy diagnostics.

    Additional Parameters (for energy calculations):
    ------------------------------------------------
    mu_s : float
        Solid shear modulus
    mu_f : float
        Fluid dynamic viscosity
    rho_s : float
        Solid density
    rho_f : float
        Fluid density
    w_t : float
        Smoothing width for Heaviside function (defaults to 2*dx if None)
    eta_s : float
        Solid viscosity (Kelvin-Voigt damping)
    time : float
        Current simulation time
    integrated_dissipation : float
        Cumulative integrated dissipation ∫₀ᵗ ε(τ) dτ

    Returns:
    --------
    float : Updated integrated dissipation value
    """
    if w_t is None:
        w_t = 2.0 * dx

    if step % vis_output_freq == 0 or step == 1:
        vmag = np.sqrt(a**2 + b**2)

        # Calculate divergence, but get the 'clean' interior version for stats
        div_field, div_interior = divergence_2d_interior(a, b, dx, dy, pad=4)

        solid_area = np.sum(solid_mask) * dx * dy

        # Compute energy quantities
        ke = compute_kinetic_energy(a, b, rho_f, rho_s, phi, w_t, dx, dy)
        se = compute_strain_energy(X1, X2, phi, mu_s, dx, dy, kappa=kappa)
        dissipation_rate = compute_viscous_dissipation(a, b, mu_f, phi, w_t, dx, dy, eta_s)

        # Note: integrated_dissipation is updated in the main loop, not here
        total_energy = ke + se + integrated_dissipation

        # Use div_interior for the log!
        print(
                f"[Step {step:05d}] t={time:.3f}, dt={dt:.2e}, "
                f"max|v|={np.max(vmag):.3f}, "
                f"KE={ke:.4e}, SE={se:.4e}, ε={dissipation_rate:.4e}, "
                f"E_tot={total_energy:.4e}, "
                f"min(J)={np.min(J):.3f}, "
                f"max|σ|={np.max(np.sqrt(sigma_sxx**2 + sigma_syy**2 + 2*sigma_sxy**2)):.2f}, "
                f"max|div|={np.max(np.abs(div_interior)):.2e}"
            )

        output_dir = os.path.join("outputs", directory_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save energy time series to CSV
        energy_file = os.path.join(output_dir, "energy_history.csv")
        file_exists = os.path.isfile(energy_file)

        with open(energy_file, 'a', newline='') as csvfile:
            fieldnames = ['step', 'time', 'dt', 'kinetic_energy', 'strain_energy',
                         'dissipation_rate', 'integrated_dissipation', 'total_energy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists or step == 1:
                writer.writeheader()

            writer.writerow({
                'step': step,
                'time': time,
                'dt': dt,
                'kinetic_energy': ke,
                'strain_energy': se,
                'dissipation_rate': dissipation_rate,
                'integrated_dissipation': integrated_dissipation,
                'total_energy': total_energy
            })

        with h5py.File(os.path.join(output_dir, f"data_{step:06d}.h5"), "w") as f:
            f.create_dataset("phi", data=phi)
            f.create_dataset("X1", data=X1)
            f.create_dataset("X2", data=X2)
            f.create_dataset("J", data=J)
            f.create_dataset("a", data=a)
            f.create_dataset("b", data=b)
            f.create_dataset("p", data=p)
            f.create_dataset("sigma_xx", data=sigma_sxx)
            f.create_dataset("sigma_yy", data=sigma_syy)
            f.create_dataset("sigma_xy", data=sigma_sxy)
            # Save the full field, even if edges are noisy
            f.create_dataset("div_vel", data=div_field)

            # Save scalar energy values as attributes
            f.attrs['time'] = time
            f.attrs['kinetic_energy'] = ke
            f.attrs['strain_energy'] = se
            f.attrs['dissipation_rate'] = dissipation_rate
            f.attrs['integrated_dissipation'] = integrated_dissipation
            f.attrs['total_energy'] = total_energy

    return integrated_dissipation