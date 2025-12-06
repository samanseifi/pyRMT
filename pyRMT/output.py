import os
import h5py
import numpy as np

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

def output_simulation_data(dx, dy, phi, solid_mask, X1, X2, a, b, p, vis_output_freq, directory_name, step, dt, sigma_sxx, sigma_sxy, sigma_syy, J):
    if step % vis_output_freq == 0 or step == 1:
        vmag = np.sqrt(a**2 + b**2)
        
        # Calculate divergence, but get the 'clean' interior version for stats
        div_field, div_interior = divergence_2d_interior(a, b, dx, dy, pad=4)
        
        solid_area = np.sum(solid_mask) * dx * dy
        
        # Use div_interior for the log!
        print(
                f"[Step {step:05d}] dt={dt:.2e}, "
                f"max|v|={np.max(vmag):.3f}, "
                f"min(J)={np.min(J):.3f}, "
                f"mean(J)={np.mean(J):.3f}, "
                f"max|σ_solid|={np.max(np.abs(sigma_sxx)):.2f}, "
                f"max div (interior) = {np.max(np.abs(div_interior)):.2e}, " 
                f"solid area = {solid_area:.4f}"
            )
            
        output_dir = os.path.join("outputs", directory_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
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