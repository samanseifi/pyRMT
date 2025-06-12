import os
import numpy as np
import h5py
from .functions import divergence_2d

def output_simulation_data(dx, dy, phi, solid_mask, X1, X2, a, b, p, vis_output_freq, directory_name, step, dt, sigma_sxx, sigma_sxy, sigma_syy, J):
    if step % vis_output_freq == 0 or step == 1:
        vmag = np.sqrt(a**2 + b**2)
        div = divergence_2d(a, b, dx, dy)
        solid_area = np.sum(solid_mask) * dx * dy
        print(
                f"[Step {step:05d}] dt={dt:.2e}, "
                f"max|v|={np.max(vmag):.3f}, "
                f"min(J)={np.min(J):.3f}, "
                f"mean(J)={np.mean(J):.3f}, "
                f"max|σ_solid|={np.max(np.abs(sigma_sxx)):.2f}, "
                f"max divergence = {np.max(np.abs(div)):.2e}, "
                f"solid area = {solid_area:.4f}"
            )
            # create output directory if it doesn't exist
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
            f.create_dataset("div_vel", data=div)