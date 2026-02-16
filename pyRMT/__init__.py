from pyRMT.functions import (
    create_grid,
    apply_phi_BCs,
    extrapolate_transverse_layers_2field,
    compute_timestep,
    advect_semi_lagrangian_rk4,
    compute_solid_stress,
    heaviside_smooth_alt,
    velocity_RK4,
    compute_curvature,
    velocity_rhs_blended_optimized,
    apply_velocity_BCs,
    build_poisson_matrix,
    pressure_projection_amg,
    _precompute_poisson_eigenvalues,
    rebuild_phi_from_reference_map,
    reinitialize_phi_PDE,
)

from pyRMT.output import (
    compute_kinetic_energy,
    compute_strain_energy,
    compute_viscous_dissipation,
    output_simulation_data,
)

from pyRMT.interpolators import (
    bilinear_interpolate,
    bicubic_interpolate,
)

from pyRMT.utils import (
    grad_central_x_2nd,
    grad_central_y_2nd,
    diff_upwind_3rd,
    lap_2nd,
    fast_solve_3x3,
)
