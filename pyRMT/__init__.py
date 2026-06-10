from pyRMT.functions import (
    create_grid,
    apply_phi_BCs,
    extrapolate_reference_map,
    compute_timestep,
    advect_semilagrangian_rk4,
    advect_weno5_rk3,
    advect_central2_rk3,
    advect_reference_map,
    solid_cauchy_stress,
    smoothed_heaviside,
    momentum_step_rk4,
    compute_curvature,
    velocity_rhs_blended_optimized,
    apply_velocity_BCs,
    build_poisson_matrix,
    pressure_projection_amg,
    _precompute_poisson_eigenvalues,
    _precompute_poisson_eigenvalues_periodic,
    rebuild_phi_from_reference_map,
    reinitialize_phi_PDE,
    reinitialize_phi_fmm,
    reinitialize_level_set,
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
    grad_central_x_4th,
    grad_central_y_4th,
    diff_upwind_3rd,
    lap_2nd,
    fast_solve_3x3,
)

# Deprecated aliases (old names kept for notebooks / external scripts)
from pyRMT.functions import (
    velocity_RK4,
    heaviside_smooth_alt,
    compute_solid_stress,
    extrapolate_transverse_layers_2field,
    advect_semi_lagrangian_rk4,
)
