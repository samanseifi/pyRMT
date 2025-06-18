# pyRMT
pyRMT is a Python package for simulating Fluid-Structure Interaction (FSI) in 2D using the Reference Map Technique (RMT). It includes a set of benchmark problems to validate the accuracy of its methods. The main features of the formulation are:

- Discretization on a collocated grid
- Semi-Lagrangian advection of the reference map with RK4 integration
- Time integration of the velocity field using RK4
- 4th order central difference schemes for advective and diffusive terms
- Extrapolation-based extension of solid regions into the fluid for stress computation
- Level set reconstruction via the advected reference map
- Mild PDE-based reinitialization of the level set
- Multigrid solver for the Poisson pressure equation
- Nearly-incompressible formulation of Neo-Hookean solid



## Examples

### 1) Lid-driven Fluid flow
Verification of the fluid solver with lid-driven fluid flow:
| <img src="vids/lid_driven_re_1000.png" alt="Lid Driven Cavity Simulation Re=1000" width="200"/> | <img src="vids/lid_driven.png" alt="Lid Driven Cavity Simulation" width="200"/> |
|:---------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------:|
| Lid Driven Cavity (Re=1000)                                                                   | Lid Driven Cavity                                                               |

### 2) Soft Disc in Lid driven flow
| <img src="vids/centroid.png" alt="Soft Disc in Lid Driven Cavity Simulation" width="250"/> | <img src="vids/lid_driven_256x256_new_2.gif" alt="Soft Disc in Lid Driven Cavity Simulation" width="250"/> |
|:------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| Centroid Trajectory                                                                         | Simulation Animation                                                                            |

## License

This project is licensed under the MIT License.
