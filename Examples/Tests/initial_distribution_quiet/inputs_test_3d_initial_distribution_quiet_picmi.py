#!/usr/bin/env python3
#
# Quiet-velocity-start unit test (PICMI).
#
# Exercises the new `warpx_velocity_quiet_start=True` kwarg on
# AnalyticDistribution, paired with GriddedLayout. The injector at the
# C++ level is `gaussian_parse_momentum_function_quiet`.
#
# At N_ppc = 2*2*2 = 8 (minimum cube, N_half = 2), the per-axis quantile
# lattice has two points at z = +/-Phi^{-1}(0.75). Tensor-product across the
# three velocity axes gives 8 antithetically-paired samples per cell whose
# per-axis mean equals the prescribed drift exactly.

from pywarpx import picmi

constants = picmi.constants

#################################
####### GENERAL PARAMETERS ######
#################################

max_steps = 1
max_grid_size = 8
nx = ny = nz = max_grid_size

xmin, xmax = -1.0, 1.0
ymin, ymax = -1.0, 1.0
zmin, zmax = -1.0, 1.0

# Drift values prescribed in normalized momentum (gamma*v/c).
ux_m = 1.0e-2
uy_m = 2.0e-2
uz_m = 3.0e-2
# Per-axis thermal spread, same units. Small enough that gamma~1.
u_th = 1.0e-3

#################################
############ NUMERICS ###########
#################################

cfl = 1.0e-8

#################################
############ PLASMA #############
#################################

# AnalyticDistribution with parser-driven drift and spread, opted in
# to the deterministic antithetic Gaussian lattice via
# warpx_velocity_quiet_start=True. The new C++ injector path is
# gaussian_parse_momentum_function_quiet.
electron_dist = picmi.AnalyticDistribution(
    density_expression="1.0e21",
    momentum_expressions=[f"{ux_m}", f"{uy_m}", f"{uz_m}"],
    warpx_momentum_spread_expressions=[f"{u_th}", f"{u_th}", f"{u_th}"],
    warpx_velocity_quiet_start=True,
)

electrons = picmi.Species(
    particle_type="electron",
    name="quiet_electrons",
    initial_distribution=electron_dist,
)

#################################
###### GRID AND SOLVER ##########
#################################

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    warpx_max_grid_size=max_grid_size,
    warpx_blocking_factor=max_grid_size,
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)
solver = picmi.ElectromagneticSolver(grid=grid, cfl=cfl)

#################################
######### DIAGNOSTICS ###########
#################################

# Per-species mean momentum is written to reducedfiles/pmom.txt.
pmom_diag = picmi.ReducedDiagnostic(
    diag_type="ParticleMomentum",
    name="pmom",
    period=1,
)

#################################
####### SIMULATION SETUP ########
#################################

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=1,
    warpx_use_filter=0,
    warpx_serialize_initial_conditions=1,
)

# GriddedLayout with 2*2*2 = 8 particles per cell. The quiet-velocity
# injector requires N_ppc to be a perfect cube; this is the minimum cube.
sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(
        grid=grid,
        n_macroparticle_per_cell=[2, 2, 2],
    ),
)

sim.add_diagnostic(pmom_diag)

sim.step(max_steps)
