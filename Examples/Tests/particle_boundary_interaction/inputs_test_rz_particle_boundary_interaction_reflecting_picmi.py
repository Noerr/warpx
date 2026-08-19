#!/usr/bin/env python3
# --- Input file for particle-boundary interaction testing in RZ.
# --- This input is a simple case of reflection
# --- of one electron on the surface of a sphere.
# ---
# --- This is the PICMI counterpart of
# --- inputs_test_rz_particle_boundary_interaction_reflecting: the specular
# --- reflection is performed by WarpX itself, via
# --- picmi.EmbeddedBoundary(particle_boundary_condition="Reflecting"),
# --- rather than being reimplemented in a Python callback (as is done in
# --- inputs_test_rz_particle_boundary_interaction_picmi.py).

from pywarpx import picmi

##########################
# numerics parameters
##########################

dt = 1.0e-11

# --- Nb time steps

max_steps = 23
diagnostic_interval = 1

# --- grid

nr = 64
nz = 64

rmin = 0.0
rmax = 2
zmin = -2
zmax = 2

##########################
# numerics components
##########################

grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    n_azimuthal_modes=1,
    lower_bound=[rmin, zmin],
    upper_bound=[rmax, zmax],
    lower_boundary_conditions=["none", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["none", "reflecting"],
    upper_boundary_conditions_particles=["absorbing", "reflecting"],
)


solver = picmi.ElectrostaticSolver(
    grid=grid, method="Multigrid", warpx_absolute_tolerance=1e-7
)

# Specularly reflect the particles when they reach the embedded boundary
# (instead of the default behavior, which is to absorb them).
embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function="-(x**2+y**2+z**2-radius**2)",
    radius=0.2,
    particle_boundary_condition="Reflecting",
)

##########################
# physics components
##########################

# one particle
e_dist = picmi.ParticleListDistribution(
    x=0.0, y=0.0, z=-0.25, ux=0.5e10, uy=0.0, uz=1.0e10, weight=1
)

electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    initial_distribution=e_dist,
    warpx_save_particles_at_eb=1,
)

##########################
# diagnostics
##########################

field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=diagnostic_interval,
    data_list=["Er", "Ez", "phi", "rho", "rho_electrons"],
    warpx_format="openpmd",
)

part_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=diagnostic_interval,
    species=[electrons],
    warpx_format="openpmd",
)

##########################
# simulation setup
##########################

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=max_steps,
    warpx_embedded_boundary=embedded_boundary,
)

sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[10, 1, 1], grid=grid),
)
sim.add_diagnostic(part_diag)
sim.add_diagnostic(field_diag)

##########################
# simulation run
##########################

sim.step(max_steps)
