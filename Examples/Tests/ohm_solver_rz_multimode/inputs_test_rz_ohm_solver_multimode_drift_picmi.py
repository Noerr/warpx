#!/usr/bin/env python3
#
# --- Uniform, magnetized plasma drifting across the field in RZ geometry with
# --- two azimuthal modes and the hybrid-PIC (Ohm's law) solver.
#
# --- Cold ions drift with u = v0 x_hat in B = B0 z_hat. With J = curl(B)/mu0 = 0
# --- and a uniform electron pressure, Ohm's law gives E = -v_i x B = v0 B0 y_hat,
# --- under which the ions are force free: the state is stationary and every
# --- field is uniform. The transverse E is a pure m = 1 field that is finite on
# --- the axis (E_r = v0 B0 sin(theta), E_theta = v0 B0 cos(theta)), so the test
# --- checks the multimode Ohm's law, including its on-axis treatment, against
# --- an exact answer, and checks that B stays uniform (curl E = 0).

import argparse
import sys

import numpy as np

from pywarpx import picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument(
    "--legacy_hall_ordering",
    help="divide the averaged Hall numerator by the averaged rho (hall_divide_at_nodes = 0)",
    action="store_true",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

# Plasma: protons, n0, B0; drift speed v0 (cold ions)
n0 = 1.0e20  # m^-3
B0 = 1.0  # T
v0 = 1.0e4  # m/s
Te = 10.0  # eV, uniform: no pressure gradient

M = constants.m_p
w_pi = np.sqrt(constants.q_e**2 * n0 / (M * constants.ep0))
d_i = constants.c / w_pi  # ion skin depth
w_ci = constants.q_e * B0 / M

Nr = 32
Nz = 32
Lr = 4.0 * d_i
Lz = 4.0 * d_i
dt = 0.01 / w_ci
max_steps = 4

simulation = picmi.Simulation(
    time_step_size=dt,
    max_steps=max_steps,
    verbose=0,
    particle_shape=1,
    warpx_current_deposition_algo="direct",
    warpx_serialize_initial_conditions=True,
)

grid = picmi.CylindricalGrid(
    number_of_cells=[Nr, Nz],
    n_azimuthal_modes=2,
    lower_bound=[0.0, 0.0],
    upper_bound=[Lr, Lz],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["reflecting", "periodic"],
    warpx_max_grid_size=16,
)

simulation.solver = picmi.HybridPICSolver(
    grid=grid,
    Te=Te,
    n0=n0,
    gamma=1.0,
    n_floor=0.01 * n0,
    plasma_resistivity=0.0,
    substeps=40,
)
simulation.add_applied_field(picmi.AnalyticInitialField(Bz_expression=B0))

ions = picmi.Species(
    name="ions",
    charge="q_e",
    mass=M,
    initial_distribution=picmi.UniformDistribution(
        density=n0, directed_velocity=[v0, 0.0, 0.0]
    ),
)
# r, theta, z
simulation.add_species(
    ions, layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[4, 8, 2])
)

simulation.add_diagnostic(
    picmi.FieldDiagnostic(
        name="diag1",
        grid=grid,
        period=max_steps,
        data_list=["B", "E", "J", "rho"],
    )
)
simulation.add_diagnostic(
    picmi.FieldDiagnostic(
        name="openpmd",
        grid=grid,
        period=max_steps,
        data_list=["B", "E"],
        warpx_format="openpmd",
    )
)

simulation.initialize_inputs()
if args.legacy_hall_ordering:
    from pywarpx import hybridpicmodel

    hybridpicmodel.hall_divide_at_nodes = 0
simulation.initialize_warpx()

# Start from the stationary E (otherwise the first ion push sees E = 0).
# Mode 1 of a uniform E_y: Im(E_r,1) = Re(E_theta,1) = E_y.
Ey = v0 * B0
Er = simulation.fields.get("Efield_fp", dir="r", level=0)
Et = simulation.fields.get("Efield_fp", dir="theta", level=0)
a = Er[...]
a[..., 2] = Ey
Er[...] = a
a = Et[...]
a[..., 1] = Ey
Et[...] = a

simulation.step(max_steps)
