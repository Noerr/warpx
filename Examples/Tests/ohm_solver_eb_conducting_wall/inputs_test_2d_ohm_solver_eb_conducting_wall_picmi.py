#!/usr/bin/env python3
#
# --- Hybrid-PIC (Ohm's law) test of an embedded boundary acting as a
# --- perfectly conducting wall.
#
# A line current along +y, offset by a from the axis of a conducting
# cylinder of radius R (2D XZ, so the problem is exact in the x-z plane),
# is ramped up from zero with the hybrid solver's external current. A cold,
# heavy, uniform ion background fills the vacuum region so that Ohm's law is
# defined everywhere; the ions barely move. With no displacement current the
# field settles by resistive diffusion to the vacuum solution: the wire plus
# an image current -I at R^2/a, which has B.n = 0 on the wall.
#
# In addition, a nonuniform E field is loaded inside the conductor at t = 0
# (Ey = E0 x / L for r > R + 2 dx, out of reach of the ions' field gather).
# The solver never updates E at EB-masked points, so this E persists there.
# B at EB-masked points must not be pushed by it: with m_eb_update_B honored
# in EvolveB, B inside the conductor stays exactly zero; if the B push ignored
# the mask, curl(E) inside the conductor would grow B there. On the default staggered grid this is
# what makes the test sensitive to the mask: with E = 0 inside the conductor,
# every E value used by a masked B point is itself masked (and zero), so the
# B mask would make no difference.
#
# analysis.py checks that the embedded boundary behaves as a conductor:
# no B inside it (while E there is nonzero), and the field in the vacuum
# matches the wire + image solution once the (stair-case) wall position is
# fitted.

import json
import math
import os

import numpy as np

from pywarpx import picmi

constants = picmi.constants
MU0 = constants.mu0

# ---------------------------------------------------------------- geometry
R = 2.0e-2  # conducting wall radius [m]
a = 0.8e-2  # wire offset from the axis [m]
w = 1.5e-3  # Gaussian wire radius [m]
half = 1.125 * R  # domain half-width
n = 80  # cells per direction (R/dx = 35.6)
dx = 2.0 * half / n

# ---------------------------------------------------------------- plasma / drive
I0 = 1.0  # final wire current [A]
eta = 1.0e-4  # resistivity [Ohm m]; resistive diffusion dominates
n0 = 1.0e22  # background density [m^-3]
Te = 0.01  # electron temperature [eV]
m_i = 100.0 * constants.m_p  # heavy ions: they barely move
substeps = 25
E0 = 100.0  # [V/m] scale of the E field loaded inside the conductor

# ---------------------------------------------------------------- timescales
lam_max = (eta / MU0) * 4.0 * 2 / dx**2  # max eigenvalue of (eta/mu0) laplacian
dt_sub = 0.5 * 2.785 / lam_max  # half of the RK4 real-axis stability limit
dt = 2 * substeps * dt_sub
tau1 = MU0 * R**2 / (2.4048**2 * eta)  # slowest resistive decay time
nramp = int(math.ceil(10.0 * tau1 / dt))
nhold = int(math.ceil(5.0 * tau1 / dt))
nsteps = int(math.ceil((nramp + nhold) / 10.0)) * 10
tau_ramp = nramp * dt

# ---------------------------------------------------------------- grid
grid = picmi.Cartesian2DGrid(
    number_of_cells=[n, n],
    lower_bound=[-half, -half],
    upper_bound=[half, half],
    lower_boundary_conditions=["dirichlet", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["reflecting", "reflecting"],
    upper_boundary_conditions_particles=["reflecting", "reflecting"],
    warpx_max_grid_size=40,
    warpx_blocking_factor=8,
)

# ---------------------------------------------------------------- driven current
J0 = I0 / (math.pi * w**2)
jy_expr = (
    f"{J0:.17g}*exp(-((x-({a:.17g}))**2+z**2)/{w**2:.17g})"
    f"*if(t<{tau_ramp:.17g},sin({0.5 * math.pi / tau_ramp:.17g}*t)**2,1.0)"
)

solver = picmi.HybridPICSolver(
    grid=grid,
    Te=Te,
    n0=n0,
    gamma=1.0,
    n_floor=0.01 * n0,
    plasma_resistivity=eta,
    substeps=substeps,
    Jx_external_function="0.0",
    Jy_external_function=jy_expr,
    Jz_external_function="0.0",
)

sim = picmi.Simulation(
    solver=solver,
    max_steps=nsteps,
    time_step_size=dt,
    particle_shape=1,
    verbose=1,
)
sim.current_deposition_algo = "direct"
sim.embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function=f"sqrt(x*x + z*z) - ({R:.17g})"
)


# ---------------------------------------------------------------- E inside the conductor
def load_conductor_E():
    """Ey = E0 x / half inside the conductor, zero elsewhere.

    It starts 2 cells inside the wall so that ions (at r < R, shape-1 gather)
    never see it; only the B update at EB-masked points can.
    """
    Ey = sim.fields.get("Efield_fp_external", dir="y", level=0)
    XM, ZM = np.meshgrid(Ey.mesh("x"), Ey.mesh("z"), indexing="ij")
    Ey[:, :] = E0 * XM / half * (np.hypot(XM, ZM) > R + 2 * dx)


sim.add_applied_field(
    picmi.LoadInitialFieldFromPython(
        load_from_python=load_conductor_E, load_E=True, load_B=False
    )
)

# ---------------------------------------------------------------- background ions
ions = picmi.Species(
    name="ions",
    charge="q_e",
    mass=m_i,
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=f"{n0:.17g}*((x*x+z*z)<{R**2:.17g})",
        momentum_expressions=["0.", "0.", "0."],
    ),
)
sim.add_species(
    ions, layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[2, 2])
)

# ---------------------------------------------------------------- diagnostics
sim.add_diagnostic(
    picmi.FieldDiagnostic(
        name="diag1",
        grid=grid,
        period=nsteps,
        data_list=["B", "E", "J"],
        write_dir="diags",
        warpx_format="openpmd",
        warpx_openpmd_backend="h5",
    )
)

# parameters for analysis.py (written once, by rank 0). The rank is read from
# the launcher's environment so that mpi4py is not needed.
rank = 0
for key in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "SLURM_PROCID"):
    if key in os.environ:
        rank = int(os.environ[key])
        break
if rank == 0:
    with open("params.json", "w") as f:
        json.dump(
            dict(
                R=R,
                a=a,
                w=w,
                dx=dx,
                I0=I0,
                dt=dt,
                nsteps=nsteps,
                tau_ramp=tau_ramp,
                tau1=tau1,
                E0=E0,
                half=half,
            ),
            f,
            indent=2,
        )

sim.step(nsteps)
