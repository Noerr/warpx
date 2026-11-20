#!/usr/bin/env python3
#
# Analysis of the RZ multimode hybrid-PIC drift equilibrium: the fields must stay
# uniform, with E = v0 B0 y_hat, i.e. in the stored azimuthal modes
#   E_r:     m=0 = 0, Re(m=1) = 0,       Im(m=1) = v0 B0
#   E_theta: m=0 = 0, Re(m=1) = v0 B0,   Im(m=1) = 0
#   E_z = 0
#   B_z:     m=0 = B0, and every other B mode = 0.
# This holds everywhere, including on the axis, where the m=1 transverse field
# is finite.
#
# The reflecting, conducting wall is not compatible with a uniform drift across
# it; the disturbance it causes is excluded by not checking the outer 6 radial
# cells. The remaining E error (a few percent, largest next to the axis) comes
# from the linear-shape deposition of the uniform m=1 current in the first
# radial cell, which is 13/12 of the exact value.

import numpy as np
import openpmd_api as io

v0 = 1.0e4
B0 = 1.0
Ey = v0 * B0
n_wall = 6

series = io.Series("diags/openpmd/openpmd_%T.bp5", io.Access.read_only)
it = series.iterations[max(series.iterations)]


def load(record, comp):
    rc = it.meshes[record][comp]
    data = rc.load_chunk()
    series.flush()
    # (mode component, z, r), without the cells next to the wall
    return data[:, :, :-n_wall] * rc.unit_SI


Er = load("E", "r")
Et = load("E", "t")
Ez = load("E", "z")
Br = load("B", "r")
Bt = load("B", "t")
Bz = load("B", "z")
assert Er.shape[0] == 3, "expected 2 azimuthal modes"

# (relative error, tolerance)
checks = {
    "E_r: Im(m=1) - v0 B0": (np.abs(Er[2] - Ey).max() / Ey, 5e-2),
    "E_r: Re(m=1)": (np.abs(Er[1]).max() / Ey, 5e-2),
    "E_theta: Re(m=1) - v0 B0": (np.abs(Et[1] - Ey).max() / Ey, 5e-2),
    "E_theta: Im(m=1)": (np.abs(Et[2]).max() / Ey, 5e-2),
    "E_r, E_theta: m=0": (max(np.abs(Er[0]).max(), np.abs(Et[0]).max()) / Ey, 2e-2),
    "E_z": (np.abs(Ez).max() / Ey, 1e-10),
    "B_r, B_theta": (max(np.abs(Br).max(), np.abs(Bt).max()) / B0, 1e-10),
    "B_z: m=0 - B0": (np.abs(Bz[0] - B0).max() / B0, 1e-6),
    "B_z: m=1": (np.abs(Bz[1:]).max() / B0, 1e-4),
}
failed = False
for key, (err, tol) in checks.items():
    ok = err < tol
    failed = failed or not ok
    print(f"{key:26s} rel. error {err:.3e}  (tolerance {tol:.0e})  {'ok' if ok else 'FAILED'}")

assert not failed
