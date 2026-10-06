#!/usr/bin/env python3
#
# --- Analysis for the hybrid-PIC embedded-boundary conducting-wall test.
#
# At the last output (current on its flat top) this checks that the
# embedded boundary acts as a perfect conductor:
#   1. no B inside the conductor, although a nonuniform E was loaded there
#      at t = 0 and persists (so B at EB-masked points must not be pushed),
#   2. no out-of-plane field (the problem is exactly 2D),
#   3. the field in the vacuum matches the analytic wire + image solution
#      once the wall position is fitted (the stair-case wall sits a fraction
#      of a cell into the vacuum), and the fitted offset is ~one cell,
#   4. the force on the wire matches the image force for the fitted wall.
#
# Usage: analysis.py [diag_dir] [params.json]

import json
import os
import sys

import numpy as np
import openpmd_api as io

MU0 = 1.25663706212e-6

diag_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("diags", "diag1")
param_file = sys.argv[2] if len(sys.argv) > 2 else "params.json"
with open(param_file) as f:
    p = json.load(f)
R, a, w, dx, I0 = p["R"], p["a"], p["w"], p["dx"], p["I0"]


# ------------------------------------------------------------------ load fields
def load(diag_dir, record):
    """x, z (1D), and the x, y, z components of `record` on (x, z), plus time."""
    series = io.Series(os.path.join(diag_dir, "openpmd_%T.h5"), io.Access.read_only)
    it_id = sorted(series.iterations)[-1]
    it = series.iterations[it_id]
    B = it.meshes[record]
    labels = list(B.axis_labels)
    spacing = np.array(B.grid_spacing) * B.grid_unit_SI
    offset = np.array(B.grid_global_offset) * B.grid_unit_SI
    if B.data_order == "F":
        labels, spacing, offset = labels[::-1], spacing[::-1], offset[::-1]
    out = {}
    for c in ("x", "y", "z"):
        rc = B[c]
        arr = rc.load_chunk()
        series.flush()
        arr = np.asarray(arr, dtype=float) * rc.unit_SI
        pos = np.array(rc.position)
        if B.data_order == "F":
            pos = pos[::-1]
        coords = {
            lab: offset[k] + (np.arange(arr.shape[k]) + pos[k]) * spacing[k]
            for k, lab in enumerate(labels)
        }
        order = [labels.index(lab) for lab in ("x", "z")]
        out[c] = np.transpose(arr, order)  # (x, z)
    t = it.time * it.time_unit_SI
    series.close()
    return coords["x"], coords["z"], out["x"], out["y"], out["z"], t


x, z, Bx, By, Bz, t = load(diag_dir, "B")
_, _, Ex, Ey, Ez, _ = load(diag_dir, "E")
X, Z = np.meshgrid(x, z, indexing="ij")
r = np.hypot(X, Z)
rwire = np.hypot(X - a, Z)


# ------------------------------------------------------------------ analytic
def analytic_B(Rw, current):
    """Gaussian wire at (a, 0) plus point image -current at (Rw^2/a, 0): (Bx, Bz)."""
    bx = np.zeros_like(X)
    bz = np.zeros_like(X)
    for x0, s, gauss in ((a, 1.0, True), (Rw**2 / a, -1.0, False)):
        rx, rz = X - x0, Z
        r2 = rx**2 + rz**2
        with np.errstate(divide="ignore", invalid="ignore"):
            coef = MU0 * s * current / (2 * np.pi * r2)
            if gauss:
                coef = coef * (-np.expm1(-r2 / w**2))
        coef = np.where(r2 > 0, coef, 0.0)
        bx += coef * rz
        bz += -coef * rx
    return bx, bz


cur = I0  # the analysis runs on the flat top (t > tau_ramp)
assert t > p["tau_ramp"], f"last output t={t} is before the end of the ramp"

vac = (R - r > 3 * dx) & (rwire > 3 * w)  # vacuum, away from the wall and the wire core
deep = r > R + 3 * dx  # well inside the conductor
Bmag = np.hypot(Bx, Bz)
bx0, bz0 = analytic_B(R, cur)
Bref_max = np.hypot(bx0, bz0)[vac].max()


def rel_err(Rw):
    bx, bz = analytic_B(Rw, cur)
    err2 = (Bx - bx) ** 2 + (Bz - bz) ** 2
    return float(np.sqrt(err2[vac].mean() / (bx**2 + bz**2)[vac].mean()))


err_true = rel_err(R)
shifts = np.linspace(-0.5, 2.0, 101)  # in cells, > 0 = wall moved into the vacuum
errs = [rel_err(R - s * dx) for s in shifts]
k = int(np.argmin(errs))
shift, err_fit = float(shifts[k]), float(errs[k])

# force per unit length on the wire, same quadrature for simulation and reference
J = cur / (np.pi * w**2) * np.exp(-(rwire**2) / w**2)
core = rwire < 6 * w
dA = (x[1] - x[0]) * (z[1] - z[0])
F_sim = np.array([(J * Bz)[core].sum(), -(J * Bx)[core].sum()]) * dA
bxf, bzf = analytic_B(R - shift * dx, cur)
F_ref = np.array([(J * bzf)[core].sum(), -(J * bxf)[core].sum()]) * dA
force_ratio = float(F_sim[0] / F_ref[0])  # the image force is along -x

conductor_B = float(Bmag[deep].max() / Bref_max) if deep.any() else 0.0
conductor_E = float(np.abs(Ey)[deep].max()) if deep.any() else 0.0
By_ratio = float(np.abs(By).max() / Bref_max)

print(f"t = {t:.4e} s (tau_ramp = {p['tau_ramp']:.4e} s), R/dx = {R / dx:.1f}")
print(
    f"max|Ey| in conductor [V/m]         {conductor_E:.3e}  (loaded: E0 = {p.get('E0', 0.0):g})"
)
print(f"max|B| in conductor / max|B_ref|   {conductor_B:.3e}")
print(f"max|By| / max|B_ref|               {By_ratio:.3e}")
print(f"field error vs true wall           {err_true:.4f}")
print(f"fitted wall shift [cells]          {shift:.3f}")
print(f"field error at fitted wall         {err_fit:.4f}")
print(f"force / image force (fitted wall)  {force_ratio:.5f}")

if "E0" in p:
    # the loaded E must still be there, or check 1 would not test the B mask
    assert conductor_E > 0.1 * p["E0"], "E loaded inside the conductor did not persist"
assert conductor_B < 1e-10, "B inside the conductor: B was pushed at EB-masked points"
assert By_ratio < 1e-5, "out-of-plane field in a 2D wire problem"
assert 0.3 < shift < 1.5, "effective wall is not within about one cell of the EB"
assert err_fit < 0.02, "field does not match the wire + image solution"
assert abs(force_ratio - 1.0) < 0.01, "force on the wire does not match the image force"
assert err_true < 0.05, "field error against the true wall position too large"
print("PASS")
