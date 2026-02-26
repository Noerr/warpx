# Pressure Tensor Diagnostic — Design Document

## Motivation

WarpX provides a scalar temperature diagnostic (`T_ions`, `T_electrons`) that
computes `T = m <(u - <u>)^2> / (3 q_e)` per cell.  This is the **trace** of
the temperature tensor, averaged over all three velocity components.  For
HybridPIC simulations with anisotropic particle distributions (e.g., near
embedded boundaries, shocks, or magnetic mirrors), the scalar temperature
obscures critical physics.  A full pressure/temperature tensor diagnostic would
expose:

- **Temperature anisotropy** (T_perp vs T_parallel)
- **Off-diagonal stress** (shear heating, velocity-space instabilities)
- **Pressure balance verification** across EB surfaces

## What Exists Today

| Diagnostic | Computes | Source |
|-----------|----------|--------|
| `T_species` | Scalar temperature in eV: `m <(u-<u>)^2> / (3 q_e)` | `TemperatureFunctor` → `DepositTotalNGPTemperature` |
| `particle_fields_to_plot` | User-defined `f(x,y,z,ux,uy,uz)` averaged per cell | `ParticleReductionFunctor` |

The `particle_fields_to_plot` mechanism could approximate individual tensor
components via `ux*uy` etc., but:
- It computes `<u_i u_j>`, not `<(u_i - <u_i>)(u_j - <u_j>)>` — the mean-flow
  contribution is not subtracted, requiring post-processing.
- Each component requires a separate `ParticleToMesh` pass; 6 components means
  6 independent depositions plus the weight sum, none sharing work.
- `ux`, `uy`, `uz` in the parser are normalized to `gamma*v/c`, which adds a
  unit conversion step.

## Proposed Diagnostic

### User Interface

**PICMI Python** (in `data_list`):
```python
data_list = ['B', 'E', 'rho', 'T_ions', 'P_ions']
```

**Classic WarpX input** (in `fields_to_plot`):
```
diag1.fields_to_plot = B E rho T_ions P_ions
```

`P_ions` produces **6 output fields** (symmetric tensor):
```
Pxx_ions  Pxy_ions  Pxz_ions  Pyy_ions  Pyz_ions  Pzz_ions
```

### Output Quantity

Each component is the **temperature tensor** in eV (intensive, per-particle
average, consistent with the existing scalar `T`):

```
P_ij = m * <(u_i - <u_i>)(u_j - <u_j>)> / q_e     [eV]
```

where `u = gamma * v` (the quantity stored in `PIdx::ux/uy/uz`).

**Consistency check**: `(Pxx + Pyy + Pzz) / 3 = T` (the existing scalar
temperature diagnostic).

To recover the physical pressure tensor in Pa, the user multiplies by the number
density: `P_ij [Pa] = n * q_e * P_ij [eV]`.  The density `n` is available from
the `rho` diagnostic (`n = rho / q`).

### Why eV and not Pa?

The existing `T_ions` outputs in eV (an intensive quantity).  Matching this
convention means:
- Direct comparison: trace/3 == T
- No cell-volume dependence (cell volume would be needed for Pa)
- Consistent with what PIC users expect

## Algorithm

Follows the same **two-pass NGP deposition** as `DepositTotalNGPTemperature`,
extended from 1 component (trace) to 6 (symmetric tensor).

### Pass 1 — Mean Velocity

Deposit into a 4-component cell-centered MultiFab `sum_mf`:
```
comp 0: sum(w)          — total weight per cell
comp 1: sum(w * ux)     — weighted ux
comp 2: sum(w * uy)     — weighted uy
comp 3: sum(w * uz)     — weighted uz
```

Then normalize: `<u_i> = sum(w * u_i) / sum(w)`.

(Identical to the existing temperature code, lines 1922–1958 of
`WarpXParticleContainer.cpp`.)

### Pass 2 — Tensor Components

For each particle, compute deviations `du_i = u_i - <u_i>` using the cell's
mean velocity from Pass 1.  Deposit into a 6-component MultiFab `ptensor_mf`:
```
comp 0: sum(w * du_x * du_x)     — xx
comp 1: sum(w * du_x * du_y)     — xy
comp 2: sum(w * du_x * du_z)     — xz
comp 3: sum(w * du_y * du_y)     — yy
comp 4: sum(w * du_y * du_z)     — yz
comp 5: sum(w * du_z * du_z)     — zz
```

### Finalize

Divide each component by `sum(w)` and multiply by `mass / q_e`:
```
P_ij = (mass / q_e) * sum(w * du_i * du_j) / sum(w)
```

### Performance Notes

- **Two particle loops** over the species (same as scalar T) — not six.
- Pass 2 deposits 6 components per particle in a single loop using 6 atomic adds.
- The deposition is **NGP** (Nearest Grid Point = cell-centered), consistent
  with the existing temperature diagnostic.  No shape-function stencil.
- Only computed at diagnostic output steps (controlled by `period`), not every
  timestep.

## Implementation Plan

### New Files

| File | Purpose |
|------|---------|
| `Source/Diagnostics/ComputeDiagFunctors/PressureTensorFunctor.H` | Header for the functor class |
| `Source/Diagnostics/ComputeDiagFunctors/PressureTensorFunctor.cpp` | Functor `operator()` — calls into `WarpXParticleContainer` |

### Modified Files

| File | Changes |
|------|---------|
| `Source/Particles/WarpXParticleContainer.H` | Declare `DepositNGPPressureTensor()` and `GetAverageNGPPressureTensor()` |
| `Source/Particles/WarpXParticleContainer.cpp` | Implement the two-pass deposition (modeled on `DepositTotalNGPTemperature`) |
| `Source/Diagnostics/Diagnostics.H` | Add `m_P_per_species_index` vector |
| `Source/Diagnostics/Diagnostics.cpp` | Parse `P_<species>` names, resolve species index |
| `Source/Diagnostics/FullDiagnostics.cpp` | Register `PressureTensorFunctor` for `P_` prefix, emit 6 output names |
| `Source/Diagnostics/CMakeLists.txt` | Add new `.cpp` to build |

### Step-by-Step

1. **`WarpXParticleContainer`**: Add `DepositNGPPressureTensor(MultiFab*, int lev)`
   - Allocate 4-comp `sum_mf` + 6-comp output MultiFab
   - Pass 1: deposit `w`, `w*ux`, `w*uy`, `w*uz`; normalize
   - Pass 2: deposit `w*du_i*du_j` for 6 components
   - Finalize: `*= mass / (sum_w * q_e)`
   - Add wrapper `GetAverageNGPPressureTensor(lev)` returning a
     `unique_ptr<MultiFab>` with 6 components

2. **`PressureTensorFunctor`**: Modeled on `TemperatureFunctor`
   - Constructor: `(int lev, IntVect crse_ratio, int ispec, int ncomp=6)`
   - `operator()`: call `pc.GetAverageNGPPressureTensor(m_lev)`, then
     `Coarsen(mf_dst, *ptensor, dcomp, 0, 6, 0, m_crse_ratio)`

3. **`Diagnostics.cpp`**: Add parsing block for `P_<species>` (copy the `T_`
   block, use `m_P_per_species_index`)

4. **`FullDiagnostics.cpp`**: Add `else if (rfind("P_", 0) == 0)` block
   - Create `PressureTensorFunctor` with `ncomp=6`
   - Push 6 output names: `Pxx_species`, `Pxy_species`, ..., `Pzz_species`

5. **PICMI**: No Python changes needed — `P_ions` passes through `data_list`
   to the C++ `fields_to_plot` parameter as-is.

## Output Format

With `P_ions` requested, the diagnostic file contains 6 fields:

| Field | Tensor element | Units |
|-------|---------------|-------|
| `Pxx_ions` | P_{xx} | eV |
| `Pxy_ions` | P_{xy} | eV |
| `Pxz_ions` | P_{xz} | eV |
| `Pyy_ions` | P_{yy} | eV |
| `Pyz_ions` | P_{yz} | eV |
| `Pzz_ions` | P_{zz} | eV |

The tensor is symmetric: `P_{xy} = P_{yx}`, so only 6 of 9 components are
stored.

## Verification

1. **Isotropic Maxwellian**: All diagonal components should equal `T_ions`;
   off-diagonal should be ~0 (within statistical noise).
2. **Trace consistency**: `(Pxx + Pyy + Pzz) / 3` should match the existing
   `T_ions` diagnostic exactly (same algorithm, same NGP deposition).
3. **Bi-Maxwellian**: Set up a distribution with `T_perp != T_parallel` and
   verify the diagonal components reflect the anisotropy.

## Future Extensions

- **Per-component scalar temperature** (`Tx_ions`, `Ty_ions`, `Tz_ions`):
  trivially extractable from the diagonal of the tensor, but could be added
  as convenience aliases.
- **Relativistic correction**: The current formula uses `u = gamma*v`.  For
  relativistic species, the kinetic energy is not `(1/2) m u^2`, so the
  "temperature" interpretation breaks down.  A relativistic pressure tensor
  would require `v = u/gamma` in the deposition.  This is the same limitation
  as the existing scalar `T` diagnostic and can be addressed later.
