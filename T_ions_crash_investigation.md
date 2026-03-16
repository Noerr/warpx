# WarpX T_ions Diagnostic Crash Investigation

## Bug Summary

A reproducible crash occurs during diagnostic output steps in WarpX simulations of shear flow Z-pinch configurations with embedded boundaries (EB). The crash affects both AMD and NVIDIA GPUs.

## Three Necessary Conditions

All three conditions must be present simultaneously to trigger the crash:

1. **Reflecting embedded boundary** (`eb_particle_boundary = Reflecting`)
   - Absorbing boundaries do not trigger the crash
   - Reflecting keeps particles alive near the EB surface; absorbing removes them

2. **T_ions diagnostic enabled** (ion temperature, two-pass deposition)
   - Disabling T_ions allows simulations to run to completion (hundreds of thousands of steps, hundreds of successful diagnostic steps)
   - Other diagnostics (rho, current density) work without issue — these are single-pass depositions

3. **High shear flow along the embedded boundary surface**
   - The test case simulates a shear flow Z-pinch with a cylindrical EB
   - Only the highest shear flow velocity triggers the crash
   - A more moderate shear level ran to completion
   - This means particles with high tangential velocity streaming along the EB surface are a trigger

## Architecture of the Two-Pass T_ions Deposition

The T_ions diagnostic uses a two-pass (DOUBLE_PASS) variance deposition algorithm (Bell 1979, WV1) implemented in:
- `Source/Particles/PhysicalParticleContainer.cpp` (lines ~1910-2005)
- `Source/Particles/Deposition/TemperatureDeposition.H`

### Pass 1 — Accumulate means
- WarpXParIter loop over tiles (multi-stream, up to 8 GPU streams)
- Each tile launches a `ParallelFor` kernel that atomically accumulates:
  - `nx_arr` (sample counts)
  - `wx_arr` (weight sums)
  - `vxbar_arr` (weighted velocity sums)
- `Gpu::streamSynchronize()` after each tile (line 511 of TemperatureDeposition.H)
- MFIter destructor syncs all streams
- Followed by `SumBoundary` to communicate guard cells across MPI ranks
- Followed by explicit `Gpu::streamSynchronize()`

### Pass 2 — Accumulate demeaned variance
- Clears `w2` arrays, syncs
- New WarpXParIter loop over tiles
- Each tile launches a `ParallelFor` kernel that:
  - **READS** `vxbar_arr`, `wx_arr`, `nx_arr` from Pass 1 to compute local mean velocity
  - Computes demeaned velocity: `vd = v - vbar`
  - Atomically accumulates `w2x_arr += w * vd * vd`
- This **read-back from accumulation arrays** is the fundamental difference from single-pass diagnostics (rho, J), which only do atomic writes

### Key code location (Pass 2 read, TemperatureDeposition.H lines 102-113):
```cpp
if (type == TemperatureDepositionType::DOUBLE_PASS)
{
    if (nx_arr(ixv[0], ixv[1], ixv[2], 0) > 0) {
        vxb = vxbar_arr(ixv[0], ixv[1], ixv[2], 0) / wx_arr(ixv[0], ixv[1], ixv[2], 0);
    }
    // ... same for y, z
}
```

## Leading Hypothesis

High tangential velocity particles reflecting off the cylindrical EB undergo grazing-angle reflections (bisection algorithm in `ParticleBoundaryProcess::Reflect`). After reflection:
- Particles remain alive and very close to the EB surface
- Large tangential displacement per timestep can move them far from their original cell
- Their positions may end up **outside the valid index range of the accumulation arrays**

Single-pass diagnostics (rho, J) only do atomic **writes** — an out-of-bounds access might hit a guard cell or silently corrupt. But T_ions Pass 2 does **reads** from accumulation arrays at the particle's grid position to retrieve the local mean velocity. An out-of-bounds read into unmapped GPU memory would crash.

This explains why all three conditions are necessary:
- **Reflect** keeps these particles alive (Absorb would remove them)
- **T_ions** is the only diagnostic that reads back from grid arrays during particle-loop deposition
- **High shear** produces the extreme tangential velocities needed for particles to end up in out-of-bounds positions

## Prior Analysis: Multi-Stream Race Condition (Debunked)

An initial hypothesis that GPU stream interleaving between `ScrapeParticlesAtEB`, `gatherParticlesFromEmbeddedBoundaries`, and `deleteInvalidParticles` could cause a race condition was investigated and **ruled out**. The AMReX `MFIter` destructor (`MFIter::Finalize()` in `AMReX_MFIter.cpp:245-256`) synchronizes all GPU streams when the particle iterator goes out of scope, and WarpX never enters a no-sync region.

The two-pass T_ions deposition also has explicit `Gpu::streamSynchronize()` calls between passes (lines 1955, 1969, 1979, 2003 of PhysicalParticleContainer.cpp), so inter-pass synchronization appears correct.

## Testing Plan — Perlmutter (NVIDIA)

### Step 1: compute-sanitizer memcheck

Run the failing case under NVIDIA's memory checker to identify the exact faulting kernel and memory address:

```bash
compute-sanitizer --tool memcheck --show-backtrace yes \
    srun -n 1 ./warpx.rz <inputs_file>
```

- Use **single rank** to keep output clean
- This requires no recompilation
- Will catch: illegal memory access, out-of-bounds, use-after-free

**What to look for in the output:**
- Which kernel name appears (look for `doVarianceDepositionShapeN` or `varianceDepositionSubKernel`)
- Whether the faulting access is a **read** (supports the hypothesis) or **write**
- The faulting memory address — wildly out of range suggests corruption; slightly off suggests off-by-one or array bounds issue
- Whether the fault is in Pass 1 or Pass 2

### Step 2: compute-sanitizer racecheck (if memcheck is clean)

```bash
compute-sanitizer --tool racecheck --show-backtrace yes \
    srun -n 1 ./warpx.rz <inputs_file>
```

- Detects data races and hazards between kernels on shared memory
- Only needed if memcheck doesn't find anything (unlikely for a crash)

### Step 3: cuda-gdb (for deeper inspection)

If sanitizer results point to a specific location, use the debugger to inspect state:
- Set breakpoints in `doVarianceDepositionShapeN`
- Inspect particle positions relative to EB surface and grid bounds
- Check accumulation array index ranges vs. actual array dimensions

### Optional Quick Validation Test

Change line 1916 of `PhysicalParticleContainer.cpp` from:
```cpp
auto depos_type = TemperatureDepositionType::DOUBLE_PASS;
```
to:
```cpp
auto depos_type = TemperatureDepositionType::SINGLE_PASS;
```

If the crash disappears with SINGLE_PASS, this confirms the issue is specific to Pass 2's read-back from accumulation arrays, not a general deposition problem.

## Key Source Files

| File | Relevance |
|------|-----------|
| `Source/Particles/Deposition/TemperatureDeposition.H` | Two-pass deposition kernels, per-tile streamSync (line 511) |
| `Source/Particles/PhysicalParticleContainer.cpp:1910-2005` | Two-pass orchestration, inter-pass sync |
| `Source/Particles/Deposition/VarianceAccumulationBuffer.cpp` | Accumulation buffer management, reset, normalization |
| `Source/EmbeddedBoundary/ParticleBoundaryProcess.H:70-190` | Reflect functor (bisection + specular reflection) |
| `Source/EmbeddedBoundary/ParticleScraper.H:168-218` | Scraper kernel that calls Reflect |
| `Source/Evolve/WarpXEvolve.cpp:689-696` | EB scrape → gather → delete pipeline |
| `build/_deps/fetchedamrex-src/Src/Base/AMReX_MFIter.cpp:225-275` | MFIter destructor stream synchronization |
