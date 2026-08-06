/* Copyright 2019-2020 Neil Zaim, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ParticleUtils.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_Dim3.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_REAL.H>
#include <AMReX_SPACE.H>

namespace ParticleUtils
{

    using namespace amrex;

    // Define shortcuts for frequently-used type names
    using ParticleType = typename WarpXParticleContainer::ParticleType;
    using ParticleBins = DenseBins<ParticleTileDataType>;

    /* Find the particles and count the particles that are in each cell.
       Note that this does *not* rearrange particle arrays */
    amrex::DenseBins<ParticleTileDataType>
    findParticlesInEachCell (amrex::Geometry const& geom_lev,
                             amrex::MFIter const & mfi,
                             ParticleTileType & ptile) {

        // Extract particle structures for this tile
        int const np = ptile.numParticles();
        auto ptd = ptile.getParticleTileData();

        // Extract box properties
        Box const& cbx = mfi.tilebox(IntVect::TheZeroVector()); //Cell-centered box
        const auto lo = lbound(cbx);
        const auto dxi = geom_lev.InvCellSizeArray();
        const auto plo = geom_lev.ProbLoArray();

        // Find particles that are in each cell;
        // results are stored in the object `bins`.
        ParticleBins bins;
        bins.build(np, ptd, cbx,
            // Pass lambda function that returns the cell index
            [=] AMREX_GPU_DEVICE (ParticleType const & p) noexcept -> amrex::IntVect
            {
                return IntVect{AMREX_D_DECL(
                                   static_cast<int>((p.pos(0)-plo[0])*dxi[0] - lo.x),
                                   static_cast<int>((p.pos(1)-plo[1])*dxi[1] - lo.y),
                                   static_cast<int>((p.pos(2)-plo[2])*dxi[2] - lo.z))};
            });

        return bins;
    }

    DenseBins<ParticleTileDataType>
    findParticlesInEachCylSubCell (amrex::Geometry const& geom_lev,
                                   amrex::MFIter const & mfi,
                                   ParticleTileType & ptile,
                                   int const n_theta_max)
    {
        using namespace amrex::literals;

        int const np = ptile.numParticles();
        auto ptd = ptile.getParticleTileData();

        Box const& cbx = mfi.tilebox(IntVect::TheZeroVector()); //Cell-centered box
        const auto lo = lbound(cbx);
        const int nr = cbx.length(0);
        const int nz = cbx.length(1);
        const auto dxi = geom_lev.InvCellSizeArray();
        const auto plo = geom_lev.ProbLoArray();
        const amrex::Real dr = geom_lev.CellSize(0);
        const amrex::Real twopi = 2.0_rt*static_cast<amrex::Real>(MathConst::pi);

        // Over-allocate n_theta_max sectors per (r,z) cell; near-axis cells use fewer
        // sectors (see N_theta(r) below) and simply leave the extra bin slots empty.
        const int nbins = nr*nz*n_theta_max;
        ParticleBins bins;
        bins.build(np, ptd, nbins,
            // Lambda returns the flat sub-cell index for particle `i`. We use the
            // (tile-data, index) form so theta can be read from the SoA array
            // (m_rdata[PIdx::theta]) — the pure-SoA superparticle has no rdata().
            [=] AMREX_GPU_DEVICE (const ParticleTileDataType & ptd_in, const int i) noexcept -> int
            {
                int const ir = static_cast<int>((ptd_in.pos(0,i)-plo[0])*dxi[0]) - lo.x;
                int const iz = static_cast<int>((ptd_in.pos(1,i)-plo[1])*dxi[1]) - lo.y;
                int isec = 0;
#if defined(WARPX_DIM_RZ)
                // r-adaptive azimuthal sector count: 1 at the axis, ramping to n_theta_max.
                amrex::Real const rc =
                    plo[0] + (static_cast<amrex::Real>(ir + lo.x) + 0.5_rt)*dr;
                int const nth = amrex::min(n_theta_max,
                                  amrex::max(1, static_cast<int>(twopi*rc/dr)));
                amrex::Real th = ptd_in.m_rdata[PIdx::theta][i];
                th -= twopi*std::floor(th/twopi);              // wrap to [0, 2*pi)
                isec = static_cast<int>(th/twopi*static_cast<amrex::Real>(nth));
                if (isec >= nth) { isec = nth-1; }
                if (isec < 0)    { isec = 0; }
#else
                amrex::ignore_unused(dr, twopi, n_theta_max);
#endif
                return (ir*nz + iz)*n_theta_max + isec;
            });

        return bins;
    }

} // namespace ParticleUtils
