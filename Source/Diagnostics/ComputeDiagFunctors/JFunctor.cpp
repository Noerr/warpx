/* This file is part of WarpX.
 *
 * Authors: Roelof Groenewald
 * License: BSD-3-Clause-LBNL
 */

#include "JFunctor.H"

#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX.H>
#include <AMReX_Extension.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

using warpx::fields::FieldType;

JFunctor::JFunctor (const int dir, int lev,
                   amrex::IntVect crse_ratio,
                   bool convertRZmodes2cartesian,
                   bool deposit_current,
                   int species_index, int ncomp)
    : ComputeDiagFunctor(ncomp, crse_ratio), m_dir(dir), m_lev(lev),
      m_convertRZmodes2cartesian(convertRZmodes2cartesian),
      m_deposit_current(deposit_current),
      m_species_index(species_index)
{ }

void
JFunctor::operator() (amrex::MultiFab& mf_dst, int dcomp, const int /*i_buffer*/) const
{
    using ablastr::fields::Direction;

    auto& warpx = WarpX::GetInstance();

    // Per-species path: redeposit just the requested species into fresh local
    // MultiFabs, then interpolate the requested direction into mf_dst. We must
    // NOT reuse warpx.m_fields current_fp slots here (that would overwrite the
    // live total current that physics needs).
    if (m_species_index >= 0) {
        amrex::MultiFab* live_jx = warpx.m_fields.get(FieldType::current_fp, Direction{0}, m_lev);
        amrex::MultiFab* live_jy = warpx.m_fields.get(FieldType::current_fp, Direction{1}, m_lev);
        amrex::MultiFab* live_jz = warpx.m_fields.get(FieldType::current_fp, Direction{2}, m_lev);

        amrex::MultiFab tmp_jx(live_jx->boxArray(), live_jx->DistributionMap(),
                               live_jx->nComp(), live_jx->nGrowVect());
        amrex::MultiFab tmp_jy(live_jy->boxArray(), live_jy->DistributionMap(),
                               live_jy->nComp(), live_jy->nGrowVect());
        amrex::MultiFab tmp_jz(live_jz->boxArray(), live_jz->DistributionMap(),
                               live_jz->nComp(), live_jz->nGrowVect());
        tmp_jx.setVal(0.0);
        tmp_jy.setVal(0.0);
        tmp_jz.setVal(0.0);

        ablastr::fields::MultiLevelVectorField jspec_temp {
            ablastr::fields::VectorField{ &tmp_jx, &tmp_jy, &tmp_jz }
        };

        auto& pc = warpx.GetPartContainer().GetParticleContainer(m_species_index);
        pc.DepositCurrent(jspec_temp, warpx.getdt(m_lev), 0.0);

        // Sum ghost-cell deposition contributions back into the owning neighbor's
        // interior cells. REQUIRED: without this, every cell along an MPI
        // partition seam shows a systematic deficit (because particles near the
        // boundary deposited weight into ghost cells that never got summed into
        // the neighbor rank). The total current_fp gets this on every physics
        // step via WarpX::SyncCurrent.
        //
        // Note: we deliberately do NOT apply WarpX::ApplyFilterMF here. The
        // bilinear filter is a numerical regularization for the field solve;
        // per-species J is a diagnostic of the kinetic first moment <q*v>(x)
        // and is more informative without the smoothing kernel folded in.
        // Consequence: in runs with warpx.use_filter = 1, the sum of per-species
        // J*_<species> will differ from the total J* by the filter residual.
        // This is by design.
        //
        // All of this runs only on diagnostic compute cycles (when this functor
        // is invoked from Diagnostics::FilterComputePackFlush), so no per-
        // physics-step overhead is incurred.
        const amrex::Periodicity& period = warpx.Geom(m_lev).periodicity();
        warpx.SumBoundaryJ(jspec_temp, m_lev, period);

        for (int idim = 0; idim < 3; ++idim) {
            jspec_temp[0][idim]->FillBoundary(period);
        }

        amrex::MultiFab* m_mf_src = jspec_temp[0][m_dir];
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_mf_src != nullptr, "m_mf_src can't be a nullptr.");
        AMREX_ASSUME(m_mf_src != nullptr);

        InterpolateMFForDiag(
            mf_dst, *m_mf_src, dcomp,
            warpx.DistributionMap(m_lev), m_convertRZmodes2cartesian);
        return;
    }

    /** pointer to source multifab (can be multi-component) */
    amrex::MultiFab* m_mf_src = warpx.m_fields.get(FieldType::current_fp,Direction{m_dir},m_lev);

    // Deposit current if no solver or the electrostatic solver is being used
    if (m_deposit_current)
    {
        // allocate temporary multifab to deposit current density into
        ablastr::fields::MultiLevelVectorField current_fp_temp {
            warpx.m_fields.get_alldirs(FieldType::current_fp, m_lev)
        };

        auto& mypc = warpx.GetPartContainer();
        mypc.DepositCurrent(current_fp_temp, warpx.getdt(m_lev), 0.0);

        // sum values in guard cells - note that this does not filter the
        // current density.
        for (int idim = 0; idim < 3; ++idim) {
            current_fp_temp[0][idim]->FillBoundary(warpx.Geom(m_lev).periodicity());
        }
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_mf_src != nullptr, "m_mf_src can't be a nullptr.");
    AMREX_ASSUME(m_mf_src != nullptr);

    InterpolateMFForDiag(
        mf_dst, *m_mf_src, dcomp,
        warpx.DistributionMap(m_lev), m_convertRZmodes2cartesian);
}
