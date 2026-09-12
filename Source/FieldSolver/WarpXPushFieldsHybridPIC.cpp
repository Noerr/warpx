/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *          S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Fluids/MultiFluidContainer.H"
#include "Fluids/WarpXFluidContainer.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/Communication.H>


using namespace amrex;

void WarpX::HybridPICEvolveFields ()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    ABLASTR_PROFILE("WarpX::HybridPICEvolveFields()");

    // The below deposition is hard coded for a single level simulation
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        finest_level == 0,
        "Ohm's law E-solve only works with a single level.");

    // Get requested number of substeps to use
    const int sub_steps = m_hybrid_pic_model->m_substeps;

    // Get flag to include external fields.
    const bool add_external_fields = m_hybrid_pic_model->m_add_external_fields;

    // Handle field splitting for Hybrid field push
    if (add_external_fields) {
        // Get the external fields
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            gett_old(0),
            0.5_rt*dt[0]);

        // If using split fields, subtract the external field at the old time
        for (int lev = 0; lev <= finest_level; ++lev) {
            for (int idim = 0; idim < 3; ++idim) {
                MultiFab::Subtract(
                    *m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev),
                    *m_fields.get(FieldType::hybrid_B_fp_external, Direction{idim}, lev),
                    0, 0, 1,
                    m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev)->nGrowVect());
            }
        }
    }

    // The particles have now been pushed to their t_{n+1} positions.
    // Perform charge deposition at t_{n+1} and current deposition at t_{n+1/2}.
    HybridPICDepositRhoAndJ();

    // Get the external current
    m_hybrid_pic_model->GetCurrentExternal();

    // Reference hybrid-PIC multifabs
    ablastr::fields::MultiLevelScalarField rho_fp_temp = m_fields.get_mr_levels(FieldType::hybrid_rho_fp_temp, finest_level);
    ablastr::fields::MultiLevelVectorField current_fp_temp = m_fields.get_mr_levels_alldirs(FieldType::hybrid_current_fp_temp, finest_level);

    // During the above deposition the charge and current density were updated
    // so that, at this time, we have rho^{n} in rho_fp_temp, rho{n+1} in the
    // 0'th index of `rho_fp`, J_i^{n-1/2} in `current_fp_temp` and J_i^{n+1/2}
    // in `current_fp`.

    // Note: E^{n} is recalculated with the accurate J_i^{n} since at the end
    // of the last step we had to "guess" it. It also needs to be
    // recalculated to include the resistivity before evolving B.

    // J_i^{n} is calculated as the average of J_i^{n-1/2} and J_i^{n+1/2}.
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        for (int idim = 0; idim < 3; ++idim) {
            // Perform a linear combination of values (with 0.5 prefactors) of
            // J_i^{n-1/2} and J_i^{n+1/2}, writing the result into
            // `current_fp_temp[lev][idim]`. All components are averaged so that
            // every azimuthal mode (m=0 and m>=1 in multi-mode RZ) is time-centered.
            MultiFab::LinComb(
                *current_fp_temp[lev][idim],
                0.5_rt, *current_fp_temp[lev][idim], 0,
                0.5_rt, *m_fields.get(FieldType::current_fp, Direction{idim}, lev), 0,
                0, current_fp_temp[lev][idim]->nComp(), current_fp_temp[lev][idim]->nGrowVect()
            );
        }
    }

    // Push the B field from t=n to t=n+1/2 using the current and density
    // at t=n, while updating the E field along with B using the electron
    // momentum equation
    const int sub_steps_per_half = sub_steps / 2;
    for (int sub_step = 0; sub_step < sub_steps_per_half; sub_step++)
    {
        m_hybrid_pic_model->BfieldEvolveRK(
            m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level),
            m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level),
            current_fp_temp, rho_fp_temp,
            m_eb_update_E,
            dt[0]/sub_steps,
            SubcyclingHalf::FirstHalf, guard_cells.ng_FieldSolver,
            WarpX::sync_nodal_points
        );
    }

    // Average rho^{n} and rho^{n+1} to get rho^{n+1/2} in rho_fp_temp
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // Perform a linear combination of values (with 0.5 prefactors) of
        // rho^{n} and rho^{n+1}, writing the result into `rho_fp_temp[lev]`.
        // All components are averaged so that every azimuthal mode is time-centered.
        MultiFab::LinComb(
            *rho_fp_temp[lev], 0.5_rt, *rho_fp_temp[lev], 0,
            0.5_rt, *m_fields.get(FieldType::rho_fp, lev), 0, 0, rho_fp_temp[lev]->nComp(), rho_fp_temp[lev]->nGrowVect()
        );
    }

    if (add_external_fields) {
        // Get the external fields at E^{n+1/2}
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            gett_old(0) + 0.5_rt*dt[0],
            0.5_rt*dt[0]);
    }

    // Now push the B field from t=n+1/2 to t=n+1 using the n+1/2 quantities
    for (int sub_step = 0; sub_step < sub_steps_per_half; sub_step++)
    {
        m_hybrid_pic_model->BfieldEvolveRK(
            m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level),
            m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level),
            m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level),
            rho_fp_temp,
            m_eb_update_E,
            dt[0]/sub_steps,
            SubcyclingHalf::SecondHalf, guard_cells.ng_FieldSolver,
            WarpX::sync_nodal_points
        );
    }

    // Extrapolate the ion current density to t=n+1 using
    // J_i^{n+1} = 1/2 * J_i^{n-1/2} + 3/2 * J_i^{n+1/2}, and recalling that
    // now current_fp_temp = J_i^{n} = 1/2 * (J_i^{n-1/2} + J_i^{n+1/2})
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        for (int idim = 0; idim < 3; ++idim) {
            // Perform a linear combination of values (with -1.0 and 2.0 prefactors)
            // of J_i^{n-1/2} and J_i^{n+1/2}, writing the result into
            // `current_fp_temp[lev][idim]`. All components are extrapolated so that
            // every azimuthal mode (m=0 and m>=1 in multi-mode RZ) is handled.
            MultiFab::LinComb(
                *current_fp_temp[lev][idim],
                -1._rt, *current_fp_temp[lev][idim], 0,
                2._rt, *m_fields.get(FieldType::current_fp, Direction{idim}, lev), 0,
                0, current_fp_temp[lev][idim]->nComp(), current_fp_temp[lev][idim]->nGrowVect()
            );
        }
    }

    if (add_external_fields) {
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            gett_new(0),
            0.5_rt*dt[0]);
    }

    // Calculate the electron pressure at t=n+1
    m_hybrid_pic_model->CalculateElectronPressure();

    // Update the E field to t=n+1 using the extrapolated J_i^n+1 value
    m_hybrid_pic_model->CalculatePlasmaCurrent(
        m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level),
        m_eb_update_E);
    m_hybrid_pic_model->HybridPICSolveE(
        m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level),
        current_fp_temp,
        m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level),
        m_fields.get_mr_levels(FieldType::rho_fp, finest_level),
        m_eb_update_E, false);
    FillBoundaryE(guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);

    // Handle field splitting for Hybrid field push
    if (add_external_fields) {
        // If using split fields, add the external field at the new time
        for (int lev = 0; lev <= finest_level; ++lev) {
            for (int idim = 0; idim < 3; ++idim) {
                MultiFab::Add(
                    *m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev),
                    *m_fields.get(FieldType::hybrid_B_fp_external, Direction{idim}, lev),
                    0, 0, 1,
                    m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev)->nGrowVect());
                MultiFab::Add(
                    *m_fields.get(FieldType::Efield_fp, Direction{idim}, lev),
                    *m_fields.get(FieldType::hybrid_E_fp_external, Direction{idim}, lev),
                    0, 0, 1,
                    m_fields.get(FieldType::Efield_fp, Direction{idim}, lev)->nGrowVect());
            }
        }
    }

    // Copy the rho^{n+1} values to rho_fp_temp and the J_i^{n+1/2} values to
    // current_fp_temp since at the next step those values will be needed as
    // rho^{n} and J_i^{n-1/2}.
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // copy all components (every azimuthal mode) starting at index 0 to index 0
        MultiFab::Copy(*rho_fp_temp[lev], *m_fields.get(FieldType::rho_fp, lev),
                        0, 0, rho_fp_temp[lev]->nComp(), rho_fp_temp[lev]->nGrowVect());
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Copy(*current_fp_temp[lev][idim], *m_fields.get(FieldType::current_fp, Direction{idim}, lev),
                           0, 0, current_fp_temp[lev][idim]->nComp(), current_fp_temp[lev][idim]->nGrowVect());
        }
    }

    // Check that the E-field does not have nan or inf values, otherwise print a clear message
    ablastr::fields::MultiLevelVectorField Efield_fp = m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level);
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        for (int idim = 0; idim < 3; ++idim) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                Efield_fp[lev][idim]->is_finite(),
                "Non-finite value detected in E-field; this indicates more substeps should be used in the field solver."
            );
        }
    }
}

void WarpX::HybridPICDepositRhoAndJ ()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Perform charge deposition in component 0 of rho_fp at current time.
    mypc->DepositCharge(m_fields.get_mr_levels(FieldType::rho_fp, finest_level), 0._rt);
    // Perform current deposition at t_{n-1/2}.
    mypc->DepositCurrent(m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level), dt[0], -0.5_rt * dt[0]);

    // TODO: Perhaps add flag here for when using temperature accumulation in Hybrid
    // Perform Temperature Deposition at time t_{n}
    mypc->DepositTemperatures(m_fields, 0.0_rt);

    // Deposit cold-relativistic fluid charge and current
    if (do_fluid_species) {
        int const lev = 0;
        myfl->DepositCharge(m_fields, *m_fields.get(FieldType::rho_fp, lev), lev);
        myfl->DepositCurrent(m_fields,
            *m_fields.get(FieldType::current_fp, Direction{0}, lev),
            *m_fields.get(FieldType::current_fp, Direction{1}, lev),
            *m_fields.get(FieldType::current_fp, Direction{2}, lev),
            lev);
    }

    // Synchronize J and rho:
    // filter (if used), exchange guard cells, interpolate across MR levels
    // and apply boundary conditions
    SyncCurrentAndRho();

    // On-axis modal BC for the charge density. A scalar density's m>=1 azimuthal
    // components must vanish on the axis (rho_m ~ r^|m| as r->0), but the RZ
    // deposition leaves a spurious value there (volume-amplified by the 1/(pi dr)
    // axis factor). Left uncleaned it contaminates the E_r/E_theta Hall term --
    // which interpolates nodal rho onto its off-axis stagger -- and the rho
    // diagnostic. ApplyRhofieldBoundary only handles the wall/PEC boundaries, so
    // zero the m>=1 modes of rho_fp at i=0 here. rho_fp_temp (used by the E-solve)
    // is copied/averaged from rho_fp, so this keeps every downstream rho correct.
    // Mode 0 (component 0) is physical on axis and left untouched.
#if defined(WARPX_DIM_RZ)
    for (int lev = 0; lev <= finest_level; ++lev) {
        amrex::MultiFab& rho = *m_fields.get(FieldType::rho_fp, lev);
        const int ncr = rho.nComp();
        if (ncr <= 1) { continue; }   // single mode: nothing to zero
        for (amrex::MFIter mfi(rho, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box tb = mfi.tilebox();
            if (tb.smallEnd(0) > 0) { continue; }   // only tiles touching the axis
            amrex::Array4<amrex::Real> const& rho_arr = rho.array(mfi);
            tb.setRange(0, 0, 1);                    // i = 0 (r = 0) column only
            amrex::ParallelFor(tb, ncr - 1,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                rho_arr(i, j, k, n + 1) = 0._rt;     // zero components 1..ncr-1 (m>=1)
            });
        }
    }
#endif

    // On-axis modal regularity for the deposited ion current J_i. J is a vector
    // like E, so this mirrors the E-field axis treatment:
    //   - J_z is scalar-like (J_{z,m} ~ r^|m|): its m>=1 modes must vanish at the
    //     axis, but the RZ deposition divides them by the tiny axis volume
    //     (~1/(pi dr)), amplifying deposition noise ~6x. Zero them (as for rho).
    //   - J_theta is a transverse vector: its m=1 mode is FINITE at the axis with
    //     J_{theta,1} = -i J_{r,1} (odd-parity regularity), not zero. The deposition
    //     forces all m>=1 transverse modes to zero on axis (laser-class); restore
    //     J_{theta,1} from J_r.
    //   - J_r is cell-centered in r (innermost sample at r=dr/2, no node at r=0); it
    //     has no on-axis pathology and is left untouched, supplying J_{r,1}(0) by
    //     linear extrapolation from its two innermost cells.
    // J_theta and J_z are nodal in r (node at i=0); J_r is cell-centered. These
    // relations are linear, so applying them once to current_fp propagates through
    // the linear time-averaging/extrapolation into current_fp_temp.
#if defined(WARPX_DIM_RZ)
    for (int lev = 0; lev <= finest_level; ++lev) {
        amrex::MultiFab& Jr = *m_fields.get(FieldType::current_fp, Direction{0}, lev);
        amrex::MultiFab& Jt = *m_fields.get(FieldType::current_fp, Direction{1}, lev);
        amrex::MultiFab& Jz = *m_fields.get(FieldType::current_fp, Direction{2}, lev);
        const int ncj = Jz.nComp();
        if (ncj <= 1) { continue; }   // single mode: nothing to do
        for (amrex::MFIter mfi(Jt, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box tb = mfi.tilebox();
            if (tb.smallEnd(0) > 0) { continue; }   // only tiles touching the axis
            amrex::Array4<amrex::Real> const& Jr_arr = Jr.array(mfi);
            amrex::Array4<amrex::Real> const& Jt_arr = Jt.array(mfi);
            amrex::Array4<amrex::Real> const& Jz_arr = Jz.array(mfi);
            tb.setRange(0, 0, 1);                    // i = 0 (r = 0) column only
            amrex::ParallelFor(tb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                for (int n = 1; n < ncj; ++n) { Jz_arr(i, j, k, n) = 0._rt; }
                if (ncj >= 3) {
                    const amrex::Real Jr1_re = 1.5_rt*Jr_arr(i,j,k,1) - 0.5_rt*Jr_arr(i+1,j,k,1);
                    const amrex::Real Jr1_im = 1.5_rt*Jr_arr(i,j,k,2) - 0.5_rt*Jr_arr(i+1,j,k,2);
                    Jt_arr(i, j, k, 1) =  Jr1_im;   // Re(J_theta,1) =  Im(J_r,1)
                    Jt_arr(i, j, k, 2) = -Jr1_re;   // Im(J_theta,1) = -Re(J_r,1)
                }
            });
        }
    }
#endif

    // SyncCurrent does not include a call to FillBoundary, but it is needed
    // for the hybrid-PIC solver since current values are interpolated to
    // a nodal grid
    for (int lev = 0; lev <= finest_level; ++lev) {
        ablastr::utils::communication::FillBoundary(
            *m_fields.get(FieldType::rho_fp, lev),
            m_fields.get(FieldType::rho_fp, lev)->nGrowVect(),
            WarpX::do_single_precision_comms,
            Geom(lev).periodicity(),
            true
        );
        for (int idim = 0; idim < 3; ++idim) {
            ablastr::utils::communication::FillBoundary(
                *m_fields.get(FieldType::current_fp, Direction{idim}, lev),
                m_fields.get(FieldType::current_fp, Direction{idim}, lev)->nGrowVect(),
                WarpX::do_single_precision_comms,
                Geom(lev).periodicity(),
                true
            );
        }
    }

#if defined(WARPX_DIM_RZ)
    // PROTOTYPE near-axis noise filter for the scalar-like m>=1 moments (rho, Jz).
    // Their deposition noise is peaked near the axis (tracks n0(r)) while the
    // physical m=1 signal ~ r vanishes there (S/N ~ r^{3/2} -> 0), so a sqrt(r)
    // variance-stabilized radial smooth (x sqrt(r) -> 3-pt binomial -> / sqrt(r))
    // suppresses near-axis noise with minimal signal loss. Parity-aware: the m>=1
    // axis node stays 0 (skipped). Applied only to m>=1 components (m=0 equilibrium
    // untouched). Gated by env HYBRID_AXIS_FILTER_PASSES (0 = off) for prototyping.
    static const int filter_passes = [](){
        const char* e = std::getenv("HYBRID_AXIS_FILTER_PASSES");
        return e ? std::atoi(e) : 0;
    }();
    if (filter_passes > 0) {
        const amrex::Real dr = Geom(0).CellSize(0);
        for (int lev = 0; lev <= finest_level; ++lev) {
            for (int t = 0; t < 2; ++t) {
                amrex::MultiFab* q = (t == 0)
                    ? m_fields.get(FieldType::rho_fp, lev)
                    : m_fields.get(FieldType::current_fp, Direction{2}, lev);
                const int nc = q->nComp();
                if (nc <= 1) { continue; }
                for (int pass = 0; pass < filter_passes; ++pass) {
                    amrex::MultiFab qcopy(q->boxArray(), q->DistributionMap(), nc, q->nGrowVect());
                    amrex::MultiFab::Copy(qcopy, *q, 0, 0, nc, q->nGrowVect());
                    for (amrex::MFIter mfi(*q, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                        amrex::Box tb = mfi.tilebox();
                        amrex::Array4<amrex::Real> const& qa = q->array(mfi);
                        amrex::Array4<amrex::Real const> const& qc = qcopy.const_array(mfi);
                        const int ilo = tb.smallEnd(0);
                        const int ihi = tb.bigEnd(0);
                        amrex::ParallelFor(tb, nc - 1,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                            const int c = n + 1;                 // m>=1 components only
                            if (i <= ilo || i >= ihi) { return; } // skip axis node & wall edge
                            const amrex::Real wi  = std::sqrt(amrex::max<amrex::Real>(i*dr,       0.25_rt*dr));
                            const amrex::Real wim = std::sqrt(amrex::max<amrex::Real>((i-1)*dr,   0.25_rt*dr));
                            const amrex::Real wip = std::sqrt(amrex::max<amrex::Real>((i+1)*dr,   0.25_rt*dr));
                            const amrex::Real sm = 0.25_rt*qc(i-1,j,k,c)*wim
                                                 + 0.5_rt *qc(i  ,j,k,c)*wi
                                                 + 0.25_rt*qc(i+1,j,k,c)*wip;
                            qa(i,j,k,c) = sm / wi;
                        });
                    }
                }
            }
        }
        for (int lev = 0; lev <= finest_level; ++lev) {
            ablastr::utils::communication::FillBoundary(
                *m_fields.get(FieldType::rho_fp, lev),
                m_fields.get(FieldType::rho_fp, lev)->nGrowVect(),
                WarpX::do_single_precision_comms, Geom(lev).periodicity(), true);
            ablastr::utils::communication::FillBoundary(
                *m_fields.get(FieldType::current_fp, Direction{2}, lev),
                m_fields.get(FieldType::current_fp, Direction{2}, lev)->nGrowVect(),
                WarpX::do_single_precision_comms, Geom(lev).periodicity(), true);
        }
    }
#endif
}

void WarpX::HybridPICInitializeRhoJandB ()
{
    // The Ohm's law solver requires two timesteps' values for the charge
    // and current densities. This function is called at the start of
    // the PIC loop (before particles have been pushed for the first time,
    // but after their positions and velocities have been de-synchronized).

    using warpx::fields::FieldType;
    using ablastr::fields::Direction;

    if (restart_chkfile.empty()) {
        // This is not a restart, so the rho_fp and current_fp multifabs are
        // still empty.
        HybridPICDepositRhoAndJ();

        // Handle field splitting for Hybrid field push
        if (m_hybrid_pic_model->m_add_external_fields) {
            // Get the external fields
            // Currently t_new is what t_old will be when entering the solver since
            // after initialization the t_old is set to t_new, then t_new is incremented by dt
            m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
                gett_new(0),
                0.5_rt*dt[0]);

            // If using split fields, add the external field at t=0
            for (int lev = 0; lev <= finest_level; ++lev) {
                for (int idim = 0; idim < 3; ++idim) {
                    // Check to make sure field only contains numeric values
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        m_fields.get(FieldType::hybrid_B_fp_external, Direction{idim}, lev)->is_finite(),
                        "Non-finite value detected in external B-field at t=0."
                    );

                    MultiFab::Add(
                        *m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev),
                        *m_fields.get(FieldType::hybrid_B_fp_external, Direction{idim}, lev),
                        0, 0, 1,
                        m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev)->nGrowVect());
                }
            }
        }
    }

    // Copy the rho_fp values to rho_fp_temp and the current_fp values to
    // current_fp_temp, since the "temp" multifabs are meant to store the
    // particle and current densities from the previous step during the field
    // solve routine and are needed when the first field solve is
    // performed after pushing the particles.
    ablastr::fields::MultiLevelScalarField rho_fp_temp = m_fields.get_mr_levels(FieldType::hybrid_rho_fp_temp, finest_level);
    ablastr::fields::MultiLevelVectorField current_fp_temp = m_fields.get_mr_levels_alldirs(FieldType::hybrid_current_fp_temp, finest_level);
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // copy all components (every azimuthal mode) starting at index 0 to index 0
        MultiFab::Copy(*rho_fp_temp[lev], *m_fields.get(FieldType::rho_fp, lev),
                        0, 0, rho_fp_temp[lev]->nComp(), rho_fp_temp[lev]->nGrowVect());
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Copy(*current_fp_temp[lev][idim], *m_fields.get(FieldType::current_fp, Direction{idim}, lev),
                        0, 0, current_fp_temp[lev][idim]->nComp(), current_fp_temp[lev][idim]->nGrowVect());
        }
    }
}

void
WarpX::CalculateExternalCurlA() {
    ABLASTR_PROFILE("WarpX::CalculateExternalCurlA()");

    auto & warpx = WarpX::GetInstance();

    // Get reference to External Field Object
    auto* ext_vector = warpx.m_hybrid_pic_model->m_external_vector_potential.get();
    ext_vector->CalculateExternalCurlA();

}
