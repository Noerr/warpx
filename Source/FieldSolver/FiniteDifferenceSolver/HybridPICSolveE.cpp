/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *          S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FiniteDifferenceSolver.H"

#include "EmbeddedBoundary/Enabled.H"
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
#   include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#elif defined(WARPX_DIM_RSPHERE)
#   include "FiniteDifferenceAlgorithms/SphericalYeeAlgorithm.H"
#else
#   include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#   include "FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H"
#endif
#include "HybridPICModel/HybridPICModel.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <cmath>

using namespace amrex;
using warpx::fields::FieldType;

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
namespace {

// Compile-time upper bound on n_rz_azimuthal_modes for the pseudo-spectral
// kernel below. Determines the size of stack-allocated theta-grid buffers
// inside the per-cell ParallelFor lambdas. Increase if you need more modes.
constexpr int MAX_PSEUDO_SPECTRAL_NMODES = 8;
constexpr int MAX_PSEUDO_SPECTRAL_NTHETA = 2 * MAX_PSEUDO_SPECTRAL_NMODES - 1;

// Inverse azimuthal Fourier transform from modal storage to theta-grid values.
// Convention matches WarpX RZ deposition/gather:
//   A(theta) = A_0 + sum_{m>=1} [Re(A_m) cos(m*theta) + Im(A_m) sin(m*theta)]
// (factor-of-2 absorbed into the stored amplitudes for m >= 1).
//
// `A_modes` has (2*nmodes - 1) entries: [A_0, Re(A_1), Im(A_1), Re(A_2), ...].
// `A_theta` is filled with `n_theta` real-space values at theta_k = 2*pi*k/n_theta.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void inverse_az_ft (int nmodes, int n_theta,
                    amrex::Real const* AMREX_RESTRICT A_modes,
                    amrex::Real* AMREX_RESTRICT A_theta)
{
    for (int k = 0; k < n_theta; ++k) {
        const amrex::Real theta_k = (2._rt * MathConst::pi * k) / n_theta;
        amrex::Real val = A_modes[0];
        for (int m = 1; m < nmodes; ++m) {
            const amrex::Real c = std::cos(m * theta_k);
            const amrex::Real s = std::sin(m * theta_k);
            val += A_modes[2*m - 1] * c + A_modes[2*m] * s;
        }
        A_theta[k] = val;
    }
}

// Forward azimuthal Fourier transform from theta-grid values to modal storage.
//   A_0    = (1/n_theta) sum_k A(theta_k)
//   A_m_re = (2/n_theta) sum_k A(theta_k) cos(m*theta_k)   for m >= 1
//   A_m_im = (2/n_theta) sum_k A(theta_k) sin(m*theta_k)   for m >= 1
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void forward_az_ft (int nmodes, int n_theta,
                    amrex::Real const* AMREX_RESTRICT A_theta,
                    amrex::Real* AMREX_RESTRICT A_modes)
{
    amrex::Real a0 = 0._rt;
    for (int k = 0; k < n_theta; ++k) { a0 += A_theta[k]; }
    A_modes[0] = a0 / n_theta;
    for (int m = 1; m < nmodes; ++m) {
        amrex::Real ar = 0._rt, ai = 0._rt;
        for (int k = 0; k < n_theta; ++k) {
            const amrex::Real theta_k = (2._rt * MathConst::pi * k) / n_theta;
            ar += A_theta[k] * std::cos(m * theta_k);
            ai += A_theta[k] * std::sin(m * theta_k);
        }
        A_modes[2*m - 1] = (2._rt * ar) / n_theta;
        A_modes[2*m    ] = (2._rt * ai) / n_theta;
    }
}

// Enforce on-axis modal boundary conditions for a scalar field (rho, P_e).
// Smoothness in (x, y) requires modes m>=1 to vanish as r^m near the axis,
// so on the axis itself they are zero.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void apply_axis_bc_scalar (int nmodes, amrex::Real* AMREX_RESTRICT A_modes)
{
    for (int m = 1; m < nmodes; ++m) {
        A_modes[2*m - 1] = 0._rt;
        A_modes[2*m    ] = 0._rt;
    }
}

// Enforce on-axis modal BCs for the (r, theta) components of a vector field.
// V_r,0(0) = V_theta,0(0) = 0; V_z behaves as a scalar (handled separately).
// For m == 1, regularity demands Re(V_r)= -Im(V_theta), Im(V_r)= +Re(V_theta);
// the *finite* m=1 amplitude must be read from the cell adjacent to the axis
// (linear-in-r extrapolation), but for a first cut we zero m>=1 here and
// rely on the surrounding kernels (Faraday, Ampere) to recover the m=1
// cross-coupling from neighboring cells. This is consistent with how the
// existing m=0 kernel treats on-axis (J, B, E) tangential components.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void apply_axis_bc_vector_rt (int nmodes,
                              amrex::Real* AMREX_RESTRICT Vr_modes,
                              amrex::Real* AMREX_RESTRICT Vt_modes)
{
    Vr_modes[0] = 0._rt;
    Vt_modes[0] = 0._rt;
    for (int m = 1; m < nmodes; ++m) {
        Vr_modes[2*m - 1] = 0._rt;
        Vr_modes[2*m    ] = 0._rt;
        Vt_modes[2*m - 1] = 0._rt;
        Vt_modes[2*m    ] = 0._rt;
    }
}

} // anonymous namespace
#endif // defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)

void FiniteDifferenceSolver::CalculateCurrentAmpere (
    ablastr::fields::VectorField & Jfield,
    ablastr::fields::VectorField const& Bfield,
    [[maybe_unused]]std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev )
{
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
        CalculateCurrentAmpereCylindrical <CylindricalYeeAlgorithm> (
            Jfield, Bfield, eb_update_E, lev
        );

#elif defined(WARPX_DIM_RSPHERE)
        CalculateCurrentAmpereSpherical <SphericalYeeAlgorithm> (
            Jfield, Bfield, lev
        );

#else
    if (WarpX::GetInstance().grid_type == GridType::Staggered)
    {
        CalculateCurrentAmpereCartesian <CartesianYeeAlgorithm> (
            Jfield, Bfield, eb_update_E, lev
        );
    } else {
        CalculateCurrentAmpereCartesian <CartesianNodalAlgorithm> (
            Jfield, Bfield, eb_update_E, lev
        );
    }

#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "CalculateCurrentAmpere: Unknown algorithm choice."));
    }
}

// /**
//   * \brief Calculate total current from Ampere's law without displacement
//   * current i.e. J = 1/mu_0 curl x B.
//   *
//   * \param[out] Jfield  vector of total current MultiFabs at a given level
//   * \param[in] Bfield   vector of magnetic field MultiFabs at a given level
//   * \param[in] eb_update_E specifies where the plasma current should be calculated.
//   * \param[in] lev refinement level
//   */
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
template<typename T_Algo>
void FiniteDifferenceSolver::CalculateCurrentAmpereCylindrical (
    ablastr::fields::VectorField& Jfield,
    ablastr::fields::VectorField const& Bfield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev
)
{
    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Jfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const& Jr = Jfield[0]->array(mfi);
        Array4<Real> const& Jtheta = Jfield[1]->array(mfi);
        Array4<Real> const& Jz = Jfield[2]->array(mfi);
        Array4<Real> const& Br = Bfield[0]->array(mfi);
        Array4<Real> const& Btheta = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries.
        // The plasma current is stored at the same locations as the E-field,
        // therefore the `eb_update_E` multifab also appropriately specifies
        // where the plasma current should be calculated.
        amrex::Array4<int> update_Jr_arr, update_Jtheta_arr, update_Jz_arr;
        if (EB::enabled()) {
            update_Jr_arr = eb_update_E[0]->array(mfi);
            update_Jtheta_arr = eb_update_E[1]->array(mfi);
            update_Jz_arr = eb_update_E[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        int const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract cylindrical specific parameters
        Real const dr = m_dr;
        int const nmodes = m_nmodes;
        Real const rmin = m_rmin;

        // Extract tileboxes for which to loop
        Box const& tjr  = mfi.tilebox(Jfield[0]->ixType().toIntVect());
        Box const& tjtheta  = mfi.tilebox(Jfield[1]->ixType().toIntVect());
        Box const& tjz  = mfi.tilebox(Jfield[2]->ixType().toIntVect());

        Real const one_over_mu0 = 1._rt / PhysConst::mu0;

        // Calculate the total current, using Ampere's law, on the same grid
        // as the E-field
        amrex::ParallelFor(tjr, tjtheta, tjz,

            // Jr calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Jr_arr && update_Jr_arr(i, j, 0) == 0) { return; }

                // Mode m=0
                Jr(i, j, 0, 0) = one_over_mu0 * (
                    - T_Algo::DownwardDz(Btheta, coefs_z, n_coefs_z, i, j, 0, 0)
                );

                // Higher-order modes
                // r on cell-centered point (Jr is cell-centered in r)
                Real const r = rmin + (i + 0.5_rt)*dr;
                for (int m=1; m<nmodes; m++) {
                    Jr(i, j, 0, 2*m-1) = one_over_mu0 * (
                        - T_Algo::DownwardDz(Btheta, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                        + m * Bz(i, j, 0, 2*m  ) / r
                    );  // Real part
                    Jr(i, j, 0, 2*m  ) = one_over_mu0 * (
                        - T_Algo::DownwardDz(Btheta, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                        - m * Bz(i, j, 0, 2*m-1) / r
                    ); // Imaginary part
                }
            },

            // Jtheta calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Jtheta_arr && update_Jtheta_arr(i, j, 0) == 0) { return; }

                // r on a nodal point (Jtheta is nodal in r)
                Real const r = rmin + i*dr;
                // Off-axis, regular curl
                if (r > 0.5_rt*dr) {
                    // Mode m=0
                    Jtheta(i, j, 0, 0) = one_over_mu0 * (
                        - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 0)
                        + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 0)
                    );

                    // Higher-order modes
                    for (int m=1 ; m<nmodes ; m++) { // Higher-order modes
                        Jtheta(i, j, 0, 2*m-1) = one_over_mu0 * (
                            - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                            + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 2*m-1)
                        ); // Real part
                        Jtheta(i, j, 0, 2*m  ) = one_over_mu0 * (
                            - T_Algo::DownwardDr(Bz, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                            + T_Algo::DownwardDz(Br, coefs_z, n_coefs_z, i, j, 0, 2*m  )
                        ); // Imaginary part
                    }
                // r==0: on-axis corrections
                } else {
                    // Ensure that Jtheta remains 0 on axis (except for m=1)
                    // Mode m=0
                    Jtheta(i, j, 0, 0) = 0.;
                    // Higher-order modes
                    for (int m=1; m<nmodes; m++) {
                        if (m == 1){
                            // The same logic as is used in the E-field update for the fully
                            // electromagnetic FDTD case is used here.
                            Jtheta(i,j,0,2*m-1) =  Jr(i,j,0,2*m  );
                            Jtheta(i,j,0,2*m  ) = -Jr(i,j,0,2*m-1);
                        } else {
                            Jtheta(i, j, 0, 2*m-1) = 0.;
                            Jtheta(i, j, 0, 2*m  ) = 0.;
                        }
                    }
                }
            },

            // Jz calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Jz_arr && update_Jz_arr(i, j, 0) == 0) { return; }

                // r on a nodal point (Jz is nodal in r)
                Real const r = rmin + i*dr;
                // Off-axis, regular curl
                if (r > 0.5_rt*dr) {
                    // Mode m=0
                    Jz(i, j, 0, 0) = one_over_mu0 * (
                       T_Algo::DownwardDrr_over_r(Btheta, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                    );
                    // Higher-order modes
                    for (int m=1 ; m<nmodes ; m++) {
                        Jz(i, j, 0, 2*m-1) = one_over_mu0 * (
                            - m * Br(i, j, 0, 2*m  ) / r
                            + T_Algo::DownwardDrr_over_r(Btheta, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m-1)
                        ); // Real part
                        Jz(i, j, 0, 2*m  ) = one_over_mu0 * (
                            m * Br(i, j, 0, 2*m-1) / r
                            + T_Algo::DownwardDrr_over_r(Btheta, r, dr, coefs_r, n_coefs_r, i, j, 0, 2*m  )
                        ); // Imaginary part
                    }
                // r==0: on-axis corrections
                } else {
                    // For m==0, Btheta is linear in r, for small r
                    // Therefore, the formula below regularizes the singularity
                    Jz(i, j, 0, 0) = one_over_mu0 * 4 * Btheta(i, j, 0, 0) / dr;
                    // Ensure that Jz remains 0 for higher-order modes
                    for (int m=1; m<nmodes; m++) {
                        Jz(i, j, 0, 2*m-1) = 0.;
                        Jz(i, j, 0, 2*m  ) = 0.;
                    }
                }
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

#elif defined(WARPX_DIM_RSPHERE)
template<typename T_Algo>
void FiniteDifferenceSolver::CalculateCurrentAmpereSpherical (
    ablastr::fields::VectorField& Jfield,
    ablastr::fields::VectorField const& Bfield,
    int lev
)
{
    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Jfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const& Jr = Jfield[0]->array(mfi);
        Array4<Real> const& Jtheta = Jfield[1]->array(mfi);
        Array4<Real> const& Jphi = Jfield[2]->array(mfi);
        Array4<Real> const& Btheta = Bfield[1]->array(mfi);
        Array4<Real> const& Bphi = Bfield[2]->array(mfi);

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        int const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());

        // Extract cylindrical specific parameters
        Real const dr = m_dr;
        Real const rmin = m_rmin;

        // Extract tileboxes for which to loop
        Box const& tjr  = mfi.tilebox(Jfield[0]->ixType().toIntVect());
        Box const& tjtheta  = mfi.tilebox(Jfield[1]->ixType().toIntVect());
        Box const& tjphi  = mfi.tilebox(Jfield[2]->ixType().toIntVect());

        Real const one_over_mu0 = 1._rt / PhysConst::mu0;

        // Calculate the total current, using Ampere's law, on the same grid
        // as the E-field
        amrex::ParallelFor(tjr, tjtheta, tjphi,

            // Jr calculation
            [=] AMREX_GPU_DEVICE (int i, int /*j*/, int /*k*/){
                Jr(i, 0, 0, 0) = 0._rt;
            },

            // Jtheta calculation
            [=] AMREX_GPU_DEVICE (int i, int /*j*/, int /*k*/){
                // r on a nodal point (Jtheta is nodal in r)
                Real const r = rmin + i*dr;
                // Off-axis, regular curl
                if (r > 0.5_rt*dr) {
                    // Mode m=0
                    Jtheta(i, 0, 0, 0) = one_over_mu0 * (
                        - T_Algo::DownwardDrr_over_r(Bphi, r, dr, coefs_r, n_coefs_r, i, 0, 0, 0));
                } else { // r==0: on-axis corrections
                    // Ensure that Jtheta remains 0 on axis
                    Jtheta(i, 0, 0, 0) = 0.;
                }
            },

            // Jphi calculation
            [=] AMREX_GPU_DEVICE (int i, int /*j*/, int /*k*/){
                // r on a nodal point (Jphi is nodal in r)
                Real const r = rmin + i*dr;
                // Off-axis, regular curl
                if (r > 0.5_rt*dr) {
                    Jphi(i, 0, 0, 0) = one_over_mu0 * (
                       T_Algo::DownwardDrr_over_r(Btheta, r, dr, coefs_r, n_coefs_r, i, 0, 0, 0)
                    );
                // r==0: on-axis corrections
                } else {
                    // Btheta is linear in r, for small r
                    // Therefore, the formula below regularizes the singularity
                    Jphi(i, 0, 0, 0) = one_over_mu0 * 4 * Btheta(i, 0, 0, 0) / dr;
                }
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

#else

template<typename T_Algo>
void FiniteDifferenceSolver::CalculateCurrentAmpereCartesian (
    ablastr::fields::VectorField& Jfield,
    ablastr::fields::VectorField const& Bfield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev
)
{
    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Jfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers) {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const &Jx = Jfield[0]->array(mfi);
        Array4<Real> const &Jy = Jfield[1]->array(mfi);
        Array4<Real> const &Jz = Jfield[2]->array(mfi);
        Array4<Real const> const &Bx = Bfield[0]->const_array(mfi);
        Array4<Real const> const &By = Bfield[1]->const_array(mfi);
        Array4<Real const> const &Bz = Bfield[2]->const_array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries.
        // The plasma current is stored at the same locations as the E-field,
        // therefore the `eb_update_E` multifab also appropriately specifies
        // where the plasma current should be calculated.
        amrex::Array4<int> update_Jx_arr, update_Jy_arr, update_Jz_arr;
        if (EB::enabled()) {
            update_Jx_arr = eb_update_E[0]->array(mfi);
            update_Jy_arr = eb_update_E[1]->array(mfi);
            update_Jz_arr = eb_update_E[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract tileboxes for which to loop
        Box const& tjx  = mfi.tilebox(Jfield[0]->ixType().toIntVect());
        Box const& tjy  = mfi.tilebox(Jfield[1]->ixType().toIntVect());
        Box const& tjz  = mfi.tilebox(Jfield[2]->ixType().toIntVect());

        Real const one_over_mu0 = 1._rt / PhysConst::mu0;

        // Calculate the total current, using Ampere's law, on the same grid
        // as the E-field
        amrex::ParallelFor(tjx, tjy, tjz,

            // Jx calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Skip field update in the embedded boundaries
                if (update_Jx_arr && update_Jx_arr(i, j, k) == 0) { return; }

                Jx(i, j, k) = one_over_mu0 * (
                    - T_Algo::DownwardDz(By, coefs_z, n_coefs_z, i, j, k)
                    + T_Algo::DownwardDy(Bz, coefs_y, n_coefs_y, i, j, k)
                );
            },

            // Jy calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Skip field update in the embedded boundaries
                if (update_Jy_arr && update_Jy_arr(i, j, k) == 0) { return; }

                Jy(i, j, k) = one_over_mu0 * (
                    - T_Algo::DownwardDx(Bz, coefs_x, n_coefs_x, i, j, k)
                    + T_Algo::DownwardDz(Bx, coefs_z, n_coefs_z, i, j, k)
                );
            },

            // Jz calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){

                // Skip field update in the embedded boundaries
                if (update_Jz_arr && update_Jz_arr(i, j, k) == 0) { return; }

                Jz(i, j, k) = one_over_mu0 * (
                    - T_Algo::DownwardDy(Bx, coefs_y, n_coefs_y, i, j, k)
                    + T_Algo::DownwardDx(By, coefs_x, n_coefs_x, i, j, k)
                );
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}
#endif


void FiniteDifferenceSolver::HybridPICSolveE (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    [[maybe_unused]]std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev, HybridPICModel const* hybrid_model,
    const bool solve_for_Faraday)
{
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)

        if (hybrid_model->m_use_pseudo_spectral_E) {
            HybridPICSolveECylindricalMultiMode <CylindricalYeeAlgorithm> (
                Efield, Jfield, Jifield, Bfield, rhofield, Pefield,
                eb_update_E, lev, hybrid_model, solve_for_Faraday
            );
        } else {
            HybridPICSolveECylindrical <CylindricalYeeAlgorithm> (
                Efield, Jfield, Jifield, Bfield, rhofield, Pefield,
                eb_update_E, lev, hybrid_model, solve_for_Faraday
            );
        }

#elif defined(WARPX_DIM_RSPHERE)

        HybridPICSolveESpherical <SphericalYeeAlgorithm> (
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield,
            lev, hybrid_model, solve_for_Faraday
        );

#else
    if (WarpX::GetInstance().grid_type == GridType::Staggered)
    {
        HybridPICSolveECartesian <CartesianYeeAlgorithm> (
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield,
            eb_update_E, lev, hybrid_model, solve_for_Faraday
        );
    } else {
        HybridPICSolveECartesian <CartesianNodalAlgorithm> (
            Efield, Jfield, Jifield, Bfield, rhofield, Pefield,
            eb_update_E, lev, hybrid_model, solve_for_Faraday
        );
    }
#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "HybridSolveE: The hybrid-PIC electromagnetic solver algorithm must be used"));
    }
}

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
template<typename T_Algo>
void FiniteDifferenceSolver::HybridPICSolveECylindrical (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev, HybridPICModel const* hybrid_model,
    const bool solve_for_Faraday )
{
    // Both steps below do not currently support m > 0 and should be
    // modified if such support wants to be added
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_nmodes == 1),
        "Ohm's law solver only support m = 0 azimuthal mode at present.");

    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    using namespace ablastr::coarsen::sample;

    // get hybrid model parameters
    const auto eta = hybrid_model->m_eta;
    const auto eta_h = hybrid_model->m_eta_h;
    const auto rho_floor = hybrid_model->m_n_floor * PhysConst::q_e;
    const auto resistivity_has_J_dependence = hybrid_model->m_resistivity_has_J_dependence;
    const auto hyper_resistivity_has_B_dependence = hybrid_model->m_hyper_resistivity_has_B_dependence;
    const bool include_hyper_resistivity_term = hybrid_model->m_include_hyper_resistivity_term;

    const bool include_external_fields = hybrid_model->m_add_external_fields;

    const bool holmstrom_vacuum_region = hybrid_model->m_holmstrom_vacuum_region;

    auto & warpx = WarpX::GetInstance();
    ablastr::fields::VectorField Bfield_external, Efield_external;
    if (include_external_fields) {
        Bfield_external = warpx.m_fields.get_alldirs(FieldType::hybrid_B_fp_external, 0); // lev=0
        Efield_external = warpx.m_fields.get_alldirs(FieldType::hybrid_E_fp_external, 0); // lev=0
    }

    // Index type required for interpolating fields from their respective
    // staggering to the Ex, Ey, Ez locations
    amrex::GpuArray<int, 3> const& Er_stag = hybrid_model->Ex_IndexType;
    amrex::GpuArray<int, 3> const& Etheta_stag = hybrid_model->Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_stag = hybrid_model->Ez_IndexType;
    amrex::GpuArray<int, 3> const& Jr_stag = hybrid_model->Jx_IndexType;
    amrex::GpuArray<int, 3> const& Jtheta_stag = hybrid_model->Jy_IndexType;
    amrex::GpuArray<int, 3> const& Jz_stag = hybrid_model->Jz_IndexType;
    amrex::GpuArray<int, 3> const& Br_stag = hybrid_model->Bx_IndexType;
    amrex::GpuArray<int, 3> const& Btheta_stag = hybrid_model->By_IndexType;
    amrex::GpuArray<int, 3> const& Bz_stag = hybrid_model->Bz_IndexType;

    // Parameters for `interp` that maps from Yee to nodal mesh and back
    amrex::GpuArray<int, 3> const& nodal = {1, 1, 1};
    // The "coarsening is just 1 i.e. no coarsening"
    amrex::GpuArray<int, 3> const& coarsen = {1, 1, 1};

    // The E-field calculation is done in 2 steps:
    // 1) The J x B term is calculated on a nodal mesh in order to ensure
    //    energy conservation.
    // 2) The nodal E-field values are averaged onto the Yee grid and the
    //    electron pressure & resistivity terms are added (these terms are
    //    naturally located on the Yee grid).

    // Create a temporary multifab to hold the nodal E-field values
    // Note the multifab has 3 values for Ex, Ey and Ez which we can do here
    // since all three components will be calculated on the same grid.
    // For multi-mode RZ each spatial direction carries (2*nmodes - 1) components.
    // Also note that enE_nodal_mf does not need to have any guard cells since
    // these values will be interpolated to the Yee mesh which is contained
    // by the nodal mesh.
    auto const& ba = convert(rhofield.boxArray(), IntVect::TheNodeVector());
    MultiFab enE_nodal_mf(ba, rhofield.DistributionMap(), 3 * (2*m_nmodes - 1), IntVect::TheZeroVector());

    // Loop through the grids, and over the tiles within each grid for the
    // initial, nodal calculation of E
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(enE_nodal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        Array4<Real> const& enE_nodal = enE_nodal_mf.array(mfi);
        Array4<Real const> const& Jr = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jtheta = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& Jir = Jifield[0]->const_array(mfi);
        Array4<Real const> const& Jit = Jifield[1]->const_array(mfi);
        Array4<Real const> const& Jiz = Jifield[2]->const_array(mfi);
        Array4<Real const> const& Br = Bfield[0]->const_array(mfi);
        Array4<Real const> const& Btheta = Bfield[1]->const_array(mfi);
        Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);

        Array4<Real> Br_ext, Btheta_ext, Bz_ext;
        if (include_external_fields) {
            Br_ext = Bfield_external[0]->array(mfi);
            Btheta_ext = Bfield_external[1]->array(mfi);
            Bz_ext = Bfield_external[2]->array(mfi);
        }

        // Loop over the cells and update the nodal E field
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

            // interpolate the total current to a nodal grid
            auto const jr_interp = Interp(Jr, Jr_stag, nodal, coarsen, i, j, 0, 0);
            auto const jtheta_interp = Interp(Jtheta, Jtheta_stag, nodal, coarsen, i, j, 0, 0);
            auto const jz_interp = Interp(Jz, Jz_stag, nodal, coarsen, i, j, 0, 0);

            // interpolate the ion current to a nodal grid
            auto const jir_interp = Interp(Jir, Jr_stag, nodal, coarsen, i, j, 0, 0);
            auto const jit_interp = Interp(Jit, Jtheta_stag, nodal, coarsen, i, j, 0, 0);
            auto const jiz_interp = Interp(Jiz, Jz_stag, nodal, coarsen, i, j, 0, 0);

            // interpolate the B field to a nodal grid
            auto Br_interp = Interp(Br, Br_stag, nodal, coarsen, i, j, 0, 0);
            auto Btheta_interp = Interp(Btheta, Btheta_stag, nodal, coarsen, i, j, 0, 0);
            auto Bz_interp = Interp(Bz, Bz_stag, nodal, coarsen, i, j, 0, 0);

            if (include_external_fields) {
                Br_interp += Interp(Br_ext, Br_stag, nodal, coarsen, i, j, 0, 0);
                Btheta_interp += Interp(Btheta_ext, Btheta_stag, nodal, coarsen, i, j, 0, 0);
                Bz_interp += Interp(Bz_ext, Bz_stag, nodal, coarsen, i, j, 0, 0);
            }

            // calculate enE = (J - Ji) x B
            enE_nodal(i, j, 0, 0) = (
                (jtheta_interp - jit_interp) * Bz_interp
                - (jz_interp - jiz_interp) * Btheta_interp
            );
            enE_nodal(i, j, 0, 1) = (
                (jz_interp - jiz_interp) * Br_interp
                - (jr_interp - jir_interp) * Bz_interp
            );
            enE_nodal(i, j, 0, 2) = (
                (jr_interp - jir_interp) * Btheta_interp
                - (jtheta_interp - jit_interp) * Br_interp
            );
        });

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    // Loop through the grids, and over the tiles within each grid again
    // for the Yee grid calculation of the E field
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const& Er = Efield[0]->array(mfi);
        Array4<Real> const& Etheta = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real const> const& Jr = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jtheta = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& enE = enE_nodal_mf.const_array(mfi);
        Array4<Real const> const& rho = rhofield.const_array(mfi);
        Array4<Real const> const& Pe = Pefield.const_array(mfi);
        Array4<Real> const& Br = Bfield[0]->array(mfi);
        Array4<Real> const& Btheta = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries
        amrex::Array4<int> update_Er_arr, update_Etheta_arr, update_Ez_arr;
        if (EB::enabled()) {
            update_Er_arr = eb_update_E[0]->array(mfi);
            update_Etheta_arr = eb_update_E[1]->array(mfi);
            update_Ez_arr = eb_update_E[2]->array(mfi);
        }

        Array4<Real> Er_ext, Etheta_ext, Ez_ext;
        if (include_external_fields) {
            Er_ext = Efield_external[0]->array(mfi);
            Etheta_ext = Efield_external[1]->array(mfi);
            Ez_ext = Efield_external[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        int const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract cylindrical specific parameters
        Real const dr = m_dr;
        Real const rmin = m_rmin;

        Box const& ter  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tet  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Loop over the cells and update the E field
        amrex::ParallelFor(ter, tet, tez,

            // Er calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Er_arr && update_Er_arr(i, j, 0) == 0) { return; }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Er_stag, coarsen, i, j, 0, 0);

                if (rho_val < rho_floor && holmstrom_vacuum_region) {
                    Er(i, j, 0) = 0._rt;
                } else {
                    // Get the gradient of the electron pressure if the longitudinal part of
                    // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                    const Real grad_Pe = (!solve_for_Faraday) ?
                        T_Algo::UpwardDr(Pe, coefs_r, n_coefs_r, i, j, 0, 0)
                        : 0._rt;

                    // interpolate the nodal neE values to the Yee grid
                    const auto enE_r = Interp(enE, nodal, Er_stag, coarsen, i, j, 0, 0);

                    // safety condition since we divide by rho
                    const auto rho_val_limited = std::max(rho_val, rho_floor);

                    Er(i, j, 0) = (enE_r - grad_Pe) / rho_val_limited;
                }

                // Add resistivity only if E field value is used to update B
                if (solve_for_Faraday) {
                    Real jtot_val = 0._rt;
                    if (resistivity_has_J_dependence) {
                        // Interpolate current to appropriate staggering to match E field
                        const Real jr_val = Jr(i, j, 0);
                        const Real jtheta_val = Interp(Jtheta, Jtheta_stag, Er_stag, coarsen, i, j, 0, 0);
                        const Real jz_val = Interp(Jz, Jz_stag, Er_stag, coarsen, i, j, 0, 0);
                        jtot_val = std::sqrt(jr_val*jr_val + jtheta_val*jtheta_val + jz_val*jz_val);
                    }

                    Er(i, j, 0) += eta(rho_val, jtot_val) * Jr(i, j, 0);

                    if (include_hyper_resistivity_term) {

                        // Interpolate B field to appropriate staggering to match E field
                        Real btot_val = 0._rt;
                        if (hyper_resistivity_has_B_dependence) {
                            const Real br_val = Interp(Br, Br_stag, Er_stag, coarsen, i, j, 0, 0);
                            const Real bt_val = Interp(Btheta, Btheta_stag, Er_stag, coarsen, i, j, 0, 0);
                            const Real bz_val = Interp(Bz, Bz_stag, Er_stag, coarsen, i, j, 0, 0);
                            btot_val = std::sqrt(br_val*br_val + bt_val*bt_val + bz_val*bz_val);
                        }

                        // r on cell-centered point (Jr is cell-centered in r)
                        const Real r = rmin + (i + 0.5_rt)*dr;
                        auto nabla2Jr = T_Algo::Dr_rDr_over_r(Jr, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                            + T_Algo::Dzz(Jr, coefs_z, n_coefs_z, i, j, 0, 0) - Jr(i, j, 0)/(r*r);

                        Er(i, j, 0) -= eta_h(rho_val, btot_val) * nabla2Jr;
                    }
                }

                if (include_external_fields && (rho_val >= rho_floor)) {
                    Er(i, j, 0) -= Er_ext(i, j, 0);
                }
            },

            // Etheta calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Etheta_arr && update_Etheta_arr(i, j, 0) == 0) { return; }

                // r on a nodal grid (Etheta is nodal in r)
                Real const r = rmin + i*dr;
                // Mode m=0: // Ensure that Etheta remains 0 on axis
                if (r < 0.5_rt*dr) {
                    Etheta(i, j, 0, 0) = 0.;
                    return;
                }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Etheta_stag, coarsen, i, j, 0, 0);

                if (rho_val < rho_floor && holmstrom_vacuum_region) {
                    Etheta(i, j, 0) = 0._rt;
                } else {
                    // Get the gradient of the electron pressure
                    // -> d/dt = 0 for m = 0
                    const auto grad_Pe = 0.0_rt;

                    // interpolate the nodal neE values to the Yee grid
                    const auto enE_t = Interp(enE, nodal, Etheta_stag, coarsen, i, j, 0, 1);

                    // safety condition since we divide by rho
                    const auto rho_val_limited = std::max(rho_val, rho_floor);

                    Etheta(i, j, 0) = (enE_t - grad_Pe) / rho_val_limited;
                }

                // Add resistivity only if E field value is used to update B
                if (solve_for_Faraday) {
                    Real jtot_val = 0._rt;
                    if(resistivity_has_J_dependence) {
                        // Interpolate current to appropriate staggering to match E field
                        const Real jr_val = Interp(Jr, Jr_stag, Etheta_stag, coarsen, i, j, 0, 0);
                        const Real jtheta_val = Jtheta(i, j, 0);
                        const Real jz_val = Interp(Jz, Jz_stag, Etheta_stag, coarsen, i, j, 0, 0);
                        jtot_val = std::sqrt(jr_val*jr_val + jtheta_val*jtheta_val + jz_val*jz_val);
                    }

                    Etheta(i, j, 0) += eta(rho_val, jtot_val) * Jtheta(i, j, 0);

                    if (include_hyper_resistivity_term) {

                        // Interpolate B field to appropriate staggering to match E field
                        Real btot_val = 0._rt;
                        if (hyper_resistivity_has_B_dependence) {
                            const Real br_val = Interp(Br, Br_stag, Etheta_stag, coarsen, i, j, 0, 0);
                            const Real bt_val = Interp(Btheta, Btheta_stag, Etheta_stag, coarsen, i, j, 0, 0);
                            const Real bz_val = Interp(Bz, Bz_stag, Etheta_stag, coarsen, i, j, 0, 0);
                            btot_val = std::sqrt(br_val*br_val + bt_val*bt_val + bz_val*bz_val);
                        }

                        // Special handling of the hyper-resistivity term on axis to avoid division by zero
                        // and ensure that Etheta remains 0 on axis for m=0 mode
                        auto nabla2Jtheta = 0.0_rt;
                        if (r > 0.0_rt) {
                            nabla2Jtheta = T_Algo::Dr_rDr_over_r(Jtheta, r, dr, coefs_r, n_coefs_r, i, j, 0, 0)
                                + T_Algo::Dzz(Jtheta, coefs_z, n_coefs_z, i, j, 0, 0) - Jtheta(i, j, 0)/(r*r);
                        }

                        Etheta(i, j, 0) -= eta_h(rho_val, btot_val) * nabla2Jtheta;
                    }
                }

                if (include_external_fields && (rho_val >= rho_floor)) {
                    Etheta(i, j, 0) -= Etheta_ext(i, j, 0);
                }
            },

            // Ez calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

                // Skip field update in the embedded boundaries
                if (update_Ez_arr && update_Ez_arr(i, j, 0) == 0) { return; }

                // Interpolate to get the appropriate charge density in space
                const Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, 0, 0);

                if (rho_val < rho_floor && holmstrom_vacuum_region) {
                    Ez(i, j, 0) = 0._rt;
                } else {
                    // Get the gradient of the electron pressure if the longitudinal part of
                    // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                    const Real grad_Pe = (!solve_for_Faraday) ?
                        T_Algo::UpwardDz(Pe, coefs_z, n_coefs_z, i, j, 0, 0)
                        : 0._rt;

                    // interpolate the nodal neE values to the Yee grid
                    const auto enE_z = Interp(enE, nodal, Ez_stag, coarsen, i, j, 0, 2);

                    // safety condition since we divide by rho
                    const auto rho_val_limited = std::max(rho_val, rho_floor);

                    Ez(i, j, 0) = (enE_z - grad_Pe) / rho_val_limited;
                }

                // Add resistivity only if E field value is used to update B
                if (solve_for_Faraday) {
                    Real jtot_val = 0._rt;
                    if (resistivity_has_J_dependence) {
                        // Interpolate current to appropriate staggering to match E field
                        const Real jr_val = Interp(Jr, Jr_stag, Ez_stag, coarsen, i, j, 0, 0);
                        const Real jtheta_val = Interp(Jtheta, Jtheta_stag, Ez_stag, coarsen, i, j, 0, 0);
                        const Real jz_val = Jz(i, j, 0);
                        jtot_val = std::sqrt(jr_val*jr_val + jtheta_val*jtheta_val + jz_val*jz_val);
                    }

                    Ez(i, j, 0) += eta(rho_val, jtot_val) * Jz(i, j, 0);

                    if (include_hyper_resistivity_term) {

                        // Interpolate B field to appropriate staggering to match E field
                        Real btot_val = 0._rt;
                        if (hyper_resistivity_has_B_dependence) {
                            const Real br_val = Interp(Br, Br_stag, Ez_stag, coarsen, i, j, 0, 0);
                            const Real bt_val = Interp(Btheta, Btheta_stag, Ez_stag, coarsen, i, j, 0, 0);
                            const Real bz_val = Interp(Bz, Bz_stag, Ez_stag, coarsen, i, j, 0, 0);
                            btot_val = std::sqrt(br_val*br_val + bt_val*bt_val + bz_val*bz_val);
                        }

                        // r on nodal point (Jz is nodal in r)
                        const Real r = rmin + i*dr;

                        auto nabla2Jz = T_Algo::Dzz(Jz, coefs_z, n_coefs_z, i, j, 0, 0);
                        if (r > 0.5_rt*dr) {
                            nabla2Jz += T_Algo::Dr_rDr_over_r(Jz, r, dr, coefs_r, n_coefs_r, i, j, 0, 0);
                        } else {
                            // Special handling of the hyper-resistivity term on axis to avoid division by zero
                            // and ensure that Jz remains well-behaved on axis for m=0 mode
                            // This works since there is a symmetry condition on axis that cancels the geometric 1/r term
                            nabla2Jz += T_Algo::Drr(Jz, coefs_r, n_coefs_r, i, j, 0, 0);
                        }

                        Ez(i, j, 0) -= eta_h(rho_val, btot_val) * nabla2Jz;
                    }
                }

                if (include_external_fields && (rho_val >= rho_floor)) {
                    Ez(i, j, 0) -= Ez_ext(i, j, 0);
                }
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

template<typename T_Algo>
void FiniteDifferenceSolver::HybridPICSolveECylindricalMultiMode (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev, HybridPICModel const* hybrid_model,
    const bool solve_for_Faraday )
{
    // Pseudo-spectral azimuthal-mode Ohm's-law E-solve.
    //
    // Structure mirrors HybridPICSolveECylindrical:
    //   Pass 1 (nodal): compute (J - J_i) x B per azimuthal mode and store
    //                   in a 3*ncomps-component nodal scratch MultiFab.
    //   Pass 2 (Yee):   for each component r/theta/z and each mode,
    //                   E = (enE - grad P_e) / (e rho), plus eta J (resistivity
    //                   on the Faraday update path).
    //
    // The nonlinear pieces ((J - J_i) x B, the 1/rho factor, eta(rho, |J|))
    // are evaluated by inverse-FT'ing each cell's modal inputs to an
    // azimuthal grid of n_theta = 2*nmodes - 1 points, doing the pointwise
    // arithmetic in real (r, z, theta) space, and forward-FT'ing back to
    // modes. At nmodes = 1 both FTs degenerate to the identity and the
    // kernel reduces to the same scalar arithmetic as the m=0-only routine.

    const int nmodes = m_nmodes;
    const int ncomps = 2 * nmodes - 1;
    const int n_theta = 2 * nmodes - 1;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        nmodes <= MAX_PSEUDO_SPECTRAL_NMODES,
        "n_rz_azimuthal_modes exceeds MAX_PSEUDO_SPECTRAL_NMODES; "
        "increase the compile-time bound in HybridPICSolveE.cpp.");

    // First-cut feature gating: lift in follow-up commits once each path is
    // generalized to multi-mode separately. Both pieces introduce additional
    // nonlinearities (Laplacian-of-J / external-field couplings).
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !hybrid_model->m_include_hyper_resistivity_term,
        "Pseudo-spectral hybrid solver: hyper-resistivity is not yet supported.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !hybrid_model->m_add_external_fields,
        "Pseudo-spectral hybrid solver: external E/B fields are not yet supported.");

    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    using namespace ablastr::coarsen::sample;

    // get hybrid model parameters
    const auto eta = hybrid_model->m_eta;
    const auto rho_floor = hybrid_model->m_n_floor * PhysConst::q_e;
    const auto resistivity_has_J_dependence = hybrid_model->m_resistivity_has_J_dependence;
    const bool holmstrom_vacuum_region = hybrid_model->m_holmstrom_vacuum_region;

    // Index type required for interpolating fields from their respective
    // staggering to the Er, Etheta, Ez locations
    amrex::GpuArray<int, 3> const& Er_stag     = hybrid_model->Ex_IndexType;
    amrex::GpuArray<int, 3> const& Etheta_stag = hybrid_model->Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_stag     = hybrid_model->Ez_IndexType;
    amrex::GpuArray<int, 3> const& Jr_stag     = hybrid_model->Jx_IndexType;
    amrex::GpuArray<int, 3> const& Jtheta_stag = hybrid_model->Jy_IndexType;
    amrex::GpuArray<int, 3> const& Jz_stag     = hybrid_model->Jz_IndexType;
    amrex::GpuArray<int, 3> const& Br_stag     = hybrid_model->Bx_IndexType;
    amrex::GpuArray<int, 3> const& Btheta_stag = hybrid_model->By_IndexType;
    amrex::GpuArray<int, 3> const& Bz_stag     = hybrid_model->Bz_IndexType;

    amrex::GpuArray<int, 3> const& nodal   = {1, 1, 1};
    amrex::GpuArray<int, 3> const& coarsen = {1, 1, 1};

    // 3*ncomps components on the nodal scratch: components
    //   [0 .. ncomps-1]            -> r-direction modal pieces
    //   [ncomps .. 2*ncomps-1]     -> theta-direction modal pieces
    //   [2*ncomps .. 3*ncomps-1]   -> z-direction modal pieces
    auto const& ba = convert(rhofield.boxArray(), IntVect::TheNodeVector());
    MultiFab enE_nodal_mf(ba, rhofield.DistributionMap(), 3 * ncomps,
                          IntVect::TheZeroVector());

    // =====================================================================
    // Pass 1 : compute modal (J - J_i) x B on the nodal grid
    // =====================================================================
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(enE_nodal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        Array4<Real> const& enE_nodal     = enE_nodal_mf.array(mfi);
        Array4<Real const> const& Jr      = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jtheta  = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz      = Jfield[2]->const_array(mfi);
        Array4<Real const> const& Jir     = Jifield[0]->const_array(mfi);
        Array4<Real const> const& Jit     = Jifield[1]->const_array(mfi);
        Array4<Real const> const& Jiz     = Jifield[2]->const_array(mfi);
        Array4<Real const> const& Br      = Bfield[0]->const_array(mfi);
        Array4<Real const> const& Btheta  = Bfield[1]->const_array(mfi);
        Array4<Real const> const& Bz      = Bfield[2]->const_array(mfi);

        amrex::ParallelFor(mfi.tilebox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) {

            // Modal samples at this nodal cell, after stagger interpolation.
            Real Br_m [MAX_PSEUDO_SPECTRAL_NMODES * 2];
            Real Bt_m [MAX_PSEUDO_SPECTRAL_NMODES * 2];
            Real Bz_m [MAX_PSEUDO_SPECTRAL_NMODES * 2];
            Real Jr_m [MAX_PSEUDO_SPECTRAL_NMODES * 2];
            Real Jt_m [MAX_PSEUDO_SPECTRAL_NMODES * 2];
            Real Jz_m [MAX_PSEUDO_SPECTRAL_NMODES * 2];
            Real Jir_m[MAX_PSEUDO_SPECTRAL_NMODES * 2];
            Real Jit_m[MAX_PSEUDO_SPECTRAL_NMODES * 2];
            Real Jiz_m[MAX_PSEUDO_SPECTRAL_NMODES * 2];

            for (int c = 0; c < ncomps; ++c) {
                Br_m [c] = Interp(Br,     Br_stag,     nodal, coarsen, i, j, 0, c);
                Bt_m [c] = Interp(Btheta, Btheta_stag, nodal, coarsen, i, j, 0, c);
                Bz_m [c] = Interp(Bz,     Bz_stag,     nodal, coarsen, i, j, 0, c);
                Jr_m [c] = Interp(Jr,     Jr_stag,     nodal, coarsen, i, j, 0, c);
                Jt_m [c] = Interp(Jtheta, Jtheta_stag, nodal, coarsen, i, j, 0, c);
                Jz_m [c] = Interp(Jz,     Jz_stag,     nodal, coarsen, i, j, 0, c);
                Jir_m[c] = Interp(Jir,    Jr_stag,     nodal, coarsen, i, j, 0, c);
                Jit_m[c] = Interp(Jit,    Jtheta_stag, nodal, coarsen, i, j, 0, c);
                Jiz_m[c] = Interp(Jiz,    Jz_stag,     nodal, coarsen, i, j, 0, c);
            }

            // On-axis modal BCs (regularity in (x, y) Cartesian coords).
            // Nodal i == 0 corresponds to r == 0 exactly.
            if (i == 0) {
                apply_axis_bc_vector_rt(nmodes, Br_m,  Bt_m);
                apply_axis_bc_vector_rt(nmodes, Jr_m,  Jt_m);
                apply_axis_bc_vector_rt(nmodes, Jir_m, Jit_m);
                // V_z behaves as a scalar: only m >= 1 modes are forced to zero.
                apply_axis_bc_scalar(nmodes, Bz_m);
                apply_axis_bc_scalar(nmodes, Jz_m);
                apply_axis_bc_scalar(nmodes, Jiz_m);
            }

            // Inverse FT each modal array to theta-grid samples.
            Real Br_th [MAX_PSEUDO_SPECTRAL_NTHETA];
            Real Bt_th [MAX_PSEUDO_SPECTRAL_NTHETA];
            Real Bz_th [MAX_PSEUDO_SPECTRAL_NTHETA];
            Real Jr_th [MAX_PSEUDO_SPECTRAL_NTHETA];
            Real Jt_th [MAX_PSEUDO_SPECTRAL_NTHETA];
            Real Jz_th [MAX_PSEUDO_SPECTRAL_NTHETA];
            Real Jir_th[MAX_PSEUDO_SPECTRAL_NTHETA];
            Real Jit_th[MAX_PSEUDO_SPECTRAL_NTHETA];
            Real Jiz_th[MAX_PSEUDO_SPECTRAL_NTHETA];

            inverse_az_ft(nmodes, n_theta, Br_m,  Br_th);
            inverse_az_ft(nmodes, n_theta, Bt_m,  Bt_th);
            inverse_az_ft(nmodes, n_theta, Bz_m,  Bz_th);
            inverse_az_ft(nmodes, n_theta, Jr_m,  Jr_th);
            inverse_az_ft(nmodes, n_theta, Jt_m,  Jt_th);
            inverse_az_ft(nmodes, n_theta, Jz_m,  Jz_th);
            inverse_az_ft(nmodes, n_theta, Jir_m, Jir_th);
            inverse_az_ft(nmodes, n_theta, Jit_m, Jit_th);
            inverse_az_ft(nmodes, n_theta, Jiz_m, Jiz_th);

            // Pointwise (J - J_i) x B at each theta_k.
            Real enE_r_th[MAX_PSEUDO_SPECTRAL_NTHETA];
            Real enE_t_th[MAX_PSEUDO_SPECTRAL_NTHETA];
            Real enE_z_th[MAX_PSEUDO_SPECTRAL_NTHETA];
            for (int k = 0; k < n_theta; ++k) {
                const Real dJr = Jr_th[k] - Jir_th[k];
                const Real dJt = Jt_th[k] - Jit_th[k];
                const Real dJz = Jz_th[k] - Jiz_th[k];
                enE_r_th[k] = dJt * Bz_th[k] - dJz * Bt_th[k];
                enE_t_th[k] = dJz * Br_th[k] - dJr * Bz_th[k];
                enE_z_th[k] = dJr * Bt_th[k] - dJt * Br_th[k];
            }

            // Forward FT back to modes and store.
            Real enE_r_m[MAX_PSEUDO_SPECTRAL_NMODES * 2];
            Real enE_t_m[MAX_PSEUDO_SPECTRAL_NMODES * 2];
            Real enE_z_m[MAX_PSEUDO_SPECTRAL_NMODES * 2];
            forward_az_ft(nmodes, n_theta, enE_r_th, enE_r_m);
            forward_az_ft(nmodes, n_theta, enE_t_th, enE_t_m);
            forward_az_ft(nmodes, n_theta, enE_z_th, enE_z_m);

            for (int c = 0; c < ncomps; ++c) {
                enE_nodal(i, j, 0, 0*ncomps + c) = enE_r_m[c];
                enE_nodal(i, j, 0, 1*ncomps + c) = enE_t_m[c];
                enE_nodal(i, j, 0, 2*ncomps + c) = enE_z_m[c];
            }
        });

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    // =====================================================================
    // Pass 2 : compute modal E on the Yee staggering
    // =====================================================================
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        Array4<Real> const& Er             = Efield[0]->array(mfi);
        Array4<Real> const& Etheta         = Efield[1]->array(mfi);
        Array4<Real> const& Ez             = Efield[2]->array(mfi);
        Array4<Real const> const& Jr       = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jtheta   = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz       = Jfield[2]->const_array(mfi);
        Array4<Real const> const& enE      = enE_nodal_mf.const_array(mfi);
        Array4<Real const> const& rho      = rhofield.const_array(mfi);
        Array4<Real const> const& Pe       = Pefield.const_array(mfi);

        amrex::Array4<int> update_Er_arr, update_Etheta_arr, update_Ez_arr;
        if (EB::enabled()) {
            update_Er_arr     = eb_update_E[0]->array(mfi);
            update_Etheta_arr = eb_update_E[1]->array(mfi);
            update_Ez_arr     = eb_update_E[2]->array(mfi);
        }

        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        int const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        Real const dr = m_dr;
        Real const rmin = m_rmin;

        Box const& ter = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tet = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez = mfi.tilebox(Efield[2]->ixType().toIntVect());

        amrex::ParallelFor(ter, tet, tez,

            // --------------------------- E_r ---------------------------
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) {

                if (update_Er_arr && update_Er_arr(i, j, 0) == 0) { return; }

                // Modal inputs interpolated to the E_r stagger.
                Real enE_r_m[MAX_PSEUDO_SPECTRAL_NMODES * 2];
                Real rho_m  [MAX_PSEUDO_SPECTRAL_NMODES * 2];
                Real gPr_m  [MAX_PSEUDO_SPECTRAL_NMODES * 2];
                Real Jr_m   [MAX_PSEUDO_SPECTRAL_NMODES * 2];
                Real Jt_m   [MAX_PSEUDO_SPECTRAL_NMODES * 2];
                Real Jz_m   [MAX_PSEUDO_SPECTRAL_NMODES * 2];

                for (int c = 0; c < ncomps; ++c) {
                    enE_r_m[c] = Interp(enE, nodal, Er_stag, coarsen, i, j, 0, 0*ncomps + c);
                    rho_m  [c] = Interp(rho, nodal, Er_stag, coarsen, i, j, 0, c);
                    // grad P_e radial component: d/dr of nodal P_e mode-by-mode.
                    gPr_m  [c] = T_Algo::UpwardDr(Pe, coefs_r, n_coefs_r, i, j, 0, c);
                    Jr_m   [c] = Jr(i, j, 0, c);
                    Jt_m   [c] = Interp(Jtheta, Jtheta_stag, Er_stag, coarsen, i, j, 0, c);
                    Jz_m   [c] = Interp(Jz,     Jz_stag,     Er_stag, coarsen, i, j, 0, c);
                }

                // Inverse FT to theta-grid.
                Real enE_r_th[MAX_PSEUDO_SPECTRAL_NTHETA];
                Real rho_th  [MAX_PSEUDO_SPECTRAL_NTHETA];
                Real gPr_th  [MAX_PSEUDO_SPECTRAL_NTHETA];
                Real Jr_th   [MAX_PSEUDO_SPECTRAL_NTHETA];
                Real Jt_th   [MAX_PSEUDO_SPECTRAL_NTHETA];
                Real Jz_th   [MAX_PSEUDO_SPECTRAL_NTHETA];
                inverse_az_ft(nmodes, n_theta, enE_r_m, enE_r_th);
                inverse_az_ft(nmodes, n_theta, rho_m,   rho_th);
                inverse_az_ft(nmodes, n_theta, gPr_m,   gPr_th);
                inverse_az_ft(nmodes, n_theta, Jr_m,    Jr_th);
                inverse_az_ft(nmodes, n_theta, Jt_m,    Jt_th);
                inverse_az_ft(nmodes, n_theta, Jz_m,    Jz_th);

                Real Er_th[MAX_PSEUDO_SPECTRAL_NTHETA];
                for (int k = 0; k < n_theta; ++k) {
                    if (rho_th[k] < rho_floor && holmstrom_vacuum_region) {
                        Er_th[k] = 0._rt;
                        continue;
                    }
                    const Real rho_lim = amrex::max(rho_th[k], rho_floor);
                    const Real grad_Pe_k = solve_for_Faraday ? 0._rt : gPr_th[k];
                    Er_th[k] = (enE_r_th[k] - grad_Pe_k) / rho_lim;

                    if (solve_for_Faraday) {
                        Real jmag = 0._rt;
                        if (resistivity_has_J_dependence) {
                            jmag = std::sqrt(Jr_th[k]*Jr_th[k]
                                           + Jt_th[k]*Jt_th[k]
                                           + Jz_th[k]*Jz_th[k]);
                        }
                        Er_th[k] += eta(rho_th[k], jmag) * Jr_th[k];
                    }
                }

                Real Er_m[MAX_PSEUDO_SPECTRAL_NMODES * 2];
                forward_az_ft(nmodes, n_theta, Er_th, Er_m);
                for (int c = 0; c < ncomps; ++c) {
                    Er(i, j, 0, c) = Er_m[c];
                }
            },

            // -------------------------- E_theta ------------------------
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) {

                if (update_Etheta_arr && update_Etheta_arr(i, j, 0) == 0) { return; }

                // r at the E_theta stagger (nodal in r): i=0 is on axis.
                const Real r = rmin + i * dr;

                if (r < 0.5_rt * dr) {
                    // On axis: V_theta,0 = 0, and for m >= 2 V_theta,m = 0.
                    // The m=1 cross-coupling with V_r is enforced elsewhere
                    // (Ampere/Faraday) via linear-in-r extrapolation, so we
                    // zero all modes here to match the existing m=0 behavior.
                    for (int c = 0; c < ncomps; ++c) { Etheta(i, j, 0, c) = 0._rt; }
                    return;
                }

                Real enE_t_m[MAX_PSEUDO_SPECTRAL_NMODES * 2];
                Real rho_m  [MAX_PSEUDO_SPECTRAL_NMODES * 2];
                Real gPt_m  [MAX_PSEUDO_SPECTRAL_NMODES * 2];
                Real Jr_m   [MAX_PSEUDO_SPECTRAL_NMODES * 2];
                Real Jt_m   [MAX_PSEUDO_SPECTRAL_NMODES * 2];
                Real Jz_m   [MAX_PSEUDO_SPECTRAL_NMODES * 2];

                // P_e at E_theta stagger (needed for the (-i m / r) factor).
                Real Pe_m   [MAX_PSEUDO_SPECTRAL_NMODES * 2];

                for (int c = 0; c < ncomps; ++c) {
                    enE_t_m[c] = Interp(enE, nodal, Etheta_stag, coarsen, i, j, 0, 1*ncomps + c);
                    rho_m  [c] = Interp(rho, nodal, Etheta_stag, coarsen, i, j, 0, c);
                    Pe_m   [c] = Interp(Pe,  nodal, Etheta_stag, coarsen, i, j, 0, c);
                    Jr_m   [c] = Interp(Jr,     Jr_stag,     Etheta_stag, coarsen, i, j, 0, c);
                    Jt_m   [c] = Jtheta(i, j, 0, c);
                    Jz_m   [c] = Interp(Jz,     Jz_stag,     Etheta_stag, coarsen, i, j, 0, c);
                }

                // grad P_e theta-component: in modes it is (-i m P_m) / r.
                // The result of (-i*m) acting on (Re + i Im) is (m*Im - i*m*Re),
                // so stored Re <- m*Im_input, Im <- -m*Re_input, divided by r.
                gPt_m[0] = 0._rt; // d/dtheta of mode 0 is zero
                for (int m = 1; m < nmodes; ++m) {
                    const Real Pe_re = Pe_m[2*m - 1];
                    const Real Pe_im = Pe_m[2*m    ];
                    gPt_m[2*m - 1] =  (m * Pe_im) / r;
                    gPt_m[2*m    ] = -(m * Pe_re) / r;
                }

                Real enE_t_th[MAX_PSEUDO_SPECTRAL_NTHETA];
                Real rho_th  [MAX_PSEUDO_SPECTRAL_NTHETA];
                Real gPt_th  [MAX_PSEUDO_SPECTRAL_NTHETA];
                Real Jr_th   [MAX_PSEUDO_SPECTRAL_NTHETA];
                Real Jt_th   [MAX_PSEUDO_SPECTRAL_NTHETA];
                Real Jz_th   [MAX_PSEUDO_SPECTRAL_NTHETA];
                inverse_az_ft(nmodes, n_theta, enE_t_m, enE_t_th);
                inverse_az_ft(nmodes, n_theta, rho_m,   rho_th);
                inverse_az_ft(nmodes, n_theta, gPt_m,   gPt_th);
                inverse_az_ft(nmodes, n_theta, Jr_m,    Jr_th);
                inverse_az_ft(nmodes, n_theta, Jt_m,    Jt_th);
                inverse_az_ft(nmodes, n_theta, Jz_m,    Jz_th);

                Real Et_th[MAX_PSEUDO_SPECTRAL_NTHETA];
                for (int k = 0; k < n_theta; ++k) {
                    if (rho_th[k] < rho_floor && holmstrom_vacuum_region) {
                        Et_th[k] = 0._rt;
                        continue;
                    }
                    const Real rho_lim = amrex::max(rho_th[k], rho_floor);
                    const Real grad_Pe_k = solve_for_Faraday ? 0._rt : gPt_th[k];
                    Et_th[k] = (enE_t_th[k] - grad_Pe_k) / rho_lim;

                    if (solve_for_Faraday) {
                        Real jmag = 0._rt;
                        if (resistivity_has_J_dependence) {
                            jmag = std::sqrt(Jr_th[k]*Jr_th[k]
                                           + Jt_th[k]*Jt_th[k]
                                           + Jz_th[k]*Jz_th[k]);
                        }
                        Et_th[k] += eta(rho_th[k], jmag) * Jt_th[k];
                    }
                }

                Real Et_m[MAX_PSEUDO_SPECTRAL_NMODES * 2];
                forward_az_ft(nmodes, n_theta, Et_th, Et_m);
                for (int c = 0; c < ncomps; ++c) {
                    Etheta(i, j, 0, c) = Et_m[c];
                }
            },

            // --------------------------- E_z ---------------------------
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) {

                if (update_Ez_arr && update_Ez_arr(i, j, 0) == 0) { return; }

                Real enE_z_m[MAX_PSEUDO_SPECTRAL_NMODES * 2];
                Real rho_m  [MAX_PSEUDO_SPECTRAL_NMODES * 2];
                Real gPz_m  [MAX_PSEUDO_SPECTRAL_NMODES * 2];
                Real Jr_m   [MAX_PSEUDO_SPECTRAL_NMODES * 2];
                Real Jt_m   [MAX_PSEUDO_SPECTRAL_NMODES * 2];
                Real Jz_m   [MAX_PSEUDO_SPECTRAL_NMODES * 2];

                for (int c = 0; c < ncomps; ++c) {
                    enE_z_m[c] = Interp(enE, nodal, Ez_stag, coarsen, i, j, 0, 2*ncomps + c);
                    rho_m  [c] = Interp(rho, nodal, Ez_stag, coarsen, i, j, 0, c);
                    gPz_m  [c] = T_Algo::UpwardDz(Pe, coefs_z, n_coefs_z, i, j, 0, c);
                    Jr_m   [c] = Interp(Jr,     Jr_stag,     Ez_stag, coarsen, i, j, 0, c);
                    Jt_m   [c] = Interp(Jtheta, Jtheta_stag, Ez_stag, coarsen, i, j, 0, c);
                    Jz_m   [c] = Jz(i, j, 0, c);
                }

                // Ez stagger is nodal in r, so i=0 is on the axis. V_z behaves
                // as a scalar: m=0 free, m>=1 zero on axis.
                const Real r = rmin + i * dr;
                if (r < 0.5_rt * dr) {
                    apply_axis_bc_scalar(nmodes, enE_z_m);
                    apply_axis_bc_scalar(nmodes, gPz_m);
                    apply_axis_bc_scalar(nmodes, Jz_m);
                    apply_axis_bc_scalar(nmodes, rho_m);
                    // Note: at i=0, rho_m[0] (m=0) is still non-zero; the axis
                    // BC only zeros m >= 1 contributions. The pointwise math
                    // proceeds normally for m=0.
                }

                Real enE_z_th[MAX_PSEUDO_SPECTRAL_NTHETA];
                Real rho_th  [MAX_PSEUDO_SPECTRAL_NTHETA];
                Real gPz_th  [MAX_PSEUDO_SPECTRAL_NTHETA];
                Real Jr_th   [MAX_PSEUDO_SPECTRAL_NTHETA];
                Real Jt_th   [MAX_PSEUDO_SPECTRAL_NTHETA];
                Real Jz_th   [MAX_PSEUDO_SPECTRAL_NTHETA];
                inverse_az_ft(nmodes, n_theta, enE_z_m, enE_z_th);
                inverse_az_ft(nmodes, n_theta, rho_m,   rho_th);
                inverse_az_ft(nmodes, n_theta, gPz_m,   gPz_th);
                inverse_az_ft(nmodes, n_theta, Jr_m,    Jr_th);
                inverse_az_ft(nmodes, n_theta, Jt_m,    Jt_th);
                inverse_az_ft(nmodes, n_theta, Jz_m,    Jz_th);

                Real Ez_th[MAX_PSEUDO_SPECTRAL_NTHETA];
                for (int k = 0; k < n_theta; ++k) {
                    if (rho_th[k] < rho_floor && holmstrom_vacuum_region) {
                        Ez_th[k] = 0._rt;
                        continue;
                    }
                    const Real rho_lim = amrex::max(rho_th[k], rho_floor);
                    const Real grad_Pe_k = solve_for_Faraday ? 0._rt : gPz_th[k];
                    Ez_th[k] = (enE_z_th[k] - grad_Pe_k) / rho_lim;

                    if (solve_for_Faraday) {
                        Real jmag = 0._rt;
                        if (resistivity_has_J_dependence) {
                            jmag = std::sqrt(Jr_th[k]*Jr_th[k]
                                           + Jt_th[k]*Jt_th[k]
                                           + Jz_th[k]*Jz_th[k]);
                        }
                        Ez_th[k] += eta(rho_th[k], jmag) * Jz_th[k];
                    }
                }

                Real Ez_m[MAX_PSEUDO_SPECTRAL_NMODES * 2];
                forward_az_ft(nmodes, n_theta, Ez_th, Ez_m);
                for (int c = 0; c < ncomps; ++c) {
                    Ez(i, j, 0, c) = Ez_m[c];
                }
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

#elif defined(WARPX_DIM_RSPHERE)
template<typename T_Algo>
void FiniteDifferenceSolver::HybridPICSolveESpherical (
    ablastr::fields::VectorField const& /*Efield*/,
    ablastr::fields::VectorField const& /*Jfield*/,
    ablastr::fields::VectorField const& /*Jifield*/,
    ablastr::fields::VectorField const& /*Bfield*/,
    amrex::MultiFab const& /*rhofield*/,
    amrex::MultiFab const& /*Pefield*/,
    int /*lev*/, HybridPICModel const* /*hybrid_model*/,
    const bool /*solve_for_Faraday*/ )
{
    WARPX_ABORT_WITH_MESSAGE("HybridPICSolveESphrical not fully implemented");
}
#else

template<typename T_Algo>
void FiniteDifferenceSolver::HybridPICSolveECartesian (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& Jfield,
    ablastr::fields::VectorField const& Jifield,
    ablastr::fields::VectorField const& Bfield,
    amrex::MultiFab const& rhofield,
    amrex::MultiFab const& Pefield,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 > const& eb_update_E,
    int lev, HybridPICModel const* hybrid_model,
    const bool solve_for_Faraday )
{
    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    using namespace ablastr::coarsen::sample;

    // get hybrid model parameters
    const auto eta = hybrid_model->m_eta;
    const auto eta_h = hybrid_model->m_eta_h;
    const auto rho_floor = hybrid_model->m_n_floor * PhysConst::q_e;
    const auto resistivity_has_J_dependence = hybrid_model->m_resistivity_has_J_dependence;
    const auto hyper_resistivity_has_B_dependence = hybrid_model->m_hyper_resistivity_has_B_dependence;
    const bool include_hyper_resistivity_term = hybrid_model->m_include_hyper_resistivity_term;

    const bool include_external_fields = hybrid_model->m_add_external_fields;

    const bool holmstrom_vacuum_region = hybrid_model->m_holmstrom_vacuum_region;

    auto & warpx = WarpX::GetInstance();
    ablastr::fields::VectorField Bfield_external, Efield_external;
    if (include_external_fields) {
        Bfield_external = warpx.m_fields.get_alldirs(FieldType::hybrid_B_fp_external, 0); // lev=0
        Efield_external = warpx.m_fields.get_alldirs(FieldType::hybrid_E_fp_external, 0); // lev=0
    }

    // Index type required for interpolating fields from their respective
    // staggering to the Ex, Ey, Ez locations
    amrex::GpuArray<int, 3> const& Ex_stag = hybrid_model->Ex_IndexType;
    amrex::GpuArray<int, 3> const& Ey_stag = hybrid_model->Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_stag = hybrid_model->Ez_IndexType;
    amrex::GpuArray<int, 3> const& Jx_stag = hybrid_model->Jx_IndexType;
    amrex::GpuArray<int, 3> const& Jy_stag = hybrid_model->Jy_IndexType;
    amrex::GpuArray<int, 3> const& Jz_stag = hybrid_model->Jz_IndexType;
    amrex::GpuArray<int, 3> const& Bx_stag = hybrid_model->Bx_IndexType;
    amrex::GpuArray<int, 3> const& By_stag = hybrid_model->By_IndexType;
    amrex::GpuArray<int, 3> const& Bz_stag = hybrid_model->Bz_IndexType;

    // Parameters for `interp` that maps from Yee to nodal mesh and back
    amrex::GpuArray<int, 3> const& nodal = {1, 1, 1};
    // The "coarsening is just 1 i.e. no coarsening"
    amrex::GpuArray<int, 3> const& coarsen = {1, 1, 1};

    // The E-field calculation is done in 2 steps:
    // 1) The J x B term is calculated on a nodal mesh in order to ensure
    //    energy conservation.
    // 2) The nodal E-field values are averaged onto the Yee grid and the
    //    electron pressure & resistivity terms are added (these terms are
    //    naturally located on the Yee grid).

    // Create a temporary multifab to hold the nodal E-field values
    // Note the multifab has 3 values for Ex, Ey and Ez which we can do here
    // since all three components will be calculated on the same grid.
    // Also note that enE_nodal_mf does not need to have any guard cells since
    // these values will be interpolated to the Yee mesh which is contained
    // by the nodal mesh.
    auto const& ba = convert(rhofield.boxArray(), IntVect::TheNodeVector());
    MultiFab enE_nodal_mf(ba, rhofield.DistributionMap(), 3, IntVect::TheZeroVector());

    // Loop through the grids, and over the tiles within each grid for the
    // initial, nodal calculation of E
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(enE_nodal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        Array4<Real> const& enE_nodal = enE_nodal_mf.array(mfi);
        Array4<Real const> const& Jx = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jy = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& Jix = Jifield[0]->const_array(mfi);
        Array4<Real const> const& Jiy = Jifield[1]->const_array(mfi);
        Array4<Real const> const& Jiz = Jifield[2]->const_array(mfi);
        Array4<Real const> const& Bx = Bfield[0]->const_array(mfi);
        Array4<Real const> const& By = Bfield[1]->const_array(mfi);
        Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);

        Array4<Real> Bx_ext, By_ext, Bz_ext;
        if (include_external_fields) {
            Bx_ext = Bfield_external[0]->array(mfi);
            By_ext = Bfield_external[1]->array(mfi);
            Bz_ext = Bfield_external[2]->array(mfi);
        }

        // Loop over the cells and update the nodal E field
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k){

            // interpolate the total plasma current to a nodal grid
            auto const jx_interp = Interp(Jx, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto const jy_interp = Interp(Jy, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto const jz_interp = Interp(Jz, Jz_stag, nodal, coarsen, i, j, k, 0);

            // interpolate the ion current to a nodal grid
            auto const jix_interp = Interp(Jix, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto const jiy_interp = Interp(Jiy, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto const jiz_interp = Interp(Jiz, Jz_stag, nodal, coarsen, i, j, k, 0);

            // interpolate the B field to a nodal grid
            auto Bx_interp = Interp(Bx, Bx_stag, nodal, coarsen, i, j, k, 0);
            auto By_interp = Interp(By, By_stag, nodal, coarsen, i, j, k, 0);
            auto Bz_interp = Interp(Bz, Bz_stag, nodal, coarsen, i, j, k, 0);

            if (include_external_fields) {
                Bx_interp += Interp(Bx_ext, Bx_stag, nodal, coarsen, i, j, k, 0);
                By_interp += Interp(By_ext, By_stag, nodal, coarsen, i, j, k, 0);
                Bz_interp += Interp(Bz_ext, Bz_stag, nodal, coarsen, i, j, k, 0);
            }

            // calculate enE = (J - Ji) x B
            enE_nodal(i, j, k, 0) = (
                (jy_interp - jiy_interp) * Bz_interp
                - (jz_interp - jiz_interp) * By_interp
            );
            enE_nodal(i, j, k, 1) = (
                (jz_interp - jiz_interp) * Bx_interp
                - (jx_interp - jix_interp) * Bz_interp
            );
            enE_nodal(i, j, k, 2) = (
                (jx_interp - jix_interp) * By_interp
                - (jy_interp - jiy_interp) * Bx_interp
            );
        });

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    // Loop through the grids, and over the tiles within each grid again
    // for the Yee grid calculation of the E field
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const& Ex = Efield[0]->array(mfi);
        Array4<Real> const& Ey = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real const> const& Jx = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jy = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& enE = enE_nodal_mf.const_array(mfi);
        Array4<Real const> const& rho = rhofield.const_array(mfi);
        Array4<Real const> const& Pe = Pefield.array(mfi);
        Array4<Real> const& Bx = Bfield[0]->array(mfi);
        Array4<Real> const& By = Bfield[1]->array(mfi);
        Array4<Real> const& Bz = Bfield[2]->array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries
        amrex::Array4<int> update_Ex_arr, update_Ey_arr, update_Ez_arr;
        if (EB::enabled()) {
            update_Ex_arr = eb_update_E[0]->array(mfi);
            update_Ey_arr = eb_update_E[1]->array(mfi);
            update_Ez_arr = eb_update_E[2]->array(mfi);
        }

        Array4<Real> Ex_ext, Ey_ext, Ez_ext;
        if (include_external_fields) {
            Ex_ext = Efield_external[0]->array(mfi);
            Ey_ext = Efield_external[1]->array(mfi);
            Ez_ext = Efield_external[2]->array(mfi);
        }

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        Box const& tex  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tey  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Loop over the cells and update the E field
        // Ex calculation
        amrex::ParallelFor(tex, [=] AMREX_GPU_DEVICE (int i, int j, int k){

            // Skip field update in the embedded boundaries
            if (update_Ex_arr && update_Ex_arr(i, j, k) == 0) { return; }

            // Interpolate to get the appropriate charge density in space
            const Real rho_val = Interp(rho, nodal, Ex_stag, coarsen, i, j, k, 0);

            if (rho_val < rho_floor && holmstrom_vacuum_region) {
                Ex(i, j, k) = 0._rt;
            } else {
                // Get the gradient of the electron pressure if the longitudinal part of
                // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                const Real grad_Pe = (!solve_for_Faraday) ?
                    T_Algo::UpwardDx(Pe, coefs_x, n_coefs_x, i, j, k)
                    : 0._rt;

                // interpolate the nodal neE values to the Yee grid
                const auto enE_x = Interp(enE, nodal, Ex_stag, coarsen, i, j, k, 0);

                // safety condition since we divide by rho
                const auto rho_val_limited = std::max(rho_val, rho_floor);

                Ex(i, j, k) = (enE_x - grad_Pe) / rho_val_limited;
            }

            // Add resistivity only if E field value is used to update B
            if (solve_for_Faraday) {
                Real jtot_val = 0._rt;
                if (resistivity_has_J_dependence) {
                    // Interpolate current to appropriate staggering to match E field
                    const Real jx_val = Jx(i, j, k);
                    const Real jy_val = Interp(Jy, Jy_stag, Ex_stag, coarsen, i, j, k, 0);
                    const Real jz_val = Interp(Jz, Jz_stag, Ex_stag, coarsen, i, j, k, 0);
                    jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);
                }

                Ex(i, j, k) += eta(rho_val, jtot_val) * Jx(i, j, k);

                if (include_hyper_resistivity_term) {

                    // Interpolate B field to appropriate staggering to match E field
                    Real btot_val = 0._rt;
                    if (hyper_resistivity_has_B_dependence) {
                        const Real bx_val = Interp(Bx, Bx_stag, Ex_stag, coarsen, i, j, k, 0);
                        const Real by_val = Interp(By, By_stag, Ex_stag, coarsen, i, j, k, 0);
                        const Real bz_val = Interp(Bz, Bz_stag, Ex_stag, coarsen, i, j, k, 0);
                        btot_val = std::sqrt(bx_val*bx_val + by_val*by_val + bz_val*bz_val);
                    }

                    auto nabla2Jx = T_Algo::Dxx(Jx, coefs_x, n_coefs_x, i, j, k)
                        + T_Algo::Dyy(Jx, coefs_y, n_coefs_y, i, j, k)
                        + T_Algo::Dzz(Jx, coefs_z, n_coefs_z, i, j, k);

                    Ex(i, j, k) -= eta_h(rho_val, btot_val) * nabla2Jx;
                }
            }

            if (include_external_fields && (rho_val >= rho_floor)) {
                Ex(i, j, k) -= Ex_ext(i, j, k);
            }
        });

        // Ey calculation
        amrex::ParallelFor(tey, [=] AMREX_GPU_DEVICE (int i, int j, int k) {

            // Skip field update in the embedded boundaries
            if (update_Ey_arr && update_Ey_arr(i, j, k) == 0) { return; }

            // Interpolate to get the appropriate charge density in space
            const Real rho_val = Interp(rho, nodal, Ey_stag, coarsen, i, j, k, 0);

            if (rho_val < rho_floor && holmstrom_vacuum_region) {
                Ey(i, j, k) = 0._rt;
            } else {
                // Get the gradient of the electron pressure if the longitudinal part of
                // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                const Real grad_Pe = (!solve_for_Faraday) ?
                    T_Algo::UpwardDy(Pe, coefs_y, n_coefs_y, i, j, k)
                    : 0._rt;

                // interpolate the nodal neE values to the Yee grid
                const auto enE_y = Interp(enE, nodal, Ey_stag, coarsen, i, j, k, 1);

                // safety condition since we divide by rho
                const auto rho_val_limited = std::max(rho_val, rho_floor);

                Ey(i, j, k) = (enE_y - grad_Pe) / rho_val_limited;
            }

            // Add resistivity only if E field value is used to update B
            if (solve_for_Faraday) {
                Real jtot_val = 0._rt;
                if (resistivity_has_J_dependence) {
                    // Interpolate current to appropriate staggering to match E field
                    const Real jx_val = Interp(Jx, Jx_stag, Ey_stag, coarsen, i, j, k, 0);
                    const Real jy_val = Jy(i, j, k);
                    const Real jz_val = Interp(Jz, Jz_stag, Ey_stag, coarsen, i, j, k, 0);
                    jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);
                }

                Ey(i, j, k) += eta(rho_val, jtot_val) * Jy(i, j, k);

                if (include_hyper_resistivity_term) {

                    // Interpolate B field to appropriate staggering to match E field
                    Real btot_val = 0._rt;
                    if (hyper_resistivity_has_B_dependence) {
                        const Real bx_val = Interp(Bx, Bx_stag, Ey_stag, coarsen, i, j, k, 0);
                        const Real by_val = Interp(By, By_stag, Ey_stag, coarsen, i, j, k, 0);
                        const Real bz_val = Interp(Bz, Bz_stag, Ey_stag, coarsen, i, j, k, 0);
                        btot_val = std::sqrt(bx_val*bx_val + by_val*by_val + bz_val*bz_val);
                    }

                    auto nabla2Jy = T_Algo::Dxx(Jy, coefs_x, n_coefs_x, i, j, k)
                        + T_Algo::Dyy(Jy, coefs_y, n_coefs_y, i, j, k)
                        + T_Algo::Dzz(Jy, coefs_z, n_coefs_z, i, j, k);

                    Ey(i, j, k) -= eta_h(rho_val, btot_val) * nabla2Jy;
                }
            }

            if (include_external_fields && (rho_val >= rho_floor)) {
                Ey(i, j, k) -= Ey_ext(i, j, k);
            }
        });

        // Ez calculation
        amrex::ParallelFor(tez, [=] AMREX_GPU_DEVICE (int i, int j, int k){

            // Skip field update in the embedded boundaries
            if (update_Ez_arr && update_Ez_arr(i, j, k) == 0) { return; }

            // Interpolate to get the appropriate charge density in space
            const Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, k, 0);

            if (rho_val < rho_floor && holmstrom_vacuum_region) {
                Ez(i, j, k) = 0._rt;
            } else {
                // Get the gradient of the electron pressure if the longitudinal part of
                // the E-field should be included, otherwise ignore it since curl x (grad Pe) = 0
                const Real grad_Pe = (!solve_for_Faraday) ?
                    T_Algo::UpwardDz(Pe, coefs_z, n_coefs_z, i, j, k)
                    : 0._rt;

                // interpolate the nodal neE values to the Yee grid
                const auto enE_z = Interp(enE, nodal, Ez_stag, coarsen, i, j, k, 2);

                // safety condition since we divide by rho
                const auto rho_val_limited = std::max(rho_val, rho_floor);

                Ez(i, j, k) = (enE_z - grad_Pe) / rho_val_limited;
            }

            // Add resistivity only if E field value is used to update B
            if (solve_for_Faraday) {
                Real jtot_val = 0._rt;
                if (resistivity_has_J_dependence) {
                    // Interpolate current to appropriate staggering to match E field
                    const Real jx_val = Interp(Jx, Jx_stag, Ez_stag, coarsen, i, j, k, 0);
                    const Real jy_val = Interp(Jy, Jy_stag, Ez_stag, coarsen, i, j, k, 0);
                    const Real jz_val = Jz(i, j, k);
                    jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);
                }

                Ez(i, j, k) += eta(rho_val, jtot_val) * Jz(i, j, k);

                if (include_hyper_resistivity_term) {

                    // Interpolate B field to appropriate staggering to match E field
                    Real btot_val = 0._rt;
                    if (hyper_resistivity_has_B_dependence) {
                        const Real bx_val = Interp(Bx, Bx_stag, Ez_stag, coarsen, i, j, k, 0);
                        const Real by_val = Interp(By, By_stag, Ez_stag, coarsen, i, j, k, 0);
                        const Real bz_val = Interp(Bz, Bz_stag, Ez_stag, coarsen, i, j, k, 0);
                        btot_val = std::sqrt(bx_val*bx_val + by_val*by_val + bz_val*bz_val);
                    }

                    auto nabla2Jz = T_Algo::Dxx(Jz, coefs_x, n_coefs_x, i, j, k)
                        + T_Algo::Dyy(Jz, coefs_y, n_coefs_y, i, j, k)
                        + T_Algo::Dzz(Jz, coefs_z, n_coefs_z, i, j, k);

                    Ez(i, j, k) -= eta_h(rho_val, btot_val) * nabla2Jz;
                }
            }

            if (include_external_fields && (rho_val >= rho_floor)) {
                Ez(i, j, k) -= Ez_ext(i, j, k);
            }
        });

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}
#endif
