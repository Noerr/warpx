/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_HYBRIDPIC_AZIMUTHAL_FT_H_
#define WARPX_HYBRIDPIC_AZIMUTHAL_FT_H_

#include "Utils/WarpXConst.H"

#include <AMReX_Array.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

/**
 * Azimuthal Fourier transforms used by the multimode RZ hybrid-PIC solver to
 * evaluate the nonlinear terms of Ohm's law and of the electron pressure
 * closure on a grid of azimuthal samples.
 *
 * Storage convention (as in the RZ deposition and gather):
 *   A(r, z, theta) = A_0 + sum_{m >= 1} [ Re(A_m) cos(m theta) + Im(A_m) sin(m theta) ]
 * with the modes stored as the flat array
 *   [ A_0, Re(A_1), Im(A_1), Re(A_2), Im(A_2), ... ]  (2*nmodes - 1 entries).
 *
 * The samples are taken at theta_k = 2 pi k / n_theta, k = 0 .. n_theta - 1.
 */

/** Compile-time bound on n_rz_azimuthal_modes for the multimode hybrid solver
 *  (sizes the per-cell stack buffers). */
constexpr int MAX_PSEUDO_SPECTRAL_NMODES = 8;

/** Number of azimuthal samples used for nmodes stored modes: 3*nmodes keeps
 *  the quadratic products of the stored modes (e.g. J x B) free of aliasing
 *  into the stored modes. Non-polynomial terms (1/rho, rho^gamma) are not
 *  alias-free; their aliasing error is small when the m >= 1 perturbation is. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr int HybridPICAzimuthalNTheta (int nmodes) { return 3 * nmodes; }

constexpr int MAX_PSEUDO_SPECTRAL_NTHETA = HybridPICAzimuthalNTheta(MAX_PSEUDO_SPECTRAL_NMODES);

/** cos(m theta_k) and sin(m theta_k) for the n_theta samples and the stored
 *  modes, tabulated once on the host and captured by value in the kernels. */
struct AzimuthalFTTable
{
    int nmodes = 1;
    int n_theta = 1;
    amrex::GpuArray<amrex::Real, MAX_PSEUDO_SPECTRAL_NMODES*MAX_PSEUDO_SPECTRAL_NTHETA> cos_mt{};
    amrex::GpuArray<amrex::Real, MAX_PSEUDO_SPECTRAL_NMODES*MAX_PSEUDO_SPECTRAL_NTHETA> sin_mt{};

    explicit AzimuthalFTTable (int a_nmodes)
        : nmodes{a_nmodes}, n_theta{HybridPICAzimuthalNTheta(a_nmodes)}
    {
        for (int m = 0; m < nmodes; ++m) {
            for (int k = 0; k < n_theta; ++k) {
                const amrex::Real theta = (amrex::Real{2} * MathConst::pi * k) / n_theta;
                cos_mt[m*n_theta + k] = std::cos(m * theta);
                sin_mt[m*n_theta + k] = std::sin(m * theta);
            }
        }
    }
};

/** Modes -> n_theta azimuthal samples. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void inverse_az_ft (AzimuthalFTTable const& tab,
                    amrex::Real const* AMREX_RESTRICT A_modes,
                    amrex::Real* AMREX_RESTRICT A_theta)
{
    for (int k = 0; k < tab.n_theta; ++k) {
        amrex::Real val = A_modes[0];
        for (int m = 1; m < tab.nmodes; ++m) {
            val += A_modes[2*m - 1] * tab.cos_mt[m*tab.n_theta + k]
                 + A_modes[2*m    ] * tab.sin_mt[m*tab.n_theta + k];
        }
        A_theta[k] = val;
    }
}

/** n_theta azimuthal samples -> the nmodes stored modes; higher harmonics of
 *  the samples are discarded. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void forward_az_ft (AzimuthalFTTable const& tab,
                    amrex::Real const* AMREX_RESTRICT A_theta,
                    amrex::Real* AMREX_RESTRICT A_modes)
{
    const int n_theta = tab.n_theta;
    auto a0 = amrex::Real{0};
    for (int k = 0; k < n_theta; ++k) { a0 += A_theta[k]; }
    A_modes[0] = a0 / n_theta;
    for (int m = 1; m < tab.nmodes; ++m) {
        auto ar = amrex::Real{0};
        auto ai = amrex::Real{0};
        for (int k = 0; k < n_theta; ++k) {
            ar += A_theta[k] * tab.cos_mt[m*n_theta + k];
            ai += A_theta[k] * tab.sin_mt[m*n_theta + k];
        }
        A_modes[2*m - 1] = (amrex::Real{2} * ar) / n_theta;
        A_modes[2*m    ] = (amrex::Real{2} * ai) / n_theta;
    }
}

/** On-axis value of a scalar (or of the z component of a vector): the m >= 1
 *  modes vanish at r = 0 (they scale as r^m). */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void apply_axis_bc_scalar (int nmodes, amrex::Real* AMREX_RESTRICT A_modes)
{
    for (int m = 1; m < nmodes; ++m) {
        A_modes[2*m - 1] = amrex::Real{0};
        A_modes[2*m    ] = amrex::Real{0};
    }
}

/** On-axis value of the (r, theta) components of a vector field. A uniform
 *  transverse vector (a, b) has V_r = a cos(theta) + b sin(theta) and
 *  V_theta = -a sin(theta) + b cos(theta), i.e. m = 1 with V_theta,1 = -i V_r,1:
 *  Re(V_theta,1) = Im(V_r,1), Im(V_theta,1) = -Re(V_r,1). The m >= 2 modes
 *  vanish on the axis. V_r,1 is kept and V_theta,1 is set from it. Mode 0 is
 *  left as interpolated (its exact on-axis value is zero). */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void apply_axis_bc_vector_rt (int nmodes,
                              amrex::Real* AMREX_RESTRICT Vr_modes,
                              amrex::Real* AMREX_RESTRICT Vt_modes)
{
    if (nmodes > 1) {
        Vt_modes[1] =  Vr_modes[2];
        Vt_modes[2] = -Vr_modes[1];
    }
    for (int m = 2; m < nmodes; ++m) {
        Vr_modes[2*m - 1] = amrex::Real{0};
        Vr_modes[2*m    ] = amrex::Real{0};
        Vt_modes[2*m - 1] = amrex::Real{0};
        Vt_modes[2*m    ] = amrex::Real{0};
    }
}

#endif // WARPX_HYBRIDPIC_AZIMUTHAL_FT_H_
