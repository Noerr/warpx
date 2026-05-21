/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

/**
 * Inline helpers for azimuthal Fourier transforms used by the multi-mode RZ
 * hybrid-PIC kernel and pressure calculation.
 *
 * Storage convention (matching WarpX RZ deposition / gather):
 *   A(r, z, theta) = A_0(r, z)
 *                  + sum_{m >= 1} [ Re(A_m) cos(m theta)
 *                                 + Im(A_m) sin(m theta) ]
 * where the stored modal amplitudes for m >= 1 already absorb the
 * negative-m mirror (factor of 2 vs. the standard bilateral spectrum).
 *
 * `A_modes` is a flat array of (2*nmodes - 1) entries:
 *   [ A_0, Re(A_1), Im(A_1), Re(A_2), Im(A_2), ... ]
 *
 * `A_theta` is a buffer of `n_theta` real-space samples taken on a uniform
 * azimuthal grid theta_k = 2*pi*k / n_theta, for k = 0 .. n_theta - 1.
 */

#ifndef WARPX_HYBRIDPIC_AZIMUTHAL_FT_H_
#define WARPX_HYBRIDPIC_AZIMUTHAL_FT_H_

#include "Utils/WarpXConst.H"

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)

/** Compile-time upper bound on `n_rz_azimuthal_modes` for stack-allocated
 *  theta-grid buffers inside per-cell ParallelFor lambdas. Increase if a
 *  larger truncation is required.
 *
 *  NTHETA = 3 * NMODES: enforces the Orszag 2/3 de-aliasing rule. The
 *  pseudo-spectral round-trip evaluates the nonlinear Ohm's-law RHS on a
 *  theta-grid that is 3x larger than the stored mode count, so the
 *  trustworthy band (lower 2/3 of grid modes) covers all N stored modes.
 *  Modes beyond N are implicitly discarded by the forward-FT write-back
 *  loop, which only stores indices [0, 2*N - 1]. */
constexpr int MAX_PSEUDO_SPECTRAL_NMODES = 8;
constexpr int MAX_PSEUDO_SPECTRAL_NTHETA = 3 * MAX_PSEUDO_SPECTRAL_NMODES;

/** Inverse azimuthal FT: modal storage -> n_theta real-space samples. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void inverse_az_ft (int nmodes, int n_theta,
                    amrex::Real const* AMREX_RESTRICT A_modes,
                    amrex::Real* AMREX_RESTRICT A_theta)
{
    for (int k = 0; k < n_theta; ++k) {
        const amrex::Real theta_k =
            (amrex::Real{2} * MathConst::pi * k) / n_theta;
        amrex::Real val = A_modes[0];
        for (int m = 1; m < nmodes; ++m) {
            const amrex::Real c = std::cos(m * theta_k);
            const amrex::Real s = std::sin(m * theta_k);
            val += A_modes[2*m - 1] * c + A_modes[2*m] * s;
        }
        A_theta[k] = val;
    }
}

/** Forward azimuthal FT: n_theta real-space samples -> modal storage.
 *  A_0    = (1 / n_theta) * sum_k A(theta_k)
 *  A_m_re = (2 / n_theta) * sum_k A(theta_k) cos(m theta_k)  (m >= 1)
 *  A_m_im = (2 / n_theta) * sum_k A(theta_k) sin(m theta_k)  (m >= 1)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void forward_az_ft (int nmodes, int n_theta,
                    amrex::Real const* AMREX_RESTRICT A_theta,
                    amrex::Real* AMREX_RESTRICT A_modes)
{
    amrex::Real a0 = amrex::Real{0};
    for (int k = 0; k < n_theta; ++k) { a0 += A_theta[k]; }
    A_modes[0] = a0 / n_theta;
    for (int m = 1; m < nmodes; ++m) {
        amrex::Real ar = amrex::Real{0};
        amrex::Real ai = amrex::Real{0};
        for (int k = 0; k < n_theta; ++k) {
            const amrex::Real theta_k =
                (amrex::Real{2} * MathConst::pi * k) / n_theta;
            ar += A_theta[k] * std::cos(m * theta_k);
            ai += A_theta[k] * std::sin(m * theta_k);
        }
        A_modes[2*m - 1] = (amrex::Real{2} * ar) / n_theta;
        A_modes[2*m    ] = (amrex::Real{2} * ai) / n_theta;
    }
}

/** On-axis modal BC for scalar fields (rho, P_e): smoothness in Cartesian
 *  coords requires modes m >= 1 to vanish as r^m near the axis, so at r=0
 *  they are exactly zero. Mode 0 is unconstrained. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void apply_axis_bc_scalar (int nmodes, amrex::Real* AMREX_RESTRICT A_modes)
{
    for (int m = 1; m < nmodes; ++m) {
        A_modes[2*m - 1] = amrex::Real{0};
        A_modes[2*m    ] = amrex::Real{0};
    }
}

/** On-axis modal BC for the (r, theta) components of a vector field.
 *
 *  Regularity of V_x +/- i V_y = (V_r +/- i V_theta) e^{+/- i theta} in
 *  Cartesian coordinates requires V_r,0(0) = V_theta,0(0) = 0, and for m=1
 *  the cross-coupling Re(V_r,1) = -Im(V_theta,1), Im(V_r,1) = +Re(V_theta,1).
 *  For m >= 2 both components vanish on axis.
 *
 *  Implementation: we only enforce modes m >= 1 here. Mode 0 is intentionally
 *  *not* forced to zero -- the existing m=0 hybrid kernel relies on the small
 *  non-zero values that come from interpolating Yee-staggered J and B through
 *  ghost cells at i=-1, and overriding them with exact zero diverges from
 *  that reference behavior. For m >= 1, both V_r and V_theta are zeroed
 *  (the m=1 cross-coupling is recovered by neighbor cells via linear-in-r
 *  extrapolation in the surrounding Faraday / Ampere kernels). */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void apply_axis_bc_vector_rt (int nmodes,
                              amrex::Real* AMREX_RESTRICT Vr_modes,
                              amrex::Real* AMREX_RESTRICT Vt_modes)
{
    for (int m = 1; m < nmodes; ++m) {
        Vr_modes[2*m - 1] = amrex::Real{0};
        Vr_modes[2*m    ] = amrex::Real{0};
        Vt_modes[2*m - 1] = amrex::Real{0};
        Vt_modes[2*m    ] = amrex::Real{0};
    }
}

#endif // defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)

#endif // WARPX_HYBRIDPIC_AZIMUTHAL_FT_H_
