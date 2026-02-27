#ifndef WARPX_HEATFLUXFUNCTOR_H_
#define WARPX_HEATFLUXFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <AMReX_BaseFwd.H>

/**
 * \brief Functor to calculate per-cell heat flux vector from particle data.
 *
 * Computes the 3 components of the heat flux vector
 * Q_i = (mass / 2) * <|u - <u>|^2 * (u_i - <u_i>)> / q_e  [eV * c]
 * where u = gamma*v is the 4-velocity.
 *
 * Output components (in order): Qx, Qy, Qz
 */
class HeatFluxFunctor final : public ComputeDiagFunctor
{
public:
    /** \brief Constructor
     * \param[in] lev level of multifab
     * \param[in] crse_ratio for interpolating field values from simulation MultiFabs
                  to the output diagnostic MultiFab mf_dst
     * \param[in] ispec index of the species over which to calculate the heat flux
     * \param[in] ncomp Number of components (must be 3 for the heat flux vector)
     */
    HeatFluxFunctor(int lev, amrex::IntVect crse_ratio, int ispec, int ncomp=3);

    /** \brief Compute the heat flux in each grid cell.
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in] dcomp first component of mf_dst in which cell-centered
     *            data is stored
     */
    void operator()(amrex::MultiFab& mf_dst, int dcomp, int /*i_buffer=0*/) const override;
private:
    int m_lev; /**< level on which mf_src is defined */
    int m_ispec; /**< index of species to average over */
};

#endif // WARPX_HEATFLUXFUNCTOR_H_
