#ifndef WARPX_JFUNCTOR_H_
#define WARPX_JFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <AMReX_BaseFwd.H>

/**
 * \brief Functor to cell-center MF for current density and store result in mf_out.
 */
class JFunctor final : public ComputeDiagFunctor
{
public:
    /** Constructor.
     *
     * \param[in] dir direction of vector field to operate on
     * \param[in] lev level of multifab. Used for averaging in rz.
     * \param[in] crse_ratio for interpolating field values from the simulation MultiFab, src_mf,
                  to the output diagnostic MultiFab, mf_dst.
     * \param[in] convertRZmodes2cartesian (in cylindrical) whether to
     *            sum all modes in mf_src before cell-centering into dst multifab.
     * \param[in] deposit_current whether to deposit current. Used to output current in cases
                  where no field solve is used. Ignored when species_index >= 0 (per-species
                  output always redeposits fresh into local temporaries).
     * \param[in] species_index index of species to dump j per species. Defaults to -1
     *            (dump the total j from current_fp). When >= 0, this functor allocates
     *            three local MultiFabs at level lev, redeposits the requested species's
     *            current into them, and interpolates the dir component into mf_dst.
     * \param[in] ncomp Number of component of mf_src to cell-center in dst multifab.
     */
    JFunctor (int dir, int lev,
             amrex::IntVect crse_ratio,
             bool convertRZmodes2cartesian=true,
             bool deposit_current=false,
             int species_index=-1,
             int ncomp=1);
    /** \brief Cell-center m_mf_src and write the result in mf_dst.
     *
     * In cylindrical geometry, by default this functor average all components
     * of a MultiFab and writes into one single component.
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in] dcomp first component of mf_dst in which cell-centered
     *            data is stored
     */
    void operator()(amrex::MultiFab& mf_dst, int dcomp, int /*i_buffer=0*/) const override;
private:
    /** direction of the current density to save */
    int m_dir;
    /** level on which mf_src is defined */
    int m_lev;
    /** (for cylindrical) whether to average all modes into 1 comp */
    bool m_convertRZmodes2cartesian;
    /** whether to deposit current density before saving */
    bool m_deposit_current;
    /** species index for per-species output; -1 means total j from current_fp */
    int m_species_index;
};

#endif // WARPX_JFUNCTOR_H_
