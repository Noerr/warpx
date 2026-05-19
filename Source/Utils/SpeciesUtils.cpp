/* Copyright 2023 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SpeciesUtils.H"
#include <ablastr/warn_manager/WarnManager.H>
#include "Utils/TextMsg.H"
#include "Utils/Parser/ParserUtils.H"

#include <cmath>

namespace SpeciesUtils {

namespace {
    // Acklam's rational approximation to the inverse standard normal CDF Phi^{-1}(p).
    // Host-only, called once per quantile at injector setup. Absolute error ~6e-9 for
    // p in (0, 1). Source: Peter Acklam,
    // https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
    double acklam_inv_normal_cdf (double p)
    {
        constexpr double a[6] = {
            -3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00};
        constexpr double b[5] = {
            -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01};
        constexpr double c[6] = {
            -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
            -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00};
        constexpr double d[4] = {
             7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
             3.754408661907416e+00};
        constexpr double p_low = 0.02425;
        constexpr double p_high = 1.0 - p_low;
        double q, r;
        if (p < p_low) {
            q = std::sqrt(-2.0 * std::log(p));
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1.0);
        } else if (p <= p_high) {
            q = p - 0.5;
            r = q * q;
            return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q /
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r + 1.0);
        } else {
            q = std::sqrt(-2.0 * std::log(1.0 - p));
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1.0);
        }
    }
}

    void StringParseAbortMessage(const std::string& var,
                                 const std::string& name) {
        std::stringstream stringstream;
        std::string string;
        stringstream << var << " string '" << name << "' not recognized.";
        string = stringstream.str();
        WARPX_ABORT_WITH_MESSAGE(string);
    }

    void extractSpeciesProperties (std::string const& species_name,
        std::string const& injection_style, amrex::Real& charge, amrex::Real& mass,
        PhysicalSpecies& physical_species )
    {
        const amrex::ParmParse pp_species_name(species_name);
        std::string physical_species_s;
        const bool species_is_specified = pp_species_name.query("species_type", physical_species_s);
        if (species_is_specified){
            const auto physical_species_from_string = species::from_string( physical_species_s );
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(physical_species_from_string,
                physical_species_s + " does not exist!");
            physical_species = physical_species_from_string.value();
            charge = species::get_charge( physical_species );
            mass = species::get_mass( physical_species );
        }

        // parse charge and mass
        const bool charge_is_specified =
            utils::parser::queryWithParser(pp_species_name, "charge", charge);
        const bool mass_is_specified =
            utils::parser::queryWithParser(pp_species_name, "mass", mass);

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            charge_is_specified ||
            species_is_specified ||
            (injection_style == "external_file"),
            "Need to specify at least one of species_type or charge for species '" +
            species_name + "'."
        );

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            mass_is_specified ||
            species_is_specified ||
            (injection_style == "external_file"),
            "Need to specify at least one of species_type or mass for species '" +
            species_name + "'."
        );

        if ( charge_is_specified && species_is_specified ){
            ablastr::warn_manager::WMRecordWarning("Species",
                "Both '" + species_name +  ".charge' and " +
                    species_name + ".species_type' are specified.\n" +
                    species_name + ".charge' will take precedence.\n");

        }

        if ( mass_is_specified && species_is_specified ){
            ablastr::warn_manager::WMRecordWarning("Species",
                "Both '" + species_name +  ".mass' and " +
                    species_name + ".species_type' are specified.\n" +
                    species_name + ".mass' will take precedence.\n");
        }
    }

    // Depending on injection type at runtime, initialize inj_rho
    // so that inj_rho->getDensity calls
    // InjectorPosition[Constant or Predefined or etc.].getDensity.
    void parseDensity (std::string const& species_name, std::string const& source_name,
        std::unique_ptr<InjectorDensity,InjectorDensityDeleter>& h_inj_rho,
        std::unique_ptr<amrex::Parser>& density_parser,
        amrex::Geometry const& geom)
    {
        const amrex::ParmParse pp_species(species_name);

        // parse density information
        std::string rho_prof_s;
        utils::parser::get(pp_species, source_name, "profile", rho_prof_s);
        std::transform(rho_prof_s.begin(), rho_prof_s.end(),
                    rho_prof_s.begin(), ::tolower);
        if (rho_prof_s == "constant") {
            amrex::Real density = 0;
            utils::parser::getWithParser(pp_species, source_name, "density", density);
            // Construct InjectorDensity with InjectorDensityConstant.
            h_inj_rho.reset(new InjectorDensity((InjectorDensityConstant*)nullptr, density));
        } else if (rho_prof_s == "parse_density_function") {
            std::string str_density_function;
            utils::parser::Store_parserString(pp_species, source_name, "density_function(x,y,z)", str_density_function);
            // Construct InjectorDensity with InjectorDensityParser.
            density_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str_density_function,{"x","y","z"}));
            h_inj_rho.reset(new InjectorDensity((InjectorDensityParser*)nullptr,
                density_parser->compile<3>()));
        } else if (rho_prof_s == "read_from_file") {
            std::string density_file;
            bool distributed = true;
            utils::parser::get(pp_species, source_name, "read_density_from_path", density_file);
            pp_species.query("read_density_distributed", distributed);
            h_inj_rho.reset(new InjectorDensity((InjectorDensityFromFile*)nullptr, density_file, geom, distributed));
        } else {
            StringParseAbortMessage("Density profile type", rho_prof_s);
        }
    }

    // Depending on injection type at runtime, initialize inj_mom
    // so that inj_mom->getMomentum calls
    // InjectorMomentum[Constant or Gaussian or etc.].getMomentum.
    void parseMomentum (std::string const& species_name, std::string const& source_name, const std::string& style,
        std::unique_ptr<InjectorMomentum,InjectorMomentumDeleter>& h_inj_mom,
        std::unique_ptr<amrex::Parser>& ux_parser,
        std::unique_ptr<amrex::Parser>& uy_parser,
        std::unique_ptr<amrex::Parser>& uz_parser,
        std::unique_ptr<amrex::Parser>& ux_th_parser,
        std::unique_ptr<amrex::Parser>& uy_th_parser,
        std::unique_ptr<amrex::Parser>& uz_th_parser,
        std::unique_ptr<TemperatureProperties>& h_mom_temp,
        std::unique_ptr<VelocityProperties>& h_mom_vel,
        int flux_normal_axis, int flux_direction)
    {
        using namespace amrex::literals;

        const amrex::ParmParse pp_species(species_name);

        // parse momentum information
        std::string mom_dist_s;
        utils::parser::get(pp_species, source_name, "momentum_distribution_type", mom_dist_s);
        std::transform(mom_dist_s.begin(),
                       mom_dist_s.end(),
                       mom_dist_s.begin(),
                       ::tolower);
        if (mom_dist_s == "at_rest") {
            constexpr amrex::Real ux = 0._rt;
            constexpr amrex::Real uy = 0._rt;
            constexpr amrex::Real uz = 0._rt;
            // Construct InjectorMomentum with InjectorMomentumConstant.
            h_inj_mom.reset(new InjectorMomentum((InjectorMomentumConstant*)nullptr, ux, uy, uz));
        } else if (mom_dist_s == "constant") {
            amrex::Real ux = 0._rt;
            amrex::Real uy = 0._rt;
            amrex::Real uz = 0._rt;
            utils::parser::queryWithParser(pp_species, source_name, "ux", ux);
            utils::parser::queryWithParser(pp_species, source_name, "uy", uy);
            utils::parser::queryWithParser(pp_species, source_name, "uz", uz);
            // Construct InjectorMomentum with InjectorMomentumConstant.
            h_inj_mom.reset(new InjectorMomentum((InjectorMomentumConstant*)nullptr, ux, uy, uz));
        } else if (mom_dist_s == "gaussian") {
            amrex::Real ux_m = 0._rt;
            amrex::Real uy_m = 0._rt;
            amrex::Real uz_m = 0._rt;
            amrex::Real ux_th = 0._rt;
            amrex::Real uy_th = 0._rt;
            amrex::Real uz_th = 0._rt;
            utils::parser::queryWithParser(pp_species, source_name, "ux_m", ux_m);
            utils::parser::queryWithParser(pp_species, source_name, "uy_m", uy_m);
            utils::parser::queryWithParser(pp_species, source_name, "uz_m", uz_m);
            utils::parser::queryWithParser(pp_species, source_name, "ux_th", ux_th);
            utils::parser::queryWithParser(pp_species, source_name, "uy_th", uy_th);
            utils::parser::queryWithParser(pp_species, source_name, "uz_th", uz_th);
            // Construct InjectorMomentum with InjectorMomentumGaussian.
            h_inj_mom.reset(new InjectorMomentum((InjectorMomentumGaussian*)nullptr,
                                                ux_m, uy_m, uz_m, ux_th, uy_th, uz_th));
        } else if (mom_dist_s == "gaussianflux") {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(style == "nfluxpercell",
                "Error: gaussianflux can only be used with injection_style = NFluxPerCell");
            amrex::Real ux_m = 0._rt;
            amrex::Real uy_m = 0._rt;
            amrex::Real uz_m = 0._rt;
            amrex::Real ux_th = 0._rt;
            amrex::Real uy_th = 0._rt;
            amrex::Real uz_th = 0._rt;
            utils::parser::queryWithParser(pp_species, source_name, "ux_m", ux_m);
            utils::parser::queryWithParser(pp_species, source_name, "uy_m", uy_m);
            utils::parser::queryWithParser(pp_species, source_name, "uz_m", uz_m);
            utils::parser::queryWithParser(pp_species, source_name, "ux_th", ux_th);
            utils::parser::queryWithParser(pp_species, source_name, "uy_th", uy_th);
            utils::parser::queryWithParser(pp_species, source_name, "uz_th", uz_th);
            // Construct InjectorMomentum with InjectorMomentumGaussianFlux.
            h_inj_mom.reset(new InjectorMomentum((InjectorMomentumGaussianFlux*)nullptr,
                                                ux_m, uy_m, uz_m, ux_th, uy_th, uz_th,
                                                flux_normal_axis, flux_direction));
        } else if (mom_dist_s == "uniform") {
            amrex::Real ux_min = 0._rt;
            amrex::Real uy_min = 0._rt;
            amrex::Real uz_min = 0._rt;
            amrex::Real ux_max = 0._rt;
            amrex::Real uy_max = 0._rt;
            amrex::Real uz_max = 0._rt;
            utils::parser::queryWithParser(pp_species, source_name, "ux_min", ux_min);
            utils::parser::queryWithParser(pp_species, source_name, "uy_min", uy_min);
            utils::parser::queryWithParser(pp_species, source_name, "uz_min", uz_min);
            utils::parser::queryWithParser(pp_species, source_name, "ux_max", ux_max);
            utils::parser::queryWithParser(pp_species, source_name, "uy_max", uy_max);
            utils::parser::queryWithParser(pp_species, source_name, "uz_max", uz_max);
            // Construct InjectorMomentum with InjectorMomentumUniform.
            h_inj_mom.reset(new InjectorMomentum((InjectorMomentumUniform*)nullptr,
                                                ux_min, uy_min, uz_min, ux_max, uy_max, uz_max));
        } else if (mom_dist_s == "maxwell_boltzmann"){
            h_mom_temp = std::make_unique<TemperatureProperties>(pp_species, source_name);
            const GetTemperature getTemp(*h_mom_temp);
            h_mom_vel = std::make_unique<VelocityProperties>(pp_species, source_name);
            const GetVelocity getVel(*h_mom_vel);
            // Construct InjectorMomentum with InjectorMomentumBoltzmann.
            h_inj_mom.reset(new InjectorMomentum((InjectorMomentumBoltzmann*)nullptr, getTemp, getVel));
        } else if (mom_dist_s == "maxwell_juttner"){
            h_mom_temp = std::make_unique<TemperatureProperties>(pp_species, source_name);
            const GetTemperature getTemp(*h_mom_temp);
            h_mom_vel = std::make_unique<VelocityProperties>(pp_species, source_name);
            const GetVelocity getVel(*h_mom_vel);
            // Construct InjectorMomentum with InjectorMomentumJuttner.
            h_inj_mom.reset(new InjectorMomentum((InjectorMomentumJuttner*)nullptr, getTemp, getVel));
        } else if (mom_dist_s == "parse_momentum_function") {
            std::string str_momentum_function_ux;
            std::string str_momentum_function_uy;
            std::string str_momentum_function_uz;
            utils::parser::Store_parserString(pp_species, source_name, "momentum_function_ux(x,y,z)", str_momentum_function_ux);
            utils::parser::Store_parserString(pp_species, source_name, "momentum_function_uy(x,y,z)", str_momentum_function_uy);
            utils::parser::Store_parserString(pp_species, source_name, "momentum_function_uz(x,y,z)", str_momentum_function_uz);
            // Construct InjectorMomentum with InjectorMomentumParser.
            ux_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str_momentum_function_ux, {"x","y","z"}));
            uy_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str_momentum_function_uy, {"x","y","z"}));
            uz_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str_momentum_function_uz, {"x","y","z"}));
            h_inj_mom.reset(new InjectorMomentum((InjectorMomentumParser*)nullptr,
                                                ux_parser->compile<3>(),
                                                uy_parser->compile<3>(),
                                                uz_parser->compile<3>()));
        } else if (mom_dist_s == "gaussian_parse_momentum_function") {
            std::string str_momentum_function_ux_m;
            std::string str_momentum_function_uy_m;
            std::string str_momentum_function_uz_m;
            std::string str_momentum_function_ux_th;
            std::string str_momentum_function_uy_th;
            std::string str_momentum_function_uz_th;
            utils::parser::Store_parserString(pp_species, source_name,
                "momentum_function_ux_m(x,y,z)", str_momentum_function_ux_m);
            utils::parser::Store_parserString(pp_species, source_name,
                "momentum_function_uy_m(x,y,z)", str_momentum_function_uy_m);
            utils::parser::Store_parserString(pp_species, source_name,
                "momentum_function_uz_m(x,y,z)", str_momentum_function_uz_m);
            utils::parser::Store_parserString(pp_species, source_name,
                "momentum_function_ux_th(x,y,z)", str_momentum_function_ux_th);
            utils::parser::Store_parserString(pp_species, source_name,
                "momentum_function_uy_th(x,y,z)", str_momentum_function_uy_th);
            utils::parser::Store_parserString(pp_species, source_name,
                "momentum_function_uz_th(x,y,z)", str_momentum_function_uz_th);
            // Construct InjectorMomentum with InjectorMomentumParser.
            ux_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str_momentum_function_ux_m, {"x","y","z"}));
            uy_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str_momentum_function_uy_m, {"x","y","z"}));
            uz_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str_momentum_function_uz_m, {"x","y","z"}));
            ux_th_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str_momentum_function_ux_th, {"x","y","z"}));
            uy_th_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str_momentum_function_uy_th, {"x","y","z"}));
            uz_th_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str_momentum_function_uz_th, {"x","y","z"}));
            h_inj_mom.reset(new InjectorMomentum((InjectorMomentumGaussianParser*)nullptr,
                                                ux_parser->compile<3>(),
                                                uy_parser->compile<3>(),
                                                uz_parser->compile<3>(),
                                                ux_th_parser->compile<3>(),
                                                uy_th_parser->compile<3>(),
                                                uz_th_parser->compile<3>()));
        } else if (mom_dist_s == "gaussian_parse_momentum_function_quiet") {
            // Deterministic antithetic Gaussian quiet-velocity start. Per-axis
            // sigma is parser-driven (same 6 momentum_function_u*_{m,th} strings
            // as the non-quiet variant). N_ppc must be a perfect cube; the
            // per-axis quantile lattice has N_half = round(N_ppc^(1/3)) points.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                style == "nuniformpercell" || style == "nrandompercell",
                "gaussian_parse_momentum_function_quiet requires "
                "injection_style = NUniformPerCell or NRandomPerCell");

            // Determine N_ppc from the calling injection style.
            int num_particles_per_cell = 0;
            if (style == "nuniformpercell") {
                // num_particles_per_cell_each_dim has AMREX_SPACEDIM entries
                // (2 in 2D Cartesian/RZ, 3 in 3D). The velocity-lattice cube
                // is independent of position-space dim; we just need the
                // total per-cell particle count = product over position dims.
                std::vector<int> ppc_each_dim(AMREX_SPACEDIM, 1);
                utils::parser::getArrWithParser(pp_species, source_name,
                                                "num_particles_per_cell_each_dim",
                                                ppc_each_dim, 0, AMREX_SPACEDIM);
                num_particles_per_cell = 1;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    num_particles_per_cell *= ppc_each_dim[d];
                }
            } else { // nrandompercell
                utils::parser::getWithParser(pp_species, source_name,
                                             "num_particles_per_cell",
                                             num_particles_per_cell);
            }

            const int N_half = static_cast<int>(std::round(
                std::cbrt(double(num_particles_per_cell))));
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                N_half >= 1 && N_half <= InjectorMomentumGaussianParserQuiet::K_MAX
                && N_half * N_half * N_half == num_particles_per_cell,
                std::string("gaussian_parse_momentum_function_quiet requires N_ppc ")
                + "to be a perfect cube in [1, "
                + std::to_string(InjectorMomentumGaussianParserQuiet::K_MAX*InjectorMomentumGaussianParserQuiet::K_MAX*InjectorMomentumGaussianParserQuiet::K_MAX)
                + "]. Got N_ppc=" + std::to_string(num_particles_per_cell)
                + ". Nearest valid: 1 8 27 64 125 216 343 512 729 1000 1331 1728 "
                + "2197 2744 3375 4096."
            );

            // Build the per-axis Phi^{-1}(p_j) quantile table at the midpoints
            // of N_half equal-probability bins on [0,1]. Symmetric about 0.
            amrex::Real quantile_table[InjectorMomentumGaussianParserQuiet::K_MAX] = {};
            for (int j = 0; j < N_half; ++j) {
                const double p_j = (2.0 * (j + 1) - 1.0) / (2.0 * N_half);
                quantile_table[j] = static_cast<amrex::Real>(acklam_inv_normal_cdf(p_j));
            }

            // Parse the 6 parser strings (same as gaussian_parse_momentum_function).
            std::string str_momentum_function_ux_m;
            std::string str_momentum_function_uy_m;
            std::string str_momentum_function_uz_m;
            std::string str_momentum_function_ux_th;
            std::string str_momentum_function_uy_th;
            std::string str_momentum_function_uz_th;
            utils::parser::Store_parserString(pp_species, source_name,
                "momentum_function_ux_m(x,y,z)", str_momentum_function_ux_m);
            utils::parser::Store_parserString(pp_species, source_name,
                "momentum_function_uy_m(x,y,z)", str_momentum_function_uy_m);
            utils::parser::Store_parserString(pp_species, source_name,
                "momentum_function_uz_m(x,y,z)", str_momentum_function_uz_m);
            utils::parser::Store_parserString(pp_species, source_name,
                "momentum_function_ux_th(x,y,z)", str_momentum_function_ux_th);
            utils::parser::Store_parserString(pp_species, source_name,
                "momentum_function_uy_th(x,y,z)", str_momentum_function_uy_th);
            utils::parser::Store_parserString(pp_species, source_name,
                "momentum_function_uz_th(x,y,z)", str_momentum_function_uz_th);
            ux_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str_momentum_function_ux_m, {"x","y","z"}));
            uy_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str_momentum_function_uy_m, {"x","y","z"}));
            uz_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str_momentum_function_uz_m, {"x","y","z"}));
            ux_th_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str_momentum_function_ux_th, {"x","y","z"}));
            uy_th_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str_momentum_function_uy_th, {"x","y","z"}));
            uz_th_parser = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(str_momentum_function_uz_th, {"x","y","z"}));
            h_inj_mom.reset(new InjectorMomentum(
                (InjectorMomentumGaussianParserQuiet*)nullptr,
                ux_parser->compile<3>(),
                uy_parser->compile<3>(),
                uz_parser->compile<3>(),
                ux_th_parser->compile<3>(),
                uy_th_parser->compile<3>(),
                uz_th_parser->compile<3>(),
                N_half, quantile_table));
        } else {
            StringParseAbortMessage("Momentum distribution type", mom_dist_s);
        }
    }

}
