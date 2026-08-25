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
#include <string>

namespace SpeciesUtils {

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
            std::string field_name = "density";
            bool distributed = true;
            utils::parser::get(pp_species, source_name, "read_density_from_path", density_file);
            utils::parser::query(pp_species, source_name, "density_mesh_name", field_name);
            pp_species.query("read_density_distributed", distributed);
            h_inj_rho.reset(new InjectorDensity((InjectorDensityFromFile*)nullptr,
                density_file, field_name, geom, distributed));
        } else {
            StringParseAbortMessage("Density profile type", rho_prof_s);
        }
    }

    // Depending on injection type at runtime, initialize inj_mom
    // so that inj_mom->getMomentum calls
    // InjectorMomentum[Constant or Gaussian or etc.].getMomentum.
    void parseMomentum (std::string const& species_name, std::string const& source_name, const std::string& style,
        std::unique_ptr<InjectorMomentum,InjectorMomentumDeleter>& h_inj_mom,
        std::unique_ptr<TemperatureProperties>& h_mom_temp,
        std::unique_ptr<VelocityProperties>& h_mom_vel,
        int flux_normal_axis, int flux_direction)
    {
        using namespace amrex::literals;

        const amrex::ParmParse pp_species(species_name);

        // Quiet (low-noise) velocity start. Read up front so that requesting it
        // on a distribution that cannot provide it is a hard error rather than
        // a silent fall back to pseudo-random sampling, which would run to
        // completion while quietly being noisy.
        int quiet_start = 0;
        utils::parser::queryWithParser(pp_species, source_name,
                                       "quiet_velocity_start", quiet_start);

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
        } else if (mom_dist_s == "maxwellian") {
            h_mom_temp = std::make_unique<TemperatureProperties>(pp_species, source_name);
            const GetTemperatureVector getTempVec(*h_mom_temp);
            h_mom_vel = std::make_unique<VelocityProperties>(pp_species, source_name);
            const GetVelocityVector getVelVec(*h_mom_vel);

            // Optional quiet (low-noise) start: replace the independent
            // pseudo-random normal draws by a stratified antithetic sample.
            // The per-axis lattice has N points, with N^3 equal to the number
            // of velocity samples taken per physical position.
            if (quiet_start != 0) {
                int M_vel = 1;
                utils::parser::queryWithParser(pp_species, source_name,
                                               "velocity_samples_per_position", M_vel);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(M_vel >= 1,
                    "velocity_samples_per_position must be >= 1");
                const int N_half = static_cast<int>(std::round(std::cbrt(double(M_vel))));
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    N_half >= 1 && N_half <= InjectorMomentumMaxwellianQuiet::K_MAX
                    && N_half * N_half * N_half == M_vel,
                    std::string("quiet_velocity_start requires ")
                    + "velocity_samples_per_position to be a perfect cube in [1, "
                    + std::to_string(InjectorMomentumMaxwellianQuiet::K_MAX
                                     * InjectorMomentumMaxwellianQuiet::K_MAX
                                     * InjectorMomentumMaxwellianQuiet::K_MAX)
                    + "]. Got velocity_samples_per_position=" + std::to_string(M_vel)
                    + ". Nearest valid: 1 8 27 64 125 216 343 512 729 1000 1331 "
                    + "1728 2197 2744 3375 4096.");
                if (M_vel == 1) {
                    ablastr::warn_manager::WMRecordWarning("Species",
                        "quiet_velocity_start was requested with "
                        "velocity_samples_per_position = 1. The sampling is "
                        "deterministic, but with a single sample per position "
                        "there is no antithetic partner to cancel against, so "
                        "none of the noise reduction is obtained. Set "
                        "velocity_samples_per_position to a perfect cube > 1 "
                        "(8, 27, 64, ...) for a quiet start.",
                        ablastr::warn_manager::WarnPriority::high);
                }
                h_inj_mom.reset(new InjectorMomentum(
                    (InjectorMomentumMaxwellianQuiet*)nullptr,
                    getTempVec, getVelVec, N_half));
            } else {
                h_inj_mom.reset(new InjectorMomentum((InjectorMomentumMaxwellian*)nullptr, getTempVec, getVelVec));
            }
        } else if (mom_dist_s == "maxwell_juttner"){
            h_mom_temp = std::make_unique<TemperatureProperties>(pp_species, source_name);
            const GetTemperature getTemp(*h_mom_temp);
            h_mom_vel = std::make_unique<VelocityProperties>(pp_species, source_name);
            const GetVelocityVector getVelVec(*h_mom_vel);
            // Construct InjectorMomentum with InjectorMomentumJuttner.
            h_inj_mom.reset(new InjectorMomentum((InjectorMomentumJuttner*)nullptr, getTemp, getVelVec));
        } else if (mom_dist_s == "parse_momentum_function") {
            // The momentum is defined by the parser functions ux_mean_function,
            // uy_mean_function, uz_mean_function, stored in VelocityProperties and
            // evaluated through GetVelocityVector (the parsers are owned by h_mom_vel).
            h_mom_vel = std::make_unique<VelocityProperties>(pp_species, source_name);
            const GetVelocityVector getVelVec(*h_mom_vel);
            // Construct InjectorMomentum with InjectorMomentumParser.
            h_inj_mom.reset(new InjectorMomentum((InjectorMomentumParser*)nullptr, getVelVec));
        } else {
            StringParseAbortMessage("Momentum distribution type", mom_dist_s);
        }

        // A quiet start that was asked for but not applied would silently
        // degrade to ordinary pseudo-random sampling, so refuse to run.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            quiet_start == 0 || mom_dist_s == "maxwellian",
            std::string("quiet_velocity_start is only supported with "
                        "momentum_distribution_type = maxwellian, but got '")
            + mom_dist_s + "'. From PICMI, warpx_velocity_quiet_start=True "
            "requires warpx_momentum_spread_expressions to be set.");
    }

}
