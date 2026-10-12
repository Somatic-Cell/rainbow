#include <rainbow/raindrop_tracer.hpp>
#include <rainbow/patch_accel.hpp>
#include <rainbow/patch_query.hpp>
#include <rainbow/patch_optics.hpp>
#include <rainbow/phase_cdf.hpp>
#include <rainbow/phase_cdf_math.hpp>
#include <rainbow/npy_writer.hpp>
#include <rainbow/folded_patch_data.hpp>
#include <rainbow/rainbow_diffraction.hpp>
#include <rainbow/water_refractive_index.hpp>
#include <rainbow/projected_area.hpp>

#ifndef RAINBOW_ENABLE_DIAGNOSTICS
#define RAINBOW_ENABLE_DIAGNOSTICS 0
#endif
#if RAINBOW_ENABLE_DIAGNOSTICS
#include <rainbow/patch_failure_report.hpp>
#endif

#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <string_view>

#ifndef RAINBOW_SOURCE_COMMIT
#define RAINBOW_SOURCE_COMMIT "unknown"
#endif
#ifndef RAINBOW_SOURCE_DIRTY
#define RAINBOW_SOURCE_DIRTY 1
#endif

namespace
{
using namespace rainbow;
struct Command
{
    RaindropSettings settings{};
    WaveOpticsSettings wave{};
    PhaseCdfPolicy policy{};
    PhaseStorageSettings storage{};
    double inclination_degrees=20;
    enum class Material { Water, Constant };
    Material material=Material::Water;
    WaterOpticalState water{};
    double temperature_celsius=20.0, pressure_pascal=101325.0;
    double wavelength_requested_nm=0.0, constant_index=0.0;
    bool wavelength_given=false, index_given=false, state_given=false;
    bool has_paper_sigma=false;
    double paper_primary_sigma_degrees=0.0;
    std::filesystem::path modules, output, patch_failure_report;
    std::uint32_t theta=360,phi=720;
    PhaseDensityStage stage=PhaseDensityStage::Diffraction;
    int device=0;
};
double number(std::string_view s)
{
    double x=0;
    const auto r=std::from_chars(s.data(),s.data()+s.size(),x);
    if(r.ec!=std::errc{} || r.ptr!=s.data()+s.size() || !std::isfinite(x))
        throw std::invalid_argument("Invalid numeric option.");
    return x;
}
std::uint32_t positive_integer(std::string_view s)
{
    std::uint32_t x=0;
    const auto r=std::from_chars(s.data(),s.data()+s.size(),x);
    if(r.ec!=std::errc{} || r.ptr!=s.data()+s.size() || x==0)
        throw std::invalid_argument("Invalid positive integer option.");
    return x;
}
const char* stage_name(PhaseDensityStage s)
{
    switch(s)
    {
        case PhaseDensityStage::Incoherent: return "incoherent";
        case PhaseDensityStage::Path: return "path";
        case PhaseDensityStage::Focal: return "focal";
        case PhaseDensityStage::Diffraction: return "diffraction";
        default: throw std::invalid_argument("Invalid optical stage.");
    }
}
// Resolve existing parent aliases even though the output leaves must be new.
// On Windows, also compare native path components without ASCII case aliases.
bool path_component_equal(const std::filesystem::path& a,const std::filesystem::path& b)
{
#if defined(_WIN32)
    const auto& x=a.native();const auto& y=b.native();
    if(x.size()!=y.size())return false;
    for(std::size_t i=0;i<x.size();++i)
        if(std::towlower(x[i])!=std::towlower(y[i]))return false;
    return true;
#else
    return a==b;
#endif
}
bool within_path(const std::filesystem::path& child,const std::filesystem::path& parent)
{
    auto c=child.begin();
    for(auto p=parent.begin();p!=parent.end();++p,++c)
        if(c==child.end() || !path_component_equal(*c,*p))return false;
    return true;
}
void validate_report_path(const Command& c)
{
    if(c.patch_failure_report.empty())return;
    if(c.patch_failure_report.filename().empty())
        throw std::invalid_argument("--patch-failure-report must name a new file, not a directory.");
    if(c.stage!=PhaseDensityStage::Focal && c.stage!=PhaseDensityStage::Diffraction)
        throw std::invalid_argument("--patch-failure-report requires --stage focal or diffraction.");
    auto report_part=c.patch_failure_report;report_part+=".part";
    if(std::filesystem::exists(std::filesystem::symlink_status(c.patch_failure_report))
       ||std::filesystem::exists(std::filesystem::symlink_status(report_part)))
        throw std::invalid_argument("Choose a new --patch-failure-report path (report or .part already exists).");
    const auto report=std::filesystem::weakly_canonical(std::filesystem::absolute(c.patch_failure_report));
    const auto temporary=std::filesystem::weakly_canonical(std::filesystem::absolute(report_part));
    const auto output=std::filesystem::weakly_canonical(std::filesystem::absolute(c.output));
    auto output_part=c.output;output_part+=".part";
    const auto staging=std::filesystem::weakly_canonical(std::filesystem::absolute(output_part));
    for(const auto& file:{report,temporary})for(const auto& directory:{output,staging})
        if(within_path(file,directory)||within_path(directory,file))
            throw std::invalid_argument("Report and its .part file must be separate from --out and the dataset .part directory.");
    if(!std::filesystem::is_directory(report.parent_path()))
        throw std::invalid_argument("The --patch-failure-report parent directory must already exist.");
}
template<class Char> Command parse(int argc,Char** argv)
{
    Command c;c.settings.radius_mm=1.0f;c.settings.grid_width=c.settings.grid_height=513;
    c.modules=std::filesystem::absolute(std::filesystem::path(argv[0])).parent_path()/"modules";

    for(int i=1;i<argc;++i)
    {
        const auto option=std::filesystem::path(argv[i]).string();
        if(option=="--sphere") {c.settings.force_sphere=true;continue;}
        if(option=="--allow-underresolved") {c.policy.allow_underresolved=true;continue;}
        if(option=="--help")
        {
            std::cout << "rainbow_generate --out DIRECTORY [--modules DIRECTORY] [--sphere]\n"
                         "  --radius-mm R --wavelength-nm L (required, vacuum nm) --grid VERTICES_PER_AXIS\n"
                         "  --inclination-deg A --query-theta T --query-phi P\n"
                         "  --stage diffraction|focal|incoherent|path\n"
                         "  --material water (default) --temperature-c T --pressure-pa P\n"
                         "  --material constant --ior N (explicit nondispersive reference)\n"
                         "  --focal-offsets 0,0,0,0\n"
                         "  diffraction width: Table II radius lookup, secondary = 2*primary\n"
                         "  --allow-underresolved  (explicitly accept flagged angular underresolution)\n"
                         "  --cdf-max-l1 E --cdf-max-lost-mass E --device ORDINAL\n"
                         "  --cdf-theta T --cdf-phi P (integer divisors of query dimensions)\n"
                         "  --max-coarsening-tv E (quality policy, not a blur width)\n"
                         "  --patch-failure-report NEW.json (requires diagnostics build; focal/diffraction)\n"
                         "  No additional/global storage Gaussian is applied.\n";
            return {};
        }
        if(++i>=argc) throw std::invalid_argument("Missing option value.");
        if(option=="--out") {c.output=std::filesystem::path(argv[i]);continue;}
        if(option=="--modules") {c.modules=std::filesystem::path(argv[i]);continue;}
        if(option=="--patch-failure-report")
        {
#if RAINBOW_ENABLE_DIAGNOSTICS
            if(!c.patch_failure_report.empty())
                throw std::invalid_argument("Duplicate --patch-failure-report.");
            c.patch_failure_report=std::filesystem::path(argv[i]);
            if(c.patch_failure_report.empty())
                throw std::invalid_argument("--patch-failure-report needs a new file path.");
            continue;
#else
            throw std::invalid_argument("Detailed reports are disabled. Rebuild with -DRAINBOW_ENABLE_DIAGNOSTICS=ON.");
#endif
        }
        const auto value=std::filesystem::path(argv[i]).string();
        if(option=="--radius-mm") c.settings.radius_mm=static_cast<float>(number(value));
        else if(option=="--wavelength-nm")
        {
            if(c.wavelength_given)throw std::invalid_argument("Duplicate --wavelength-nm.");
            c.wavelength_requested_nm=number(value);c.wavelength_given=true;
            if(!(c.wavelength_requested_nm>0 && c.wavelength_requested_nm<=(std::numeric_limits<float>::max)()))
                throw std::invalid_argument("Wavelength must be positive and representable by the solver.");
            c.settings.wavelength_nm=static_cast<float>(c.wavelength_requested_nm);
            if(!(c.settings.wavelength_nm>0))throw std::invalid_argument("Wavelength underflows solver precision.");
        }
        else if(option=="--material")
        {
            if(value=="water")c.material=Command::Material::Water;
            else if(value=="constant")c.material=Command::Material::Constant;
            else throw std::invalid_argument("Material must be water or constant.");
        }
        else if(option=="--temperature-c") {c.temperature_celsius=number(value);c.state_given=true;}
        else if(option=="--pressure-pa") {c.pressure_pascal=number(value);c.state_given=true;}
        else if(option=="--ior")
        {
            c.constant_index=number(value);c.index_given=true;
        }
        else if(option=="--grid") c.settings.grid_width=c.settings.grid_height=positive_integer(value);
        else if(option=="--query-theta") c.theta=positive_integer(value);
        else if(option=="--query-phi") c.phi=positive_integer(value);
        else if(option=="--inclination-deg") c.inclination_degrees=number(value);
        else if(option=="--cdf-theta") c.storage.theta_count=positive_integer(value);
        else if(option=="--cdf-phi") c.storage.phi_count=positive_integer(value);
        else if(option=="--storage-gaussian-sigma-deg" || option=="--storage-gaussian-support")
            throw std::invalid_argument("Global storage blur has been removed from rainbow_generate. Remove this option; only local Table-II diffraction and cell-mass aggregation are applied.");
        else if(option=="--max-coarsening-tv") c.storage.maximum_coarsening_tv=number(value);
        else if(option=="--diffraction-sigma-deg")
            throw std::invalid_argument("Manual diffraction width is not supported by rainbow_generate. Remove this option; Table II is selected automatically.");
        else if(option=="--transition-contrast") c.wave.transition_contrast=number(value);
        else if(option=="--cdf-max-l1") c.policy.maximum_l1_error=number(value);
        else if(option=="--cdf-max-lost-mass") c.policy.maximum_lost_mass=number(value);
        else if(option=="--device")
        {
            const auto r=std::from_chars(value.data(),value.data()+value.size(),c.device);
            if(r.ec!=std::errc{} || r.ptr!=value.data()+value.size() || c.device<0)
                throw std::invalid_argument("Invalid device ordinal.");
        }
        else if(option=="--stage")
        {
            if(value=="incoherent")c.stage=PhaseDensityStage::Incoherent;
            else if(value=="path")c.stage=PhaseDensityStage::Path;
            else if(value=="focal")c.stage=PhaseDensityStage::Focal;
            else if(value=="diffraction")c.stage=PhaseDensityStage::Diffraction;
            else throw std::invalid_argument("Unknown stage.");
        }
        else if(option=="--focal-offsets")
        {
            std::size_t begin=0;
            for(unsigned k=0;k<4;++k)
            {
                const auto end=value.find(',',begin);
                if((k<3 && end==std::string::npos) || (k==3 && end!=std::string::npos))
                    throw std::invalid_argument("Expected four focal offsets.");
                const auto length=end==std::string::npos?value.size()-begin:end-begin;
                const auto part=std::string_view(value).substr(begin,length);
                auto& x=c.wave.focal_quarter_turn_offsets[k];
                const auto r=std::from_chars(part.data(),part.data()+part.size(),x);
                if(r.ec!=std::errc{} || r.ptr!=part.data()+part.size() || x>3)
                    throw std::invalid_argument("Focal offsets must be 0..3.");
                begin=end==std::string::npos?value.size():end+1;
            }
        }
        else throw std::invalid_argument("Unknown option: "+option);
    }
    if(c.output.empty()) throw std::invalid_argument("--out is required.");
    if(!c.wavelength_given)throw std::invalid_argument("--wavelength-nm is required; there is no implicit wavelength.");
    if(c.material==Command::Material::Water)
    {
        if(c.index_given)throw std::invalid_argument("Do not specify --ior for water. It is evaluated from wavelength and state; use --material constant only for explicit reference calculations.");
        // Evaluate at the wavelength actually represented by the existing solver.
        c.water=evaluate_water_optics(double(c.settings.wavelength_nm),c.temperature_celsius,c.pressure_pascal);
        c.settings.interior_index=static_cast<float>(c.water.refractive_index);
    }
    else
    {
        if(!c.index_given || !(c.constant_index>0 && c.constant_index<=(std::numeric_limits<float>::max)()))
            throw std::invalid_argument("Constant material requires a finite positive --ior.");
        if(c.state_given)throw std::invalid_argument("Temperature/pressure do not define a constant-index material.");
        c.settings.interior_index=static_cast<float>(c.constant_index);
        if(!(c.settings.interior_index>0))throw std::invalid_argument("Index underflows solver precision.");
    }
    // Absolute indices + vacuum wavelength; no unannounced conversion to air-relative n.
    // RaindropSettings::make_config() already fixes the exterior index to 1.
    c.wave.primary_sigma_degrees=0.0; // Existing solver resolves Table II.
    c.storage.gaussian_sigma_degrees=0.0; // NO second/global application of Table II.
    c.storage.gaussian_support_sigma=4.0; // Inactive legacy library field.
    c.has_paper_sigma=c.stage==PhaseDensityStage::Focal || c.stage==PhaseDensityStage::Diffraction;
    if(c.has_paper_sigma && !RainbowDiffraction::table_sigma_degrees(c.settings.radius_mm,c.paper_primary_sigma_degrees))
        throw std::invalid_argument("Table II covers radius 0.1..1.0 mm only. No bandwidth extrapolation is performed. The existing focal path also computes diffraction; use incoherent/path for out-of-table reference runs.");
    const double angle=c.inclination_degrees*phase_cdf_pi/180.0;
    if(c.storage.theta_count==0)c.storage.theta_count=c.theta;
    if(c.storage.phi_count==0)c.storage.phi_count=c.phi;
    if(c.storage.theta_count>c.theta || c.storage.phi_count>c.phi ||
       c.theta%c.storage.theta_count || c.phi%c.storage.phi_count)
        throw std::invalid_argument("CDF dimensions must divide the query dimensions without upsampling.");
    if(!(c.storage.maximum_coarsening_tv>=0 && c.storage.maximum_coarsening_tv<=1))
        throw std::invalid_argument("Coarsening TV limit must lie in [0,1].");
    c.settings.incident_direction={static_cast<float>(std::cos(angle)),static_cast<float>(-std::sin(angle)),0};
    static_cast<void>(c.settings.make_config());
    if(std::filesystem::exists(c.output)) throw std::runtime_error("Output already exists.");
    auto partial=c.output;partial+=".part";
    if(std::filesystem::exists(partial)) throw std::runtime_error("Previous .part directory exists; inspect it before recomputing.");
    validate_report_path(c);
    return c;
}
using D3=std::array<double,3>;
double dot(D3 a,D3 b) {return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
D3 normalized(D3 a)
{
    const double n=std::sqrt(dot(a,a));
    if(!(n>0 && std::isfinite(n)))throw std::runtime_error("Invalid frame.");
    for(auto& v:a)v/=n;
    return a;
}
void write_metadata(const std::filesystem::path& path,const Command& c,
    const RaindropTraceConfig& config,const PatchAccel& patches,const PhaseCdf& cdf,
    const ProjectedAreaResult& area)
{
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    if(!out)throw std::runtime_error("Cannot open metadata.json.");
    out.imbue(std::locale::classic());out<<std::setprecision(17);
    const auto& s=cdf.input_statistics();const auto& a=cdf.audit_statistics();
    const auto& storage=cdf.storage_statistics();
    auto k=normalized({config.incident_direction.x,config.incident_direction.y,config.incident_direction.z});
    D3 e{config.incident_basis_x.x,config.incident_basis_x.y,config.incident_basis_x.z};
    const double d=dot(e,k);for(unsigned i=0;i<3;++i)e[i]-=d*k[i];e=normalized(e);
    const D3 f{k[1]*e[2]-k[2]*e[1],k[2]*e[0]-k[0]*e[2],k[0]*e[1]-k[1]*e[0]};
    const FoldedPatchConfig folded{};
    out << "{\n"
        << "  \"coordinate_contract_id\": \"rainbow.phase_cdf.coordinates.v1\",\n"
        << "  \"direction_convention\": \"physical_propagation\",\n"
        << "  \"particle_frame_handedness\": \"right\",\n"
        << "  \"particle_up_axis\": [0,1,0],\n"
        << "  \"shape_polar_axis\": [0,-1,0],\n"
        << "  \"sampling_frame_layout\": \"rows_xyz_columns_e0_e1_k\",\n"
        << "  \"sampling_frame_map\": \"local_column_to_particle_column\",\n"
        << "  \"theta_zero\": \"forward\",\n"
        << "  \"theta_pi\": \"backward\",\n"
        << "  \"azimuth_periodic\": true,\n"
        << "  \"inclination_definition\": \"ki=(cos(alpha),-sin(alpha),0); alpha=0 is +x; positive alpha points downward\",\n"
        << "  \"geometry\": {\n"
        << "    \"projected_area_mm2\": " << area.area_mm2 << ",\n"
        << "    \"projected_area_m2\": " << area.area_mm2*1e-6 << ",\n"
        << "    \"projected_area_method\": \"" << (area.analytic_sphere?"analytic_sphere":"convex_surface_integral_azimuth_analytic_gk15") << "\",\n"
        << "    \"projected_area_shape\": \"solver_shape_coefficients_promoted_to_fp64\",\n"
        << "    \"projected_area_converged\": true,\n"
        << "    \"projected_area_absolute_error_estimate_mm2\": " << area.estimated_absolute_error_mm2 << ",\n"
        << "    \"projected_area_relative_error_estimate\": " << area.estimated_relative_error << ",\n"
        << "    \"projected_area_error_is_rigorous_bound\": false,\n"
        << "    \"convexity_verified\": true,\n"
        << "    \"convexity_method\": \"" << (area.analytic_sphere?"analytic_sphere":"outward_rounded_interval_curvature") << "\",\n"
        << "    \"convexity_intervals\": " << area.convexity_intervals << ",\n"
        << "    \"quadrature_evaluations\": " << area.quadrature_evaluations << ",\n"
        << "    \"incident_axis_particle_frame\": [" << k[0] << ',' << k[1] << ',' << k[2] << "]\n  },\n"
        << "  \"cross_sections\": {\n"
        << "    \"model\": \"geometric_nonabsorbing\",\n"
        << "    \"scattering_mm2\": " << area.area_mm2 << ",\n"
        << "    \"extinction_mm2\": " << area.area_mm2 << ",\n"
        << "    \"absorption_mm2\": 0,\n"
        << "    \"scattering_m2\": " << area.area_mm2*1e-6 << ",\n"
        << "    \"extinction_m2\": " << area.area_mm2*1e-6 << ",\n"
        << "    \"absorption_m2\": 0,\n"
        << "    \"assignment\": \"Csca=Cext=Aproj; Cabs=0 (transport model assumption)\",\n"
        << "    \"wave_scattering_cross_section_computed\": false\n  },\n"
        << "  \"material\": {\n";
    out << "    \"model\": \"" << (c.material==Command::Material::Water?"water_iapws_r9_97":"constant") << "\",\n"
        << "    \"wavelength_convention\": \"vacuum_nm\",\n"
        << "    \"wavelength_requested_nm\": " << c.wavelength_requested_nm << ",\n"
        << "    \"wavelength_evaluated_nm\": " << config.wavelength_nm << ",\n"
        << "    \"ambient_model\": \"unit_absolute_index_air_approximation\",\n"
        << "    \"absorption_model\": \"ignored_k_equals_zero\",\n"
        << "    \"interior_index_evaluated_fp64\": " << (c.material==Command::Material::Water?c.water.refractive_index:c.constant_index) << ",\n"
        << "    \"interior_index_solver_fp32\": " << config.interior_index << ",\n"
        << "    \"exterior_index_solver\": " << config.exterior_index;
    if(c.material==Command::Material::Water)
        out << ",\n    \"refractive_index_release\": \"IAPWS R9-97\",\n"
            << "    \"density_model\": \"IAPWS R6-95(2018) liquid pressure EOS inversion\",\n"
            << "    \"temperature_kelvin\": " << c.water.temperature_kelvin << ",\n"
            << "    \"pressure_pascal\": " << c.water.pressure_pascal << ",\n"
            << "    \"density_kg_m3\": " << c.water.density_kg_m3 << ",\n"
            << "    \"density_pressure_residual_pascal\": " << c.water.density_pressure_residual_pascal << ",\n"
            << "    \"density_bracket_width_kg_m3\": " << c.water.density_bracket_width_kg_m3;
    out << "\n  },\n  \"diffraction_policy\": {\n"
        << "    \"name\": \"Sadeghi2012_TableII_local_v1\",\n"
        << "    \"computed\": " << (c.has_paper_sigma?"true":"false") << ",\n"
        << "    \"selected_output_includes_diffraction\": " << (c.stage==PhaseDensityStage::Diffraction?"true":"false") << ",\n"
        << "    \"primary_sigma_degrees\": ";
    if(c.has_paper_sigma)out<<c.paper_primary_sigma_degrees;else out<<"null";
    out << ",\n    \"secondary_sigma_degrees\": ";
    if(c.has_paper_sigma)out<<2*c.paper_primary_sigma_degrees;else out<<"null";
    out << ",\n"
        << "    \"bandwidth_source\": \"paper Table II; secondary sigma x2\",\n"
        << "    \"between_table_radii\": \"linear interpolation (project convention)\",\n"
        << "    \"outside_table_radii\": \"reject\",\n"
        << "    \"wavelength_scaling\": \"none; no rule specified in Table II\",\n"
        << "    \"support_sigma\": 4,\n"
        << "    \"support_and_transition_detection_are_project_conventions\": true,\n"
        << "    \"physical_sigma_depends_on_grid\": false,\n"
        << "    \"query_dtheta_degrees\": " << 180.0/c.theta << ",\n"
        << "    \"query_dphi_degrees\": " << 360.0/c.phi << "\n  },\n";
    out << "  \"schema\": \"rainbow.phase_cdf.numpy.v2\",\n  \"complete\": true,\n"
        << "  \"dtype\": \"<f8\",\n  \"order\": \"C\",\n"
        << "  \"theta_count\": "<<cdf.theta_count()<<",\n  \"phi_count\": "<<cdf.phi_count()<<",\n"
        << "  \"density_measure\": \"solid_angle_sr\",\n"
        << "  \"cell_model\": \"constant_density_per_spherical_cell\",\n"
        << "  \"coordinates\": \"u=(1-cos(theta))/2; v=(phi+pi)/(2*pi)\",\n"
        << "  \"theta_definition\": \"angle_between_incident_and_outgoing_propagation_directions\",\n"
        << "  \"cdf_axis_order\": [\"phi_cell\", \"theta_edge\"],\n"
        << "  \"phi_range_rad\": ["<<-phase_cdf_pi<<','<<phase_cdf_pi<<"],\n"
        << "  \"normalization\": \"integral_p_domega_equals_one\",\n"
        << "  \"input_polarization\": \"unpolarized\",\n"
        << "  \"incident_total_intensity\": 1,\n  \"single_scattering_albedo\": 1,\n"
        << "  \"collision_model\": \"geometric_projected_area\",\n"
        << "  \"projected_area_storage\": \"metadata.geometry.projected_area_mm2\",\n"
        << "  \"stage\": \""<<stage_name(c.stage)<<"\",\n"
        << "  \"radius_mm\": "<<config.radius_mm<<",\n"
        << "  \"wavelength_nm\": "<<config.wavelength_nm<<",\n"
        << "  \"exterior_index\": "<<config.exterior_index<<",\n"
        << "  \"interior_index\": "<<config.interior_index<<",\n"
        << "  \"force_sphere\": "<<(c.settings.force_sphere?"true":"false")<<",\n"
        << "  \"incident_grid\": ["<<config.grid_width<<','<<config.grid_height<<"],\n"
        << "  \"incident_grid_half_extent\": "<<config.grid_half_extent<<",\n"
        << "  \"reference_distances\": ["<<config.reference_distance<<','<<config.outgoing_reference_distance<<"],\n"
        << "  \"query_frame_axis\": ["<<config.incident_direction.x<<','<<config.incident_direction.y<<','<<config.incident_direction.z<<"],\n"
        << "  \"query_frame_e0\": ["<<config.incident_basis_x.x<<','<<config.incident_basis_x.y<<','<<config.incident_basis_x.z<<"],\n"
        << "  \"sampling_frame_columns\": [";
    for(unsigned r=0;r<3;++r)out<<(r?",":"")<<'['<<e[r]<<','<<f[r]<<','<<k[r]<<']';
    out<<"],\n  \"shape_coefficients\": [";
    for(unsigned i=0;i<8;++i)out<<(i?",":"")<<config.shape.coefficients[i];
    out<<"],\n  \"focal_quarter_turn_offsets\": [";
    for(unsigned i=0;i<4;++i)out<<(i?",":"")<<c.wave.focal_quarter_turn_offsets[i];
    out<<"],\n  \"diffraction_primary_sigma_degrees_setting\": "<<c.wave.primary_sigma_degrees
       <<",\n  \"diffraction_zero_setting_selects_Table_II\": true,\n"
       <<"  \"diffraction_transition_contrast\": "<<c.wave.transition_contrast<<",\n"
       <<"  \"folded_branch_model\": \"finite_branch_area_ratio_v1\",\n"
       <<"  \"shared_patch_boundary_policy\": \"exact_fp32_edge_planes_regular_same_orientation_min_patch_id_v1\",\n"
       <<"  \"folded_relative_tolerance\": "<<folded.relative_tolerance<<",\n"
       <<"  \"source_commit\": \""<<RAINBOW_SOURCE_COMMIT<<"\",\n"
       <<"  \"source_dirty_at_configure\": "<<(RAINBOW_SOURCE_DIRTY?"true":"false")<<",\n"
       <<"  \"generator_version\": \"numpy_cdf_water_geometry_v1\",\n"
       <<"  \"quality\": {\n    \"source_coverage_certified\": false,\n"
       <<"    \"angular_convergence_certified\": false,\n"
       <<"    \"missing_source_cells\": "<<patches.statistics().count(PatchCellStatus::MissingCorners)<<",\n"
       <<"    \"invalid_values\": "<<s.invalid_values<<",\n"
       <<"    \"incomplete_directions\": "<<s.incomplete<<",\n"
       <<"    \"underresolved_directions\": "<<s.underresolved<<",\n"
       <<"    \"allow_underresolved\": "<<(c.policy.allow_underresolved?"true":"false")<<",\n"
       <<"    \"cdf_l1_mass_error\": "<<a.l1_error<<",\n"
       <<"    \"cdf_maximum_cell_mass_error\": "<<a.maximum_cell_error<<",\n"
       <<"    \"cdf_lost_positive_cells\": "<<a.lost_cells<<",\n"
       <<"    \"cdf_lost_probability_mass\": "<<a.lost_mass<<",\n"
       <<"    \"maximum_accepted_l1_mass_error\": "<<c.policy.maximum_l1_error<<",\n"
       <<"    \"maximum_accepted_lost_probability_mass\": "<<c.policy.maximum_lost_mass<<"\n  },\n"
       <<"  \"incident_inclination_degrees\": "<<c.inclination_degrees<<",\n"
       <<"  \"query_grid\": ["<<c.theta<<','<<c.phi<<"],\n"
       <<"  \"storage_processing\": {\n"
       <<"    \"additional_filter\": \""<<(c.storage.gaussian_sigma_degrees>0?"spherical_geodesic_gaussian_row_normalized":"none")<<"\",\n"
       <<"    \"additional_filter_is_paper_diffraction\": false,\n"
       <<"    \"sigma_degrees\": "<<c.storage.gaussian_sigma_degrees<<",\n"
       <<"    \"support_sigma\": "<<c.storage.gaussian_support_sigma<<",\n"
       <<"    \"gaussian_underresolved\": "<<(storage.gaussian_underresolved?"true":"false")<<",\n"
       <<"    \"aggregation\": \"nested_solid_angle_mass_preserving\",\n"
       <<"    \"gaussian_integral_relative_change\": "<<storage.gaussian_integral_relative_change<<",\n"
       <<"    \"aggregation_integral_relative_change\": "<<storage.aggregation_integral_relative_change<<",\n"
       <<"    \"coarsening_tv\": "<<storage.coarsening_tv<<",\n"
       <<"    \"maximum_accepted_coarsening_tv\": "<<c.storage.maximum_coarsening_tv<<"\n  },\n"
       <<"  \"hg\": {\n    \"g\": "<<storage.g_stored<<",\n"
       <<"    \"method\": \"first_moment_of_saved_cell_pdf\",\n"
       <<"    \"target\": \"saved_cdf\",\n"
       <<"    \"cosine_convention\": \"dot(incident_propagation,outgoing_propagation)\",\n"
       <<"    \"axis_particle_frame\": ["<<k[0]<<','<<k[1]<<','<<k[2]<<"],\n"
       <<"    \"g_source\": "<<storage.g_source<<",\n"
       <<"    \"g_filtered\": "<<storage.g_filtered<<",\n"
       <<"    \"cdf_mass_check\": "<<storage.cdf_mass<<"\n  }\n}\n";
    out.flush();if(!out)throw std::runtime_error("Metadata write failed.");
    out.close();if(out.fail())throw std::runtime_error("Metadata close failed.");
}
template<class Char> int run(int argc,Char** argv)
{
    const auto c=parse(argc,argv);
    if(c.output.empty())return 0;
    // Independent CPU geometric integral; validate BEFORE expensive GPU work.
    // The coefficients and direction are exactly those supplied to the solver.
    const auto geometry_config=c.settings.make_config();
    std::array<double,8> area_coefficients{};
    for(unsigned n=0;n<8;++n)area_coefficients[n]=geometry_config.shape.coefficients[n];
    const auto projected_area=compute_projected_area(area_coefficients,
        double(geometry_config.radius_mm),
        {double(geometry_config.incident_direction.x),
         double(geometry_config.incident_direction.y),
         double(geometry_config.incident_direction.z)});
    std::cout<<std::setprecision(17)
        <<"Material="<<(c.material==Command::Material::Water?"water_iapws_r9_97":"constant")
        <<" wavelength_vacuum_nm="<<c.settings.wavelength_nm
        <<" interior_index_solver="<<c.settings.interior_index
        <<" exterior_index=1; additional_storage_gaussian=none\n";
    if(c.has_paper_sigma)
        std::cout<<"Local diffraction sigma_primary_deg="<<c.paper_primary_sigma_degrees
            <<" sigma_secondary_deg="<<2*c.paper_primary_sigma_degrees
            <<" query_step_theta_deg="<<180.0/c.theta<<" query_step_phi_deg="<<360.0/c.phi
            <<" (phi arc step additionally varies with theta)\n";
    const auto start=std::chrono::steady_clock::now();
    CudaContext context(c.device);
    RaindropTracer tracer(context);
    PatchAccel patches(context,tracer.optix_context());
    PatchQuery query(context,tracer.optix_context());
    PatchOptics optics(context);
    PhaseCdf cdf(context);
    tracer.create_pipeline(c.modules/"raindrop_trace.optixir");
    query.create_pipeline(c.modules/"patch_query.optixir");
    patches.load_module(c.modules/"patch_build.fatbin");
    optics.load_module(c.modules/"patch_optics.fatbin");
    cdf.load_module(c.modules/"phase_cdf.fatbin");
    tracer.trace_unpolarized(c.settings);
    cdf.validate_trace(tracer); // Small GPU reduction, no vertex CSV or bulk readback.
    patches.build(tracer);
    query.query_grid(patches,tracer.config(),c.theta,c.phi);
    const auto& query_stats=query.statistics();
    if(query_stats.boundary_resolved_directions || query_stats.boundary_directions)
        std::cout<<"Patch boundary ownership: resolved_directions="<<query_stats.boundary_resolved_directions
            <<" duplicate_reports_removed="<<query_stats.boundary_reports_removed
            <<" pending_directions="<<query_stats.boundary_directions<<'\n';
    if(query.statistics().error_directions)throw std::runtime_error("Patch query has numerical failures.");
    if(c.stage==PhaseDensityStage::Incoherent || c.stage==PhaseDensityStage::Path)
        optics.evaluate_device(patches,query,tracer.config());
    else optics.evaluate_wave_device(patches,query,tracer.config(),c.wave);
#if RAINBOW_ENABLE_DIAGNOSTICS
    const auto write_requested_report=[&]
    {
        if(c.patch_failure_report.empty())return;
        // Read the SAME device result used by CDF construction. Do not retrace,
        // rerun optics, replace flags, or switch to the legacy CSV CLI.
        const auto report=PatchFailureReport::capture(
            context,patches,query,optics,tracer.config(),c.wave);
        report.write_json(c.patch_failure_report);
        std::cout<<"Patch failure report: directions="<<report.snapshot().directions.size()
            <<", patches="<<report.snapshot().patches.size()
            <<", gpu_read_bytes="<<report.snapshot().gpu_read_bytes
            <<"; saved="<<c.patch_failure_report<<'\n';
    };
    try
    {
        cdf.build(optics,c.theta,c.phi,c.stage,c.policy,c.storage);
    }
    catch(...)
    {
        // A failed diagnostic write must not replace the original CDF failure.
        try {write_requested_report();}
        catch(const std::exception& e)
        {std::cerr<<"Patch failure report could not be saved: "<<e.what()<<'\n';}
        catch(...)
        {std::cerr<<"Patch failure report could not be saved: unknown diagnostic failure.\n";}
        throw;
    }
    write_requested_report();
#else
    cdf.build(optics,c.theta,c.phi,c.stage,c.policy,c.storage);
#endif
    DatasetDirectory destination(c.output);
    cdf.write_arrays(destination.staging());
    write_metadata(destination.staging()/"metadata.json",c,tracer.config(),patches,cdf,projected_area);
    destination.commit();
    const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout<<"Saved NumPy CDF dataset: "<<c.output<<"; elapsed_seconds="<<elapsed
        <<"; CDF_L1_mass_error="<<cdf.audit_statistics().l1_error
        <<"; lost_probability_mass="<<cdf.audit_statistics().lost_mass
        <<"; saved_grid="<<cdf.theta_count()<<'x'<<cdf.phi_count()
        <<"; hg_g="<<cdf.storage_statistics().g_stored
        <<"; projected_area_mm2="<<projected_area.area_mm2
        <<"; coarsening_TV="<<cdf.storage_statistics().coarsening_tv<<'\n';
    return 0;
}
}
#if defined(_WIN32)
int wmain(int argc,wchar_t** argv)
#else
int main(int argc,char** argv)
#endif
{
    try{return run(argc,argv);}
    catch(const std::exception& e){std::cerr<<"rainbow_generate: "<<e.what()<<'\n';return 1;}
}
