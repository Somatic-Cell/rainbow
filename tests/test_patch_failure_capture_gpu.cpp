#include <rainbow/patch_failure_report.hpp>
#include <rainbow/raindrop_tracer.hpp>
#include <rainbow/patch_accel.hpp>
#include <rainbow/patch_query.hpp>
#include <rainbow/patch_optics.hpp>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
void require(bool condition,const char* message)
{if(!condition)throw std::runtime_error(message);}
template<class T> std::vector<T> download(const rainbow::DeviceBuffer<T>& buffer)
{std::vector<T> out(buffer.element_count());buffer.download(std::span<T>{out});return out;}
template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b)
{return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
struct TemporaryReports
{
    std::filesystem::path directory;
    TemporaryReports()
        : directory(std::filesystem::temp_directory_path()/
            ("rainbow_patch_capture_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
    {
        if(!std::filesystem::create_directory(directory))
            throw std::runtime_error("Could not reserve a fresh report test directory");
    }
    ~TemporaryReports()
    {
        std::error_code ignored;
        std::filesystem::remove_all(directory,ignored);
    }
};
std::string contents(const std::filesystem::path& path)
{
    std::ifstream input(path,std::ios::binary);
    if(!input)throw std::runtime_error("Could not read report test output");
    return {std::istreambuf_iterator<char>{input},std::istreambuf_iterator<char>{}};
}
}
int main(int argc,char** argv)
{
    try
    {
        if(argc!=5)throw std::invalid_argument("Expected raindrop.optixir patch_build.fatbin patch_query.optixir patch_optics.fatbin");
        rainbow::CudaContext cuda{0};rainbow::RaindropTracer tracer(cuda);
        rainbow::RaindropSettings settings{};
        settings.radius_mm=1.0f;settings.grid_width=settings.grid_height=129;
        const float inclination=20.0f*0.01745329251994329577f;
        settings.incident_direction={std::cos(inclination),-std::sin(inclination),0};
        tracer.create_pipeline(std::filesystem::path(argv[1]));tracer.trace(settings);
        rainbow::PatchAccel patches(cuda,tracer.optix_context());
        patches.load_module(std::filesystem::path(argv[2]));patches.build(tracer.vertices(),tracer.config());
        rainbow::PatchQuery query(cuda,tracer.optix_context());
        query.create_pipeline(std::filesystem::path(argv[3]));query.query_grid(patches,tracer.config(),90,180);
        rainbow::PatchOptics optics(cuda);optics.load_module(std::filesystem::path(argv[4]));
        rainbow::WaveOpticsSettings wave{};optics.evaluate_wave(patches,query,tracer.config(),wave);
        const auto before_vertices=download(tracer.vertices());
        const auto before_patches=download(patches.patches());
        const auto before_status=download(patches.cell_statuses());
        const auto before_hits=download(query.hits());const auto before_optics=download(optics.results());
        const auto report=rainbow::PatchFailureReport::capture(cuda,patches,query,optics,tracer.config(),wave);
        require(same(before_vertices,download(tracer.vertices())),"Capture altered source vertices");
        require(same(before_patches,download(patches.patches())),"Capture altered patch records");
        require(same(before_status,download(patches.cell_statuses())),"Capture altered cell classification");
        require(same(before_hits,download(query.hits())),"Capture altered query hits");
        require(same(before_optics,download(optics.results())),"Capture altered optical outputs");
        require(report.snapshot().origin==rainbow::PatchWitnessOrigin::GpuCapture,"Wrong provenance");
        require(report.snapshot().focal_unavailable_directions>=optics.wave_statistics().focal_errors,"Focal summary mismatch");

        // The generator uses this device-only evaluation of the SAME existing
        // trace/patch/query/configuration. Compare its captured witnesses with
        // the host-backed reference above, without changing capture's API.
        rainbow::PatchOptics device_optics(cuda);
        device_optics.load_module(std::filesystem::path(argv[4]));
        device_optics.evaluate_wave_device(patches,query,tracer.config(),wave);
        require(!device_optics.has_host_result() && device_optics.host_results().empty()
                && device_optics.host_focal_results().empty() && device_optics.host_diffraction_results().empty(),
                "Device-only evaluation unexpectedly published host results");
        const auto before_device_optics=download(device_optics.results());
        const auto before_device_focal=download(device_optics.focal_results());
        const auto before_device_diffraction=download(device_optics.diffraction_results());
        const auto optical_address=device_optics.results().address();
        const auto focal_address=device_optics.focal_results().address();
        const auto diffraction_address=device_optics.diffraction_results().address();
        const auto device_report=rainbow::PatchFailureReport::capture(
            cuda,patches,query,device_optics,tracer.config(),wave);
        require(!device_optics.has_host_result() && device_optics.host_results().empty()
                && device_optics.host_focal_results().empty() && device_optics.host_diffraction_results().empty(),
                "Device capture changed the host-result state");
        require(optical_address==device_optics.results().address()
                && focal_address==device_optics.focal_results().address()
                && diffraction_address==device_optics.diffraction_results().address(),
                "Device capture replaced optical output buffers");
        require(same(before_device_optics,download(device_optics.results())),"Device capture altered optical outputs");
        require(same(before_device_focal,download(device_optics.focal_results())),"Device capture altered focal outputs");
        require(same(before_device_diffraction,download(device_optics.diffraction_results())),
                "Device capture altered diffraction outputs");
        const auto& host_snapshot=report.snapshot();
        const auto& device_snapshot=device_report.snapshot();
        const auto scan_bytes=static_cast<std::uint64_t>(query.host_directions().size())
            *(sizeof(rainbow::PatchOpticalResult)+sizeof(rainbow::FocalOpticalResult)+sizeof(rainbow::DiffractionResult));
        require(device_snapshot.gpu_read_bytes==host_snapshot.gpu_read_bytes+scan_bytes,
                "Device capture did not scan each existing result exactly once");
        require(device_snapshot.gpu_read_calls>host_snapshot.gpu_read_calls+3u,
                "Device capture used full-size result readbacks instead of bounded chunks");

        // No exact count of pending patches is assumed: future correct classifier
        // improvements may recover them. Compare every serialized witness field
        // after removing only the documented difference in diagnostic readback
        // accounting. Serialization avoids comparing unspecified C++ padding.
        TemporaryReports files;
        const auto host_path=files.directory/"host.json";
        const auto device_path=files.directory/"device.json";
        report.write_json(host_path);require(std::filesystem::file_size(host_path)>0,"Empty report");
        auto comparable=device_snapshot;
        comparable.gpu_read_bytes=host_snapshot.gpu_read_bytes;
        comparable.gpu_read_calls=host_snapshot.gpu_read_calls;
        rainbow::PatchFailureReport(std::move(comparable)).write_json(device_path);
        require(contents(host_path)==contents(device_path),"Host-backed/device-only witness data differ");

        // Make a deterministic failure at the final (partial-chunk) record of
        // this TEST-OWNED output. Preserving this marker in both the witness and
        // GPU buffer proves capture reads the existing computation instead of
        // silently rerunning/repairing optics, even if every natural hit is now
        // valid. Restore the original record after checking the capture.
        require(!before_device_optics.empty(),"Missing result for the read-only capture regression");
        const auto marker_index=before_device_optics.size()-1u;
        auto marker=before_device_optics[marker_index];
        marker.flags|=rainbow::PatchOpticalBoundaryPending;
        marker.regular_partial_incoherent_s=-17.25;
        const auto marker_address=optical_address+marker_index*sizeof(marker);
        RAINBOW_CUDA_CHECK(cuMemcpyHtoD(marker_address,&marker,sizeof(marker)));
        const auto marked_device_optics=download(device_optics.results());
        const auto marked_report=rainbow::PatchFailureReport::capture(
            cuda,patches,query,device_optics,tracer.config(),wave);
        bool found_marker=false;
        for(const auto& direction:marked_report.snapshot().directions)
            if(direction.direction_id==marker_index)
            {
                require((direction.optical.flags&rainbow::PatchOpticalBoundaryPending)!=0u
                        && direction.optical.regular_partial_incoherent_s==-17.25,
                        "Capture reran optics or replaced the recorded failure");
                found_marker=true;
            }
        require(found_marker,"Device capture missed the final partial-chunk failure");
        require(!device_optics.has_host_result(),"Marked capture published host results");
        require(same(marked_device_optics,download(device_optics.results())),"Capture repaired the marked GPU record");
        require(same(before_device_focal,download(device_optics.focal_results())),"Marked capture changed focal outputs");
        require(same(before_device_diffraction,download(device_optics.diffraction_results())),
                "Marked capture changed diffraction outputs");
        RAINBOW_CUDA_CHECK(cuMemcpyHtoD(marker_address,&before_device_optics[marker_index],sizeof(marker)));
        require(same(before_device_optics,download(device_optics.results())),"Test did not restore the original optical result");
        require(same(before_vertices,download(tracer.vertices())),"Device capture altered source vertices");
        require(same(before_patches,download(patches.patches())),"Device capture altered patch records");
        require(same(before_status,download(patches.cell_statuses())),"Device capture altered cell classification");
        require(same(before_hits,download(query.hits())),"Device capture altered query hits");
        device_optics.close();
        optics.close();query.close();patches.close();tracer.close();
        std::cout<<"Host-backed/device-only GPU witnesses match; capture preserved buffers and did not reevaluate optics.\n";
        return 0;
    }
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
