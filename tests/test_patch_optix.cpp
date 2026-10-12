#include "patch_validation.hpp"
#include "regular64_regression.hpp"

#include <rainbow/patch_accel.hpp>
#include <rainbow/raindrop_tracer.hpp>
#include <rainbow/cuda_error.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <span>
#include <vector>

namespace
{
using namespace rainbow;
using namespace rainbow::tests;

void upload_vertices(DeviceBuffer<OutgoingVertex>& buffer,std::span<const OutgoingVertex> vertices,
                     const CudaContext& context)
{
    try
    {
        buffer.upload_async(vertices,context.stream());
        RAINBOW_CUDA_CHECK(cuStreamSynchronize(context.stream()));
    }
    catch(...)
    {
        static_cast<void>(detail::report_cuda_cleanup_result(
            cuStreamSynchronize(context.stream()),"cuStreamSynchronize(test upload cleanup)"));
        throw;
    }
}

void verify_download(const PatchAccel& accel,std::span<const OutgoingVertex> vertices)
{
    std::vector<PatchCellStatus> statuses(accel.cell_statuses().element_count());
    std::vector<OutgoingPatch> patches(accel.patches().element_count());
    std::vector<PatchAabb> aabbs(accel.aabbs().element_count());
    accel.cell_statuses().download(std::span<PatchCellStatus>{statuses});
    accel.patches().download(std::span<OutgoingPatch>{patches});
    accel.aabbs().download(std::span<PatchAabb>{aabbs});
    verify_patch_buffer(vertices,accel.layout(),statuses,patches,aabbs);
    std::array<std::uint64_t,patch_status_count> counts{};
    for(const auto status:statuses) ++counts[static_cast<unsigned>(status)];
    require_patch(counts==accel.statistics().cells,"GPU block histogram mismatch.");
    require_patch(accel.statistics().patch_count==patches.size(),"Patch statistics mismatch.");
    require_patch((accel.handle()!=0)==!patches.empty(),"Empty/nonempty GAS handle mismatch.");
}

int run(const std::filesystem::path& patch_module,const std::filesystem::path& trace_module)
{
    CudaContext cuda_context{0};
    {
        OptixContext optix_context(cuda_context);
        PatchAccel accel(cuda_context,optix_context);
        accel.load_module(patch_module);
        RaindropSettings settings;settings.grid_width=2;settings.grid_height=2;
        const auto config=settings.make_config();
        auto vertices=synthetic_patch_vertices();
        DeviceBuffer<OutgoingVertex> source(cuda_context);
        source.allocate(vertices.size());
        upload_vertices(source,vertices,cuda_context);
        accel.build(source,config);
        verify_download(accel,vertices);
        require_patch(accel.statistics().patch_count==2,"Folded geometry was silently discarded.");

        // Exercise the production classify/compact CUDA kernels on the exact
        // stored regression corners; no full 3001-grid retrace is needed here.
        vertices=regular64_regression_vertices();
        upload_vertices(source,vertices,cuda_context);
        accel.build(source,config);
        verify_download(accel,vertices);
        require_patch(accel.statistics().count(PatchCellStatus::RegularNegative)==1
                   && accel.statistics().count(PatchCellStatus::RegularPositive)==1
                   && accel.statistics().count(PatchCellStatus::NeedsRefinement)==2,
                      "CUDA FP64 fallback failed or accepted a fold/degeneracy.");

        // Rebuild with no outgoing corners. An empty result is not a dummy GAS.
        for(auto& vertex:vertices) vertex.status=VertexStatus::Miss;
        upload_vertices(source,vertices,cuda_context);
        accel.build(source,config);
        verify_download(accel,vertices);
        require_patch(accel.has_result() && accel.handle()==0,"Empty build failed.");

        // Invalid vertices are not treated as ordinary absent corners.
        vertices[0].status=VertexStatus::UnresolvedIntersection;
        upload_vertices(source,vertices,cuda_context);
        bool rejected=false;
        try {accel.build(source,config);} catch(const std::runtime_error&) {rejected=true;}
        require_patch(rejected && !accel.has_result(),"Invalid vertex did not fail the build.");
        require_patch(accel.statistics().count(PatchCellStatus::InvalidVertex)!=0,"Failure cause was lost.");

        // Ensure a failed build is followed by a valid rebuild without stale handles.
        vertices=synthetic_patch_vertices();
        upload_vertices(source,vertices,cuda_context);
        accel.build(source,config);
        verify_download(accel,vertices);
        accel.close();
        source.close();
    }
    {
        RaindropTracer tracer(cuda_context);
        tracer.create_pipeline(trace_module);
        PatchAccel accel(cuda_context,tracer.optix_context());
        accel.load_module(patch_module);
        // One grid has <256 cells/family; the other crosses several blocks.
        // Counts deliberately are not exact multiples of a CUDA block.
        for(unsigned test=0;test<2;++test)
        {
            RaindropSettings settings;
            settings.radius_mm=test==0?0.4f:1.0f;
            settings.grid_width=test==0?10u:18u;
            settings.grid_height=test==0?12u:20u;
            settings.incident_direction=test==0?Vec3{1,0,0}:Vec3{0.93969262f,-0.34202014f,0};
            tracer.trace(settings);
            accel.build(tracer.vertices(),tracer.config());
            const auto vertices=tracer.download_vertices(); // Reference check only, not build input.
            verify_download(accel,vertices);
            std::cout<<"Patch integration case "<<test<<": cells="<<accel.statistics().logical_cell_count
                     <<", patches="<<accel.statistics().patch_count<<'\n';
        }
        accel.close();
        tracer.close();
    }
    std::cout<<"Patch CUDA construction / OptiX GAS build: passed (no directional query yet).\n";
    return EXIT_SUCCESS;
}
} // namespace

#if defined(_WIN32)
int wmain(const int argc,wchar_t* argv[])
#else
int main(const int argc,char* argv[])
#endif
{
    if(argc!=3)
    {
        std::cerr<<"Usage: rainbow_patch_optix_tests <patch_build.fatbin> <raindrop_trace.optixir>\n";
        return EXIT_FAILURE;
    }
    try {return run(std::filesystem::path(argv[1]),std::filesystem::path(argv[2]));}
    catch(const std::exception& e){std::cerr<<"Error: "<<e.what()<<'\n';return EXIT_FAILURE;}
}
