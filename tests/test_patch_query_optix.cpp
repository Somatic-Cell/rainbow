#include "patch_query_validation.hpp"
#include <rainbow/patch_query.hpp>
#include <rainbow/cuda_error.hpp>

#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <span>
#include <vector>

namespace
{
int run(const std::filesystem::path& build_module,const std::filesystem::path& query_module)
{
    using namespace rainbow;
    using rainbow::tests::require_query;
    CudaContext cuda{0};
    OptixContext optix(cuda);
    auto host=tests::query_test_vertices();
    DeviceBuffer<OutgoingVertex> vertices(cuda);
    vertices.allocate(host.size());
    vertices.upload_async(std::span<const OutgoingVertex>{host},cuda.stream());
    // ホスト転送元を保持したまま同期し，例外時にも寿命を明確にする．
    RAINBOW_CUDA_CHECK(cuStreamSynchronize(cuda.stream()));
    RaindropTraceConfig config{};config.grid_width=2;config.grid_height=2;
    PatchAccel accel(cuda,optix);
    accel.load_module(build_module);
    accel.build(vertices,config);
    require_query(accel.statistics().patch_count==4,"Synthetic patch count.");
    PatchQuery query(cuda,optix);
    query.create_pipeline(query_module);
    const auto base=tests::front_patch();
    const std::vector<Vec3> directions={{0,0,1},{0,0,-1},{1,0,0},base.corners[0],
        Vec3{.11f,.05f,1}.normalized(),Vec3{-.1f,.03f,1}.normalized()};
    query.query(accel,directions);
    const auto first=query.download_hits();
    const auto first_summaries=query.host_summaries();
    const auto first_offsets=query.host_offsets();
    require_query(query.statistics().error_directions==0,"OptiX query returned a numerical/collection error.");
    require_query(first_summaries[0].hit_count==4 && first_summaries[0].refinement_hits==2,
        "All-hit collection must preserve two overlapping regular patches and both roots of a folded patch.");
    require_query(first_summaries[1].hit_count==1 && first_summaries[2].hit_count==0,
        "Directionality / miss handling.");
    for(std::size_t direction=0;direction<directions.size();++direction)
    {
        // Independent full scan (no BVH). Same geometry, independent long-double equations.
        std::uint32_t expected=0;
        for(unsigned family=0;family<4;++family)
        {
            BilinearPatchGeometry geometry{};
            for(unsigned k=0;k<4;++k)geometry.corners[k]=host[4*family+k].direction_drop;
            const auto reference=tests::ReferenceBilinearQuery::solve(geometry,directions[direction]);
            const auto cpu=BilinearPatchIntersector::intersect(geometry,directions[direction]);
            // Exact shared-corner convention is tested by CPU output; all interior roots use independent reference.
            const std::size_t roots=direction==3 ? cpu.count : reference.size();
            expected+=static_cast<std::uint32_t>(roots);
            for(std::size_t r=0;r<roots;++r)
            {
                bool found=false;
                for(std::uint32_t j=0;j<first_summaries[direction].hit_count;++j)
                {
                    const auto& h=first[static_cast<std::size_t>(first_offsets[direction]+j)];
                    const float u=direction==3 ? cpu.hits[r].u : float(reference[r].u);
                    const float v=direction==3 ? cpu.hits[r].v : float(reference[r].v);
                    if(h.patch_id==family && std::abs(h.u-u)<3e-5f && std::abs(h.v-v)<3e-5f)found=true;
                }
                require_query(found,"GPU omitted a full-scan reference intersection.");
            }
        }
        require_query(first_summaries[direction].hit_count==expected,"Unexpected GPU intersection count.");
    }
    // 同じ GAS を再問い合わせ．固定順序で比較できることを確認する．
    query.query(accel,directions);
    const auto second=query.download_hits();
    for(std::size_t i=0;i<first_summaries.size();++i)
    {
        require_query(query.host_summaries()[i].hit_count==first_summaries[i].hit_count,"Repeated query count.");
        for(std::uint32_t j=0;j<first_summaries[i].hit_count;++j)
        {
            const auto& a=first[static_cast<std::size_t>(first_offsets[i]+j)];
            const auto& b=second[static_cast<std::size_t>(query.host_offsets()[i]+j)];
            require_query(a.patch_id==b.patch_id&&a.root_index==b.root_index&&a.u==b.u&&a.v==b.v&&a.t==b.t,
                "Repeated query deterministic ordering/value.");
        }
    }
    // Two distinct path families, each with four regular patches sharing a
    // source-grid corner. GPU count/fill must keep all closed hits long enough
    // to choose one source-feature owner per family, with no traversal tie.
    config.grid_width=config.grid_height=3;
    host.assign(4u*9u,OutgoingVertex{});
    for(auto& vertex:host) vertex.status=VertexStatus::Miss;
    for(unsigned family=0;family<2;++family)
        for(unsigned y=0;y<3;++y) for(unsigned x=0;x<3;++x)
        {
            auto& vertex=host[family*9u+y*3u+x];
            vertex.direction_drop=Vec3{(float(x)-1.0f)*0.125f,(float(y)-1.0f)*0.125f,1}.normalized();
            vertex.basis_x=TransverseFrame::from_direction(vertex.direction_drop).e0;
            vertex.field.x={1,0};vertex.status=VertexStatus::Valid;
        }
    vertices.close(); // All preceding queries are synchronized; allocate the larger fixture.
    vertices.allocate(host.size());
    vertices.upload_async(std::span<const OutgoingVertex>{host},cuda.stream());
    RAINBOW_CUDA_CHECK(cuStreamSynchronize(cuda.stream()));
    accel.build(vertices,config);
    const std::vector<Vec3> boundary_directions={{0,0,1},Vec3{0.0625f,0,1}.normalized()};
    query.query(accel,boundary_directions);
    const auto owned_hits=query.download_hits();
    require_query(query.statistics().error_directions==0 && query.statistics().boundary_directions==0,
                  "GPU shared regular boundaries remained pending or erroneous.");
    for(unsigned i=0;i<boundary_directions.size();++i)
    {
        const auto& summary=query.host_summaries()[i];
        require_query(summary.hit_count==2 && (summary.flags&PatchQueryBoundaryResolved),
                      "GPU ownership removed an independent path family or retained a duplicate.");
        require_query(summary.boundary_reports_removed()==(i==0?6u:2u),
                      "GPU shared-corner/shared-edge duplicate accounting.");
        for(unsigned family=0;family<2;++family)
        {
            const auto& hit=owned_hits[std::size_t(query.host_offsets()[i])+family];
            require_query(hit.patch_id==family*4u+i && (hit.flags&BilinearBoundaryOwned),
                          "GPU boundary owner is not the smallest source patch ID.");
        }
    }
    for(auto& v:host)v.status=VertexStatus::Miss;
    vertices.upload_async(std::span<const OutgoingVertex>{host},cuda.stream());
    RAINBOW_CUDA_CHECK(cuStreamSynchronize(cuda.stream()));
    accel.build(vertices,config);
    require_query(accel.handle()==0,"Empty source GAS.");
    query.query(accel,directions);
    require_query(query.statistics().hits==0&&query.statistics().error_directions==0,"Empty GAS query.");
    query.query(accel,std::span<const Vec3>{});
    require_query(query.statistics().directions==0,"Zero query directions.");
    query.close();accel.close();vertices.close();
    std::cout<<"Patch query OptiX: all roots, overlap, orientation, repeat, empty GAS passed.\n";
    return EXIT_SUCCESS;
}
}
#if defined(_WIN32)
int wmain(int argc,wchar_t* argv[])
#else
int main(int argc,char* argv[])
#endif
{
    try
    {
        if(argc!=3)throw std::invalid_argument("Usage: query_test <patch_build.fatbin> <patch_query.optixir>");
        return run(std::filesystem::path{argv[1]},std::filesystem::path{argv[2]});
    }
    catch(const std::exception& e){std::cerr<<"Error: "<<e.what()<<'\n';return EXIT_FAILURE;}
}
