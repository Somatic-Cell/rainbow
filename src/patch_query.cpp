#include <rainbow/patch_query.hpp>
#include <rainbow/cuda_error.hpp>
#include <rainbow/optix_error.hpp>
#include <rainbow/read_binary_file.hpp>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rainbow
{
namespace
{
// caller の呼出し直後に SDK log を表示してから失敗を通知する．
// 既存の function-table 定義/optixInit/context を複製しない．
class QueryCompileLog
{
public:
    void reset() noexcept {
        std::fill(bytes_.begin(), bytes_.end(), '\0');
        size_=bytes_.size();
    }
    char* data() noexcept {return bytes_.data();}
    std::size_t* size_address() noexcept {return &size_;}
    void check(const OptixResult result,const char* operation)
    {
        const auto end=std::find(bytes_.begin(),bytes_.end(),'\0');
        const auto count=static_cast<std::size_t>(end-bytes_.begin());
        if(count) {std::fprintf(stderr,"[%s]\n",operation);std::fwrite(bytes_.data(),1,count,stderr);std::fputc('\n',stderr);}
        if(size_>bytes_.size())std::fprintf(stderr,"[%s log truncated: %zu bytes required]\n",operation,size_);
        detail::check_optix(result,operation,__FILE__,__LINE__);
    }
private:
    std::vector<char> bytes_ =
        std::vector<char>(1024u * 1024u, '\0');
    std::size_t size_ = bytes_.size();
};
}

PatchQuery::PatchQuery(const CudaContext& cuda_context,const OptixContext& optix_context) noexcept
    :cuda_context_(cuda_context),optix_context_(optix_context),
     raygen_record_(cuda_context),miss_record_(cuda_context),hit_record_(cuda_context),
     device_params_(cuda_context),directions_(cuda_context),offsets_(cuda_context),
     summaries_(cuda_context),hits_(cuda_context)
{}
PatchQuery::~PatchQuery() noexcept {static_cast<void>(close_noexcept());}
void PatchQuery::synchronize()
{
    RAINBOW_CUDA_CHECK(cuStreamSynchronize(cuda_context_.stream()));
    has_pending_work_=false;
}
void PatchQuery::create_pipeline(const std::filesystem::path& path)
{
    if(is_closed_ || module_) throw std::logic_error("Query pipeline already initialized or closed.");
    if(optix_context_.handle()==nullptr) throw std::logic_error("Query requires a live OptiX context.");
    cuda_context_.make_current();
    const auto ir=read_binary_file(path);
    OptixModuleCompileOptions mo{};
    mo.maxRegisterCount=OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    mo.optLevel=OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mo.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    OptixPipelineCompileOptions po{};
    po.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    po.numPayloadValues=2;
    po.numAttributeValues=0; // root pair は caller 所有の payload pointee に置く．
    po.pipelineLaunchParamsVariableName="patch_query_params";
    po.pipelineLaunchParamsSizeInBytes=sizeof(PatchQueryLaunchParams);
    po.usesPrimitiveTypeFlags=OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    QueryCompileLog log;
    auto result=optixModuleCreate(optix_context_.handle(),&mo,&po,ir.data(),ir.size(),
        log.data(),log.size_address(),&module_);
    log.check(result,"optixModuleCreate(patch query)");

    OptixProgramGroupOptions go{};
    OptixProgramGroupDesc desc{};
    desc.kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen.module=module_;desc.raygen.entryFunctionName="__raygen__patch_query";
    log.reset();result=optixProgramGroupCreate(optix_context_.handle(),&desc,1,&go,log.data(),log.size_address(),&raygen_);
    log.check(result,"optixProgramGroupCreate(query raygen)");
    desc={};desc.kind=OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc.miss.module=module_;desc.miss.entryFunctionName="__miss__patch_query";
    log.reset();result=optixProgramGroupCreate(optix_context_.handle(),&desc,1,&go,log.data(),log.size_address(),&miss_);
    log.check(result,"optixProgramGroupCreate(query miss)");
    desc={};desc.kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    desc.hitgroup.moduleIS=module_;desc.hitgroup.entryFunctionNameIS="__intersection__patch_query";
    desc.hitgroup.moduleAH=module_;desc.hitgroup.entryFunctionNameAH="__anyhit__patch_query";
    // CH は不要．最寄りの一枚ではなく AH で全交差を処理する．
    log.reset();result=optixProgramGroupCreate(optix_context_.handle(),&desc,1,&go,log.data(),log.size_address(),&hitgroup_);
    log.check(result,"optixProgramGroupCreate(query hitgroup)");
    OptixProgramGroup groups[]={raygen_,miss_,hitgroup_};
    OptixPipelineLinkOptions link{};link.maxTraceDepth=1;
    log.reset();result=optixPipelineCreate(optix_context_.handle(),&po,&link,groups,3,
        log.data(),log.size_address(),&pipeline_);
    log.check(result,"optixPipelineCreate(patch query)");
    OptixStackSizes stacks{};
    for(auto group:groups) RAINBOW_OPTIX_CHECK(optixUtilAccumulateStackSizes(group,&stacks,pipeline_));
    unsigned traversal=0,state=0,continuation=0;
    RAINBOW_OPTIX_CHECK(optixUtilComputeStackSizes(&stacks,1,0,0,&traversal,&state,&continuation));
    RAINBOW_OPTIX_CHECK(optixPipelineSetStackSize(pipeline_,traversal,state,continuation,1));
    RAINBOW_OPTIX_CHECK(optixSbtRecordPackHeader(raygen_,&host_raygen_));
    RAINBOW_OPTIX_CHECK(optixSbtRecordPackHeader(miss_,&host_miss_));
    RAINBOW_OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_,&host_hit_));
    raygen_record_.allocate(1);miss_record_.allocate(1);hit_record_.allocate(1);device_params_.allocate(1);
    const auto stream=cuda_context_.stream();has_pending_work_=true;
    raygen_record_.upload_async(std::span<const HeaderRecord>{&host_raygen_,1},stream);
    miss_record_.upload_async(std::span<const HeaderRecord>{&host_miss_,1},stream);
    hit_record_.upload_async(std::span<const HeaderRecord>{&host_hit_,1},stream);
    synchronize();
}

void PatchQuery::clear_result()
{
    has_result_=false;has_grid_=false;statistics_={};
    hits_.close();summaries_.close();offsets_.close();directions_.close();
    host_offsets_.clear();host_summaries_.clear();host_directions_.clear();
}
void PatchQuery::launch()
{
    const auto stream=cuda_context_.stream();has_pending_work_=true;
    device_params_.upload_async(std::span<const PatchQueryLaunchParams>{&params_,1},stream);
    OptixShaderBindingTable sbt{};
    sbt.raygenRecord=raygen_record_.address();
    sbt.missRecordBase=miss_record_.address();sbt.missRecordCount=1;sbt.missRecordStrideInBytes=sizeof(HeaderRecord);
    sbt.hitgroupRecordBase=hit_record_.address();sbt.hitgroupRecordCount=1;sbt.hitgroupRecordStrideInBytes=sizeof(HeaderRecord);
    RAINBOW_OPTIX_CHECK(optixLaunch(pipeline_,stream,device_params_.address(),sizeof(params_),&sbt,
        params_.direction_count,1,1));
    synchronize();
}
void PatchQuery::query(const PatchAccel& source,std::span<const Vec3> directions)
{
    if(is_closed_ || !pipeline_) throw std::logic_error("Initialize the query pipeline first.");
    if(!source.has_result()) throw std::logic_error("PatchAccel has no completed result.");
    if(directions.size()>0xffffffffull) throw std::length_error("Too many query directions.");
    for(const auto d:directions)
        if(!d.is_finite() || !(d.dot(d)>0.99f && d.dot(d)<1.01f))
            throw std::invalid_argument("Query directions must be finite unit vectors.");
    cuda_context_.make_current();
    if(has_pending_work_)synchronize();
    clear_result();
    host_directions_.assign(directions.begin(),directions.end());
    const auto n=static_cast<std::uint32_t>(directions.size());
    statistics_.directions=n;
    statistics_.missing_source_cells=source.statistics().count(PatchCellStatus::MissingCorners);
    statistics_.no_outgoing_source_cells=source.statistics().count(PatchCellStatus::NoOutgoingCorners);
    host_offsets_.resize(std::size_t(n)+1,0);host_summaries_.resize(n);
    offsets_.allocate(std::size_t(n)+1);
    if(n==0)
    {
        has_pending_work_=true;offsets_.upload_async(std::span<const std::uint64_t>{host_offsets_},cuda_context_.stream());synchronize();
        has_result_=true;return;
    }
    directions_.allocate(n);summaries_.allocate(n);
    params_={};
    params_.traversable=static_cast<std::uint64_t>(source.handle());
    params_.vertices=reinterpret_cast<const OutgoingVertex*>(source.source_vertices_address());
    params_.patches=reinterpret_cast<const OutgoingPatch*>(source.patches().address());
    params_.directions=directions_.data();params_.summaries=summaries_.data();
    params_.patch_count=source.statistics().patch_count;
    params_.vertex_count=source.layout().vertices_per_path*4u;
    params_.direction_count=n;
    has_pending_work_=true;
    directions_.upload_async(std::span<const Vec3>{host_directions_},cuda_context_.stream());
    launch(); // count pass: hit storage is not yet allocated.
    summaries_.download(std::span<PatchQuerySummary>{host_summaries_});
    std::uint64_t total=0;
    for(std::uint32_t i=0;i<n;++i)
    {
        const auto& summary=host_summaries_[i];
        if(summary.flags&PatchQueryCountOverflow) throw std::overflow_error("Per-direction hit count overflow.");
        host_offsets_[i]=total;total+=summary.candidate_count;
        if(total>std::uint64_t((std::numeric_limits<std::size_t>::max)()/sizeof(PatchQueryHit)))
            throw std::length_error("Query hit storage exceeds addressable memory.");
    }
    host_offsets_[n]=total;
    hits_.allocate(static_cast<std::size_t>(total)); // bad_alloc/CUDA error, never silently truncate.
    statistics_.allocated_hits=total;
    params_.offsets=offsets_.data();params_.hits=hits_.data();params_.write_pass=1;
    has_pending_work_=true;
    offsets_.upload_async(std::span<const std::uint64_t>{host_offsets_},cuda_context_.stream());
    launch(); // fill, then deterministic per-direction ordering on GPU.
    summaries_.download(std::span<PatchQuerySummary>{host_summaries_});
    for(std::uint32_t i=0;i<n;++i)
    {
        const auto& s=host_summaries_[i];
        if(s.hit_count>host_offsets_[i+1]-host_offsets_[i])
            throw std::runtime_error("Query result exceeds its reserved segment.");
        statistics_.hits+=s.hit_count;
        statistics_.nonempty_directions+=s.hit_count!=0;
        statistics_.error_directions+=(s.flags&patch_query_error_mask)!=0;
        statistics_.refinement_directions+=(s.flags&PatchQueryRefinementHit)!=0;
        statistics_.boundary_directions+=(s.flags&PatchQueryBoundaryHit)!=0;
        statistics_.boundary_resolved_directions+=(s.flags&PatchQueryBoundaryResolved)!=0;
        statistics_.boundary_reports_removed+=s.boundary_reports_removed();
        statistics_.fp64_directions+=(s.flags&PatchQueryFp64Used)!=0;
        statistics_.max_hits=(std::max)(statistics_.max_hits,s.hit_count);
    }
    // Unresolved は結果に明示する．CLI は CSV 保存後に非ゼロ終了する．
    has_result_=true;
}
void PatchQuery::query_grid(const PatchAccel& patches,const RaindropTraceConfig& source,
    const std::uint32_t theta_count,const std::uint32_t phi_count)
{
    const auto grid=QueryDirectionGrid::make(source,theta_count,phi_count);
    const auto directions=grid.directions();
    query(patches,std::span<const Vec3>{directions});
    grid_=grid;source_config_=source;has_grid_=true;
}
std::vector<PatchQueryHit> PatchQuery::download_hits() const
{
    if(!has_result_)throw std::logic_error("No completed query.");
    std::vector<PatchQueryHit> result(hits_.element_count());
    hits_.download(std::span<PatchQueryHit>{result});return result;
}
void PatchQuery::write_csv(const std::filesystem::path& path) const
{
    if(!has_result_)throw std::logic_error("No completed query.");
    std::ofstream out(path);
    if(!out)throw std::runtime_error("Cannot open query CSV.");
    out<<std::setprecision(17)
       <<"# format=rainbow_patch_queries_v1\n# quantity=geometric_intersection_count_NOT_intensity\n"
       <<"# optical_complete=false\n# closed_patch_boundaries=true\n"
       <<"# missing_source_cells="<<statistics_.missing_source_cells
       <<"\n# no_outgoing_source_cells="<<statistics_.no_outgoing_source_cells<<'\n';
    if(has_grid_)
    {
        out<<"# theta_count="<<grid_.theta_count<<"\n# phi_count="<<grid_.phi_count
           <<"\n# order=theta_major_phi_minor\n# angle_units=radians\n"
           <<"# theta=acos(incident_propagation_direction dot outgoing_direction)\n"
           <<"# phi=atan2(outgoing dot incident_basis_y,outgoing dot incident_basis_x)\n"
           <<"# radius_mm="<<source_config_.radius_mm<<"\n# wavelength_nm="<<source_config_.wavelength_nm
           <<"\n# interior_index="<<source_config_.interior_index<<"\n# exterior_index="<<source_config_.exterior_index
           <<"\n# incident_grid="<<source_config_.grid_width<<','<<source_config_.grid_height<<'\n';
        out<<"# incident_direction="<<grid_.axis.x<<','<<grid_.axis.y<<','<<grid_.axis.z
           <<"\n# incident_basis_x="<<grid_.e0.x<<','<<grid_.e0.y<<','<<grid_.e0.z
           <<"\n# incident_field="<<source_config_.incident_field.x.real<<','<<source_config_.incident_field.x.imag
           <<','<<source_config_.incident_field.y.real<<','<<source_config_.incident_field.y.imag<<'\n';
        out<<"# coefficients=";
        for(unsigned i=0;i<8;++i)out<<(i?",":"")<<source_config_.shape.coefficients[i];
        out<<"\n# incident_grid_half_extent_drop="<<source_config_.grid_half_extent<<'\n';
    }
    out<<"direction_id,theta_index,phi_index,theta_rad,phi_rad,solid_angle_sr,wx,wy,wz,offset,capacity,hit_count,regular_hits,refinement_hits,boundary_hits,flags,first_problem_patch_id\n";
    for(std::size_t i=0;i<host_directions_.size();++i)
    {
        const auto& w=host_directions_[i];const auto& s=host_summaries_[i];
        out<<i<<',';
        if(has_grid_)
        {
            const auto row=static_cast<std::uint32_t>(i/grid_.phi_count),col=static_cast<std::uint32_t>(i%grid_.phi_count);
            out<<row<<','<<col<<','<<grid_.theta(row)<<','<<grid_.phi(col)<<','<<grid_.solid_angle(row)<<',';
        }
        else out<<"-1,-1,nan,nan,nan,";
        out<<w.x<<','<<w.y<<','<<w.z<<','<<host_offsets_[i]<<','<<s.candidate_count<<','<<s.hit_count<<','
           <<s.regular_hits<<','<<s.refinement_hits<<','<<s.boundary_hits<<','<<s.flags<<',';
        if(s.first_problem_patch_id==0xffffffffu)out<<-1;else out<<s.first_problem_patch_id;
        out<<'\n';
    }
    out.flush();if(!out)throw std::runtime_error("Writing query CSV failed.");
}
void PatchQuery::write_hits_csv(const std::filesystem::path& path) const
{
    const auto data=download_hits();
    std::ofstream out(path);
    if(!out)throw std::runtime_error("Cannot open query-hit CSV.");
    out<<std::setprecision(std::numeric_limits<float>::max_digits10)
       <<"# format=rainbow_patch_query_hits_v1\n# quantity=geometry_NOT_optical_contribution\n"
       <<"direction_id,hit_index,compact_index,patch_id,root_index,u,v,t,residual,flags\n";
    for(std::size_t i=0;i<host_summaries_.size();++i)
        for(std::uint32_t j=0;j<host_summaries_[i].hit_count;++j)
        {
            const auto& h=data[static_cast<std::size_t>(host_offsets_[i]+j)];
            out<<i<<','<<j<<','<<h.compact_index<<','<<h.patch_id<<','<<h.root_index<<','
               <<h.u<<','<<h.v<<','<<h.t<<','<<h.residual<<','<<h.flags<<'\n';
        }
    out.flush();if(!out)throw std::runtime_error("Writing query-hit CSV failed.");
}
bool PatchQuery::close_noexcept() noexcept
{
    if(is_closed_)return true;
    if(!detail::report_cuda_cleanup_result(cuCtxSetCurrent(cuda_context_.handle()),"cuCtxSetCurrent(query cleanup)"))return false;
    bool ok=true;
    if(has_pending_work_)
    {ok=detail::report_cuda_cleanup_result(cuStreamSynchronize(cuda_context_.stream()),"cuStreamSynchronize(query cleanup)")&&ok;has_pending_work_=false;}
    ok=hits_.close_noexcept()&&ok;ok=summaries_.close_noexcept()&&ok;
    ok=offsets_.close_noexcept()&&ok;ok=directions_.close_noexcept()&&ok;
    ok=device_params_.close_noexcept()&&ok;
    ok=raygen_record_.close_noexcept()&&ok;ok=miss_record_.close_noexcept()&&ok;ok=hit_record_.close_noexcept()&&ok;
    if(pipeline_){ok=detail::report_optix_cleanup_result(optixPipelineDestroy(pipeline_),"optixPipelineDestroy(query)")&&ok;pipeline_=nullptr;}
    for(auto* group:{&hitgroup_,&miss_,&raygen_})
        if(*group){ok=detail::report_optix_cleanup_result(optixProgramGroupDestroy(*group),"optixProgramGroupDestroy(query)")&&ok;*group=nullptr;}
    if(module_){ok=detail::report_optix_cleanup_result(optixModuleDestroy(module_),"optixModuleDestroy(query)")&&ok;module_=nullptr;}
    has_result_=false;is_closed_=true;return ok;
}
void PatchQuery::close(){if(!close_noexcept())throw std::runtime_error("Patch query cleanup failed; see stderr.");}
} // namespace rainbow
