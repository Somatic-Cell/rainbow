#include <rainbow/patch_failure_report.hpp>
#include <rainbow/patch_accel.hpp>
#include <rainbow/patch_query.hpp>
#include <rainbow/patch_optics.hpp>
#include <rainbow/cuda_error.hpp>
#include <cuda.h>

#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <utility>

namespace rainbow
{
namespace
{
// Read a checked subrange. Does not require a full patch/vertex download.
class WitnessReader
{
public:
    explicit WitnessReader(PatchFailureSnapshot& snapshot) : snapshot_(snapshot) {}
    template<class T> void read(const CUdeviceptr base,const std::size_t capacity,
                                const std::size_t first,const std::span<T> destination)
    {
        if(first>capacity || destination.size()>capacity-first)
            throw std::out_of_range("Patch audit: GPU read exceeds its source buffer.");
        if(destination.empty())return;
        const auto maximum=(std::numeric_limits<std::size_t>::max)();
        if(base==0 || first>maximum/sizeof(T) || destination.size()>maximum/sizeof(T))
            throw std::overflow_error("Patch audit: invalid device range.");
        const auto offset=first*sizeof(T),bytes=destination.size()*sizeof(T);
        const auto last=(std::numeric_limits<CUdeviceptr>::max)();
        if(offset>last-base || bytes>last-(base+offset))
            throw std::overflow_error("Patch audit: device address overflow.");
        RAINBOW_CUDA_CHECK(cuMemcpyDtoH(destination.data(),base+offset,bytes));
        snapshot_.gpu_read_bytes+=bytes; ++snapshot_.gpu_read_calls;
    }
    template<class T> T one(const CUdeviceptr base,const std::size_t capacity,const std::size_t index)
    {
        T result{};read(base,capacity,index,std::span<T>{&result,1});return result;
    }
private:
    PatchFailureSnapshot& snapshot_;
};
bool finite_optical(const PatchOpticalResult& o) noexcept
{
    const auto& f=o.regular_partial_path_field;
    const auto& g=o.regular_partial_path_field_second;
    return std::isfinite(o.regular_partial_incoherent_s)&&std::isfinite(o.regular_partial_incoherent_p)
        &&std::isfinite(o.regular_partial_path_s)&&std::isfinite(o.regular_partial_path_p)
        &&std::isfinite(f.s_real)&&std::isfinite(f.s_imag)&&std::isfinite(f.p_real)&&std::isfinite(f.p_imag)
        &&std::isfinite(g.s_real)&&std::isfinite(g.s_imag)&&std::isfinite(g.p_real)&&std::isfinite(g.p_imag);
}
bool finite_focal(const FocalOpticalResult& f) noexcept
{
    return std::isfinite(f.intensity_s)&&std::isfinite(f.intensity_p)
        &&std::isfinite(f.field.s_real)&&std::isfinite(f.field.s_imag)
        &&std::isfinite(f.field.p_real)&&std::isfinite(f.field.p_imag)
        &&std::isfinite(f.field_second.s_real)&&std::isfinite(f.field_second.s_imag)
        &&std::isfinite(f.field_second.p_real)&&std::isfinite(f.field_second.p_imag);
}
}

PatchFailureReport PatchFailureReport::capture(
    const CudaContext& context,const PatchAccel& source,const PatchQuery& query,
    const PatchOptics& optics,const RaindropTraceConfig& config,
    const WaveOpticsSettings& wave_settings)
{
    if(!source.has_result() || !query.matches_source(source) || !optics.has_result()
       || !optics.has_wave_result() || query.direction_grid()==nullptr)
        throw std::logic_error("Patch audit requires one unchanged completed trace/build/query/wave result.");
    if(source.is_unpolarized()!=optics.is_unpolarized())
        throw std::invalid_argument("Patch audit: incident-state representation mismatch.");
    const auto* grid=query.direction_grid();
    const auto n=query.host_summaries().size();
    const bool host_results=optics.has_host_result();
    if(grid->size()!=n || n!=query.host_directions().size() || query.host_offsets().size()!=n+1
       || optics.results().element_count()!=n || optics.focal_results().element_count()!=n
       || optics.diffraction_results().element_count()!=n
       || (host_results && (optics.host_results().size()!=n || optics.host_focal_results().size()!=n
                            || optics.host_diffraction_results().size()!=n))
       || config.grid_width!=source.layout().grid_width || config.grid_height!=source.layout().grid_height)
        throw std::invalid_argument("Patch audit: source/result layout mismatch.");
    context.make_current();
    // Both wave evaluation modes synchronize; this explicit boundary also protects
    // sparse readback if a future implementation changes its blocking behavior.
    RAINBOW_CUDA_CHECK(cuStreamSynchronize(context.stream()));
    PatchFailureSnapshot result{};
    result.origin=PatchWitnessOrigin::GpuCapture;
    result.input_polarization = source.is_unpolarized() ? IncidentPolarization::Unpolarized : IncidentPolarization::SingleJones;
    result.config=config;result.wave_settings=wave_settings;
    result.theta_count=grid->theta_count;result.phi_count=grid->phi_count;
    result.stored_patch_count=source.statistics().patch_count;
    result.missing_source_cells=source.statistics().count(PatchCellStatus::MissingCorners);
    result.no_outgoing_source_cells=source.statistics().count(PatchCellStatus::NoOutgoingCorners);
    WitnessReader reader(result);
    std::map<std::uint32_t,OutgoingPatch> patch_cache;
    std::map<std::uint32_t,OutgoingVertex> vertex_cache;
    std::map<std::uint32_t,Field32> second_cache;
    std::set<std::uint32_t> wanted_compact,problem_ids;
    // Device-only generation deliberately has no full optical host copy. Scan
    // those already-computed results in bounded chunks and retain only failed
    // directions. This is diagnostic readback, never another optical evaluation.
    // The legacy host-backed path below performs no additional result scan.
    constexpr std::size_t scan_capacity=4096;
    std::vector<PatchOpticalResult> optical_chunk;
    std::vector<FocalOpticalResult> focal_chunk;
    std::vector<DiffractionResult> diffraction_chunk;
    if(!host_results)
    {
        const auto capacity=n<scan_capacity?n:scan_capacity;
        optical_chunk.resize(capacity);focal_chunk.resize(capacity);diffraction_chunk.resize(capacity);
    }
    std::size_t chunk_begin=0,chunk_end=0;
    const auto read_patch=[&](const std::uint32_t compact)->const OutgoingPatch&
    {
        auto it=patch_cache.find(compact);
        if(it==patch_cache.end())
        {
            auto p=reader.one<OutgoingPatch>(source.patches().address(),source.patches().element_count(),compact);
            it=patch_cache.emplace(compact,p).first;
        }
        return it->second;
    };
    for(std::size_t i=0;i<n;++i)
    {
        if(!host_results && i==chunk_end)
        {
            chunk_begin=i;
            const auto count=n-i<scan_capacity?n-i:scan_capacity;
            reader.read(optics.results().address(),optics.results().element_count(),i,
                        std::span<PatchOpticalResult>{optical_chunk.data(),count});
            reader.read(optics.focal_results().address(),optics.focal_results().element_count(),i,
                        std::span<FocalOpticalResult>{focal_chunk.data(),count});
            reader.read(optics.diffraction_results().address(),optics.diffraction_results().element_count(),i,
                        std::span<DiffractionResult>{diffraction_chunk.data(),count});
            chunk_end=i+count;
        }
        const auto& q=query.host_summaries()[i];
        const auto& o=host_results?optics.host_results()[i]:optical_chunk[i-chunk_begin];
        const auto& f=host_results?optics.host_focal_results()[i]:focal_chunk[i-chunk_begin];
        const auto& d=host_results?optics.host_diffraction_results()[i]:diffraction_chunk[i-chunk_begin];
        const bool qe=(q.flags&patch_query_error_mask)!=0u;
        const bool op=!o.known_hits_complete(), fp=!f.valid(), dp=!d.valid();
        const bool nf=!finite_optical(o)||!finite_focal(f)
                      ||!std::isfinite(d.intensity_s)||!std::isfinite(d.intensity_p);
        result.query_error_directions+=qe;result.optical_incomplete_directions+=op;
        result.focal_unavailable_directions+=fp;result.diffraction_unavailable_directions+=dp;
        result.nonfinite_directions+=nf;
        result.underresolved_directions+=(d.flags&DiffractionUnderresolved)!=0u;
        if(!(qe||op||fp||dp||nf))continue; // underresolution alone is not a NaN witness
        FailedDirectionWitness record{};
        record.direction_id=static_cast<std::uint32_t>(i);record.direction=query.host_directions()[i];
        record.query=q;record.optical=o;record.focal=f;record.diffraction=d;
        const std::uint64_t first=query.host_offsets()[i],next=query.host_offsets()[i+1];
        if(next<first || first>query.hits().element_count() || next>query.hits().element_count()
           || q.hit_count>next-first)
            throw std::runtime_error("Patch audit: unsafe hit range; no out-of-bounds witness read performed.");
        record.hits.resize(q.hit_count); // only hit_count, NOT reserved capacity
        reader.read(query.hits().address(),query.hits().element_count(),
                    static_cast<std::size_t>(first),std::span<PatchQueryHit>{record.hits});
        for(const auto& hit:record.hits)
        {
            const auto& patch=read_patch(hit.compact_index);
            if(patch.patch_id!=hit.patch_id)
                throw std::runtime_error("Patch audit: hit/patch identity mismatch.");
            wanted_compact.insert(hit.compact_index);
        }
        for(const auto id:{q.first_problem_patch_id,o.first_problem_patch_id,f.first_problem_patch_id})
            if(id!=0xffffffffu)problem_ids.insert(id);
        result.directions.push_back(std::move(record));
    }
    // Some query errors have a first_problem ID but no accepted hit. Stable
    // compaction is ordered by patch_id; binary-search only those named records.
    for(const auto id:problem_ids)
    {
        std::uint32_t lo=0,hi=source.statistics().patch_count;
        while(lo<hi)
        {
            const auto mid=lo+(hi-lo)/2;
            if(read_patch(mid).patch_id<id)lo=mid+1;else hi=mid;
        }
        if(lo<source.statistics().patch_count && read_patch(lo).patch_id==id)wanted_compact.insert(lo);
        else result.absent_problem_patch_ids.push_back(id);
    }
    const auto vertex_count=std::size_t(source.layout().vertices_per_path)*4;
    for(const auto compact:wanted_compact)
    {
        FailedPatchWitness record{};record.compact_index=compact;record.patch=read_patch(compact);
        for(unsigned k=0;k<4;++k)
        {
            const auto id=record.patch.vertex_indices[k];
            auto it=vertex_cache.find(id);
            if(it==vertex_cache.end())
                it=vertex_cache.emplace(id,reader.one<OutgoingVertex>(source.source_vertices_address(),vertex_count,id)).first;
            record.vertices[k]=it->second;
            if(source.is_unpolarized())
            {
                auto jt=second_cache.find(id);
                if(jt==second_cache.end()) jt=second_cache.emplace(id,
                    reader.one<Field32>(source.source_second_fields_address(),vertex_count,id)).first;
                record.second_input_fields[k]=jt->second;
            }
        }
        result.patches.push_back(record);
    }
    return PatchFailureReport(std::move(result));
}
} // namespace rainbow
