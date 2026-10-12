#pragma once

#include <rainbow/patch_accel.hpp>
#include <rainbow/patch_query_launch_params.hpp>
#include <rainbow/query_direction_grid.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace rainbow
{
struct PatchQueryStatistics
{
    std::uint64_t allocated_hits=0, hits=0;
    std::uint32_t directions=0, nonempty_directions=0, error_directions=0;
    std::uint32_t refinement_directions=0, boundary_directions=0, fp64_directions=0;
    std::uint32_t boundary_resolved_directions=0;
    std::uint64_t boundary_reports_removed=0;
    std::uint32_t max_hits=0;
    std::uint64_t missing_source_cells=0, no_outgoing_source_cells=0;
};

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324) // SBT record に必要な alignment による padding．
#endif
// context を借用し，query 用 pipeline / SBT / 入出力だけを所有する．
// GAS は再構築せず PatchAccel のものを使う．query() は同期して戻る．
// 2-pass count/fill -> per-direction sorting．全体の hit 上限で切り捨てない．
// 同じ context の資源を渡し，query の間は patches/vertices を変更しないこと．
// 戻り後も patch_id を解釈する利用側は，対応する元の配列を保持すること．
class PatchQuery final
{
public:
    PatchQuery(const CudaContext& cuda_context,const OptixContext& optix_context) noexcept;
    ~PatchQuery() noexcept;
    PatchQuery(const PatchQuery&)=delete;
    PatchQuery& operator=(const PatchQuery&)=delete;
    PatchQuery(PatchQuery&&)=delete;
    PatchQuery& operator=(PatchQuery&&)=delete;

    void create_pipeline(const std::filesystem::path& optixir_path);
    void query(const PatchAccel& patches,std::span<const Vec3> directions);
    void query_grid(const PatchAccel& patches,const RaindropTraceConfig& source,
        std::uint32_t theta_count,std::uint32_t phi_count);

    // Read-only access for synchronous downstream optical evaluation.
    [[nodiscard]] bool has_result() const noexcept { return has_result_; }
    [[nodiscard]] const DeviceBuffer<Vec3>& directions() const noexcept { return directions_; }
    [[nodiscard]] const std::vector<Vec3>& host_directions() const noexcept { return host_directions_; }
    [[nodiscard]] const QueryDirectionGrid* direction_grid() const noexcept
    { return has_grid_ ? &grid_ : nullptr; }
    [[nodiscard]] bool matches_source(const PatchAccel& source) const noexcept
    {
        // An empty direction set has no referenced geometry. For nonempty
        // results this checks identity/layout, not in-place changes or address
        // reuse. Do not retrace/rebuild before the downstream consumer finishes.
        return has_result_ && source.has_result()
            && (host_directions_.empty()
                || (params_.vertices == reinterpret_cast<const OutgoingVertex*>(source.source_vertices_address())
                    && params_.patches == reinterpret_cast<const OutgoingPatch*>(source.patches().address())
                    && params_.traversable == static_cast<std::uint64_t>(source.handle())
                    && params_.patch_count == source.statistics().patch_count
                    && params_.vertex_count == source.layout().vertices_per_path * 4u));
    }
    [[nodiscard]] const PatchQueryStatistics& statistics() const noexcept {return statistics_;}
    [[nodiscard]] const DeviceBuffer<PatchQueryHit>& hits() const noexcept {return hits_;}
    [[nodiscard]] const DeviceBuffer<std::uint64_t>& offsets() const noexcept {return offsets_;}
    [[nodiscard]] const DeviceBuffer<PatchQuerySummary>& summaries() const noexcept {return summaries_;}
    [[nodiscard]] std::vector<PatchQueryHit> download_hits() const;
    [[nodiscard]] const std::vector<PatchQuerySummary>& host_summaries() const noexcept {return host_summaries_;}
    [[nodiscard]] const std::vector<std::uint64_t>& host_offsets() const noexcept {return host_offsets_;}
    void write_csv(const std::filesystem::path& path) const;
    void write_hits_csv(const std::filesystem::path& path) const;
    void close();
    [[nodiscard]] bool close_noexcept() noexcept;

private:
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HeaderRecord
    {char header[OPTIX_SBT_RECORD_HEADER_SIZE];};
    static_assert(sizeof(HeaderRecord)%OPTIX_SBT_RECORD_ALIGNMENT==0);
    void synchronize();
    void launch();
    void clear_result();

    const CudaContext& cuda_context_;
    const OptixContext& optix_context_;
    OptixModule module_=nullptr;
    OptixProgramGroup raygen_=nullptr,miss_=nullptr,hitgroup_=nullptr;
    OptixPipeline pipeline_=nullptr;
    DeviceBuffer<HeaderRecord> raygen_record_,miss_record_,hit_record_;
    DeviceBuffer<PatchQueryLaunchParams> device_params_;
    DeviceBuffer<Vec3> directions_;
    DeviceBuffer<std::uint64_t> offsets_;
    DeviceBuffer<PatchQuerySummary> summaries_;
    DeviceBuffer<PatchQueryHit> hits_;
    // async コピー元は同期・例外 cleanup 完了まで保持する．
    HeaderRecord host_raygen_{},host_miss_{},host_hit_{};
    PatchQueryLaunchParams params_{};
    std::vector<Vec3> host_directions_;
    std::vector<std::uint64_t> host_offsets_;
    std::vector<PatchQuerySummary> host_summaries_;
    PatchQueryStatistics statistics_{};
    QueryDirectionGrid grid_{};
    RaindropTraceConfig source_config_{};
    bool has_grid_=false,has_pending_work_=false,has_result_=false,is_closed_=false;
};
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
} // namespace rainbow
