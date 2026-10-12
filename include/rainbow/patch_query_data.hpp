#pragma once

#include <rainbow/bilinear_patch_intersection.hpp>
#include <cstdint>
#include <cstddef>
#include <type_traits>

namespace rainbow
{

// 幾何の問い合わせ結果．強度・位相関数ではない．
enum PatchQueryFlags : std::uint32_t
{
    PatchQueryNone = 0,
    PatchQueryUnresolved = 1u << 0,       // 少なくとも一候補の孤立解を確定できない．
    PatchQueryInvalidData = 1u << 1,
    PatchQueryCountMismatch = 1u << 2,    // count/fill 不一致．容量を越えた書込みは行わない．
    PatchQueryRefinementHit = 1u << 3,    // NeedsRefinement に少なくとも一つ交差．
    PatchQueryBoundaryHit = 1u << 4,      // 少なくとも一つの境界の ownership が未解決．
    PatchQueryFp64Used = 1u << 5,
    PatchQueryDuplicateReport = 1u << 6,  // 同じ primitive/root の重複を除去した．
    PatchQuerySingularHit = 1u << 7,
    PatchQueryCountOverflow = 1u << 8,
    PatchQueryBoundaryResolved = 1u << 9  // 正則な入射格子の辺/頂点に一意の owner を割当てた．
};
inline constexpr std::uint32_t patch_query_error_mask =
    PatchQueryUnresolved | PatchQueryInvalidData | PatchQueryCountMismatch | PatchQueryCountOverflow;

struct PatchQueryHit
{
    std::uint32_t compact_index = 0;
    std::uint32_t patch_id = 0;     // 元の family/cell 番号（compact index と区別）．
    std::uint32_t root_index = 0;   // 同じ patch 内の u,v 辞書順．二つの根を保持．
    std::uint32_t flags = 0;        // BilinearHitFlags．
    float u = 0.0f, v = 0.0f, t = 0.0f, residual = 0.0f;
};

struct PatchQuerySummary
{
    std::uint32_t candidate_count = 0;   // count pass が確保した容量（方向ごと）．
    std::uint32_t hit_count = 0;         // 重複報告と，所有権が解決した共有境界の重複を除く．
    std::uint32_t regular_hits = 0;
    std::uint32_t refinement_hits = 0;
    std::uint32_t boundary_hits = 0;
    std::uint32_t flags = 0;
    std::uint32_t first_problem_patch_id = 0xffffffffu;
    std::uint32_t reserved = 0; // ABI-retained count of shared-boundary reports removed.
    [[nodiscard]] HOST_DEVICE std::uint32_t boundary_reports_removed() const noexcept
    { return reserved; }
};

// A feature of the SOURCE grid, not a coordinate in outgoing direction space.
// Equal first/second identifies a vertex; otherwise the sorted pair is an edge.
// Global vertex indices are path-major, so distinct optical families and
// nonadjacent overlapping sheets never acquire the same feature key.
struct PatchBoundaryFeature
{
    std::uint32_t first=0, second=0;
    bool valid=false;
    [[nodiscard]] HOST_DEVICE bool same(const PatchBoundaryFeature b) const noexcept
    { return valid && b.valid && first==b.first && second==b.second; }
};

[[nodiscard]] HOST_DEVICE inline PatchBoundaryFeature patch_boundary_feature(
    const OutgoingPatch& patch, const PatchQueryHit& hit) noexcept
{
    if(!(hit.flags&BilinearBoundary)) return {};
    const auto edges=(hit.flags&bilinear_boundary_edge_mask)>>bilinear_boundary_edge_shift;
    const bool u0=(edges&BilinearEdgeU0)!=0, u1=(edges&BilinearEdgeU1)!=0;
    const bool v0=(edges&BilinearEdgeV0)!=0, v1=(edges&BilinearEdgeV1)!=0;
    if(!edges || (u0&&u1) || (v0&&v1)
       || (u0&&hit.u!=0.0f) || (u1&&hit.u!=1.0f)
       || (v0&&hit.v!=0.0f) || (v1&&hit.v!=1.0f)) return {};
    for(unsigned i=0;i<4;++i) for(unsigned j=0;j<i;++j)
        if(patch.vertex_indices[i]==patch.vertex_indices[j]) return {};
    unsigned a=0,b=0;
    if((u0||u1)&&(v0||v1)) a=b=(u1?1u:0u)+(v1?2u:0u);
    else if(u0||u1) {a=u1?1u:0u;b=a+2u;}
    else {a=v1?2u:0u;b=a+1u;}
    auto first=patch.vertex_indices[a],second=patch.vertex_indices[b];
    if(second<first) {const auto tmp=first;first=second;second=tmp;}
    return {first,second,true};
}

// 1 direction を一つの raygen invocation だけが所有する．atomic/固定長 hit list は不要．
// count/fill 両方から同じ consume() を使い，分岐条件を別実装にしない．
struct PatchHitCollector
{
    PatchQueryHit* storage = nullptr; // count pass では nullptr．
    std::uint32_t capacity = 0;
    std::uint32_t count = 0;
    std::uint32_t flags = 0;
    bool writing = false;
    std::uint32_t first_problem_patch_id = 0xffffffffu;

    HOST_DEVICE void note(const std::uint32_t geometry_flags, const std::uint32_t patch_id = 0xffffffffu) noexcept
    {
        if((geometry_flags & (BilinearUnresolved | BilinearInvalidInput)) != 0
           && patch_id < first_problem_patch_id) first_problem_patch_id=patch_id;
        if(geometry_flags & BilinearUnresolved) flags |= PatchQueryUnresolved;
        if(geometry_flags & BilinearInvalidInput) flags |= PatchQueryInvalidData;
        if(geometry_flags & BilinearFp64Evaluation) flags |= PatchQueryFp64Used;
    }

    HOST_DEVICE void consume(
        const OutgoingPatch& patch, const std::uint32_t compact_index,
        const BilinearRayHits& intersections) noexcept
    {
        note(intersections.flags,patch.patch_id);
        for(std::uint32_t root=0;root<intersections.count;++root)
        {
            const auto& h=intersections.hits[root];
            if(count==0xffffffffu) {flags|=PatchQueryCountOverflow;return;}
            if(writing)
            {
                if(count>=capacity || storage==nullptr) flags|=PatchQueryCountMismatch;
                else storage[count]={compact_index,patch.patch_id,root,h.flags,h.u,h.v,h.t,h.residual};
            }
            ++count;
        }
    }

    // ソートは「近い順」ではなく patch_id/root_index 順．BVH の訪問順序に依存しない．
    // 通常の小さな交差集合向けの in-place insertion sort．固定個数で打切らない．
    [[nodiscard]] HOST_DEVICE PatchQuerySummary finish(
        const OutgoingPatch* patches, const std::uint32_t patch_count) noexcept
    {
        PatchQuerySummary result{};
        result.candidate_count=capacity;
        if(count!=capacity) flags|=PatchQueryCountMismatch;
        result.flags=flags;
        result.first_problem_patch_id=first_problem_patch_id;
        if(count>capacity || (capacity!=0 && storage==nullptr)) return result;
        for(std::uint32_t i=1;i<count;++i)
        {
            const auto value=storage[i]; std::uint32_t j=i;
            while(j!=0 && less(value,storage[j-1])) {storage[j]=storage[j-1];--j;}
            storage[j]=value;
        }
        std::uint32_t kept=0;
        for(std::uint32_t i=0;i<count;++i)
        {
            const auto h=storage[i];
            // Validate before either kind of duplicate removal. Otherwise a
            // malformed higher-ID report could disappear behind a valid owner.
            if(h.compact_index>=patch_count || !patches
               || h.patch_id!=patches[h.compact_index].patch_id
               || (h.flags&~bilinear_known_hit_flags)!=0u
               || (h.flags&BilinearBoundaryOwned)!=0u || h.root_index>1u
               || !(h.u>=0.0f && h.u<=1.0f && h.v>=0.0f && h.v<=1.0f)
               || !(h.t>0.0f && h.t<=0x1.fffffep127f)
               || !(h.residual>=0.0f && h.residual<=0x1.fffffep127f))
            {result.flags|=PatchQueryInvalidData;continue;}
            if((h.flags&bilinear_boundary_edge_mask)
               && !patch_boundary_feature(patches[h.compact_index],h).valid)
            {result.flags|=PatchQueryInvalidData;continue;}
            if(kept!=0 && h.compact_index==storage[kept-1].compact_index
                       && h.root_index==storage[kept-1].root_index)
            {result.flags|=PatchQueryDuplicateReport;continue;}
            storage[kept++]=h;
        }
        // The bilinear interpolant is piecewise defined on the incident grid.
        // A true shared edge/vertex is ONE preimage with several closed-chart
        // reports. Give it to the smallest patch_id, independently of BVH order.
        // The paper does not specify this measure-zero boundary convention.
        // A fold/singular or opposite-orientation meeting is deliberately left
        // pending: it is not made complete by deleting one of its branches.
        for(std::uint32_t i=0;i<kept;++i)
        {
            auto& h=storage[i];
            const auto& patch=patches[h.compact_index];
            const auto feature=patch_boundary_feature(patch,h);
            if(!feature.valid || !patch.has_regular_spherical_map()
               || (h.flags&BilinearSingular)) continue;
            bool compatible=true;
            for(std::uint32_t j=0;j<kept;++j)
            {
                if(i==j) continue;
                const auto& other=storage[j];
                const auto& adjacent=patches[other.compact_index];
                const auto other_feature=patch_boundary_feature(adjacent,other);
                if(!other_feature.valid && (other.flags&(BilinearBoundary|BilinearSingular)))
                {
                    // A retained but unclassified neighboring report cannot
                    // certify ownership of this shared source feature.
                    bool has_first=false,has_second=false;
                    for(const auto vertex:adjacent.vertex_indices)
                    {has_first|=vertex==feature.first;has_second|=vertex==feature.second;}
                    if(has_first&&has_second) compatible=false;
                }
                if(!feature.same(other_feature)) continue;
                if(adjacent.status!=patch.status || (other.flags&BilinearSingular)
                   || other.patch_id==h.patch_id) compatible=false;
            }
            if(compatible) h.flags|=BilinearBoundaryOwned;
        }
        std::uint32_t unique=0;
        for(std::uint32_t i=0;i<kept;++i)
        {
            const auto h=storage[i];
            if(h.flags&BilinearBoundaryOwned)
            {
                const auto feature=patch_boundary_feature(patches[h.compact_index],h);
                bool duplicate=false;
                for(std::uint32_t j=0;j<unique;++j)
                    if((storage[j].flags&BilinearBoundaryOwned)
                       && feature.same(patch_boundary_feature(patches[storage[j].compact_index],storage[j])))
                    {duplicate=true;break;}
                result.flags|=PatchQueryBoundaryResolved;
                if(duplicate) {++result.reserved;continue;}
            }
            storage[unique++]=h;
            const auto status=patches[h.compact_index].status;
            if(status==PatchCellStatus::NeedsRefinement)
            {++result.refinement_hits;result.flags|=PatchQueryRefinementHit;}
            else if(status==PatchCellStatus::RegularPositive || status==PatchCellStatus::RegularNegative)
                ++result.regular_hits;
            else result.flags|=PatchQueryInvalidData;
            if((h.flags&BilinearBoundary) && !(h.flags&BilinearBoundaryOwned))
            {++result.boundary_hits;result.flags|=PatchQueryBoundaryHit;}
            if(h.flags&BilinearSingular) result.flags|=PatchQuerySingularHit;
        }
        result.hit_count=unique;
        return result;
    }

private:
    HOST_DEVICE static bool less(const PatchQueryHit& a,const PatchQueryHit& b) noexcept
    {
        if(a.patch_id!=b.patch_id) return a.patch_id<b.patch_id;
        if(a.compact_index!=b.compact_index) return a.compact_index<b.compact_index;
        return a.root_index<b.root_index;
    }
};

static_assert(sizeof(PatchQueryHit)==32 && sizeof(PatchQuerySummary)==32);
static_assert(std::is_trivially_copyable_v<PatchQueryHit>);
static_assert(std::is_trivially_copyable_v<PatchQuerySummary>);
} // namespace rainbow
