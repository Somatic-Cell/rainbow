#pragma once

#include <rainbow/patch_optical_evaluator.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace rainbow::tests
{
namespace boundary_ownership_detail
{
inline void require(bool condition, const char* message)
{
    if(!condition) throw std::runtime_error(message);
}
inline bool near(double a, double b)
{
    return std::isfinite(a) && std::isfinite(b)
        && std::abs(a-b) <= 2e-6*(1.0+std::abs(b));
}

struct Mesh
{
    std::vector<OutgoingVertex> vertices;
    std::vector<Field32> second_fields;
    std::vector<OutgoingPatch> patches;
    std::vector<BilinearPatchGeometry> geometry;
};
inline void append_vertex(Mesh& m, Vec3 direction)
{
    OutgoingVertex v{};
    v.direction_drop=direction;
    v.basis_x=TransverseFrame::from_direction(direction).e0;
    v.field={{2,0},{0,0}};
    v.optical_cycles={0,0.25f};
    v.status=VertexStatus::Valid;
    m.vertices.push_back(v);
    m.second_fields.push_back({{0,0},{3,0}});
}
inline void append_patch(Mesh& m, std::uint32_t id,
                         const std::array<std::uint32_t,4>& indices,
                         PatchCellStatus status=PatchCellStatus::RegularPositive)
{
    OutgoingPatch p{};p.patch_id=id;p.status=status;
    // Synthetic optical metrics are explicit arithmetic fixtures. They are
    // not assertions about the solid angle of the synthetic geometry.
    p.incident_area_drop2=1.0f;p.signed_solid_angle_sr=0.25f;
    BilinearPatchGeometry g{};
    for(unsigned k=0;k<4;++k)
    {
        p.vertex_indices[k]=indices[k];
        g.corners[k]=m.vertices[indices[k]].direction_drop;
    }
    m.patches.push_back(p);m.geometry.push_back(g);
}
inline Mesh corner_mesh()
{
    Mesh m;
    for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x)
        append_vertex(m,Vec3{float(x)*0.125f,float(y)*0.125f,1}.normalized());
    m.vertices[4].basis_x={0,-1,0}; // Scattering s axis for incident +x, output +z.
    append_patch(m,0,{0,1,3,4});append_patch(m,1,{1,2,4,5});
    append_patch(m,2,{3,4,6,7});append_patch(m,3,{4,5,7,8});
    return m;
}
inline Mesh edge_mesh()
{
    Mesh m;
    for(int y : {-1,1}) for(int x=-1;x<=1;++x)
        append_vertex(m,Vec3{float(x)*0.125f,float(y)*0.125f,1}.normalized());
    append_patch(m,10,{0,1,3,4});append_patch(m,11,{1,2,4,5});
    return m;
}
struct Collected
{
    // Storage retains the count-pass capacity; only summary.hit_count is live.
    std::vector<PatchQueryHit> storage;
    PatchQuerySummary summary{};
};
inline Collected collect(const Mesh& m, const std::vector<BilinearRayHits>& raw,
                          const std::vector<unsigned>& traversal)
{
    require(raw.size()==m.patches.size(),"Ownership test input size mismatch.");
    PatchHitCollector counter{};
    for(unsigned i : traversal) counter.consume(m.patches[i],i,raw[i]);
    Collected out;out.storage.resize(counter.count);
    PatchHitCollector writer{out.storage.data(),counter.count,0,0,true};
    for(unsigned i : traversal) writer.consume(m.patches[i],i,raw[i]);
    out.summary=writer.finish(m.patches.data(),static_cast<unsigned>(m.patches.size()));
    return out;
}
inline std::vector<BilinearRayHits> intersections(const Mesh& m, Vec3 direction)
{
    std::vector<BilinearRayHits> raw;
    for(const auto& g:m.geometry)raw.push_back(BilinearPatchIntersector::intersect(g,direction));
    return raw;
}
inline Collected query(const Mesh& m, Vec3 direction={0,0,1})
{
    std::vector<unsigned> traversal(m.patches.size());
    std::iota(traversal.begin(),traversal.end(),0u);
    return collect(m,intersections(m,direction),traversal);
}
inline void require_owned(const Collected& q, unsigned count)
{
    require(q.summary.hit_count==count,"Shared source feature did not retain the expected owner count.");
    require((q.summary.flags&patch_query_error_mask)==0u,"Ownership introduced a query error.");
    require((q.summary.flags&PatchQueryBoundaryHit)==0u,"Certified regular ownership remains pending.");
    require((q.summary.flags&PatchQueryBoundaryResolved)!=0u,"Resolved ownership lost its diagnostic.");
    for(unsigned i=0;i<count;++i)
        require((q.storage[i].flags&BilinearBoundary)!=0u
                &&(q.storage[i].flags&BilinearBoundaryOwned)!=0u,
                "The retained boundary owner must preserve its boundary provenance.");
}
inline PatchOpticalResult optical_result(const Mesh& m, const Collected& q)
{
    const Vec3 direction{0,0,1};
    const std::uint64_t offsets[2]={0,static_cast<std::uint64_t>(q.storage.size())};
    PatchOpticsParams p{};
    p.vertices=m.vertices.data();p.vertex_count=static_cast<unsigned>(m.vertices.size());
    p.second_input_fields=m.second_fields.data();p.second_input_count=p.vertex_count;
    p.input_polarization=IncidentPolarization::Unpolarized;
    p.patches=m.patches.data();p.patch_count=static_cast<unsigned>(m.patches.size());
    p.directions=&direction;p.direction_count=1;p.offsets=offsets;
    p.summaries=&q.summary;p.hits=q.storage.data();p.hit_storage_count=q.storage.size();
    p.incident_direction={1,0,0};p.incident_basis_x={0,0,1};
    return PatchOpticalEvaluator::evaluate_direction(p,0);
}

inline void test_edge_and_single_owner()
{
    auto m=edge_mesh();auto q=query(m);
    require(q.summary.candidate_count==2u,"Exact edge must be reported by both closed patches.");
    require_owned(q,1);
    require(q.storage[0].patch_id==10u,"Exact edge owner is not the smallest source patch id.");
    // A valid boundary chart can exist without an adjacent stored chart.
    // Owning this reported hit does not certify missing source-cell coverage.
    m.patches.erase(m.patches.begin());m.geometry.erase(m.geometry.begin());
    q=query(m);require_owned(q,1);
    require(q.storage[0].patch_id==11u,"A single valid boundary chart was discarded.");
}

inline void test_corner_order_and_optics()
{
    auto m=corner_mesh();
    const std::uint32_t ids[4]={31,7,19,43};
    for(unsigned i=0;i<4;++i)
    {
        m.patches[i].patch_id=ids[i];
        m.patches[i].signed_solid_angle_sr=i==1u?0.25f:float(i+2u);
    }
    const auto raw=intersections(m,{0,0,1});
    std::array<unsigned,4> order{0,1,2,3};
    do
    {
        const auto q=collect(m,raw,{order.begin(),order.end()});
        require(q.summary.candidate_count==4u,"Four-way corner count-pass capacity changed.");
        require_owned(q,1);
        require(q.storage[0].patch_id==7u,"Corner ownership depends on traversal order.");
    }while(std::next_permutation(order.begin(),order.end()));
    const auto q=collect(m,raw,{3,1,0,2,1});
    require_owned(q,1);
    require(q.summary.candidate_count==5u&&(q.summary.flags&PatchQueryDuplicateReport)!=0u,
            "Exact primitive duplicate removal was lost during boundary ownership.");
    const auto r=optical_result(m,q);
    require(r.flags==0u&&r.known_hits_complete()&&r.evaluated_hits==1u&&r.rejected_hits==0u,
            "Owned regular corner did not reach complete optical evaluation.");
    // At the selected vertex E0=(2,0), E1=(0,3), phase=1/4 turn,
    // area/omega=4. These are independent arithmetic expectations. The two
    // input columns are incoherently averaged; four charts are not four paths.
    require(near(r.regular_partial_path_field.s_real,0)
            &&near(r.regular_partial_path_field.s_imag,4)
            &&near(r.regular_partial_path_field.p_real,0)
            &&near(r.regular_partial_path_field.p_imag,0)
            &&near(r.regular_partial_path_field_second.p_real,0)
            &&near(r.regular_partial_path_field_second.p_imag,6),
            "Boundary owner changed the optical phase, polarization or amplitude.");
    require(near(r.regular_partial_incoherent_s,8)&&near(r.regular_partial_incoherent_p,18)
            &&near(r.regular_partial_path_s,8)&&near(r.regular_partial_path_p,18),
            "Shared charts were summed, averaged, or scaled as separate physical paths.");
}

inline void test_independent_source_features()
{
    auto m=corner_mesh();const auto originals=m.patches;
    // Two path families with the same output directions have distinct
    // path-major vertex IDs. Their four-way source corners must remain two hits.
    for(unsigned i=0;i<9;++i)
    {m.vertices.push_back(m.vertices[i]);m.second_fields.push_back(m.second_fields[i]);}
    for(unsigned i=0;i<4;++i)
    {
        std::array<std::uint32_t,4> ids{};
        for(unsigned k=0;k<4;++k)ids[k]=originals[i].vertex_indices[k]+9u;
        append_patch(m,4u+i,ids);
    }
    auto q=query(m);require_owned(q,2);
    require(q.storage[0].patch_id==0u&&q.storage[1].patch_id==4u,
            "Coincident outputs from different path families were merged.");

    // Equal geometric edge positions from disjoint source cells also represent
    // distinct paths even if they are assigned to the same path family.
    m=edge_mesh();m.patches.resize(1);m.geometry.resize(1);
    const auto first=m.patches[0];
    for(unsigned i=0;i<6;++i)
    {m.vertices.push_back(m.vertices[i]);m.second_fields.push_back(m.second_fields[i]);}
    std::array<std::uint32_t,4> disjoint{};
    for(unsigned k=0;k<4;++k)disjoint[k]=first.vertex_indices[k]+6u;
    append_patch(m,100,disjoint);
    q=query(m);require_owned(q,2);
    require(q.storage[0].patch_id==10u&&q.storage[1].patch_id==100u,
            "Geometric coincidence was used in place of shared source IDs.");
}

inline void test_packed_interior_is_not_owned()
{
    constexpr float extent=0x1p-10f,tiny=0x1p-37f;
    Mesh m;
    for(float y : {-extent,extent})for(float x : {-extent,tiny,extent})
        append_vertex(m,{x,y,1});
    append_patch(m,0,{0,1,3,4});append_patch(m,1,{1,2,4,5});
    const auto q=query(m);
    require(q.summary.hit_count==1u&&q.storage[0].patch_id==0u,
            "Packed interior endpoint acquired an adjacent patch hit.");
    require(q.storage[0].u==1.0f&&q.storage[0].v==0.5f,
            "Packed endpoint witness changed.");
    // Exact u=2^27/(2^27+1)<1, although binary32 storage rounds it to one.
    require((q.storage[0].flags&(BilinearBoundary|BilinearBoundaryOwned))==0u
            &&(q.summary.flags&(PatchQueryBoundaryHit|PatchQueryBoundaryResolved))==0u,
            "Packed coordinates were mistaken for an exact shared source feature.");
}

inline void test_pending_and_error_contracts()
{
    auto m=edge_mesh();
    // Reverse one local parameterization and its source indices. The shared
    // edge still has the same two global endpoints but opposite orientation.
    std::swap(m.patches[1].vertex_indices[0],m.patches[1].vertex_indices[1]);
    std::swap(m.patches[1].vertex_indices[2],m.patches[1].vertex_indices[3]);
    std::swap(m.geometry[1].corners[0],m.geometry[1].corners[1]);
    std::swap(m.geometry[1].corners[2],m.geometry[1].corners[3]);
    m.patches[1].status=PatchCellStatus::RegularNegative;
    m.patches[1].signed_solid_angle_sr=-0.25f;
    auto q=query(m);
    require(q.summary.hit_count==2u&&(q.summary.flags&PatchQueryBoundaryHit)!=0u,
            "Opposite orientations across a feature were silently merged.");
    for(unsigned i=0;i<q.summary.hit_count;++i)
        require((q.storage[i].flags&BilinearBoundaryOwned)==0u,
                "Opposite-orientation feature was falsely certified.");

    m=edge_mesh();m.patches[1].status=PatchCellStatus::NeedsRefinement;
    q=query(m);
    require(q.summary.hit_count==2u&&(q.summary.flags&PatchQueryRefinementHit)!=0u
            &&(q.summary.flags&PatchQueryBoundaryHit)!=0u,
            "Folded/refinement boundary was dropped or certified as regular.");
    require(!optical_result(m,q).known_hits_complete(),"Unprepared boundary fold became complete optics.");

    m=edge_mesh();m.patches.resize(1);m.geometry.resize(1);
    auto raw=intersections(m,{0,0,1});
    require(raw[0].count==1u,"Legacy boundary fixture has no isolated root.");
    raw[0].hits[0].flags&=~(bilinear_boundary_edge_mask|BilinearBoundaryOwned);
    q=collect(m,raw,{0});
    require(q.summary.hit_count==1u&&(q.summary.flags&PatchQueryBoundaryHit)!=0u
            &&(q.storage[0].flags&BilinearBoundaryOwned)==0u,
            "Legacy unannotated boundary was trusted as certified ownership.");
    require(!optical_result(m,q).known_hits_complete(),"Legacy boundary silently passed the optical gate.");

    raw=intersections(m,{0,0,1});raw[0].hits[0].flags|=BilinearSingular;
    q=collect(m,raw,{0});
    require((q.summary.flags&PatchQuerySingularHit)!=0u
            &&(q.storage[0].flags&BilinearBoundaryOwned)==0u,
            "A singular boundary was promoted to a regular owner.");
    require(!optical_result(m,q).known_hits_complete(),"Singular boundary silently passed optics.");

    raw=intersections(m,{0,0,1});raw[0].flags|=BilinearUnresolved;
    q=collect(m,raw,{0});
    const auto unresolved=optical_result(m,q);
    require((q.summary.flags&PatchQueryUnresolved)!=0u
            &&(unresolved.flags&PatchOpticalQueryError)!=0u&&!unresolved.known_hits_complete(),
            "Ownership erased an independently unresolved query.");
    raw[0].flags=BilinearInvalidInput;q=collect(m,raw,{0});
    require((q.summary.flags&PatchQueryInvalidData)!=0u
            &&!optical_result(m,q).known_hits_complete(),
            "Ownership erased invalid query input.");
}

inline void test_unannotated_neighbor_cannot_certify_a_subset()
{
    // These records deliberately model legacy/unsupported reports without
    // exact feature annotations. The source patches still share their edge.
    // Resolve neither a subset of that feature nor the regular report alone.
    for(unsigned unsupported=0;unsupported<3;++unsupported)
    {
        auto m=edge_mesh();auto raw=intersections(m,{0,0,1});
        raw[1].hits[0].flags&=~(bilinear_boundary_edge_mask|BilinearBoundaryOwned);
        if(unsupported==1u)raw[1].hits[0].flags|=BilinearSingular;
        if(unsupported==2u)m.patches[1].status=PatchCellStatus::NeedsRefinement;
        const auto q=collect(m,raw,{1,0});
        require(q.summary.hit_count==2u&&(q.summary.flags&PatchQueryBoundaryHit)!=0u,
                "An unsupported adjacent boundary report was discarded.");
        require((q.summary.flags&PatchQueryBoundaryResolved)==0u,
                "A subset of a source feature with an unsupported report was certified.");
        for(unsigned i=0;i<q.summary.hit_count;++i)
            require((q.storage[i].flags&BilinearBoundaryOwned)==0u,
                    "Regular boundary acquired ownership despite unsupported adjacent evidence.");
        require(!optical_result(m,q).known_hits_complete(),
                "Unsupported adjacent boundary evidence was hidden from optical evaluation.");
    }
}

inline void test_malformed_neighbor_cannot_be_hidden_by_compaction()
{
    const auto m=edge_mesh();const auto raw=intersections(m,{0,0,1});
    const float nan=std::numeric_limits<float>::quiet_NaN();
    for(unsigned corruption=0;corruption<10;++corruption)
    {
        Collected q;q.storage.resize(2);
        PatchHitCollector writer{q.storage.data(),2,0,0,true};
        writer.consume(m.patches[0],0,raw[0]);writer.consume(m.patches[1],1,raw[1]);
        require(writer.count==2u,"Malformed boundary fixture must have two raw reports.");
        // The valid smaller-ID hit would win ownership. Validate its higher-ID
        // neighbor before eliminating any duplicate feature report. Mutating
        // the collected record also permits root_index>1 without ever reading
        // outside BilinearRayHits' fixed two-root storage.
        auto& bad=q.storage[1];
        switch(corruption)
        {
            case 0: bad.t=nan;break;
            case 1: bad.u=nan;break;
            case 2: bad.v=nan;break;
            case 3: bad.u=-0.25f;break;
            case 4: bad.v=1.25f;break;
            case 5: bad.residual=-0.125f;break;
            case 6: bad.root_index=2u;break;
            case 7: bad.t=0.0f;break;
            case 8: bad.t=std::numeric_limits<float>::infinity();break;
            default: bad.residual=nan;break;
        }
        q.summary=writer.finish(m.patches.data(),static_cast<unsigned>(m.patches.size()));
        require((q.summary.flags&PatchQueryInvalidData)!=0u,
                "Malformed non-owner report was hidden by boundary compaction.");
        const auto optical=optical_result(m,q);
        require((optical.flags&PatchOpticalQueryError)!=0u&&!optical.known_hits_complete(),
                "Malformed eliminated boundary report became complete optical input.");
    }
}

inline void test_two_fold_roots_remain_distinct()
{
    // P=(u+v-1, (u-1/2)(v-1/2)+1/16, 1): exact +z roots
    // (u,v)=(1/4,3/4),(3/4,1/4), belonging to two folded branches.
    Mesh m;
    for(Vec3 p : std::array<Vec3,4>{{{-1,0.3125f,1},{0,-0.1875f,1},
                                    {0,-0.1875f,1},{1,0.3125f,1}}})append_vertex(m,p);
    append_patch(m,99,{0,1,2,3},PatchCellStatus::NeedsRefinement);
    const auto raw=intersections(m,{0,0,1});
    require(raw[0].count==2u,"Independent folded-root fixture lost one root.");
    const auto q=collect(m,raw,{0,0});
    require(q.summary.candidate_count==4u&&q.summary.hit_count==2u,
            "Ownership merged two roots or retained duplicate primitive reports.");
    require(q.storage[0].root_index==0u&&q.storage[1].root_index==1u
            &&near(q.storage[0].u,0.25)&&near(q.storage[1].u,0.75)
            &&near(q.storage[0].v,0.75)&&near(q.storage[1].v,0.25),
            "Distinct folded roots or their deterministic ordering changed.");
}

inline void test_cpu_reconstructed_exact_corner()
{
    // CPU reconstruction, NOT a GPU capture, from a473f614723a6ac4ebc68dcee18e1856f9208314.
    // RaindropPathTracer / IAPWS water: radius=1 mm, wavelength=785 nm,
    // inclination=-89.7 deg, T=20 C, p=101325 Pa, source grid=3001^2.
    // QueryDirectionGrid 1800x3600, direction_id=1687286 (46.85,68.65 deg).
    // The query is exactly TRRT vertex (1689,1986), global index 32979678.
    // The four patches below reproduce the four pending boundary reports in
    // the repository's log fixture. Only local IDs are compacted for this test.
    // Exact dyadic arithmetic proves each incident edge determinant is zero:
    // query and the central stored corner are componentwise identical.
    constexpr std::uint32_t bits[9][3]={
        {0x3f2e89dbu,0x3f2ea197u,0x3e874bcbu},
        {0x3f2e8781u,0x3f2e807eu,0x3e880248u},
        {0x3f2e8521u,0x3f2e5f36u,0x3e88b8b4u},
        {0x3f2ee010u,0x3f2e4c1cu,0x3e874769u},
        {0x3f2eddaeu,0x3f2e2afeu,0x3e87fddeu},
        {0x3f2edb4fu,0x3f2e09abu,0x3e88b441u},
        {0x3f2f3630u,0x3f2df666u,0x3e8742fau},
        {0x3f2f33cdu,0x3f2dd53bu,0x3e87f969u},
        {0x3f2f3162u,0x3f2db3e7u,0x3e88afc1u}};
    Mesh original;
    for(const auto& b:bits)
        append_vertex(original,{std::bit_cast<float>(b[0]),std::bit_cast<float>(b[1]),std::bit_cast<float>(b[2])});
    append_patch(original,32956688,{0,1,3,4});append_patch(original,32956689,{1,2,4,5});
    append_patch(original,32959688,{3,4,6,7});append_patch(original,32959689,{4,5,7,8});
    constexpr std::uint32_t omega[4]={0x36343f2du,0x3634481cu,0x36346a3cu,0x36346b51u};
    for(unsigned i=0;i<4;++i)
    {
        original.patches[i].incident_area_drop2=std::bit_cast<float>(0x3505cb5cu);
        original.patches[i].signed_solid_angle_sr=std::bit_cast<float>(omega[i]);
    }
    const Vec3 direction=original.vertices[4].direction_drop;
    const auto raw=intersections(original,direction);
    for(unsigned i=0;i<4;++i)
    {
        require(raw[i].count==1u&&(raw[i].flags&BilinearUnresolved)==0u
                &&(raw[i].hits[0].flags&BilinearBoundary)!=0u,
                "The reconstructed true shared corner was lost or left numerically unresolved.");
        require(raw[i].hits[0].u==(i%2u?0.0f:1.0f)
                &&raw[i].hits[0].v==(i/2u?0.0f:1.0f)
                &&raw[i].hits[0].residual==0.0f,
                "Reconstructed shared-corner coordinates changed.");
    }
    // Exercise compact-index permutations independently of original patch IDs.
    std::array<unsigned,4> permutation{0,1,2,3};
    do
    {
        Mesh shuffled=original;
        for(unsigned i=0;i<4;++i)
        {shuffled.patches[i]=original.patches[permutation[i]];shuffled.geometry[i]=original.geometry[permutation[i]];}
        const auto q=collect(shuffled,intersections(shuffled,direction),{3,2,1,0});
        require_owned(q,1);
        require(q.summary.candidate_count==4u&&q.storage[0].patch_id==32956688u,
                "Reconstructed four-patch corner did not choose its canonical owner.");
    }while(std::next_permutation(permutation.begin(),permutation.end()));
}
} // namespace boundary_ownership_detail

inline void test_bilinear_boundary_ownership()
{
    using namespace boundary_ownership_detail;
    test_edge_and_single_owner();
    test_corner_order_and_optics();
    test_independent_source_features();
    test_packed_interior_is_not_owned();
    test_pending_and_error_contracts();
    test_unannotated_neighbor_cannot_certify_a_subset();
    test_malformed_neighbor_cannot_be_hidden_by_compaction();
    test_two_fold_roots_remain_distinct();
    test_cpu_reconstructed_exact_corner();
}
} // namespace rainbow::tests
