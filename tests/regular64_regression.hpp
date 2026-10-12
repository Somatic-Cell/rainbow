#pragma once
#include <rainbow/patch_geometry.hpp>
#include <array>
#include <bit>
#include <cstdint>
#include <vector>

namespace rainbow::tests
{
// i0033, R cell (2028,2848), patch_id 8546028 at grid 3001.
// CPU-reconstructed source rays from analysis_results.json, NOT a GPU snapshot.
inline BilinearPatchGeometry regular64_regression_patch()
{
    constexpr std::uint32_t bits[4][3] = {
        {1054474498u,1063745435u,1024252710u},
        {1054085668u,1063838341u,1022485300u},
        {1053274012u,1064021247u,1016974475u},
        {1052325291u,1064216558u,1005935086u}};
    BilinearPatchGeometry g{};
    for(unsigned i=0;i<4;++i)g.corners[i]={std::bit_cast<float>(bits[i][0]),
        std::bit_cast<float>(bits[i][1]),std::bit_cast<float>(bits[i][2])};
    return g;
}
inline std::vector<OutgoingVertex> regular64_regression_vertices()
{
    std::vector<OutgoingVertex> vertices(16);
    auto a=regular64_regression_patch(),b=a;
    std::swap(b.corners[0],b.corners[1]);std::swap(b.corners[2],b.corners[3]);
    BilinearPatchGeometry folded{{{-0.5f,-0.5f,1},{0.5f,-0.5f,1},
                                  {0.5f,0.5f,1},{-0.5f,0.5f,1}}};
    for(auto& c:folded.corners)c=c.normalized();
    const BilinearPatchGeometry degenerate{{{0,0,1},{0,0,1},{0,0,1},{0,0,1}}};
    const BilinearPatchGeometry cases[4]={a,b,folded,degenerate};
    for(unsigned f=0;f<4;++f)for(unsigned i=0;i<4;++i)
    {
        auto& v=vertices[4*f+i];v.status=VertexStatus::Valid;
        v.direction_drop=cases[f].corners[i];v.basis_x={1,0,0};
        v.field={{1,0},{0,0}};
    }
    return vertices;
}
} // namespace rainbow::tests
