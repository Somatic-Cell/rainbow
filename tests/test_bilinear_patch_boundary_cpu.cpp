#include <rainbow/bilinear_patch_boundary.hpp>

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace
{
using rainbow::BilinearBoundaryMembership;
using rainbow::BilinearPatchBoundary;
using rainbow::Vec3;

void require(const bool condition, const char* message)
{
    if(!condition) throw std::runtime_error(message);
}

void determinant_cases()
{
    const Vec3 a{1, 0, 0}, b{0, 1, 0}, c{0, 0, 1};
    require(BilinearPatchBoundary::determinant_sign(a,b,c) == 1, "Positive determinant lost.");
    require(BilinearPatchBoundary::determinant_sign(a,c,b) == -1, "Negative determinant lost.");
    require(BilinearPatchBoundary::determinant_sign(a,b,b) == 0, "Exact coplanarity lost.");

    // Before uniform scaling by 2^-23, these integer rows are
    // (n,n-1,n-2), (n+1,n,n-1), (n,n-1,n-1), with n=2^23.
    // Subtracting row 1 from row 3 gives (0,0,1); the remaining minor
    // is n*n-(n-1)*(n+1)=1. Therefore the exact determinant is 2^-69.
    // Its sign must survive cancellation of triple products near one.
    constexpr float d = 0x1p-23f;
    const Vec3 u{1.0f, 1.0f-d, 1.0f-2.0f*d};
    const Vec3 v{1.0f+d, 1.0f, 1.0f-d};
    const Vec3 w{1.0f, 1.0f-d, 1.0f-d};
    require(BilinearPatchBoundary::determinant_sign(u,v,w) == 1,
            "Exact 2^-69 determinant was rounded away.");
    require(BilinearPatchBoundary::determinant_sign(u,w,v) == -1,
            "Permutation did not negate the exact determinant.");
    require(BilinearPatchBoundary::determinant_sign(u,v,u) == 0,
            "Cancellation fixture failed its exact-zero companion.");

    constexpr float tiny = 0x1p-149f;
    constexpr float huge = 0x1.fffffep127f;
    require(BilinearPatchBoundary::determinant_sign({tiny,0,0},{0,tiny,0},{0,0,tiny}) == 1,
            "Smallest nonzero binary32 triple was lost.");
    require(BilinearPatchBoundary::determinant_sign({huge,0,0},{0,huge,0},{0,0,huge}) == 1,
            "Largest finite binary32 triple overflowed.");
}

void membership_cases()
{
    constexpr float e = 0x1p-10f;
    constexpr float tiny = 0x1p-100f;
    const Vec3 inside[4] = {{-e,-e,1},{tiny,-e,1},{-e,e,1},{tiny,e,1}};
    const Vec3 outside[4] = {{-e,-e,1},{-tiny,-e,1},{-e,e,1},{-tiny,e,1}};
    // Exact u = 2^90/(2^90+1) is interior but rounds to 1 even in FP64.
    // Original input bits still decide the membership exactly.
    const auto i = BilinearPatchBoundary::classify_regular(inside,{0,0,1},1);
    require(i.membership == BilinearBoundaryMembership::Interior && i.edges == 0,
            "Interior root within a double rounding interval became an edge.");
    require(BilinearPatchBoundary::classify_regular(outside,{0,0,1},1).membership
                == BilinearBoundaryMembership::Outside,
            "Exterior root within a double rounding interval was included.");
    require(BilinearPatchBoundary::classify_regular(inside,{0,0,-1},1).membership
                == BilinearBoundaryMembership::Outside,
            "Negative ray branch was included.");
    require(BilinearPatchBoundary::classify_regular(inside,{0,0,1},0).membership
                == BilinearBoundaryMembership::Uncertified,
            "Uncertified/folded patch was treated as regular.");

    const Vec3 reversed[4] = {inside[0],inside[2],inside[1],inside[3]};
    require(BilinearPatchBoundary::classify_regular(reversed,{0,0,1},-1).membership
                == BilinearBoundaryMembership::Interior,
            "Negative patch orientation was not respected.");

    const Vec3 centered[4] = {{-e,-e,1},{e,-e,1},{-e,e,1},{e,e,1}};
    const Vec3 rays[8] = {{-e,0,1},{e,0,1},{0,-e,1},{0,e,1},
                         {-e,-e,1},{e,-e,1},{-e,e,1},{e,e,1}};
    const std::uint32_t masks[8] = {
        rainbow::BilinearEdgeU0, rainbow::BilinearEdgeU1,
        rainbow::BilinearEdgeV0, rainbow::BilinearEdgeV1,
        rainbow::BilinearEdgeU0 | rainbow::BilinearEdgeV0,
        rainbow::BilinearEdgeU1 | rainbow::BilinearEdgeV0,
        rainbow::BilinearEdgeU0 | rainbow::BilinearEdgeV1,
        rainbow::BilinearEdgeU1 | rainbow::BilinearEdgeV1};
    for(unsigned k = 0; k < 8; ++k)
    {
        const auto boundary = BilinearPatchBoundary::classify_regular(centered,rays[k],1);
        require(boundary.membership == BilinearBoundaryMembership::Boundary
                    && boundary.edges == masks[k],
                "Exact edge/corner topology was lost.");
    }

    const Vec3 adjacent[4] = {{e,-e,1},{2*e,-e,1},{e,e,1},{2*e,e,1}};
    const auto neighbor = BilinearPatchBoundary::classify_regular(adjacent,{e,0,1},1);
    require(neighbor.membership == BilinearBoundaryMembership::Boundary
                && neighbor.edges == rainbow::BilinearEdgeU0,
            "Adjacent patch disagreed about its original shared edge.");
}

// The Python regression sends exact FP32 input bits and independently checks
// the returned signs using standard-library arbitrary-precision integers.
// This mode adds no library dependency to the C++ or CUDA solver.
void batch_predicates()
{
    for(;;)
    {
        std::uint32_t bits[9]{};
        if(!(std::cin >> std::hex >> bits[0])) return;
        for(unsigned k = 1; k < 9; ++k)
            if(!(std::cin >> std::hex >> bits[k])) throw std::runtime_error("Truncated predicate fixture.");
        const auto f = [&](unsigned k) { return std::bit_cast<float>(bits[k]); };
        const Vec3 a{f(0),f(1),f(2)}, b{f(3),f(4),f(5)}, c{f(6),f(7),f(8)};
        if(!a.is_finite() || !b.is_finite() || !c.is_finite())
            throw std::runtime_error("Nonfinite predicate fixture.");
        std::cout << std::dec << BilinearPatchBoundary::determinant_sign(a,b,c) << '\n';
    }
}
}

int main(const int argc, char** argv)
{
    try
    {
        if(argc == 2 && std::string_view(argv[1]) == "--batch-predicates") batch_predicates();
        else
        {
            if(argc != 1) throw std::runtime_error("Unexpected boundary-test argument.");
            determinant_cases();
            membership_cases();
            std::cout << "Bilinear boundary CPU: exact cancellation, input extrema, near edges and topology passed.\n";
        }
        return EXIT_SUCCESS;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
