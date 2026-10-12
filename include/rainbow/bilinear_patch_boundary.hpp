#pragma once

#include <rainbow/vec3.hpp>
#include <cmath>
#include <cstdint>

namespace rainbow
{

enum class BilinearBoundaryMembership : std::uint32_t
{
    Uncertified = 0,
    Interior = 1,
    Boundary = 2,
    Outside = 3
};

enum BilinearPatchEdge : std::uint32_t
{
    BilinearEdgeU0 = 1u << 0,
    BilinearEdgeU1 = 1u << 1,
    BilinearEdgeV0 = 1u << 2,
    BilinearEdgeV1 = 1u << 3
};

struct BilinearBoundaryClassification
{
    BilinearBoundaryMembership membership = BilinearBoundaryMembership::Uncertified;
    std::uint32_t edges = 0; // Exact zero edge predicates, not rounded u/v values.
};

// Exact boundary predicates for the ORIGINAL stored binary32 directions.
// This is a membership test, not a replacement for solving the bilinear patch.
// Use it only with a separately certified regular, common-hemisphere patch.
// A folded patch cannot be classified by four common edge halfspaces.
struct BilinearPatchBoundary
{
private:
    // Grow-expansion addition. Each component is a binary64 number and the
    // expansion sum is exact. TwoSum requires ordinary round-to-nearest
    // arithmetic: keep fast-math/reassociation disabled, as for the solver.
    struct Expansion
    {
        double values[12]{};
        unsigned count = 0;

        HOST_DEVICE void add(const double value) noexcept
        {
            if(value == 0.0) return;
            double q = value;
            unsigned kept = 0;
            for(unsigned i = 0; i < count; ++i)
            {
                const double b = values[i];
                const double sum = q + b;
                const double b_virtual = sum - q;
                const double a_virtual = sum - b_virtual;
                const double b_roundoff = b - b_virtual;
                const double a_roundoff = q - a_virtual;
                const double error = a_roundoff + b_roundoff;
                if(error != 0.0) values[kept++] = error;
                q = sum;
            }
            if(q != 0.0) values[kept++] = q;
            count = kept;
        }

        HOST_DEVICE void triple(const float a, const float b, const float c,
                                const double sign) noexcept
        {
            // A product of two finite binary32 values is exact in binary64
            // (at most 48 significant bits). FMA then recovers the complete
            // rounding residual of multiplication by the third input.
            // Even the extrema of finite binary32 triples lie strictly inside
            // the normal binary64 exponent range, so no under/overflow occurs.
            const double ab = double(a) * double(b);
            const double high = ab * double(c);
            const double low = ::fma(ab, double(c), -high);
            add(sign * low);
            add(sign * high);
        }

        [[nodiscard]] HOST_DEVICE int sign() const noexcept
        {
            if(count == 0) return 0;
            return values[count - 1] > 0.0 ? 1 : -1;
        }
    };

public:
    // Preconditions: all nine components are finite binary32 numbers. The
    // returned sign is that of the exact scalar triple product a.(b x c).
    // Six triples contribute at most twelve expansion components; no heap,
    // epsilon threshold, platform long double, or external library is used.
    [[nodiscard]] HOST_DEVICE static int determinant_sign(
        const Vec3 a, const Vec3 b, const Vec3 c) noexcept
    {
        Expansion result{};
        result.triple(a.x, b.y, c.z, 1.0);
        result.triple(a.y, b.z, c.x, 1.0);
        result.triple(a.z, b.x, c.y, 1.0);
        result.triple(a.x, b.z, c.y, -1.0);
        result.triple(a.y, b.x, c.z, -1.0);
        result.triple(a.z, b.y, c.x, -1.0);
        return result.sign();
    }

    // certified_orientation must be +1 or -1 only after the caller has
    // certified nonzero radial Jacobian of that sign over the entire patch
    // and a common open hemisphere for the four corners. Passing 0 preserves
    // Uncertified. Tensor-product corner order is (00,10,01,11).
    [[nodiscard]] HOST_DEVICE static BilinearBoundaryClassification classify_regular(
        const Vec3 (&corners)[4], const Vec3 direction,
        const int certified_orientation) noexcept
    {
        BilinearBoundaryClassification result{};
        if((certified_orientation != 1 && certified_orientation != -1)
           || !direction.is_finite()
           || (direction.x == 0.0f && direction.y == 0.0f && direction.z == 0.0f))
            return result;
        for(unsigned i = 0; i < 4; ++i)
            if(!corners[i].is_finite()) return result;

        // The regular radial image is the convex spherical quadrilateral
        // with boundary 00 -> 10 -> 11 -> 01. Each original chord edge and
        // the origin define the same plane as its spherical edge. Thus these
        // exact signs classify the ray without changing its direction or
        // triangulating the patch/optical interpolation.
        const unsigned first[4] = {0u, 1u, 3u, 2u};
        const unsigned second[4] = {1u, 3u, 2u, 0u};
        const std::uint32_t edge[4] = {
            BilinearEdgeV0, BilinearEdgeU1, BilinearEdgeV1, BilinearEdgeU0};
        bool outside = false;
        for(unsigned i = 0; i < 4; ++i)
        {
            const int side = certified_orientation
                * determinant_sign(direction, corners[first[i]], corners[second[i]]);
            outside |= side < 0;
            if(side == 0) result.edges |= edge[i];
        }
        result.membership = outside ? BilinearBoundaryMembership::Outside
            : (result.edges ? BilinearBoundaryMembership::Boundary
                            : BilinearBoundaryMembership::Interior);
        return result;
    }
};
} // namespace rainbow
