#pragma once

#include <rainbow/outgoing_patch.hpp>
#include <rainbow/patch_regular64.hpp>
#include <cmath>
#include <limits>

namespace rainbow
{
namespace patch_detail
{
inline constexpr float infinity = std::numeric_limits<float>::infinity();
inline constexpr float quiet_nan = std::numeric_limits<float>::quiet_NaN();

// Narrow finite-input interval type used ONLY to certify the sign of the
// radial Jacobian and common-hemisphere tests. Ordinary arithmetic stays FP32.
// Keep FTZ/fast-math off. nextafterf receives TWO floats (no mixed overload).
struct Interval
{
    float lo, hi;
    HOST_DEVICE static Interval point(const float x) noexcept { return {x,x}; }
    HOST_DEVICE Interval operator+(const Interval b) const noexcept
    { return {::nextafterf(lo+b.lo,-infinity), ::nextafterf(hi+b.hi,infinity)}; }
    HOST_DEVICE Interval operator-(const Interval b) const noexcept
    { return {::nextafterf(lo-b.hi,-infinity), ::nextafterf(hi-b.lo,infinity)}; }
    HOST_DEVICE Interval operator*(const Interval b) const noexcept
    {
        const float p0=lo*b.lo, p1=lo*b.hi, p2=hi*b.lo, p3=hi*b.hi;
        return {::nextafterf(::fminf(::fminf(p0,p1),::fminf(p2,p3)),-infinity),
                ::nextafterf(::fmaxf(::fmaxf(p0,p1),::fmaxf(p2,p3)),infinity)};
    }
};
struct IntervalVector
{
    Interval x,y,z;
    HOST_DEVICE static IntervalVector point(const Vec3 p) noexcept
    { return {Interval::point(p.x),Interval::point(p.y),Interval::point(p.z)}; }
    HOST_DEVICE IntervalVector operator-(const IntervalVector b) const noexcept
    { return {x-b.x,y-b.y,z-b.z}; }
    HOST_DEVICE IntervalVector cross(const IntervalVector b) const noexcept
    { return {y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x}; }
    HOST_DEVICE Interval dot(const IntervalVector b) const noexcept
    { return x*b.x+y*b.y+z*b.z; }
};

HOST_DEVICE inline bool finite(const float x) noexcept
{ return x >= -0x1.fffffep127f && x <= 0x1.fffffep127f; }

// Cancellation-aware determinant: a.(b x c) = a.((b-a) x (c-a)).
// The interval certificate is separate; this is the metric evaluation.
HOST_DEVICE inline float difference_of_products(
    const float x,const float y,const float z,const float w) noexcept
{
    const float zw=z*w;
    const float residual=::fmaf(-z,w,zw);
    return ::fmaf(x,y,-zw)+residual;
}

HOST_DEVICE inline float determinant(const Vec3 a,const Vec3 b,const Vec3 c) noexcept
{
    const Vec3 u=b-a, v=c-a;
    return ::fmaf(a.x,difference_of_products(u.y,v.z,u.z,v.y),
           ::fmaf(a.y,difference_of_products(u.z,v.x,u.x,v.z),
                     a.z*difference_of_products(u.x,v.y,u.y,v.x)));
}

// Solid angle of a triangle as seen from the origin. The length factors make
// this valid for stored directions whose FP32 lengths differ slightly from 1.
HOST_DEVICE inline float triangle_angle(const Vec3 a,const Vec3 b,const Vec3 c) noexcept
{
    const float la=a.length(),lb=b.length(),lc=c.length();
    const float denominator=::fmaf(la,lb*lc,
        ::fmaf(a.dot(b),lc,::fmaf(b.dot(c),la,c.dot(a)*lb)));
    return 2.0f*::atan2f(determinant(a,b,c),denominator);
}
} // namespace patch_detail

// This is a Euclidean bilinear CHORD patch. Do not normalize evaluate().
// The ray query in the next stage will solve P(u,v)=t*direction, t>0.
// The spherical image P/|P| is used for solid angle, NOT for the GAS AABB.
struct BilinearPatchGeometry
{
    Vec3 corners[4]; // 00,10,01,11.

    [[nodiscard]] HOST_DEVICE Vec3 evaluate(const float u,const float v) const noexcept
    {
        const Vec3 lower=corners[0]*(1.0f-u)+corners[1]*u;
        const Vec3 upper=corners[2]*(1.0f-u)+corners[3]*u;
        return lower*(1.0f-v)+upper*v;
    }

    [[nodiscard]] HOST_DEVICE PatchAabb bounds() const noexcept
    {
        Vec3 lo=corners[0],hi=corners[0];
        for(unsigned i=1;i<4;++i)
        {
            lo={::fminf(lo.x,corners[i].x),::fminf(lo.y,corners[i].y),::fminf(lo.z,corners[i].z)};
            hi={::fmaxf(hi.x,corners[i].x),::fmaxf(hi.y,corners[i].y),::fmaxf(hi.z,corners[i].z)};
        }
        // Exact bilinear convex combinations lie inside corner min/max.
        // A small unit-direction-space guard also covers rounded evaluation.
        // It does not clamp a solid angle or change the patch geometry.
        constexpr float padding=0x1p-19f;
        using patch_detail::infinity;
        return {::nextafterf(lo.x-padding,-infinity),::nextafterf(lo.y-padding,-infinity),
                ::nextafterf(lo.z-padding,-infinity),::nextafterf(hi.x+padding,infinity),
                ::nextafterf(hi.y+padding,infinity),::nextafterf(hi.z+padding,infinity)};
    }

    // Conservative certificate. Failure means "requires further handling",
    // not "no intersection" and not necessarily "mathematically folded".
    [[nodiscard]] HOST_DEVICE int certified_orientation() const noexcept
    {
        using V=patch_detail::IntervalVector;
        const V p0=V::point(corners[0]),p1=V::point(corners[1]);
        const V p2=V::point(corners[2]),p3=V::point(corners[3]);
        const auto d0=p0.dot((p1-p0).cross(p2-p0));
        const auto d1=p1.dot((p1-p0).cross(p3-p1));
        const auto d2=p2.dot((p3-p2).cross(p2-p0));
        const auto d3=p3.dot((p3-p2).cross(p3-p1));
        // J=P.(P_u x P_v) is bilinear in u,v. Four strictly equal signs
        // therefore certify no zero of J anywhere in the unit square.
        const bool positive=d0.lo>0 && d1.lo>0 && d2.lo>0 && d3.lo>0;
        const bool negative=d0.hi<0 && d1.hi<0 && d2.hi<0 && d3.hi<0;
        if(!positive && !negative) return 0;
        // A common open hemisphere excludes origin crossings and the ambiguity
        // of choosing a spherical complement. A failed certificate is retained.
        const Vec3 axis=(corners[0]+corners[1])+(corners[2]+corners[3]);
        const V a=V::point(axis);
        if(!(a.dot(p0).lo>0 && a.dot(p1).lo>0 && a.dot(p2).lo>0 && a.dot(p3).lo>0)) return 0;
        return positive?1:-1;
    }

    [[nodiscard]] HOST_DEVICE float signed_solid_angle() const noexcept
    {
        // Only an integration identity, NOT a replacement of the bilinear
        // primitive by two triangles. Field/phase interpolation stays bilinear.
        return patch_detail::triangle_angle(corners[0],corners[1],corners[3])
             + patch_detail::triangle_angle(corners[0],corners[3],corners[2]);
    }
};

struct PatchConstructionResult
{
    OutgoingPatch patch{};
    PatchAabb aabb{};
};

struct PatchConstruction
{
    [[nodiscard]] HOST_DEVICE static bool usable_vertex(const OutgoingVertex& v) noexcept
    {
        const float norm2=v.direction_drop.dot(v.direction_drop);
        const auto& f=v.field;
        return v.position_drop.is_finite() && v.direction_drop.is_finite() && v.basis_x.is_finite()
            && ::fabsf(norm2-1.0f)<=0x1p-16f && v.optical_cycles.is_valid()
            && patch_detail::finite(f.x.real) && patch_detail::finite(f.x.imag)
            && patch_detail::finite(f.y.real) && patch_detail::finite(f.y.imag);
    }

    [[nodiscard]] HOST_DEVICE static PatchConstructionResult make(
        const OutgoingVertex* vertices,const PatchBuildLayout layout,const std::uint32_t id) noexcept
    {
        PatchConstructionResult result{};
        auto& patch=result.patch;
        patch.patch_id=id;
        patch.incident_area_drop2=layout.incident_area_drop2;
        patch.signed_solid_angle_sr=patch_detail::quiet_nan;
        layout.corner_indices(id,patch.vertex_indices);
        unsigned valid_count=0;
        bool invalid_vertex=false,invalid_geometry=false;
        BilinearPatchGeometry geometry{};
        for(unsigned i=0;i<4;++i)
        {
            const auto& vertex=vertices[patch.vertex_indices[i]];
            if(vertex.status==VertexStatus::Valid)
            {
                ++valid_count;
                invalid_geometry|=!usable_vertex(vertex);
                geometry.corners[i]=vertex.direction_drop;
            }
            else if(vertex.status!=VertexStatus::Miss && vertex.status!=VertexStatus::TotalInternalReflection)
                invalid_vertex=true;
        }
        // Do not let a missing corner hide a numerical failure at another corner.
        if(invalid_vertex) patch.status=PatchCellStatus::InvalidVertex;
        else if(invalid_geometry) patch.status=PatchCellStatus::InvalidGeometry;
        else if(valid_count==0) patch.status=PatchCellStatus::NoOutgoingCorners;
        else if(valid_count<4) patch.status=PatchCellStatus::MissingCorners;
        else
        {
            result.aabb=geometry.bounds();
            const int orientation=geometry.certified_orientation();
            patch.status=PatchCellStatus::NeedsRefinement;
            if(orientation!=0)
            {
                const float angle=geometry.signed_solid_angle();
                // No epsilon floor or fabricated area. Unusable metric -> retain
                // geometry and mark it for refinement before any wave evaluation.
                if(patch_detail::finite(angle) && angle*float(orientation)>0.0f)
                {
                    patch.status=orientation>0?PatchCellStatus::RegularPositive:PatchCellStatus::RegularNegative;
                    patch.signed_solid_angle_sr=angle;
                }
            }
            // Preserve the existing FP32 fast path, including its metric bits.
            // NeedsRefinement means "not certified", not "proven folded".
            // Recheck the SAME stored corners before invoking folded handling.
            if(patch.status==PatchCellStatus::NeedsRefinement)
            {
                const int precise_orientation=RegularPatch64::certified_orientation(geometry.corners);
                float precise_angle=0.0f;
                if(precise_orientation!=0
                   && RegularPatch64::try_metric(geometry.corners,precise_orientation,precise_angle))
                {
                    patch.status=precise_orientation>0
                        ?PatchCellStatus::RegularPositive:PatchCellStatus::RegularNegative;
                    patch.signed_solid_angle_sr=precise_angle;
                }
            }
        }
        return result;
    }
};

} // namespace rainbow
