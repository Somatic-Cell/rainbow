#pragma once

#include <rainbow/vec3.hpp>
#include <cmath>
#include <limits>

namespace rainbow::regular64_detail
{
// A fallback certificate for the stored FP32 corners, NOT for the underlying
// analytic optical map. No corner normalization, subdivision or retracing.
// CUDA uses explicitly directed operations. Host operations are rounded to
// binary64 before nextafter expands their bounds; do not enable fast-math.
struct Interval
{
    double lo, hi;
    HOST_DEVICE static Interval point(double x) noexcept { return {x,x}; }
    HOST_DEVICE static double add_down(double a,double b) noexcept
    {
#if defined(__CUDA_ARCH__)
        return __dadd_rd(a,b);
#else
        volatile double r=a+b;
        return ::nextafter(r,-std::numeric_limits<double>::infinity());
#endif
    }
    HOST_DEVICE static double add_up(double a,double b) noexcept
    {
#if defined(__CUDA_ARCH__)
        return __dadd_ru(a,b);
#else
        volatile double r=a+b;
        return ::nextafter(r,std::numeric_limits<double>::infinity());
#endif
    }
    HOST_DEVICE static double mul_down(double a,double b) noexcept
    {
#if defined(__CUDA_ARCH__)
        return __dmul_rd(a,b);
#else
        volatile double r=a*b;
        return ::nextafter(r,-std::numeric_limits<double>::infinity());
#endif
    }
    HOST_DEVICE static double mul_up(double a,double b) noexcept
    {
#if defined(__CUDA_ARCH__)
        return __dmul_ru(a,b);
#else
        volatile double r=a*b;
        return ::nextafter(r,std::numeric_limits<double>::infinity());
#endif
    }
    HOST_DEVICE Interval operator+(Interval b) const noexcept
    { return {add_down(lo,b.lo),add_up(hi,b.hi)}; }
    HOST_DEVICE Interval operator-(Interval b) const noexcept
    { return {add_down(lo,-b.hi),add_up(hi,-b.lo)}; }
    HOST_DEVICE Interval operator*(Interval b) const noexcept
    {
        return {::fmin(::fmin(mul_down(lo,b.lo),mul_down(lo,b.hi)),
                       ::fmin(mul_down(hi,b.lo),mul_down(hi,b.hi))),
                ::fmax(::fmax(mul_up(lo,b.lo),mul_up(lo,b.hi)),
                       ::fmax(mul_up(hi,b.lo),mul_up(hi,b.hi)))};
    }
};
struct Vector
{
    Interval x,y,z;
    HOST_DEVICE static Vector point(Vec3 p) noexcept
    { return {Interval::point(double(p.x)),Interval::point(double(p.y)),Interval::point(double(p.z))}; }
    HOST_DEVICE Vector operator+(Vector b) const noexcept { return {x+b.x,y+b.y,z+b.z}; }
    HOST_DEVICE Vector operator-(Vector b) const noexcept { return {x-b.x,y-b.y,z-b.z}; }
    HOST_DEVICE Vector cross(Vector b) const noexcept
    { return {y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x}; }
    HOST_DEVICE Interval dot(Vector b) const noexcept { return x*b.x+y*b.y+z*b.z; }
};
HOST_DEVICE inline bool finite(double x) noexcept
{ return x>=-0x1.fffffffffffffp1023 && x<=0x1.fffffffffffffp1023; }

// Numerically evaluate a spherical-triangle area, separately from the sign
// certificate. FP32 corners are promoted BEFORE subtracting their coordinates.
HOST_DEVICE inline double difference_of_products(double x,double y,double z,double w) noexcept
{
    const double zw=z*w;
    return ::fma(x,y,-zw)+::fma(-z,w,zw);
}
HOST_DEVICE inline double triangle_angle(Vec3 a0,Vec3 b0,Vec3 c0) noexcept
{
    const auto a=a0.cast<double>(),b=b0.cast<double>(),c=c0.cast<double>();
    const auto u=b-a,v=c-a;
    const double det=::fma(a.x,difference_of_products(u.y,v.z,u.z,v.y),
                     ::fma(a.y,difference_of_products(u.z,v.x,u.x,v.z),
                               a.z*difference_of_products(u.x,v.y,u.y,v.x)));
    const double la=a.length(),lb=b.length(),lc=c.length();
    const double den=::fma(la,lb*lc,::fma(a.dot(b),lc,::fma(b.dot(c),la,c.dot(a)*lb)));
    return 2.0*::atan2(det,den);
}
} // namespace rainbow::regular64_detail

namespace rainbow
{
struct RegularPatch64
{
    // -1/+1: strictly uniform orientation AND a common open hemisphere.
    // 0: not certified. Zero is never interpreted as a miss or a true fold.
    [[nodiscard]] HOST_DEVICE static int certified_orientation(const Vec3* corners) noexcept
    {
        using V=regular64_detail::Vector;
        if(!corners)return 0;
        for(unsigned i=0;i<4;++i)if(!corners[i].is_finite())return 0;
        const V p0=V::point(corners[0]),p1=V::point(corners[1]);
        const V p2=V::point(corners[2]),p3=V::point(corners[3]);
        const auto d0=p0.dot((p1-p0).cross(p2-p0));
        const auto d1=p1.dot((p1-p0).cross(p3-p1));
        const auto d2=p2.dot((p3-p2).cross(p2-p0));
        const auto d3=p3.dot((p3-p2).cross(p3-p1));
        const bool pos=d0.lo>0 && d1.lo>0 && d2.lo>0 && d3.lo>0;
        const bool neg=d0.hi<0 && d1.hi<0 && d2.hi<0 && d3.hi<0;
        if(!pos&&!neg)return 0;
        const V axis=(p0+p1)+(p2+p3);
        if(!(axis.dot(p0).lo>0 && axis.dot(p1).lo>0
          && axis.dot(p2).lo>0 && axis.dot(p3).lo>0))return 0;
        return pos?1:-1;
    }

    // Keep OutgoingPatch's FP32 area field/ABI. Compute in FP64 first and
    // accept only a finite, nonzero, correctly signed representable result.
    // A failed metric does NOT acquire an epsilon floor or fabricated area.
    [[nodiscard]] HOST_DEVICE static bool try_metric(
        const Vec3* corners,int orientation,float& result) noexcept
    {
        if(!corners || (orientation!=1 && orientation!=-1))return false;
        const double a=regular64_detail::triangle_angle(corners[0],corners[1],corners[3]);
        const double b=regular64_detail::triangle_angle(corners[0],corners[3],corners[2]);
        const double omega=a+b;
        if(!(regular64_detail::finite(a)&&regular64_detail::finite(b)
          && regular64_detail::finite(omega)&&a*orientation>0&&b*orientation>0
          && ::fabs(omega)<6.283185307179587))return false;
        const float stored=static_cast<float>(omega);
        if(!(stored>=-0x1.fffffep127f && stored<=0x1.fffffep127f
          && stored*float(orientation)>0))return false;
        result=stored;
        return true;
    }
};
} // namespace rainbow
