#pragma once

#include <rainbow/patch_geometry.hpp>
#include <rainbow/bilinear_patch_boundary.hpp>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace rainbow
{

// 保存された四隅を結ぶ「正規化しない」双線形面と，原点からの半直線の交差．
// 二つの孤立解を両方返す．面を三角形に置換しない．
// 共有辺は閉区間として保持する．正確な辺の情報を残し，収集後に同じ入射格子の
// feature だけを一意の owner に割り当てる．異なる波面の重なりは除去しない．
enum BilinearHitFlags : std::uint32_t
{
    BilinearHitNone = 0,
    BilinearBoundary = 1u << 0,
    BilinearSingular = 1u << 1,
    BilinearUsedFp64 = 1u << 2,
    // Bits 3..6 encode BilinearPatchEdge, certified from the original corners.
    // They are NEVER inferred from u/v after packing into binary32.
    BilinearBoundaryOwned = 1u << 7 // Set only after topological ownership.
};
inline constexpr unsigned bilinear_boundary_edge_shift = 3;
inline constexpr std::uint32_t bilinear_boundary_edge_mask = 0xfu << bilinear_boundary_edge_shift;
inline constexpr std::uint32_t bilinear_known_hit_flags =
    BilinearBoundary | BilinearSingular | BilinearUsedFp64
    | bilinear_boundary_edge_mask | BilinearBoundaryOwned;

enum BilinearQueryFlags : std::uint32_t
{
    BilinearQueryNone = 0,
    BilinearUnresolved = 1u << 0,
    BilinearInvalidInput = 1u << 1,
    BilinearFp64Evaluation = 1u << 2
};

struct BilinearRayHit
{
    float u = 0.0f;
    float v = 0.0f;
    float t = 0.0f;
    float residual = 0.0f; // |P(u,v)-t*w| の最大成分．保存後の FP32 値で評価する．
    std::uint32_t flags = BilinearHitNone;
};

struct BilinearRayHits
{
    BilinearRayHit hits[2]{};
    std::uint32_t count = 0;
    std::uint32_t flags = BilinearQueryNone;
};

namespace bilinear_detail
{
template<class T> HOST_DEVICE T abs_value(const T x) noexcept { return x < T(0) ? -x : x; }
template<class T> HOST_DEVICE T max_value(const T a, const T b) noexcept { return a > b ? a : b; }
template<class T> HOST_DEVICE T clamp_unit(const T x) noexcept
{ return x < T(0) ? T(0) : (x > T(1) ? T(1) : x); }
template<class T> HOST_DEVICE constexpr T epsilon() noexcept
{
    if constexpr(std::is_same_v<T, float>) return T(0x1p-23f);
    else return T(0x1p-52);
}
template<class T> HOST_DEVICE T multiply_add(const T a, const T b, const T c) noexcept
{
    if constexpr(std::is_same_v<T, float>) return ::fmaf(a, b, c);
    else return ::fma(a, b, c);
}
template<class T> HOST_DEVICE T square_root(const T x) noexcept
{
    if constexpr(std::is_same_v<T, float>) return ::sqrtf(x);
    else return ::sqrt(x);
}
template<class T> HOST_DEVICE T copy_sign(const T x, const T y) noexcept
{
    if constexpr(std::is_same_v<T, float>) return ::copysignf(x, y);
    else return ::copysign(x, y);
}
template<class T> HOST_DEVICE T component(const Vec3T<T> p, const unsigned axis) noexcept
{ return axis == 0 ? p.x : (axis == 1 ? p.y : p.z); }

// xy-zw: 一方の積の丸め残差も回収する．極端な任意の浮動小数点入力ではなく，
// 単位方向近傍の四隅を対象にする（通常の積の overflow を許容する型ではない）．
template<class T> HOST_DEVICE T difference_of_products(
    const T x, const T y, const T z, const T w) noexcept
{
    const T zw = z*w;
    return multiply_add(x, y, -zw) + multiply_add(-z, w, zw);
}

template<class T> struct Point2
{
    T x, y;
    HOST_DEVICE Point2 operator+(const Point2 b) const noexcept { return {x+b.x, y+b.y}; }
    HOST_DEVICE Point2 operator*(const T s) const noexcept { return {x*s, y*s}; }
    HOST_DEVICE T largest() const noexcept { return max_value(abs_value(x), abs_value(y)); }
    HOST_DEVICE T cross(const Point2 b) const noexcept
    { return difference_of_products(x, b.y, y, b.x); }
};

template<class T> HOST_DEVICE Point2<T> project(
    const Vec3T<T> p, const Vec3T<T> w, const unsigned kz) noexcept
{
    const unsigned kx = (kz+1)%3, ky = (kz+2)%3;
    const T z = component(w, kz);
    return {difference_of_products(z, component(p,kx), component(w,kx), component(p,kz)),
            difference_of_products(z, component(p,ky), component(w,ky), component(p,kz))};
}

template<class T> struct Attempt
{
    BilinearRayHits result{};
    bool retry = false;
};

// P=a+b*u+c*v+d*u*v を ray direction に直交する2成分へ写す．
// v を消去すると cross(b,d)u^2 + [cross(a,d)+cross(b,c)]u + cross(a,c)=0．
// Newton 単独探索ではないので，初期値によって片方の孤立解を失わない．
template<class T> HOST_DEVICE Attempt<T> solve(
    const BilinearPatchGeometry& geometry, const Vec3 direction) noexcept
{
    Attempt<T> out{};
    const auto w = direction.template cast<T>();
    unsigned kz = 0;
    if(abs_value(w.y) > abs_value(w.x)) kz = 1;
    if(abs_value(w.z) > abs_value(component(w,kz))) kz = 2;
    const auto p0 = geometry.corners[0].template cast<T>();
    const auto p1 = geometry.corners[1].template cast<T>();
    const auto p2 = geometry.corners[2].template cast<T>();
    const auto p3 = geometry.corners[3].template cast<T>();
    const auto edge_u = p1-p0, edge_v = p2-p0;
    const auto twist = (p3-p2)-edge_u;
    auto a = project(p0,w,kz), b = project(edge_u,w,kz);
    auto c = project(edge_v,w,kz), d = project(twist,w,kz);

    T scale = max_value(max_value(a.largest(),b.largest()),max_value(c.largest(),d.largest()));
    if(!(scale > T(0))) { out.retry = true; return out; }
    a = a*(T(1)/scale); b = b*(T(1)/scale);
    c = c*(T(1)/scale); d = d*(T(1)/scale);

    const T eps = epsilon<T>();
    // Four corner projected values bound every bilinear convex combination.
    // A guard avoids deciding a boundary miss from roundoff alone.
    const Point2<T> corners[4] = {a,a+b,a+c,a+b+c+d};
    T xlo=corners[0].x, xhi=xlo, ylo=corners[0].y, yhi=ylo;
    for(unsigned i=1;i<4;++i)
    {
        xlo = xlo < corners[i].x ? xlo : corners[i].x;
        xhi = max_value(xhi,corners[i].x);
        ylo = ylo < corners[i].y ? ylo : corners[i].y;
        yhi = max_value(yhi,corners[i].y);
    }
    const T guard = T(64)*eps;
    if(xlo>guard || xhi<-guard || ylo>guard || yhi<-guard) return out;

    const T A=b.cross(d), B=a.cross(d)+b.cross(c), C=a.cross(c);
    const T magnitude=max_value(abs_value(A),max_value(abs_value(B),abs_value(C)));
    if(!(magnitude > T(0))) { out.retry = true; return out; }
    const T qa=A/magnitude, qb=B/magnitude, qc=C/magnitude;
    if constexpr(std::is_same_v<T,float>)
    {
        // A が厳密なゼロなら線形方程式．近いだけなら次数を勝手に落とさない．
        if(A!=T(0) && abs_value(qa)<T(64)*eps) {out.retry=true;return out;}
    }
    T roots[2]{};
    unsigned root_count=0;
    bool repeated=false;
    if(A==T(0))
    {
        if(B==T(0))
        {
            if(abs_value(qc)<=guard) out.retry=true;
            return out;
        }
        roots[root_count++]=-qc/qb;
    }
    else
    {
        const T discriminant=difference_of_products(qb,qb,T(4)*qa,qc);
        const T error=T(64)*eps*(qb*qb+abs_value(T(4)*qa*qc));
        if(discriminant<T(0))
        {
            if(discriminant>=-error) out.retry=true;
            return out;
        }
        if(discriminant == T(0))
        {
            if constexpr(std::is_same_v<T, float>)
            {
                // FP32 では，判別式が丸めによってゼロになった可能性がある
                // この精度では重解と確定せず，FP64 での再評価を要求する
                out.retry = true;
                return out;
            }
            else
            {
                // FP64 側で判別式がゼロと評価された場合の既存処理
                // 候補を一つ登録し，後続の検証へ進める
                roots[root_count++] = -qb / (T(2) * qa);
                repeated = true;
            }
        }
        else
        {
            if(discriminant<=error) {out.retry=true;return out;}
            const T q=-T(0.5)*(qb+copy_sign(square_root(discriminant),qb));
            roots[0]=q/qa; roots[1]=qc/q;
            if(roots[1]<roots[0]) {const T tmp=roots[0];roots[0]=roots[1];roots[1]=tmp;}
            root_count=2;
        }
    }

    for(unsigned root=0;root<root_count;++root)
    {
        T u=roots[root];
        if(!(u==u)) {out.retry=true;continue;}
        if(u<-guard || u>T(1)+guard) continue;
        const auto numerator=a+b*u, denominator=c+d*u;
        if(denominator.largest()<=guard)
        {
            // 消去式の余分な解や，ray と iso-line の一致．孤立解として捏造しない．
            out.retry=true;continue;
        }
        T v=abs_value(denominator.x)>=abs_value(denominator.y)
            ? -numerator.x/denominator.x : -numerator.y/denominator.y;
        if(!(v==v)) {out.retry=true;continue;}
        if(v<-guard || v>T(1)+guard) continue;

        // A few Newton corrections refine an already enumerated algebraic root.
        // They are not used as a replacement for enumeration of both roots.
        T conditioning=T(1);
        for(unsigned iteration=0;iteration<3;++iteration)
        {
            const auto fu=b+d*v, fv=c+d*u;
            const T determinant=fu.cross(fv);
            const auto residual=a+b*u+c*v+d*(u*v);
            if(abs_value(determinant)<=T(64)*eps*fu.largest()*fv.largest())
            {
                if(!repeated) out.retry=true;
                break;
            }
            conditioning=max_value(T(1),max_value(fu.largest(),fv.largest())/abs_value(determinant));
            const T du=residual.cross(fv)/determinant;
            const T dv=fu.cross(residual)/determinant;
            u-=du; v-=dv;
        }
        const T parameter_error=T(32)*eps*conditioning;
        if constexpr(std::is_same_v<T,float>)
        {
            // 近接した根・境界付近では FP64 で入力四隅から再計算する．
            // Match the inclusive boundary test below. In FP32, rounding can make
            // u == 1-parameter_error even for a root strictly inside the patch.
            // That case must be reevaluated in FP64, not returned as a boundary.
            if(parameter_error>T(2e-5) || u<=parameter_error || u>=T(1)-parameter_error
               || v<=parameter_error || v>=T(1)-parameter_error)
            {out.retry=true;continue;}
        }
        if(parameter_error>T(1e-5) && !repeated) {out.retry=true;continue;}
        if(u<-parameter_error || u>T(1)+parameter_error
           || v<-parameter_error || v>T(1)+parameter_error) continue;
        bool boundary=u<=parameter_error || u>=T(1)-parameter_error
                   || v<=parameter_error || v>=T(1)-parameter_error;
        std::uint32_t exact_edges=0;
        if constexpr(std::is_same_v<T,double>)
        {
            if(boundary && !repeated)
            {
                // The numerical root's error band is not proof of an edge.
                // For a certified regular map, use exact signs of the four
                // ORIGINAL binary32 edge planes to distinguish in/out/on.
                // Folded/uncertified maps retain their existing pending path.
                const int orientation=RegularPatch64::certified_orientation(geometry.corners);
                const auto location=BilinearPatchBoundary::classify_regular(
                    geometry.corners,direction,orientation);
                if(location.membership==BilinearBoundaryMembership::Outside) continue;
                if(location.membership==BilinearBoundaryMembership::Interior) boundary=false;
                if(location.membership==BilinearBoundaryMembership::Boundary)
                {
                    exact_edges=location.edges;
                    // These coordinates are mathematically equal to endpoints;
                    // this does not jitter the ray or widen patch membership.
                    if(exact_edges&BilinearEdgeU0) u=T(0);
                    if(exact_edges&BilinearEdgeU1) u=T(1);
                    if(exact_edges&BilinearEdgeV0) v=T(0);
                    if(exact_edges&BilinearEdgeV1) v=T(1);
                }
            }
        }
        if(u<T(0) || u>T(1)) u=clamp_unit(u);
        if(v<T(0) || v>T(1)) v=clamp_unit(v);
        const auto residual=a+b*u+c*v+d*(u*v);
        if(residual.largest()>T(256)*eps) {out.retry=true;continue;}
        const auto lower=p0*(T(1)-u)+p1*u;
        const auto upper=p2*(T(1)-u)+p3*u;
        const auto point=lower*(T(1)-v)+upper*v;
        const T t=component(point,kz)/component(w,kz);
        if(!(t>T(0)))
        {
            if(abs_value(t)<=T(64)*eps) out.retry=true;
            continue;
        }
        BilinearRayHit hit{};
        hit.u=static_cast<float>(u); hit.v=static_cast<float>(v); hit.t=static_cast<float>(t);
        // Membership is decided before packing u/v into FP32. An interior
        // FP64 coordinate can round to 1.0f without lying on a shared edge.
        // Preserve the pre-packing boundary decision and the residual check.
        if(boundary)
            hit.flags|=BilinearBoundary | (exact_edges << bilinear_boundary_edge_shift);
        if(repeated) hit.flags|=BilinearSingular;
        if constexpr(std::is_same_v<T,double>) hit.flags|=BilinearUsedFp64;
        const Vec3 error3=geometry.evaluate(hit.u,hit.v)-direction*hit.t;
        hit.residual=max_value(abs_value(error3.x),max_value(abs_value(error3.y),abs_value(error3.z)));
        if(!(hit.residual<=0x1p-21f && hit.t<4.0f)) {out.retry=true;continue;}
        if(out.result.count<2) out.result.hits[out.result.count++]=hit;
        else out.retry=true;
    }
    return out;
}
} // namespace bilinear_detail

struct BilinearPatchIntersector
{
    [[nodiscard]] HOST_DEVICE static BilinearRayHits intersect(
        const BilinearPatchGeometry& patch, const Vec3 direction) noexcept
    {
        const float direction_norm=direction.dot(direction);
        if(!direction.is_finite() || !(direction_norm>0.99f && direction_norm<1.01f))
        {BilinearRayHits r{};r.flags=BilinearInvalidInput;return r;}
        for(unsigned i=0;i<4;++i)
        {
            // Query geometry is in unit collecting-direction space.
            const auto& p=patch.corners[i];
            if(!p.is_finite() || !(p.dot(p)<=4.0f))
            {BilinearRayHits r{};r.flags=BilinearInvalidInput;return r;}
        }
        const auto ordinary=bilinear_detail::solve<float>(patch,direction);
        auto result=ordinary.result;
        if(ordinary.retry)
        {
            const auto precise=bilinear_detail::solve<double>(patch,direction);
            result=precise.result;
            result.flags|=BilinearFp64Evaluation;
            if(precise.retry) result.flags|=BilinearUnresolved;
        }
        // root_index は u,v の辞書順で安定化する．traversal order は使用しない．
        if(result.count==2)
        {
            auto& a=result.hits[0]; auto& b=result.hits[1];
            if(b.u<a.u || (b.u==a.u && b.v<a.v)) {const auto t=a;a=b;b=t;}
        }
        return result;
    }
};

static_assert(sizeof(BilinearRayHit)==20);
static_assert(std::is_trivially_copyable_v<BilinearRayHits>);
} // namespace rainbow
