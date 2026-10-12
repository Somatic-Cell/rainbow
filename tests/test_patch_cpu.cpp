#include "patch_validation.hpp"
#include "regular64_regression.hpp"

#include <rainbow/raindrop_settings.hpp>
#include <cstdlib>
#include <iostream>
#include <limits>

int main()
{
    using namespace rainbow;
    using namespace rainbow::tests;
    try
    {
        // The reconstructed near-degenerate R patch is regular, not folded.
        const auto regression=regular64_regression_patch();
        require_patch(regression.certified_orientation()==0,"Regression no longer exercises FP32 uncertainty.");
        require_patch(RegularPatch64::certified_orientation(regression.corners)==-1,"FP64 regular certificate failed.");
        float precise=0;
        require_patch(RegularPatch64::try_metric(regression.corners,-1,precise),"FP64 metric failed.");
        const double reference=integrated_patch_angle(regression,512);
        require_patch(std::abs(std::abs(double(precise))-reference)<2e-12,"FP64 area disagrees with independent quadrature.");
        auto fallback_vertices=regular64_regression_vertices();
        const PatchBuildLayout fallback_layout{2,2,4,1,4,1.0f};
        const PatchCellStatus expected_fallback[]={PatchCellStatus::RegularNegative,PatchCellStatus::RegularPositive,
            PatchCellStatus::NeedsRefinement,PatchCellStatus::NeedsRefinement};
        for(unsigned i=0;i<4;++i)
            require_patch(PatchConstruction::make(fallback_vertices.data(),fallback_layout,i).patch.status==expected_fallback[i],
                          "FP64 fallback changed folded/degenerate semantics.");

        // Signed Cartesian permutations preserve the stored geometry exactly.
        // Check both orientations; the difficult corner remains nonzero.
        const unsigned permutations[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
        const int parity[6]={1,-1,-1,1,1,-1};
        unsigned variant_checks=0;
        for(unsigned permutation=0;permutation<6;++permutation)for(unsigned mask=0;mask<8;++mask)
        {
            auto g=regression;
            int determinant=parity[permutation];
            for(unsigned k=0;k<3;++k)if(mask&(1u<<k))determinant=-determinant;
            for(auto& corner:g.corners)
            {
                const float c[3]={corner.x,corner.y,corner.z};
                float d[3];
                for(unsigned k=0;k<3;++k)d[k]=c[permutations[permutation][k]]*((mask&(1u<<k))?-1.0f:1.0f);
                corner={d[0],d[1],d[2]};
            }
            const int orientation=-determinant;
            require_patch(RegularPatch64::certified_orientation(g.corners)==orientation,"Transformed FP64 certificate failed.");
            float metric=0;
            require_patch(RegularPatch64::try_metric(g.corners,orientation,metric),"Transformed metric failed.");
            require_patch(std::abs(std::abs(double(metric))-reference)<2e-12,"Transformed area changed.");
            ++variant_checks;
        }
        // Explicitly retain all pre-existing successful FP32 metric bits.
        std::uint32_t random=17;
        const auto uniform=[&]() {random=1664525u*random+1013904223u;return double(random)/4294967296.0;};
        unsigned fast_checks=0;
        for(unsigned k=0;k<2000;++k)
        {
            const float h=float(0.01+0.7*uniform());
            const float cx=float(-0.4+0.8*uniform()),cy=float(-0.4+0.8*uniform());
            BilinearPatchGeometry g{{Vec3{cx-h,cy-h,1}.normalized(),Vec3{cx+h,cy-h,1}.normalized(),
                                     Vec3{cx-h,cy+h,1}.normalized(),Vec3{cx+h,cy+h,1}.normalized()}};
            if(k&1u){std::swap(g.corners[0],g.corners[1]);std::swap(g.corners[2],g.corners[3]);}
            const int old_orientation=g.certified_orientation();
            const float old_metric=g.signed_solid_angle();
            require_patch(old_orientation!=0&&old_metric*float(old_orientation)>0,"Fast-path test fixture invalid.");
            auto v=regular64_regression_vertices();
            for(unsigned j=0;j<4;++j)v[j].direction_drop=g.corners[j];
            const auto r=PatchConstruction::make(v.data(),fallback_layout,0);
            require_patch(r.patch.has_regular_spherical_map(),"Existing FP32 success was rejected.");
            require_patch(std::bit_cast<std::uint32_t>(r.patch.signed_solid_angle_sr)
                         ==std::bit_cast<std::uint32_t>(old_metric),"Existing FP32 area bits changed.");
            ++fast_checks;
        }
        require_patch(RegularPatch64::certified_orientation(nullptr)==0,"Null geometry certified.");
        auto invalid=regression;invalid.corners[0].x=std::numeric_limits<float>::quiet_NaN();
        require_patch(RegularPatch64::certified_orientation(invalid.corners)==0,"NaN geometry certified.");
        std::cout<<"FP64 regression: signed variants="<<variant_checks
                 <<", unchanged FP32 fast-path cases="<<fast_checks<<".\n";

        auto geometry=square_patch();
        require_patch(geometry.certified_orientation()==1,"Square orientation is not positive.");
        const double exact=2.0*std::acos(-1.0)/3.0; // One cube face occupies one sixth of the sphere.
        require_patch(std::abs(geometry.signed_solid_angle()-exact)<3e-6,"Analytic solid angle failed.");
        require_patch(std::abs(integrated_patch_angle(geometry)-exact)<5e-5,"Independent integral failed.");
        std::swap(geometry.corners[0],geometry.corners[1]);
        std::swap(geometry.corners[2],geometry.corners[3]);
        require_patch(geometry.certified_orientation()==-1,"Negative orientation was discarded.");
        require_patch(std::abs(geometry.signed_solid_angle()+exact)<3e-6,"Signed area failed.");
        std::swap(geometry.corners[1],geometry.corners[3]);
        require_patch(geometry.certified_orientation()==0,"Fold was not flagged.");
        geometry=square_patch();
        for(auto& corner:geometry.corners) corner={1,0,0};
        require_patch(geometry.certified_orientation()==0,"Degenerate patch was certified.");

        RaindropSettings settings;settings.grid_width=2;settings.grid_height=2;
        PatchBuildLayout layout{};
        require_patch(PatchBuildLayout::try_make(settings.make_config(),layout),"Layout construction failed.");
        auto vertices=synthetic_patch_vertices();
        const PatchCellStatus expected[]={PatchCellStatus::RegularPositive,PatchCellStatus::NeedsRefinement,
                                         PatchCellStatus::MissingCorners,PatchCellStatus::NoOutgoingCorners};
        std::vector<PatchCellStatus> statuses(4);
        std::vector<OutgoingPatch> records;
        std::vector<PatchAabb> bounds;
        for(unsigned id=0;id<4;++id)
        {
            const auto result=PatchConstruction::make(vertices.data(),layout,id);
            require_patch(result.patch.status==expected[id],"Synthetic classification failed.");
            statuses[id]=result.patch.status;
            if(has_patch_geometry(statuses[id])) {records.push_back(result.patch);bounds.push_back(result.aabb);}
        }
        verify_patch_buffer(vertices,layout,statuses,records,bounds);
        // Distinct failure causes must survive neighboring absent corners.
        vertices[9].status=VertexStatus::PhaseOverflow;
        require_patch(PatchConstruction::make(vertices.data(),layout,2).patch.status==PatchCellStatus::InvalidVertex,
                      "Numerical failure was hidden by missing corner.");
        vertices[0].direction_drop.x=std::numeric_limits<float>::quiet_NaN();
        require_patch(PatchConstruction::make(vertices.data(),layout,0).patch.status==PatchCellStatus::InvalidGeometry,
                      "NaN direction was accepted.");

        // Independent analytic row-major mapping on a nonsquare grid.
        settings.grid_width=3;settings.grid_height=4;
        require_patch(PatchBuildLayout::try_make(settings.make_config(),layout),"Nonsquare layout failed.");
        require_patch(layout.vertices_per_path==12 && layout.cells_per_path==6 && layout.cell_count==24,
                      "Nonsquare counts failed.");
        std::uint32_t ids[4];layout.corner_indices(23,ids);
        require_patch(ids[0]==43 && ids[1]==44 && ids[2]==46 && ids[3]==47,"Corner IDs failed.");
        auto bad=settings.make_config();bad.grid_width=std::numeric_limits<std::uint32_t>::max();
        require_patch(!PatchBuildLayout::try_make(bad,layout),"Grid overflow was not rejected.");

        // Longitudes need no seam handling: geometry remains Cartesian.
        geometry=square_patch();
        for(auto& p:geometry.corners) p={-p.z,p.y,p.x};
        require_patch(geometry.certified_orientation()==1,"Longitude seam changed orientation.");
        require_patch(std::abs(geometry.signed_solid_angle()-exact)<3e-6,"Longitude seam changed area.");
        std::cout<<"Patch CPU geometry: passed (orientation, area, bounds, IDs, invalid/boundary cases).\n";
        return EXIT_SUCCESS;
    }
    catch(const std::exception& e){std::cerr<<"Error: "<<e.what()<<'\n';return EXIT_FAILURE;}
}
