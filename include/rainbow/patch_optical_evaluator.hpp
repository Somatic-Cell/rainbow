#pragma once

#include <rainbow/patch_optics_data.hpp>
#include <rainbow/focal_phase.hpp>
#include <rainbow/folded_patch_geometry.hpp>
#include <rainbow/phase_interpolation.hpp>
#include <rainbow/polarization_transport.hpp>
#include <cmath>

namespace rainbow
{

// Per-hit optical arithmetic and per-direction accumulation, shared by CPU/CUDA.
// Stored Field32 includes Fresnel/interface phases, but not propagation phase.
// Parallel transport is our declared frame-identification convention, not an
// algorithm explicitly specified by Sadeghi et al. (2012).
struct PatchOpticalEvaluator
{
    struct Contribution
    {
        OpticalField64 path_field{};
        OpticalField64 path_field_second{};
        double incoherent_s = 0.0, incoherent_p = 0.0;
    };

    [[nodiscard]] HOST_DEVICE static std::uint32_t evaluate_patch(
        const OutgoingVertex* vertices, const std::uint32_t vertex_count,
        const OutgoingPatch& patch, const PatchQueryHit& hit,
        const Vec3 direction, const TransverseFrame target_frame,
        Contribution& output, const FoldedPatchRecord* folded = nullptr,
        const FoldedBranch** used_branch = nullptr,
        const Field32* second_input_fields = nullptr) noexcept
    {
        if(used_branch) *used_branch = nullptr;
        if(!vertices || (hit.flags & ~bilinear_known_hit_flags) != 0u
           || hit.patch_id != patch.patch_id || hit.root_index > 1u
           || !(hit.u >= 0.0f && hit.u <= 1.0f && hit.v >= 0.0f && hit.v <= 1.0f)
           || !(hit.t > 0.0f && finite(hit.t)) || !(hit.residual >= 0.0f && finite(hit.residual)))
            return PatchOpticalInvalidInput;

        const bool boundary_owned=(hit.flags&BilinearBoundaryOwned)!=0u;
        if(boundary_owned && (!patch.has_regular_spherical_map()
           || (hit.flags&BilinearSingular) || !patch_boundary_feature(patch,hit).valid))
            return PatchOpticalInvalidInput;
        if((hit.flags&bilinear_boundary_edge_mask) && !(hit.flags&BilinearBoundary))
            return PatchOpticalInvalidInput;

        std::uint32_t pending = 0;
        const FoldedBranch* branch = nullptr;
        if(patch.status == PatchCellStatus::NeedsRefinement)
        {
            if(!folded || folded->patch_id != patch.patch_id || folded->compact_index != hit.compact_index
               || folded->flags != FoldedReady)
                pending |= PatchOpticalRefinementPending;
            else
            {
                const int b = FoldedPatchGeometry::branch_at(*folded, double(hit.u), double(hit.v));
                if(b < 0) pending |= PatchOpticalSingularPending;
                else branch = &folded->branches[b];
            }
        }
        else if(!patch.has_regular_spherical_map()) return PatchOpticalInvalidInput;
        if((hit.flags & BilinearBoundary) && !boundary_owned) pending |= PatchOpticalBoundaryPending;
        if(hit.flags & BilinearSingular) pending |= PatchOpticalSingularPending;
        if(pending != 0u) return pending; // no epsilon-area or zero-field substitution

        const double area = branch ? double(patch.incident_area_drop2) * branch->area_fraction
                                   : static_cast<double>(patch.incident_area_drop2);
        const double omega = branch ? branch->solid_angle_sr
                                    : ::fabs(static_cast<double>(patch.signed_solid_angle_sr));
        if(!(area > 0.0 && finite(area) && omega > 0.0 && finite(omega)))
            return PatchOpticalInvalidInput;

        const unsigned columns = second_input_fields ? 2u : 1u;
        const double input_weight = second_input_fields ? 0.5 : 1.0;
        Field32 transported[2][4]{};
        PhaseCycles paths[4]{};
        for(unsigned corner = 0; corner < 4; ++corner)
        {
            const auto index = patch.vertex_indices[corner];
            if(index >= vertex_count) return PatchOpticalInvalidInput;
            const auto& vertex = vertices[index];
            if(vertex.status != VertexStatus::Valid || !vertex.position_drop.is_finite()
               || !vertex.direction_drop.is_finite() || !vertex.basis_x.is_finite()
               || !finite_field(vertex.field)) return PatchOpticalInvalidInput;
            if(!vertex.optical_cycles.is_valid()) return PatchOpticalPhaseFailure;

            // Recover the stored right-handed transverse frame. Normalize only
            // the reconstructed second axis; the stored Jones components and
            // the first axis keep their existing representation.
            const auto second = vertex.direction_drop.cross(vertex.basis_x);
            const double length2 = second.cast<double>().dot(second.cast<double>());
            if(!(length2 > 0.0 && finite(length2))) return PatchOpticalFrameFailure;
            const TransverseFrame source_frame{vertex.basis_x, second.normalized()};
            if(second_input_fields)
            {
                const JonesResponse32 response{{vertex.field, second_input_fields[index]}};
                JonesResponse32 target{};
                if(!response.is_finite()) return PatchOpticalInvalidInput;
                if(!PolarizationTransport::try_transport_response(
                       vertex.direction_drop, source_frame, response,
                       direction, target_frame, target)) return PatchOpticalFrameFailure;
                transported[0][corner] = target.column[0];
                transported[1][corner] = target.column[1];
            }
            else if(!PolarizationTransport::try_transport(
                       vertex.direction_drop, source_frame, vertex.field,
                       direction, target_frame, transported[0][corner])) return PatchOpticalFrameFailure;
            paths[corner] = vertex.optical_cycles;
        }

        PhaseCycles interpolated_path{};
        if(!PhaseInterpolation::try_bilinear(paths[0], paths[1], paths[2], paths[3],
                                            hit.u, hit.v, interpolated_path))
            return PatchOpticalPhaseFailure;
        Complex32 phasor{};
        if(!interpolated_path.try_unit_phasor(phasor)) return PatchOpticalPhaseFailure;

        // Double interpolation/area scaling and accumulation; the existing
        // transport result and unit-phasor result are still FP32 by contract.
        const double amplitude = ::sqrt(area / omega);
        const double c = static_cast<double>(phasor.real), s = static_cast<double>(phasor.imag);
        Contribution candidate{};
        for(unsigned j = 0; j < columns; ++j)
        {
            const auto* f = transported[j];
            const double sr = amplitude * bilinear(f[0].x.real, f[1].x.real, f[2].x.real, f[3].x.real, hit.u, hit.v);
            const double si = amplitude * bilinear(f[0].x.imag, f[1].x.imag, f[2].x.imag, f[3].x.imag, hit.u, hit.v);
            const double pr = amplitude * bilinear(f[0].y.real, f[1].y.real, f[2].y.real, f[3].y.real, hit.u, hit.v);
            const double pi = amplitude * bilinear(f[0].y.imag, f[1].y.imag, f[2].y.imag, f[3].y.imag, hit.u, hit.v);
            // Input columns are statistically independent; output s/p are summed,
            // not averaged. No propagation or focal phase enters incoherent.
            candidate.incoherent_s += input_weight * ::fma(sr, sr, si * si);
            candidate.incoherent_p += input_weight * ::fma(pr, pr, pi * pi);
            auto& field = j == 0u ? candidate.path_field : candidate.path_field_second;
            field = {::fma(sr, c, -si * s), ::fma(sr, s, si * c),
                     ::fma(pr, c, -pi * s), ::fma(pr, s, pi * c)};
        }
        if(!finite(candidate.incoherent_s) || !finite(candidate.incoherent_p)
           || !finite_field(candidate.path_field) || !finite_field(candidate.path_field_second))
            return PatchOpticalArithmeticFailure;
        output = candidate; // success-only update
        if(used_branch) *used_branch = branch;
        return PatchOpticalNone;
    }

    [[nodiscard]] HOST_DEVICE static PatchOpticalResult evaluate_direction(
        const PatchOpticsParams& input, const std::uint32_t index,
        const FocalPhaseConfig* focal_config = nullptr,
        FocalOpticalResult* focal_output = nullptr,
        const FoldedPatchView folded = {}) noexcept
    {
        PatchOpticalResult result{};
        const bool dual = input.input_polarization == IncidentPolarization::Unpolarized;
        const unsigned columns = dual ? 2u : 1u;
        const double input_weight = dual ? 0.5 : 1.0;
        // Companion output is invalid until the entire known-hit evaluation finishes.
        // Early error exits must never leave a seemingly valid zero focal result.
        if(focal_output)
        {
            *focal_output = {};
            focal_output->flags = FocalInputError;
            invalidate_focal(*focal_output);
        }
        if((input.input_polarization != IncidentPolarization::SingleJones && !dual)
           || (dual && (!input.second_input_fields || input.second_input_count != input.vertex_count))
           || (!dual && (input.second_input_fields || input.second_input_count != 0u)))
        { result.flags = PatchOpticalInvalidInput; invalidate(result); return result; }
        if(index >= input.direction_count || !input.directions || !input.offsets || !input.summaries)
        { result.flags = PatchOpticalInvalidInput; invalidate(result); return result; }

        const auto& summary = input.summaries[index];
        result.hit_count = summary.hit_count;
        result.query_flags = summary.flags;
        if(summary.flags & patch_query_error_mask)
        {
            result.flags = PatchOpticalQueryError;
            result.first_problem_patch_id = summary.first_problem_patch_id;
            result.rejected_hits = summary.hit_count;
            invalidate(result); return result;
        }
        const std::uint64_t begin = input.offsets[index], end = input.offsets[index + 1u];
        if(begin > end || end > input.hit_storage_count
           || std::uint64_t(summary.hit_count) > end - begin
           || std::uint64_t(summary.candidate_count) != end - begin
           || (summary.hit_count != 0u && (!input.hits || !input.patches || !input.vertices)))
        { result.flags = PatchOpticalInvalidInput; result.rejected_hits = summary.hit_count;
          invalidate(result); return result; }

        // Pending flags in the summary are also preserved, even if supplied
        // externally without a corresponding record. Never certify such input.
        if((summary.flags & PatchQueryRefinementHit) && !folded.enabled())
            result.flags |= PatchOpticalRefinementPending;
        if(summary.flags & PatchQueryBoundaryHit) result.flags |= PatchOpticalBoundaryPending;
        if(summary.flags & PatchQuerySingularHit) result.flags |= PatchOpticalSingularPending;
        TransverseFrame target_frame{};
        if(!PolarizationTransport::try_make_scattering_frame(input.incident_direction,
               input.incident_basis_x, input.directions[index], target_frame))
        { result.flags |= PatchOpticalFrameFailure; result.rejected_hits = summary.hit_count;
          invalidate(result); return result; }

        Sum sr[2]{}, si[2]{}, pr[2]{}, pi[2]{}, incoherent_s{}, incoherent_p{};
        const bool with_focal = focal_config && focal_output;
        FocalOpticalResult focal{};
        Sum focal_sr[2]{}, focal_si[2]{}, focal_pr[2]{}, focal_pi[2]{}, family_envelope[4]{};
        if(with_focal && !FocalPhase::layout_valid(*focal_config)) focal.flags |= FocalInvalidGeometry;

        std::uint32_t recorded_nonregular_hits = 0;
        for(std::uint32_t i = 0; i < summary.hit_count; ++i)
        {
            const auto& hit = input.hits[begin + i];
            std::uint32_t flags = 0;
            Contribution contribution{};
            const FoldedBranch* used_branch = nullptr;
            // The upstream query sorts, removes exact duplicate reports and
            // selects one owner for certified regular source-grid boundaries.
            // Reject a broken ordering/duplicate contract instead of summing twice.
            if(i != 0u && !strictly_before(input.hits[begin + i - 1u], hit))
                flags = PatchOpticalInvalidInput;
            else if(hit.compact_index >= input.patch_count) flags = PatchOpticalInvalidInput;
            else
            {
                const auto& patch = input.patches[hit.compact_index];
                recorded_nonregular_hits += patch.status == PatchCellStatus::NeedsRefinement;
                flags = evaluate_patch(input.vertices, input.vertex_count, patch, hit,
                    input.directions[index], target_frame, contribution,
                    folded.find(hit.compact_index), &used_branch, input.second_input_fields);
            }

            if(flags != 0u)
            {
                result.flags |= flags; ++result.rejected_hits;
                result.refinement_hits += (flags & PatchOpticalRefinementPending) != 0u;
                result.boundary_hits += (flags & PatchOpticalBoundaryPending) != 0u;
                result.singular_hits += (flags & PatchOpticalSingularPending) != 0u;
                if(hit.patch_id < result.first_problem_patch_id) result.first_problem_patch_id = hit.patch_id;
                continue;
            }
            ++result.evaluated_hits;
            result.reserved += used_branch != nullptr; // ABI-retained folded-hit counter
            for(unsigned j = 0; j < columns; ++j)
            {
                const auto& f = j == 0u ? contribution.path_field : contribution.path_field_second;
                sr[j].add(f.s_real); si[j].add(f.s_imag);
                pr[j].add(f.p_real); pi[j].add(f.p_imag);
            }
            incoherent_s.add(contribution.incoherent_s); incoherent_p.add(contribution.incoherent_p);
            if(with_focal)
            {
                FocalPhaseEstimate estimate{};
                std::uint32_t code = FocalNone;
                if(used_branch)
                {
                    code = used_branch->focal_flags;
                    if(FocalPhase::layout_valid(*focal_config))
                    {
                        const auto cells = (focal_config->grid_width-1u)*(focal_config->grid_height-1u);
                        estimate.family = hit.patch_id/cells;
                        if(estimate.family >= 4) code |= FocalInvalidGeometry;
                        estimate.quarter_turns = used_branch->quarter_turns;
                        estimate.extra_quarter_turn = used_branch->extra_quarter_turn;
                    }
                    else code |= FocalInvalidGeometry;
                }
                else code = FocalPhase::estimate(input.vertices, input.vertex_count,
                    input.patches[hit.compact_index], input.incident_direction, *focal_config, estimate);
                if(code != FocalNone)
                {
                    focal.flags |= code;
                    if(hit.patch_id < focal.first_problem_patch_id) focal.first_problem_patch_id = hit.patch_id;
                }
                else
                {
                    for(unsigned j = 0; j < columns; ++j)
                    {
                        const auto& field = j == 0u ? contribution.path_field : contribution.path_field_second;
                        const auto f = FocalPhase::apply(field, estimate.quarter_turns);
                        focal_sr[j].add(f.s_real); focal_si[j].add(f.s_imag);
                        focal_pr[j].add(f.p_real); focal_pi[j].add(f.p_imag);
                    }
                    family_envelope[estimate.family].add(contribution.incoherent_s + contribution.incoherent_p);
                    ++focal.family_hits[estimate.family];
                    ++focal.corrected_hits;
                    focal.extra_quarter_turn_hits += estimate.extra_quarter_turn;
                }
            }

        }
        // Defer ONLY refinement that is accounted for by actual records.
        // Boundary/singular/query-error flags are never cleared by this feature.
        if(folded.enabled() && (summary.flags & PatchQueryRefinementHit))
        {
            if(recorded_nonregular_hits == 0u) result.flags |= PatchOpticalRefinementPending;
            if(recorded_nonregular_hits != summary.refinement_hits) result.flags |= PatchOpticalInvalidInput;
        }
        result.regular_partial_path_field = {sr[0].value(), si[0].value(), pr[0].value(), pi[0].value()};
        result.regular_partial_path_field_second = {sr[1].value(), si[1].value(), pr[1].value(), pi[1].value()};
        result.regular_partial_incoherent_s = incoherent_s.value();
        result.regular_partial_incoherent_p = incoherent_p.value();
        for(unsigned j = 0; j < columns; ++j)
        {
            result.regular_partial_path_s += input_weight * ::fma(sr[j].value(), sr[j].value(), si[j].value() * si[j].value());
            result.regular_partial_path_p += input_weight * ::fma(pr[j].value(), pr[j].value(), pi[j].value() * pi[j].value());
        }
        if(!finite_field(result.regular_partial_path_field)
           || !finite_field(result.regular_partial_path_field_second)
           || !finite(result.regular_partial_incoherent_s) || !finite(result.regular_partial_incoherent_p)
           || !finite(result.regular_partial_path_s) || !finite(result.regular_partial_path_p))
            result.flags |= PatchOpticalArithmeticFailure;
        if(result.flags & patch_optical_error_mask) invalidate(result);
        if(with_focal)
        {
            if(result.flags & patch_optical_error_mask) focal.flags |= FocalInputError;
            if(result.flags & patch_optical_pending_mask) focal.flags |= FocalInputPending;
            if(result.first_problem_patch_id < focal.first_problem_patch_id)
                focal.first_problem_patch_id = result.first_problem_patch_id;
            focal.field = {focal_sr[0].value(), focal_si[0].value(), focal_pr[0].value(), focal_pi[0].value()};
            focal.field_second = {focal_sr[1].value(), focal_si[1].value(), focal_pr[1].value(), focal_pi[1].value()};
            for(unsigned j = 0; j < columns; ++j)
            {
                const auto& f = j == 0u ? focal.field : focal.field_second;
                focal.intensity_s += input_weight * ::fma(f.s_real, f.s_real, f.s_imag * f.s_imag);
                focal.intensity_p += input_weight * ::fma(f.p_real, f.p_real, f.p_imag * f.p_imag);
            }
            for(unsigned f = 0; f < 4; ++f) focal.family_incoherent[f] = family_envelope[f].value();
            if(!finite_field(focal.field) || !finite_field(focal.field_second)
               || !finite(focal.intensity_s + focal.intensity_p)) focal.flags |= FocalArithmeticError;
            // No complete-looking focal intensity assembled from only some hits.
            if(!focal.valid()) invalidate_focal(focal);
            *focal_output = focal;
        }
        return result;
    }

private:
    HOST_DEVICE static void invalidate_focal(FocalOpticalResult& r) noexcept
    {
        r.field = {patch_optical_nan, patch_optical_nan, patch_optical_nan, patch_optical_nan};
        r.field_second = r.field;
        r.intensity_s = r.intensity_p = patch_optical_nan;
    }
    // Neumaier compensated summation. No atomic operations; one thread owns
    // one direction and consumes every hit in the existing deterministic order.
    struct Sum
    {
        double sum = 0.0, correction = 0.0;
        HOST_DEVICE void add(const double x) noexcept
        {
            const double next = sum + x;
            correction += ::fabs(sum) >= ::fabs(x) ? (sum - next) + x : (x - next) + sum;
            sum = next;
        }
        [[nodiscard]] HOST_DEVICE double value() const noexcept { return sum + correction; }
    };
    [[nodiscard]] HOST_DEVICE static double bilinear(
        const double a, const double b, const double c, const double d,
        const double u, const double v) noexcept
    {
        const double lower = ::fma(u, b - a, a), upper = ::fma(u, d - c, c);
        return ::fma(v, upper - lower, lower);
    }
    [[nodiscard]] HOST_DEVICE static bool finite(const double x) noexcept
    { return x >= -0x1.fffffffffffffp1023 && x <= 0x1.fffffffffffffp1023; }
    [[nodiscard]] HOST_DEVICE static bool finite_field(const Field32 f) noexcept
    { return finite(f.x.real) && finite(f.x.imag) && finite(f.y.real) && finite(f.y.imag); }
    [[nodiscard]] HOST_DEVICE static bool finite_field(const OpticalField64 f) noexcept
    { return finite(f.s_real) && finite(f.s_imag) && finite(f.p_real) && finite(f.p_imag); }
    [[nodiscard]] HOST_DEVICE static bool strictly_before(const PatchQueryHit& a, const PatchQueryHit& b) noexcept
    {
        if(a.patch_id != b.patch_id) return a.patch_id < b.patch_id;
        if(a.compact_index != b.compact_index) return a.compact_index < b.compact_index;
        return a.root_index < b.root_index;
    }
    HOST_DEVICE static void invalidate(PatchOpticalResult& result) noexcept
    {
        result.regular_partial_path_field = {patch_optical_nan, patch_optical_nan,
                                             patch_optical_nan, patch_optical_nan};
        result.regular_partial_path_field_second = result.regular_partial_path_field;
        result.regular_partial_incoherent_s = result.regular_partial_incoherent_p = patch_optical_nan;
        result.regular_partial_path_s = result.regular_partial_path_p = patch_optical_nan;
    }
};

} // namespace rainbow
