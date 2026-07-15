#include "exx_tile_policy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace hamilt
{
namespace
{
constexpr std::size_t BYTES_PER_MIB = 1024u * 1024u;
constexpr double DEFAULT_BUDGET_MIB = 1024.0;

bool append_product(std::size_t& total, std::initializer_list<std::size_t> factors)
{
    std::size_t bytes = 0;
    if (!checked_exx_size_product(factors, bytes))
    {
        return false;
    }
    return checked_exx_size_sum({total, bytes}, total);
}

std::size_t complex_bytes(const std::string& precision)
{
    return precision == "single" ? sizeof(float) * 2 : sizeof(double) * 2;
}

std::size_t real_bytes(const std::string& precision)
{
    return precision == "single" ? sizeof(float) : sizeof(double);
}

bool budget_bytes(double requested_mib, std::size_t& result)
{
    const double budget_mib = requested_mib == 0.0 ? DEFAULT_BUDGET_MIB : requested_mib;
    if (!std::isfinite(budget_mib) || budget_mib <= 0.0)
    {
        return false;
    }
    const long double bytes = static_cast<long double>(budget_mib) * static_cast<long double>(BYTES_PER_MIB);
    if (bytes > static_cast<long double>(std::numeric_limits<std::size_t>::max()))
    {
        return false;
    }
    result = static_cast<std::size_t>(bytes);
    return result > 0;
}

std::vector<int> batch_candidates(const ExxTilePolicyInput& input)
{
    if (input.requested_batch_fft_size > 0)
    {
        return {input.requested_batch_fft_size};
    }
    if (input.device != "gpu")
    {
        return {1};
    }

    const int limit = std::max(1, std::min(128, input.nbands));
    int candidate = 1;
    while (candidate <= limit / 2)
    {
        candidate *= 2;
    }
    std::vector<int> candidates;
    for (; candidate >= 1; candidate /= 2)
    {
        candidates.push_back(candidate);
        if (candidate == 1)
        {
            break;
        }
    }
    return candidates;
}

std::vector<int> band_candidates(const ExxTilePolicyInput& input, int batch)
{
    const int nbands = std::max(1, input.nbands);
    if (input.requested_band_tile_size > 0)
    {
        return {std::min(input.requested_band_tile_size, nbands)};
    }

    std::vector<int> candidates;
    if (batch > nbands)
    {
        candidates.push_back(nbands);
        return candidates;
    }
    for (int band = (nbands / batch) * batch; band >= batch; band -= batch)
    {
        candidates.push_back(band);
    }
    return candidates;
}

std::vector<int> q_candidates(const ExxTilePolicyInput& input)
{
    const int active_q = std::max(1, input.active_q_count);
    if (input.requested_q_tile_size > 0)
    {
        return {std::min(input.requested_q_tile_size, active_q)};
    }

    std::vector<int> candidates;
    candidates.reserve(active_q);
    for (int q = active_q; q >= 1; --q)
    {
        candidates.push_back(q);
    }
    return candidates;
}
} // namespace

bool checked_exx_size_sum(std::initializer_list<std::size_t> terms, std::size_t& result)
{
    result = 0;
    for (const std::size_t term: terms)
    {
        if (term > std::numeric_limits<std::size_t>::max() - result)
        {
            return false;
        }
        result += term;
    }
    return true;
}

bool estimate_exx_managed_bytes(const ExxTilePolicyInput& input,
                                int batch_fft_size,
                                int band_tile_size,
                                int q_tile_size,
                                ExxPotentialCacheMode cache_mode,
                                ExxTileMemoryEstimate& estimate)
{
    estimate = ExxTileMemoryEstimate{};
    if (batch_fft_size <= 0 || band_tile_size <= 0 || q_tile_size <= 0)
    {
        return false;
    }

    const std::size_t batch = static_cast<std::size_t>(batch_fft_size);
    const std::size_t band = static_cast<std::size_t>(band_tile_size);
    const std::size_t q = static_cast<std::size_t>(q_tile_size);
    const std::size_t cbytes = complex_bytes(input.precision);
    const std::size_t rbytes = real_bytes(input.precision);
    std::size_t fft_grid = 0;
    if (!checked_exx_size_product({static_cast<std::size_t>(std::max(1, input.fft_nx)),
                                   static_cast<std::size_t>(std::max(1, input.fft_ny)),
                                   static_cast<std::size_t>(std::max(1, input.fft_nz))},
                                  fft_grid))
    {
        return false;
    }
    const std::size_t nrxx = input.nrxx > 0 ? input.nrxx : fft_grid;
    const std::size_t npw = input.npw > 0 ? input.npw : fft_grid;
    const std::size_t npwk_max = input.npwk_max > 0 ? input.npwk_max : npw;

    std::size_t total = input.fixed_scratch_bytes;
    if (input.device == "gpu")
    {
        std::size_t batch_fft_buffer_bytes = 0;
        std::size_t base_fft_buffer_bytes = 0;
        if (!checked_exx_size_product({input.fft_bundle_count, fft_grid, cbytes},
                                      base_fft_buffer_bytes))
        {
            return false;
        }
        if (batch_fft_size > 1
            && (!checked_exx_size_product({2, input.fft_bundle_count, batch, fft_grid, cbytes},
                                          batch_fft_buffer_bytes)
                || !checked_exx_size_product({input.fft_bundle_count, batch, fft_grid, cbytes},
                                             estimate.cufft_workspace_bytes)))
        {
            return false;
        }
        if (!checked_exx_size_sum({batch_fft_buffer_bytes, base_fft_buffer_bytes},
                                  estimate.fft_buffer_bytes)
            || !checked_exx_size_sum({total, estimate.fft_buffer_bytes, estimate.cufft_workspace_bytes}, total))
        {
            return false;
        }
    }

    if (input.workload == ExxTileWorkload::stress)
    {
        if (!checked_exx_size_product(2, q, estimate.potential_cache_entries)
            || !checked_exx_size_product({estimate.potential_cache_entries, npw, rbytes},
                                         estimate.potential_bytes)
            || !checked_exx_size_sum({total, estimate.potential_bytes}, total)
            || !append_product(total, {band, nrxx, cbytes})
            || !append_product(total, {q, band, nrxx, cbytes})
            || !append_product(total, {band, rbytes})
            || !append_product(total, {q, band, rbytes}))
        {
            return false;
        }
    }
    else
    {
        if (!append_product(total, {batch, nrxx, cbytes})
            || !append_product(total, {batch, npwk_max, cbytes})
            || !append_product(total, {batch, nrxx, cbytes})
            || !append_product(total, {batch, npw, cbytes})
            || !append_product(total, {batch, npw, rbytes})
            || !append_product(total, {batch, rbytes})
            || !append_product(total, {2, band, nrxx, cbytes})
            || !append_product(total, {q, band, nrxx, cbytes})
            || !append_product(total, {q, band, rbytes})
            || !append_product(total, {2, q, band, cbytes}))
        {
            return false;
        }

        std::size_t direct_entries = 0;
        if (!checked_exx_size_product(input.max_star_size, q, direct_entries))
        {
            return false;
        }
        const std::size_t operator_entries = cache_mode == ExxPotentialCacheMode::full_target_k
                                                 ? static_cast<std::size_t>(std::max(1, input.active_q_count))
                                                 : q;
        estimate.potential_cache_entries = std::max(direct_entries, operator_entries);
        if (!checked_exx_size_product({estimate.potential_cache_entries, npw, rbytes},
                                      estimate.potential_bytes)
            || !checked_exx_size_sum({total, estimate.potential_bytes}, total))
        {
            return false;
        }
    }
    estimate.total_bytes = total;
    return true;
}

ExxTilePolicyResult choose_exx_tiles(const ExxTilePolicyInput& input)
{
    ExxTilePolicyResult result;
    if (!budget_bytes(input.memory_budget_mb, result.budget_bytes))
    {
        return result;
    }
    if (!input.auto_tiling
        && (input.requested_batch_fft_size <= 0
            || input.requested_band_tile_size <= 0
            || input.requested_q_tile_size <= 0))
    {
        return result;
    }

    ExxTileMemoryEstimate minimum;
    if (estimate_exx_managed_bytes(input, 1, 1, 1, ExxPotentialCacheMode::q_tile, minimum))
    {
        result.minimum_required_bytes = minimum.total_bytes;
    }

    const bool has_requested_constraint = input.requested_batch_fft_size > 0
                                          || input.requested_band_tile_size > 0
                                          || input.requested_q_tile_size > 0;
    const std::vector<int> batches = batch_candidates(input);
    const std::vector<int> qs = q_candidates(input);
    const std::vector<ExxPotentialCacheMode> cache_modes
        = input.workload == ExxTileWorkload::stress
              ? std::vector<ExxPotentialCacheMode>{ExxPotentialCacheMode::q_tile}
              : std::vector<ExxPotentialCacheMode>{ExxPotentialCacheMode::full_target_k,
                                                   ExxPotentialCacheMode::q_tile};
    for (const int batch: batches)
    {
        const std::vector<int> bands = band_candidates(input, batch);
        for (const int band: bands)
        {
            for (const int q: qs)
            {
                for (const ExxPotentialCacheMode mode: cache_modes)
                {
                    ExxTileMemoryEstimate estimate;
                    if (!estimate_exx_managed_bytes(input, batch, band, q, mode, estimate))
                    {
                        continue;
                    }
                    if (has_requested_constraint
                        && (result.requested_bytes == 0 || estimate.total_bytes < result.requested_bytes))
                    {
                        result.requested_bytes = estimate.total_bytes;
                    }
                    if (estimate.total_bytes > result.budget_bytes)
                    {
                        continue;
                    }
                    result.fits = true;
                    result.batch_fft_size = batch;
                    result.band_tile_size = band;
                    result.q_tile_size = q;
                    result.cache_mode = mode;
                    result.estimated_peak_bytes = estimate.total_bytes;
                    result.cufft_workspace_bytes = estimate.cufft_workspace_bytes;
                    return result;
                }
            }
        }
    }
    return result;
}

bool operator_exx_potential_cache_entry_limit(ExxPotentialCacheMode mode,
                                               std::size_t active_q_count,
                                               std::size_t q_tile_count,
                                               std::size_t& result)
{
    result = mode == ExxPotentialCacheMode::full_target_k ? active_q_count : q_tile_count;
    return true;
}

bool direct_exx_potential_cache_entry_limit(std::size_t star_size,
                                             std::size_t q_tile_count,
                                             std::size_t& result)
{
    return checked_exx_size_product(star_size, q_tile_count, result);
}

bool checked_exx_qtile_workspace_counts(std::size_t target_tile_size,
                                        std::size_t source_tile_size,
                                        std::size_t q_tile_size,
                                        std::size_t real_size,
                                        std::size_t& target_count,
                                        std::size_t& q_count,
                                        std::size_t& weight_count)
{
    return checked_exx_size_product({target_tile_size, real_size}, target_count)
           && checked_exx_size_product({q_tile_size, source_tile_size, real_size}, q_count)
           && checked_exx_size_product({q_tile_size, source_tile_size}, weight_count);
}

bool checked_exx_allocation_bytes(std::size_t element_count, std::size_t element_bytes)
{
    std::size_t allocation_bytes = 0;
    return checked_exx_size_product(element_count, element_bytes, allocation_bytes);
}

bool checked_exx_size_to_int(std::size_t value, int& result)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    result = static_cast<int>(value);
    return true;
}

const char* exx_potential_cache_mode_name(ExxPotentialCacheMode mode)
{
    return mode == ExxPotentialCacheMode::full_target_k ? "full-target-k" : "q-tile";
}

} // namespace hamilt
