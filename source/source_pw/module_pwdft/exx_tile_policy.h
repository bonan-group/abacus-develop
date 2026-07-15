#ifndef EXX_TILE_POLICY_H
#define EXX_TILE_POLICY_H

#include "source_pw/module_pwdft/exx_size_utils.h"

#include <cstddef>
#include <initializer_list>
#include <string>

namespace hamilt
{

enum class ExxPotentialCacheMode
{
    full_target_k,
    q_tile
};

enum class ExxTileWorkload
{
    operator_apply,
    stress
};

struct ExxTilePolicyInput
{
    bool auto_tiling = true;
    ExxTileWorkload workload = ExxTileWorkload::operator_apply;
    std::string device = "cpu";
    std::string precision = "double";
    double memory_budget_mb = 0.0;
    int nbands = 1;
    int active_q_count = 1;
    std::size_t max_star_size = 1;
    std::size_t nrxx = 0;
    std::size_t npw = 0;
    std::size_t npwk_max = 0;
    std::size_t target_npwk_max = 0;
    int fft_nx = 0;
    int fft_ny = 0;
    int fft_nz = 0;
    std::size_t fft_bundle_count = 2;
    int requested_batch_fft_size = 0;
    int requested_band_tile_size = 0;
    int requested_q_tile_size = 0;
    std::size_t fixed_scratch_bytes = 0;
};

struct ExxTileMemoryEstimate
{
    std::size_t total_bytes = 0;
    std::size_t fft_buffer_bytes = 0;
    std::size_t cufft_workspace_bytes = 0;
    std::size_t potential_bytes = 0;
    std::size_t potential_cache_entries = 0;
};

struct ExxTilePolicyResult
{
    bool fits = false;
    int batch_fft_size = 1;
    int band_tile_size = 1;
    int q_tile_size = 1;
    ExxPotentialCacheMode cache_mode = ExxPotentialCacheMode::q_tile;
    std::size_t budget_bytes = 0;
    std::size_t estimated_peak_bytes = 0;
    std::size_t cufft_workspace_bytes = 0;
    std::size_t minimum_required_bytes = 0;
    std::size_t requested_bytes = 0;
};

bool checked_exx_size_sum(std::initializer_list<std::size_t> terms, std::size_t& result);

bool estimate_exx_managed_bytes(const ExxTilePolicyInput& input,
                                int batch_fft_size,
                                int band_tile_size,
                                int q_tile_size,
                                ExxPotentialCacheMode cache_mode,
                                ExxTileMemoryEstimate& estimate);

ExxTilePolicyResult choose_exx_tiles(const ExxTilePolicyInput& input);

bool operator_exx_potential_cache_entry_limit(ExxPotentialCacheMode mode,
                                               std::size_t active_q_count,
                                               std::size_t q_tile_count,
                                               std::size_t& result);

bool direct_exx_potential_cache_entry_limit(std::size_t star_size,
                                             std::size_t q_tile_count,
                                             std::size_t& result);

bool checked_exx_qtile_workspace_counts(std::size_t target_tile_size,
                                        std::size_t source_tile_size,
                                        std::size_t q_tile_size,
                                        std::size_t real_size,
                                        std::size_t& target_count,
                                        std::size_t& q_count,
                                        std::size_t& weight_count);

bool checked_exx_allocation_bytes(std::size_t element_count, std::size_t element_bytes);

bool checked_exx_size_to_int(std::size_t value, int& result);

const char* exx_potential_cache_mode_name(ExxPotentialCacheMode mode);

} // namespace hamilt

#endif
