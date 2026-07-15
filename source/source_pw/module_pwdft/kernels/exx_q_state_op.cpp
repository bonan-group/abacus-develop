#include "source_pw/module_pwdft/kernels/exx_q_state_op.h"

#include "source_base/constants.h"
#include "source_base/tool_quit.h"
#include "source_basis/module_pw/pw_basis_k.h"

#include <algorithm>
#include <complex>
#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <vector>
#include <unordered_map>

#ifdef __MPI
#include <mpi.h>
#endif

namespace hamilt
{
namespace
{
struct IntGKey
{
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const IntGKey& rhs) const
    {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }
};

struct IntGKeyHash
{
    std::size_t operator()(const IntGKey& key) const
    {
        std::size_t h = std::hash<int>()(key.x) * static_cast<std::size_t>(73856093);
        h ^= std::hash<int>()(key.y) * static_cast<std::size_t>(19349663);
        h ^= std::hash<int>()(key.z) * static_cast<std::size_t>(83492791);
        return h;
    }
};

int checked_rounded_g_component(double value)
{
    const double rounded = std::round(value);
    if (!std::isfinite(rounded)
        || rounded < static_cast<double>(std::numeric_limits<int>::min())
        || rounded > static_cast<double>(std::numeric_limits<int>::max()))
    {
        ModuleBase::WARNING_QUIT("build_exx_symmetry_remap",
                                 "reciprocal-grid component cannot be represented as int");
    }
    return static_cast<int>(rounded);
}

IntGKey make_int_g_key(const ModuleBase::Vector3<double>& g)
{
    IntGKey key;
    key.x = checked_rounded_g_component(g.x);
    key.y = checked_rounded_g_component(g.y);
    key.z = checked_rounded_g_component(g.z);
    return key;
}

ModuleBase::Vector3<double> int_g_key_to_direct(const IntGKey& key)
{
    return ModuleBase::Vector3<double>(key.x, key.y, key.z);
}

double exx_symmetry_remap_cutoff_tolerance(double gk_ecut)
{
    return std::max(1.0e-8 * std::abs(gk_ecut), 1.0e-4);
}

ModuleBase::Vector3<double> generic_g_direct_from_ig(const ModulePW::PW_Basis_K* wfcpw, int ig)
{
    const int isz = wfcpw->ig2isz[ig];
    const int iz_raw = isz % wfcpw->nz;
    const int is = isz / wfcpw->nz;
    const int ixy = wfcpw->is2fftixy[is];
    int ix = ixy / wfcpw->fftny;
    int iy = ixy % wfcpw->fftny;
    int iz = iz_raw;
    if (ix >= int(wfcpw->nx / 2) + 1)
    {
        ix -= wfcpw->nx;
    }
    if (iy >= int(wfcpw->ny / 2) + 1)
    {
        iy -= wfcpw->ny;
    }
    if (iz >= int(wfcpw->nz / 2) + 1)
    {
        iz -= wfcpw->nz;
    }
    return ModuleBase::Vector3<double>(ix, iy, iz);
}

ModuleBase::Vector3<double> generic_g_cartesian_from_ig(const ModulePW::PW_Basis_K* wfcpw, int ig)
{
    return generic_g_direct_from_ig(wfcpw, ig) * wfcpw->G;
}

ModuleBase::Vector3<double> k_g_direct_from_igl(const ModulePW::PW_Basis_K* wfcpw, int ik, int igl)
{
    return generic_g_direct_from_ig(wfcpw, wfcpw->igl2ig_k[ik * wfcpw->npwk_max + igl]);
}

int positive_mod(int value, int modulus)
{
    int result = value % modulus;
    if (result < 0)
    {
        result += modulus;
    }
    return result;
}

ModuleBase::Vector3<double> real_grid_direct_from_global_index(const ModulePW::PW_Basis_K* wfcpw, int global_index)
{
    const int iz = global_index % wfcpw->nz;
    const int ixy = global_index / wfcpw->nz;
    const int iy = ixy % wfcpw->ny;
    const int ix = ixy / wfcpw->ny;
    return ModuleBase::Vector3<double>(static_cast<double>(ix) / wfcpw->nx,
                                       static_cast<double>(iy) / wfcpw->ny,
                                       static_cast<double>(iz) / wfcpw->nz);
}

ModuleBase::Vector3<double> negative_direct(const ModuleBase::Vector3<double>& direct)
{
    return ModuleBase::Vector3<double>(-direct.x, -direct.y, -direct.z);
}

int global_real_index_from_grid(const ModulePW::PW_Basis_K* wfcpw, int ix, int iy, int iz)
{
    return positive_mod(iz, wfcpw->nz)
           + positive_mod(iy, wfcpw->ny) * wfcpw->nz
           + positive_mod(ix, wfcpw->nx) * wfcpw->ny * wfcpw->nz;
}

int global_real_index_from_direct(const ModulePW::PW_Basis_K* wfcpw, const ModuleBase::Vector3<double>& direct)
{
    const int ix = static_cast<int>(std::lround(direct.x * wfcpw->nx));
    const int iy = static_cast<int>(std::lround(direct.y * wfcpw->ny));
    const int iz = static_cast<int>(std::lround(direct.z * wfcpw->nz));
    return global_real_index_from_grid(wfcpw, ix, iy, iz);
}

bool is_exx_realspace_symmetry_grid_compatible_impl(const ModulePW::PW_Basis_K* wfcpw,
                                                    const K_Vectors::ExxFullPoint& full_point)
{
    auto is_integral = [](double value) {
        return std::abs(value - std::round(value)) < 1.0e-8;
    };

    if (!is_integral(full_point.gmatrix.e21 * wfcpw->nx / wfcpw->ny)
        || !is_integral(full_point.gmatrix.e31 * wfcpw->nx / wfcpw->nz)
        || !is_integral(full_point.gmatrix.e12 * wfcpw->ny / wfcpw->nx)
        || !is_integral(full_point.gmatrix.e32 * wfcpw->ny / wfcpw->nz)
        || !is_integral(full_point.gmatrix.e13 * wfcpw->nz / wfcpw->nx)
        || !is_integral(full_point.gmatrix.e23 * wfcpw->nz / wfcpw->ny))
    {
        return false;
    }

    if (!is_integral(full_point.gtrans.x * wfcpw->nx)
        || !is_integral(full_point.gtrans.y * wfcpw->ny)
        || !is_integral(full_point.gtrans.z * wfcpw->nz))
    {
        return false;
    }
    return true;
}

std::complex<double> exx_realspace_bloch_phase(const K_Vectors::ExxFullPoint& full_point,
                                               const ModuleBase::Vector3<double>& rep_kvec_d,
                                               const ModuleBase::Vector3<double>& full_direct,
                                               const ModuleBase::Vector3<double>& rep_direct)
{
    const double phase_arg = ModuleBase::TWO_PI * (rep_kvec_d * rep_direct - full_point.full_kvec_d * full_direct);
    return std::complex<double>(std::cos(phase_arg), std::sin(phase_arg));
}

#ifdef __MPI
template <typename T>
MPI_Datatype mpi_complex_type();

template <>
MPI_Datatype mpi_complex_type<std::complex<float>>()
{
    return MPI_COMPLEX;
}

template <>
MPI_Datatype mpi_complex_type<std::complex<double>>()
{
    return MPI_DOUBLE_COMPLEX;
}
#endif
} // namespace

bool is_exx_realspace_symmetry_grid_compatible(const ModulePW::PW_Basis_K* wfcpw,
                                               const K_Vectors::ExxFullPoint& full_point)
{
    return wfcpw != nullptr && wfcpw->nx > 0 && wfcpw->ny > 0 && wfcpw->nz > 0
           && is_exx_realspace_symmetry_grid_compatible_impl(wfcpw, full_point);
}

void validate_exx_realspace_symmetry_grid(const ModulePW::PW_Basis_K* wfcpw,
                                          const K_Vectors::ExxFullPoint& full_point)
{
    if (!is_exx_realspace_symmetry_grid_compatible(wfcpw, full_point))
    {
        ModuleBase::WARNING_QUIT("validate_exx_realspace_symmetry_grid",
                                 "PW EXX real-space symmetry operation is incompatible with the FFT grid");
    }
}

ExxSymmetryRemap build_exx_symmetry_remap(const ModulePW::PW_Basis_K* wfcpw,
                                          const K_Vectors::ExxFullPoint& full_point,
                                          int rep_spin_index,
                                          bool need_gpu_fft_index)
{
    if (wfcpw->get_device() == "cpu" && wfcpw->poolnproc > 1)
    {
        ModuleBase::WARNING_QUIT("build_exx_symmetry_remap",
                                 "PW EXX symmetry-remapped wavefunctions are not supported with "
                                 "multi-rank CPU plane-wave distribution. CPU callers should use "
                                 "the real-space EXX symmetry rotation path.");
    }

    const ModuleBase::Vector3<double> raw_rep = full_point.full_kvec_d * full_point.kgmatrix;
    const ModuleBase::Vector3<double> rep_shift = raw_rep - wfcpw->kvec_d[rep_spin_index];
    const double cutoff_tolerance = exx_symmetry_remap_cutoff_tolerance(wfcpw->gk_ecut);

    std::unordered_map<IntGKey, int, IntGKeyHash> rep_g_to_ig;
    rep_g_to_ig.reserve(wfcpw->npwk[rep_spin_index]);
    for (int ig_rep = 0; ig_rep < wfcpw->npwk[rep_spin_index]; ++ig_rep)
    {
        rep_g_to_ig.emplace(make_int_g_key(k_g_direct_from_igl(wfcpw, rep_spin_index, ig_rep)), ig_rep);
    }

    ExxSymmetryRemap remap;
    remap.rep_igl.reserve(wfcpw->npw);
    remap.fft_isz.reserve(wfcpw->npw);
    remap.phase.reserve(wfcpw->npw);
    if (need_gpu_fft_index)
    {
        remap.fft_ixyz.reserve(wfcpw->npw);
    }
    int skipped_boundary_misses = 0;
    double max_abs_full_boundary_delta = 0.0;
    double max_abs_rep_boundary_delta = 0.0;

    for (int ig = 0; ig < wfcpw->npw; ++ig)
    {
        const ModuleBase::Vector3<double> g_cart = generic_g_cartesian_from_ig(wfcpw, ig);
        const ModuleBase::Vector3<double> gplus_full = g_cart + full_point.full_kvec_c;
        const double gplus_full_norm2 = gplus_full.norm2();
        if (gplus_full_norm2 > wfcpw->gk_ecut)
        {
            continue;
        }
        const ModuleBase::Vector3<double> g_full = generic_g_direct_from_ig(wfcpw, ig);
        const ModuleBase::Vector3<double> g_rep = g_full * full_point.kgmatrix + rep_shift;
        const IntGKey g_rep_key = make_int_g_key(g_rep);
        const auto it = rep_g_to_ig.find(g_rep_key);
        if (it == rep_g_to_ig.end())
        {
            const ModuleBase::Vector3<double> gplus_rep
                = (int_g_key_to_direct(g_rep_key) + wfcpw->kvec_d[rep_spin_index]) * wfcpw->G;
            const double full_cutoff_delta = gplus_full_norm2 - wfcpw->gk_ecut;
            const double rep_cutoff_delta = gplus_rep.norm2() - wfcpw->gk_ecut;
            if (std::abs(full_cutoff_delta) <= cutoff_tolerance
                && std::abs(rep_cutoff_delta) <= cutoff_tolerance)
            {
                ++skipped_boundary_misses;
                max_abs_full_boundary_delta = std::max(max_abs_full_boundary_delta, std::abs(full_cutoff_delta));
                max_abs_rep_boundary_delta = std::max(max_abs_rep_boundary_delta, std::abs(rep_cutoff_delta));
                continue;
            }

            std::ostringstream message;
            message << "failed to map full-point G vector to representative G vector"
                    << "; ig = " << ig
                    << ", gplus_full.norm2() - gk_ecut = " << full_cutoff_delta
                    << ", gplus_rep.norm2() - gk_ecut = " << rep_cutoff_delta
                    << ", cutoff_tolerance = " << cutoff_tolerance
                    << ", g_full = (" << g_full.x << ", " << g_full.y << ", " << g_full.z << ")"
                    << ", g_rep = (" << g_rep.x << ", " << g_rep.y << ", " << g_rep.z << ")"
                    << ", full_index = " << full_point.full_index
                    << ", rep_index = " << full_point.rep_index
                    << ", rep_local_index = " << full_point.rep_local_index
                    << ", rep_pool = " << full_point.rep_pool
                    << ", symop = " << full_point.symop
                    << ", identity = " << (full_point.identity ? "true" : "false")
                    << ", conjugate_only = " << (full_point.conjugate_only ? "true" : "false")
                    << ", time_reversal = " << (full_point.time_reversal ? "true" : "false")
                    << ", need_gpu_fft_index = " << (need_gpu_fft_index ? "true" : "false");
            ModuleBase::WARNING_QUIT("build_exx_symmetry_remap",
                                     message.str());
        }

        const int ig_rep = it->second;
        remap.rep_igl.push_back(ig_rep);
        remap.fft_isz.push_back(wfcpw->ig2isz[ig]);
        const ModuleBase::Vector3<double> gk_rep
            = k_g_direct_from_igl(wfcpw, rep_spin_index, ig_rep) + wfcpw->kvec_d[rep_spin_index];
        const double phase_arg = ModuleBase::TWO_PI * (gk_rep * full_point.gtrans);
        remap.phase.push_back(std::complex<double>(std::cos(phase_arg), std::sin(phase_arg)));

        if (need_gpu_fft_index)
        {
            const int isz = wfcpw->ig2isz[ig];
            const int iz = isz % wfcpw->nz;
            const int is = isz / wfcpw->nz;
            const int ixy = wfcpw->is2fftixy[is];
            const int iy = ixy % wfcpw->ny;
            const int ix = ixy / wfcpw->ny;
            remap.fft_ixyz.push_back(iz + iy * wfcpw->nz + ix * wfcpw->ny * wfcpw->nz);
        }
    }

    if (skipped_boundary_misses > 0)
    {
        std::ostringstream message;
        message << "skipped " << skipped_boundary_misses
                << " EXX symmetry-remap G-vector misses on the cutoff boundary"
                << "; max_abs_full_delta = " << max_abs_full_boundary_delta
                << ", max_abs_rep_delta = " << max_abs_rep_boundary_delta
                << ", cutoff_tolerance = " << cutoff_tolerance
                << ", full_index = " << full_point.full_index
                << ", rep_index = " << full_point.rep_index
                << ", rep_local_index = " << full_point.rep_local_index
                << ", symop = " << full_point.symop
                << ", need_gpu_fft_index = " << (need_gpu_fft_index ? "true" : "false");
        ModuleBase::WARNING("build_exx_symmetry_remap", message.str());
    }

    if (remap.rep_igl.empty())
    {
        ModuleBase::WARNING_QUIT("build_exx_symmetry_remap", "empty full-point G-vector map");
    }
    if (need_gpu_fft_index && remap.fft_ixyz.size() != remap.rep_igl.size())
    {
        ModuleBase::WARNING_QUIT("build_exx_symmetry_remap", "incomplete full-point GPU FFT map");
    }
    return remap;
}

template <typename T>
void rotate_exx_realspace_symmetry_cpu(const ModulePW::PW_Basis_K* wfcpw,
                                       const K_Vectors::ExxFullPoint& full_point,
                                       int rep_spin_index,
                                       const T* representative_real,
                                       T* full_real)
{
    validate_exx_realspace_symmetry_grid(wfcpw, full_point);

    const int local_size = wfcpw->nrxx;
    const int global_size = wfcpw->nxyz;
    std::vector<T> representative_global(global_size, T(0));
    const ModuleBase::Vector3<double> rep_kvec_d = wfcpw->kvec_d[rep_spin_index];

    for (int ir = 0; ir < local_size; ++ir)
    {
        const int local_iz = ir % wfcpw->nplane;
        const int ixy = ir / wfcpw->nplane;
        const int global_iz = wfcpw->startz_current + local_iz;
        representative_global[ixy * wfcpw->nz + global_iz] = representative_real[ir];
    }

#ifdef __MPI
    if (wfcpw->poolnproc > 1)
    {
        MPI_Allreduce(MPI_IN_PLACE,
                      representative_global.data(),
                      global_size,
                      mpi_complex_type<T>(),
                      MPI_SUM,
                      wfcpw->pool_world);
    }
#endif

    for (int ir = 0; ir < local_size; ++ir)
    {
        const int local_iz = ir % wfcpw->nplane;
        const int ixy = ir / wfcpw->nplane;
        const int global_iz = wfcpw->startz_current + local_iz;
        const int global_ir = ixy * wfcpw->nz + global_iz;
        const ModuleBase::Vector3<double> full_direct = real_grid_direct_from_global_index(wfcpw, global_ir);
        const ModuleBase::Vector3<double> source_direct
            = full_point.time_reversal ? negative_direct(full_direct) : full_direct;
        const ModuleBase::Vector3<double> rep_direct = source_direct * full_point.gmatrix + full_point.gtrans;
        ModuleBase::Vector3<double> rep_direct_wrapped = rep_direct;
        rep_direct_wrapped.x -= std::floor(rep_direct_wrapped.x);
        rep_direct_wrapped.y -= std::floor(rep_direct_wrapped.y);
        rep_direct_wrapped.z -= std::floor(rep_direct_wrapped.z);
        const T phase = static_cast<T>(exx_realspace_bloch_phase(full_point, rep_kvec_d, source_direct, rep_direct));
        T value = phase * representative_global[global_real_index_from_direct(wfcpw, rep_direct_wrapped)];
        if (full_point.time_reversal)
        {
            value = std::conj(value);
        }
        full_real[ir] = value;
    }
}

template void rotate_exx_realspace_symmetry_cpu<std::complex<float>>(const ModulePW::PW_Basis_K* wfcpw,
                                                                     const K_Vectors::ExxFullPoint& full_point,
                                                                     int rep_spin_index,
                                                                     const std::complex<float>* representative_real,
                                                                     std::complex<float>* full_real);
template void rotate_exx_realspace_symmetry_cpu<std::complex<double>>(const ModulePW::PW_Basis_K* wfcpw,
                                                                      const K_Vectors::ExxFullPoint& full_point,
                                                                      int rep_spin_index,
                                                                      const std::complex<double>* representative_real,
                                                                      std::complex<double>* full_real);

template <typename FPTYPE>
struct exx_rotate_realspace_op<std::complex<FPTYPE>, base_device::DEVICE_CPU>
{
    using T = std::complex<FPTYPE>;
    void operator()(const ModulePW::PW_Basis_K* wfcpw,
                    const K_Vectors::ExxFullPoint& full_point,
                    int rep_spin_index,
                    const T* representative_real,
                    T* full_real,
                    int batch_count)
    {
        for (int ib = 0; ib < batch_count; ++ib)
        {
            rotate_exx_realspace_symmetry_cpu(wfcpw,
                                               full_point,
                                               rep_spin_index,
                                               representative_real + static_cast<std::size_t>(ib) * wfcpw->nrxx,
                                               full_real + static_cast<std::size_t>(ib) * wfcpw->nrxx);
        }
    }
};

template struct exx_rotate_realspace_op<std::complex<float>, base_device::DEVICE_CPU>;
template struct exx_rotate_realspace_op<std::complex<double>, base_device::DEVICE_CPU>;

template <typename T>
void rotate_exx_realspace_symmetry_adjoint_cpu(const ModulePW::PW_Basis_K* wfcpw,
                                               const K_Vectors::ExxFullPoint& full_point,
                                               int rep_spin_index,
                                               const T* full_real,
                                               T* representative_real)
{
    if (wfcpw->get_device() != "cpu")
    {
        ModuleBase::WARNING_QUIT("rotate_exx_realspace_symmetry_adjoint_cpu",
                                 "real-space EXX symmetry rotation is implemented only for CPU");
    }
    validate_exx_realspace_symmetry_grid(wfcpw, full_point);

    const int local_size = wfcpw->nrxx;
    const int global_size = wfcpw->nxyz;
    std::vector<T> full_global(global_size, T(0));
    std::vector<T> representative_global(global_size, T(0));
    const ModuleBase::Vector3<double> rep_kvec_d = wfcpw->kvec_d[rep_spin_index];

    for (int ir = 0; ir < local_size; ++ir)
    {
        const int local_iz = ir % wfcpw->nplane;
        const int ixy = ir / wfcpw->nplane;
        const int global_iz = wfcpw->startz_current + local_iz;
        full_global[ixy * wfcpw->nz + global_iz] = full_real[ir];
    }

#ifdef __MPI
    if (wfcpw->poolnproc > 1)
    {
        MPI_Allreduce(MPI_IN_PLACE,
                      full_global.data(),
                      global_size,
                      mpi_complex_type<T>(),
                      MPI_SUM,
                      wfcpw->pool_world);
    }
#endif

    for (int global_ir = 0; global_ir < global_size; ++global_ir)
    {
        const ModuleBase::Vector3<double> full_direct = real_grid_direct_from_global_index(wfcpw, global_ir);
        const ModuleBase::Vector3<double> source_direct
            = full_point.time_reversal ? negative_direct(full_direct) : full_direct;
        const ModuleBase::Vector3<double> rep_direct = source_direct * full_point.gmatrix + full_point.gtrans;
        ModuleBase::Vector3<double> rep_direct_wrapped = rep_direct;
        rep_direct_wrapped.x -= std::floor(rep_direct_wrapped.x);
        rep_direct_wrapped.y -= std::floor(rep_direct_wrapped.y);
        rep_direct_wrapped.z -= std::floor(rep_direct_wrapped.z);
        const T phase = static_cast<T>(exx_realspace_bloch_phase(full_point, rep_kvec_d, source_direct, rep_direct));
        T value = full_global[global_ir];
        if (full_point.time_reversal)
        {
            value = phase * std::conj(value);
        }
        else
        {
            value = std::conj(phase) * value;
        }
        representative_global[global_real_index_from_direct(wfcpw, rep_direct_wrapped)] += value;
    }

    for (int ir = 0; ir < local_size; ++ir)
    {
        const int local_iz = ir % wfcpw->nplane;
        const int ixy = ir / wfcpw->nplane;
        const int global_iz = wfcpw->startz_current + local_iz;
        representative_real[ir] = representative_global[ixy * wfcpw->nz + global_iz];
    }
}

template void rotate_exx_realspace_symmetry_adjoint_cpu<std::complex<float>>(
    const ModulePW::PW_Basis_K* wfcpw,
    const K_Vectors::ExxFullPoint& full_point,
    int rep_spin_index,
    const std::complex<float>* full_real,
    std::complex<float>* representative_real);
template void rotate_exx_realspace_symmetry_adjoint_cpu<std::complex<double>>(
    const ModulePW::PW_Basis_K* wfcpw,
    const K_Vectors::ExxFullPoint& full_point,
    int rep_spin_index,
    const std::complex<double>* full_real,
    std::complex<double>* representative_real);

template <typename FPTYPE>
struct exx_conjugate_real_op<std::complex<FPTYPE>, base_device::DEVICE_CPU>
{
    using T = std::complex<FPTYPE>;
    void operator()(const T* in, T* out, std::size_t nrxx)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::size_t ir = 0; ir < nrxx; ++ir)
        {
            out[ir] = std::conj(in[ir]);
        }
    }
};

template struct exx_conjugate_real_op<std::complex<float>, base_device::DEVICE_CPU>;
template struct exx_conjugate_real_op<std::complex<double>, base_device::DEVICE_CPU>;

template <typename FPTYPE>
struct exx_gather_recip_op<std::complex<FPTYPE>, base_device::DEVICE_CPU>
{
    using T = std::complex<FPTYPE>;
    void operator()(const T* in, T* out, const int* map, int nout)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int ig = 0; ig < nout; ++ig)
        {
            const int src = map[ig];
            out[ig] = src >= 0 ? in[src] : T(0);
        }
    }
};

template <typename FPTYPE>
struct exx_scatter_add_recip_op<std::complex<FPTYPE>, base_device::DEVICE_CPU>
{
    using T = std::complex<FPTYPE>;
    void operator()(const T* in, T* out, const int* map, int nin, T factor)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int ig = 0; ig < nin; ++ig)
        {
            const int dst = map[ig];
            if (dst >= 0)
            {
                out[dst] += factor * in[ig];
            }
        }
    }
};

template struct exx_gather_recip_op<std::complex<float>, base_device::DEVICE_CPU>;
template struct exx_gather_recip_op<std::complex<double>, base_device::DEVICE_CPU>;
template struct exx_scatter_add_recip_op<std::complex<float>, base_device::DEVICE_CPU>;
template struct exx_scatter_add_recip_op<std::complex<double>, base_device::DEVICE_CPU>;
} // namespace hamilt
