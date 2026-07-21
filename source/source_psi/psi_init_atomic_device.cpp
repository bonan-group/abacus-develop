#include "psi_init_atomic_device.h"

#include "source_psi/psi_init_atomic.h"

#if defined __CUDA
#include "source_base/math_ylmreal.h"
#include "source_base/module_device/memory_op.h"
#include "source_psi/kernels/psi_init_op.h"

#include <vector>
#endif

#include <complex>

namespace psi
{

template <typename T>
class AtomicGpuInitializer<T>::Impl
{
  public:
#if defined __CUDA
    using Real = typename GetTypeReal<T>::type;
    using resize_real_op = base_device::memory::resize_memory_op<Real, base_device::DEVICE_GPU>;
    using resize_int_op = base_device::memory::resize_memory_op<int, base_device::DEVICE_GPU>;
    using resize_complex_op = base_device::memory::resize_memory_op<T, base_device::DEVICE_GPU>;
    using sync_real_h2d_op
        = base_device::memory::synchronize_memory_op<Real, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using sync_int_h2d_op
        = base_device::memory::synchronize_memory_op<int, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using delete_real_op = base_device::memory::delete_memory_op<Real, base_device::DEVICE_GPU>;
    using delete_int_op = base_device::memory::delete_memory_op<int, base_device::DEVICE_GPU>;
    using delete_complex_op = base_device::memory::delete_memory_op<T, base_device::DEVICE_GPU>;

    Impl(const ::psi_init_atomic<T>* atomic_initializer,
         const ModulePW::PW_Basis_K& pw_wfc,
         const UnitCell& ucell,
         const Structure_Factor& structure_factor,
         const bool enabled,
         const int random_seed)
        : pw_wfc_(pw_wfc),
          ucell_(ucell),
          structure_factor_(structure_factor),
          random_seed_(random_seed),
          mixing_coef_(atomic_initializer == nullptr ? Real(0)
                                                     : static_cast<Real>(atomic_initializer->mixing_coef_)),
          perturb_atomic_(atomic_initializer != nullptr && atomic_initializer->method() == "atomic+random")
    {
        if (!enabled || atomic_initializer == nullptr)
        {
            return;
        }

        const ModuleBase::realArray& overlap_table = atomic_initializer->ovlp_pswfcjlq_;
        dq_ = static_cast<Real>(atomic_initializer->table_interval_);
        nchi_max_ = overlap_table.getBound2();
        nqx_ = overlap_table.getBound3();
        total_lm_ = (ucell_.lmax_ppwf + 1) * (ucell_.lmax_ppwf + 1);

        std::vector<Real> table(overlap_table.getSize());
        for (int i = 0; i < overlap_table.getSize(); ++i)
        {
            table[i] = static_cast<Real>(overlap_table.ptr[i]);
        }

        std::vector<int> iw2iat;
        std::vector<int> iw2it;
        std::vector<int> iw2ic;
        std::vector<int> iw2lm;
        std::vector<int> iw2l;
        iw2iat.reserve(ucell_.natomwfc);
        iw2it.reserve(ucell_.natomwfc);
        iw2ic.reserve(ucell_.natomwfc);
        iw2lm.reserve(ucell_.natomwfc);
        iw2l.reserve(ucell_.natomwfc);
        int iat = 0;
        for (int it = 0; it < ucell_.ntype; ++it)
        {
            for (int ia = 0; ia < ucell_.atoms[it].na; ++ia, ++iat)
            {
                for (int ic = 0; ic < ucell_.atoms[it].ncpp.nchi; ++ic)
                {
                    if (ucell_.atoms[it].ncpp.oc[ic] < 0.0)
                    {
                        continue;
                    }
                    const int l = ucell_.atoms[it].ncpp.lchi[ic];
                    for (int m = 0; m < 2 * l + 1; ++m)
                    {
                        iw2iat.push_back(iat);
                        iw2it.push_back(it);
                        iw2ic.push_back(ic);
                        iw2lm.push_back(l * l + m);
                        iw2l.push_back(l);
                    }
                }
            }
        }
        natomwfc_ = static_cast<int>(iw2iat.size());
        if (natomwfc_ != ucell_.natomwfc)
        {
            return;
        }

        resize_real_op()(table_, table.size());
        resize_real_op()(gk_, pw_wfc_.npwk_max * 3);
        resize_real_op()(ylm_, total_lm_ * pw_wfc_.npwk_max);
        resize_complex_op()(sk_, ucell_.nat * pw_wfc_.npwk_max);
        resize_int_op()(iw2iat_, natomwfc_);
        resize_int_op()(iw2it_, natomwfc_);
        resize_int_op()(iw2ic_, natomwfc_);
        resize_int_op()(iw2lm_, natomwfc_);
        resize_int_op()(iw2l_, natomwfc_);
        sync_real_h2d_op()(table_, table.data(), table.size());
        sync_int_h2d_op()(iw2iat_, iw2iat.data(), natomwfc_);
        sync_int_h2d_op()(iw2it_, iw2it.data(), natomwfc_);
        sync_int_h2d_op()(iw2ic_, iw2ic.data(), natomwfc_);
        sync_int_h2d_op()(iw2lm_, iw2lm.data(), natomwfc_);
        sync_int_h2d_op()(iw2l_, iw2l.data(), natomwfc_);
        available_ = true;
    }

    ~Impl()
    {
        delete_real_op()(table_);
        delete_real_op()(gk_);
        delete_real_op()(ylm_);
        delete_complex_op()(sk_);
        delete_int_op()(iw2iat_);
        delete_int_op()(iw2it_);
        delete_int_op()(iw2ic_);
        delete_int_op()(iw2lm_);
        delete_int_op()(iw2l_);
    }

    void initialize_k(T* psi, const int nbands_start, const int ik, const int ik_tot)
    {
        base_device::DEVICE_GPU* ctx = {};
        const int npwk = pw_wfc_.npwk[ik];
        build_gk_op<Real, base_device::DEVICE_GPU>()(ctx,
                                                     gk_,
                                                     pw_wfc_.template get_gcar_data<Real>(),
                                                     pw_wfc_.template get_kvec_c_data<Real>(),
                                                     ik,
                                                     npwk,
                                                     pw_wfc_.npwk_max);
        ModuleBase::YlmReal::Ylm_Real<Real, base_device::DEVICE_GPU>(ctx, total_lm_, npwk, gk_, ylm_);
        structure_factor_.template get_sk<Real, base_device::DEVICE_GPU>(ctx, ik, &pw_wfc_, sk_);
        const AtomicInitTableView<Real> table_view = {total_lm_,
                                                     nchi_max_,
                                                     nqx_,
                                                     dq_,
                                                     static_cast<Real>(ucell_.tpiba),
                                                     table_,
                                                     iw2iat_,
                                                     iw2it_,
                                                     iw2ic_,
                                                     iw2lm_,
                                                     iw2l_};
        init_atomic_op<T, base_device::DEVICE_GPU>()(ctx,
                                                     psi,
                                                     natomwfc_,
                                                     npwk,
                                                     pw_wfc_.npwk_max,
                                                     gk_,
                                                     ylm_,
                                                     sk_,
                                                     table_view);

        const int nbands_complem = nbands_start - natomwfc_;
        if (nbands_complem > 0)
        {
            init_random_op<T, base_device::DEVICE_GPU>()(ctx,
                                                         psi + natomwfc_ * pw_wfc_.npwk_max,
                                                         nbands_complem,
                                                         npwk,
                                                         pw_wfc_.npwk_max,
                                                         1,
                                                         ik,
                                                         ik_tot,
                                                         random_seed_,
                                                         pw_wfc_.template get_gk2_data<Real>(),
                                                         pw_wfc_.get_igl2isz_data(),
                                                         pw_wfc_.d_is2fftixy,
                                                         pw_wfc_.fftnxy,
                                                         pw_wfc_.nz);
        }
        if (perturb_atomic_)
        {
            perturb_atomic_op<T, base_device::DEVICE_GPU>()(ctx,
                                                            psi,
                                                            nbands_start,
                                                            npwk,
                                                            pw_wfc_.npwk_max,
                                                            1,
                                                            ik,
                                                            ik_tot,
                                                            random_seed_,
                                                            mixing_coef_,
                                                            pw_wfc_.template get_gk2_data<Real>(),
                                                            pw_wfc_.get_igl2isz_data(),
                                                            pw_wfc_.d_is2fftixy,
                                                            pw_wfc_.fftnxy,
                                                            pw_wfc_.nz);
        }
    }

    bool available() const
    {
        return available_;
    }

  private:
    const ModulePW::PW_Basis_K& pw_wfc_;
    const UnitCell& ucell_;
    const Structure_Factor& structure_factor_;
    Real dq_ = 0;
    int random_seed_ = 0;
    Real mixing_coef_ = 0;
    bool perturb_atomic_ = false;
    bool available_ = false;
    int natomwfc_ = 0;
    int total_lm_ = 0;
    int nchi_max_ = 0;
    int nqx_ = 0;
    Real* table_ = nullptr;
    Real* gk_ = nullptr;
    Real* ylm_ = nullptr;
    T* sk_ = nullptr;
    int* iw2iat_ = nullptr;
    int* iw2it_ = nullptr;
    int* iw2ic_ = nullptr;
    int* iw2lm_ = nullptr;
    int* iw2l_ = nullptr;
#else
    Impl(const ::psi_init_atomic<T>*,
         const ModulePW::PW_Basis_K&,
         const UnitCell&,
         const Structure_Factor&,
         const bool,
         const int)
    {
    }

    bool available() const
    {
        return false;
    }

    void initialize_k(T*, const int, const int, const int)
    {
    }
#endif
};

template <typename T>
AtomicGpuInitializer<T>::AtomicGpuInitializer(const ::psi_init_atomic<T>* atomic_initializer,
                                               const ModulePW::PW_Basis_K& pw_wfc,
                                               const UnitCell& ucell,
                                               const Structure_Factor& structure_factor,
                                               const bool enabled,
                                               const int random_seed)
    : impl_(new Impl(atomic_initializer, pw_wfc, ucell, structure_factor, enabled, random_seed))
{
}

template <typename T>
AtomicGpuInitializer<T>::~AtomicGpuInitializer()
{
}

template <typename T>
bool AtomicGpuInitializer<T>::available() const
{
    return impl_->available();
}

template <typename T>
void AtomicGpuInitializer<T>::initialize_k(T* psi,
                                           const int nbands_start,
                                           const int ik,
                                           const int ik_tot)
{
    impl_->initialize_k(psi, nbands_start, ik, ik_tot);
}

template class AtomicGpuInitializer<std::complex<float>>;
template class AtomicGpuInitializer<std::complex<double>>;

} // namespace psi
