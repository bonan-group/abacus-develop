#include "psi_prepare.h"

#include "source_base/macros.h"
#include "source_base/math_ylmreal.h"
#include "source_base/memory_recorder.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/parallel_device.h"
#include "source_base/parallel_global.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_hsolver/diago_iter_assist.h"
#include "source_io/module_parameter/parameter.h"
#include "source_psi/psi_init_policy.h"
#include "source_psi/kernels/psi_init_op.h"
#include "source_psi/psi_init_atomic.h"
#include "source_psi/psi_init_atomic_random.h"
#include "source_psi/psi_init_file.h"
#include "source_psi/psi_init_nao.h"
#include "source_psi/psi_init_nao_random.h"
#include "source_psi/psi_init_random.h"

#include <cstdlib>

namespace psi
{

namespace
{

bool psi_init_cpu_debug_enabled()
{
    const char* value = std::getenv("ABACUS_PSI_INIT_CPU_DEBUG");
    if (value == nullptr)
    {
        return false;
    }
    const std::string flag(value);
    return !(flag.empty() || flag == "0" || flag == "false" || flag == "False" || flag == "FALSE"
             || flag == "off" || flag == "Off" || flag == "OFF");
}

template <typename T, typename Device>
struct GpuResidentInit
{
    static bool enabled(const psi_initializer<T>* psi_initer, const std::string& ks_solver, const int npol)
    {
        return false;
    }
};

#if defined __CUDA
template <typename T>
struct GpuResidentInit<T, base_device::DEVICE_GPU>
{
    static bool enabled(const psi_initializer<T>* psi_initer, const std::string& ks_solver, const int npol)
    {
        return gpu_resident_init_policy(PARAM.inp.device == "gpu",
                                        psi_init_cpu_debug_enabled(),
                                        ks_solver,
                                        psi_initer != nullptr,
                                        psi_initer == nullptr ? "" : psi_initer->method(),
                                        npol);
    }
};
#endif

template <typename T, typename Device>
class AtomicGpuWorkspace
{
  public:
    AtomicGpuWorkspace(const psi_initializer<T>*,
                       const ModulePW::PW_Basis_K&,
                       const UnitCell&,
                       const Structure_Factor&,
                       const bool,
                       const int)
    {
    }

    void init(psi::Psi<T, Device>*, const int, const int, const int)
    {
    }

    bool available() const
    {
        return false;
    }
};

#if defined __CUDA
template <typename T>
class AtomicGpuWorkspace<T, base_device::DEVICE_GPU>
{
  private:
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

  public:
    AtomicGpuWorkspace(const psi_initializer<T>* psi_initer,
                       const ModulePW::PW_Basis_K& pw_wfc,
                       const UnitCell& ucell,
                       const Structure_Factor& sf,
                       const bool enabled,
                       const int random_seed)
        : psi_initer_(psi_initer),
          pw_wfc_(pw_wfc),
          ucell_(ucell),
          sf_(sf),
          random_seed_(random_seed),
          mixing_coef_(psi_initer == nullptr ? Real(0) : static_cast<Real>(psi_initer->mixing_coef()))
    {
        if (!enabled || psi_initer_ == nullptr)
        {
            return;
        }
        if (psi_initer_->method() == "random")
        {
            available_ = true;
            return;
        }

        const psi_init_atomic<T>* atomic_initer = dynamic_cast<const psi_init_atomic<T>*>(psi_initer_);
        if (atomic_initer == nullptr)
        {
            return;
        }
        const ModuleBase::realArray& overlap_table = atomic_initer->overlap_table();
        dq_ = static_cast<Real>(atomic_initer->table_interval());
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

    AtomicGpuWorkspace(const AtomicGpuWorkspace&) = delete;
    AtomicGpuWorkspace& operator=(const AtomicGpuWorkspace&) = delete;

    ~AtomicGpuWorkspace()
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

    void init(psi::Psi<T, base_device::DEVICE_GPU>* psi_device,
              const int nbands_start,
              const int ik,
              const int ik_tot)
    {
        base_device::DEVICE_GPU* ctx = {};
        const int npwk = pw_wfc_.npwk[ik];
        psi::build_gk_op<Real, base_device::DEVICE_GPU>()(ctx,
                                                          gk_,
                                                          pw_wfc_.template get_gcar_data<Real>(),
                                                          pw_wfc_.template get_kvec_c_data<Real>(),
                                                          ik,
                                                          npwk,
                                                          pw_wfc_.npwk_max);
        ModuleBase::YlmReal::Ylm_Real<Real, base_device::DEVICE_GPU>(ctx, total_lm_, npwk, gk_, ylm_);
        sf_.template get_sk<Real, base_device::DEVICE_GPU>(ctx, ik, &pw_wfc_, sk_);
        psi::init_atomic_op<T, base_device::DEVICE_GPU>()(ctx,
                                                          psi_device->get_pointer(),
                                                          natomwfc_,
                                                          npwk,
                                                          pw_wfc_.npwk_max,
                                                          total_lm_,
                                                          nchi_max_,
                                                          nqx_,
                                                          dq_,
                                                          static_cast<Real>(ucell_.tpiba),
                                                          gk_,
                                                          ylm_,
                                                          sk_,
                                                          table_,
                                                          iw2iat_,
                                                          iw2it_,
                                                          iw2ic_,
                                                          iw2lm_,
                                                          iw2l_);

        const int nbands_complem = nbands_start - natomwfc_;
        if (nbands_complem > 0)
        {
            psi::init_random_op<T, base_device::DEVICE_GPU>()(
                ctx,
                psi_device->get_pointer() + natomwfc_ * pw_wfc_.npwk_max,
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
        if (psi_initer_->method() == "atomic+random")
        {
            psi::perturb_atomic_op<T, base_device::DEVICE_GPU>()(ctx,
                                                                 psi_device->get_pointer(),
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
    const psi_initializer<T>* psi_initer_;
    const ModulePW::PW_Basis_K& pw_wfc_;
    const UnitCell& ucell_;
    const Structure_Factor& sf_;
    Real dq_ = 0;
    int random_seed_ = 0;
    Real mixing_coef_ = 0;
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
};
#endif

template <typename T, typename Device>
struct RandomDeviceInitializer
{
    static void init(const ModulePW::PW_Basis_K& pw_wfc,
                     psi::Psi<T, Device>* psi_device,
                     const int nbands_start,
                     const int nbasis,
                     const int ik,
                     const int ik_tot,
                     const int random_seed)
    {
    }
};

#if defined __CUDA
template <typename T>
struct RandomDeviceInitializer<T, base_device::DEVICE_GPU>
{
    static void init(const ModulePW::PW_Basis_K& pw_wfc,
                     psi::Psi<T, base_device::DEVICE_GPU>* psi_device,
                     const int nbands_start,
                     const int nbasis,
                     const int ik,
                     const int ik_tot,
                     const int random_seed)
    {
        using Real = typename GetTypeReal<T>::type;
        const int npol = PARAM.globalv.npol;
        psi::init_random_op<T, base_device::DEVICE_GPU>()(psi_device->get_device(),
                                                          psi_device->get_pointer(),
                                                          nbands_start,
                                                          pw_wfc.npwk[ik],
                                                          pw_wfc.npwk_max,
                                                          npol,
                                                          ik,
                                                          ik_tot,
                                                          random_seed,
                                                          pw_wfc.template get_gk2_data<Real>(),
                                                          pw_wfc.get_igl2isz_data(),
                                                          pw_wfc.d_is2fftixy,
                                                          pw_wfc.fftnxy,
                                                          pw_wfc.nz);
    }
};
#endif

} // namespace

template <typename T, typename Device>
PSIPrepare<T, Device>::PSIPrepare(const std::string& init_wfc_in,
                            const std::string& ks_solver_in,
                            const std::string& basis_type_in,
                            const int& rank_in,
                            const UnitCell& ucell_in,
                            const Structure_Factor& sf_in,
                            const K_Vectors& kv_in,
                            const pseudopot_cell_vnl& nlpp_in,
                            const ModulePW::PW_Basis_K& pw_wfc_in)
    : ucell(ucell_in), sf(sf_in), nlpp(nlpp_in), kv(kv_in), pw_wfc(pw_wfc_in), rank(rank_in)
{
    this->init_wfc = init_wfc_in;
    this->ks_solver = ks_solver_in;
    this->basis_type = basis_type_in;
}

template <typename T, typename Device>
void PSIPrepare<T, Device>::prepare_init(const int& random_seed)
{

    // under restriction of C++11, std::unique_ptr can not be allocate via std::make_unique
    // use new instead, but will cause asymmetric allocation and deallocation, in literal aspect
    ModuleBase::timer::start("PSIPrepare", "prepare_init");
    this->psi_initer.reset();
    if (this->init_wfc == "random")
    {
        this->psi_initer = std::unique_ptr<psi_initializer<T>>(new psi_init_random<T>());
        GlobalV::ofs_running << "\n Using RANDOM starting wave functions for all " << PARAM.inp.nbands << " bands\n";
    }
    else if (this->init_wfc == "file")
    {
        this->psi_initer = std::unique_ptr<psi_initializer<T>>(new psi_init_file<T>());
        GlobalV::ofs_running << "\n Using FILE starting wave functions\n";
    }
    else if ((this->init_wfc.substr(0, 6) == "atomic") && (this->ucell.natomwfc == 0))
    {
        std::cout << " WARNING: init_wfc = " + this->init_wfc +
            " requires atomic pseudo wavefunctions(PP_PSWFC),\n but none available."
            " Automatically switch to random initialization." << std::endl;
        GlobalV::ofs_running << "\n Using RANDOM starting wave functions for all " << PARAM.inp.nbands << " bands\n";
        GlobalV::ofs_running << "\n WARNING:\n init_wfc = " + this->init_wfc + " requires atomic pseudo wavefunctions(PP_PSWFC), but none available. \n"
            " Automatically switch to random initialization.\n"
            " Note: Random starting wavefunctions may slow down convergence.\n"
            "      For faster convergence, consider using:\n"
            "      1) A pseudopotential file that includes atomic wavefunctions (with PP_PSWFC), or\n"
            "      2) Numerical atomic orbitals with 'init_wfc = nao' or 'nao+random' if available.\n"
            << std::endl;
        this->psi_initer = std::unique_ptr<psi_initializer<T>>(new psi_init_random<T>());
    }
    else if (this->init_wfc == "atomic"
             || (this->init_wfc == "atomic+random" && this->ucell.natomwfc < PARAM.inp.nbands))
    {
        if (this->ucell.natomwfc < PARAM.inp.nbands)
        {
            int nrandom = PARAM.inp.nbands - this->ucell.natomwfc;
            GlobalV::ofs_running << "\n Using ATOMIC starting wave functions with " << this->ucell.natomwfc << " atomic orbitals"
            << " + " << nrandom << " random orbitals"
            << " (total " << PARAM.inp.nbands << " bands)\n";
        }
        else
        {
            GlobalV::ofs_running << "\n Using ATOMIC starting wave functions for all " << this->ucell.natomwfc << " atomic orbitals"
                << " (covers " << PARAM.inp.nbands << " bands)\n";
        }
        this->psi_initer = std::unique_ptr<psi_initializer<T>>(new psi_init_atomic<T>());
    }
    else if (this->init_wfc == "atomic+random")
    {
        this->psi_initer = std::unique_ptr<psi_initializer<T>>(new psi_init_atomic_random<T>());
        GlobalV::ofs_running << "\n Using ATOMIC+RANDOM starting wave functions with "
                             << this->ucell.natomwfc << " atomic orbitals\n";
    }
    else if (this->init_wfc == "nao")
    {
        this->psi_initer = std::unique_ptr<psi_initializer<T>>(new psi_init_nao<T>());
        GlobalV::ofs_running << "\n Using NAO starting wave functions\n";
    }
    else if (this->init_wfc == "nao+random")
    {
        this->psi_initer = std::unique_ptr<psi_initializer<T>>(new psi_init_nao_random<T>());
        GlobalV::ofs_running << "\n Using NAO+RANDOM starting wave functions\n";
    }
    else
    {
        ModuleBase::WARNING_QUIT("PSIInit::prepare_init", "for new psi initializer, init_wfc type not supported");
    }

    this->psi_initer->initialize(&sf, &pw_wfc, &ucell, &kv, random_seed, &nlpp, rank);
    this->psi_initer->tabulate();

    ModuleBase::timer::end("PSIPrepare", "prepare_init");
}

template <typename T, typename Device>
void PSIPrepare<T, Device>::initialize_psi(Psi<std::complex<double>>* psi,
                                        psi::Psi<T, Device>* kspw_psi,
                                        hamilt::Hamilt<T, Device>* p_hamilt,
                                        std::ofstream& ofs_running)
{
    if (kspw_psi->get_nbands() == 0 || (!PARAM.globalv.ks_run))
    {
        return;
    }
    if (this->basis_type == "lcao_in_pw")
    {
        return;
    }
    ModuleBase::timer::start("PSIPrepare", "initialize_psi");

    const int nbands_start = this->psi_initer->nbands_start();
    const int random_seed = PARAM.inp.pw_seed;
    const int nbands_l = psi->get_nbands();
    const int nbasis = psi->get_nbasis();
    const int npol = nbasis / this->pw_wfc.npwk_max;
    const bool gpu_init_policy = GpuResidentInit<T, Device>::enabled(this->psi_initer.get(), this->ks_solver, npol);
    AtomicGpuWorkspace<T, Device> atomic_gpu_workspace(
        this->psi_initer.get(),
        this->pw_wfc,
        this->ucell,
        this->sf,
        gpu_init_policy,
        random_seed);
    const bool gpu_resident_init = gpu_init_policy && atomic_gpu_workspace.available();
    const bool not_equal = (nbands_start != nbands_l);

    Psi<T>* psi_cpu = reinterpret_cast<psi::Psi<T>*>(psi);
    Psi<T, Device>* psi_device = kspw_psi;

    bool fill = PARAM.inp.ks_solver != "bpcg" || GlobalV::MY_BNDGROUP == 0;
    if (fill)
    {
        if (not_equal)
        {
            if (gpu_resident_init)
            {
                psi_device = new psi::Psi<T, Device>(1, nbands_start, nbasis, nbasis, true);
            }
            else
            {
                psi_cpu = new Psi<T>(1, nbands_start, nbasis, nbasis, true);
                psi_device = PARAM.inp.device == "gpu" ? new psi::Psi<T, Device>(psi_cpu[0])
                                                       : reinterpret_cast<psi::Psi<T, Device>*>(psi_cpu);
            }
        }
        else if (PARAM.inp.precision == "single")
        {
            if (PARAM.inp.device == "cpu")
            {
                psi_cpu = reinterpret_cast<psi::Psi<T>*>(kspw_psi);
                psi_device = kspw_psi;
            }
            else
            {
                if (!gpu_resident_init)
                {
                    psi_cpu = new Psi<T>(1, nbands_start, nbasis, nbasis, true);
                }
                psi_device = kspw_psi;
            }
        }
    }

    // loop over kpoints, make it possible to only allocate memory for psig at the only one kpt
    // like (1, nbands, npwx), in which npwx is the maximal npw of all kpoints
    for (int ik = 0; ik < this->pw_wfc.nks; ik++)
    {
        if(PARAM.inp.use_k_continuity && ik > 0) continue;
        const int ik_tot = this->kv.ik2iktot.empty() ? ik : this->kv.ik2iktot[ik];
        //! Fix the wavefunction to initialize at given kpoint
        psi->fix_k(ik);
        kspw_psi->fix_k(ik);

        //! Update Hamiltonian from other kpoint to the given one
        p_hamilt->updateHk(ik);
        if (fill)
        {
            //! initialize psi_cpu
            if (gpu_resident_init)
            {
                if (this->psi_initer->method() == "random")
                {
                    RandomDeviceInitializer<T, Device>::init(this->pw_wfc,
                                                             psi_device,
                                                             nbands_start,
                                                             nbasis,
                                                             ik,
                                                             ik_tot,
                                                             random_seed);
                }
                else
                {
                    atomic_gpu_workspace.init(psi_device, nbands_start, ik, ik_tot);
                }
            }
            else
            {
                this->psi_initer->init_psig(psi_cpu->get_pointer(), ik);
                if (psi_device->get_pointer() != psi_cpu->get_pointer())
                {
                    syncmem_h2d_op()(psi_device->get_pointer(), psi_cpu->get_pointer(), nbands_start * nbasis);
                }
            }


            if (this->ks_solver == "cg")
            {
                std::vector<typename GetTypeReal<T>::type> etatom(nbands_start, 0.0);
                if (not_equal)
                {
                    // for diagH_subspace_init, psi_device->get_pointer() and kspw_psi->get_pointer() should be
                    // different
                    hsolver::DiagoIterAssist<T, Device>::diag_subspace_init(p_hamilt,
                                                                             psi_device->get_pointer(),
                                                                             nbands_start,
                                                                             nbasis,
                                                                             *(kspw_psi),
                                                                             etatom.data());
                }
                else
                {
                    // for diagH_subspace, psi_device->get_pointer() and kspw_psi->get_pointer() can be the same
                    hsolver::DiagoIterAssist<T, Device>::diag_subspace(p_hamilt,
                                                                        *psi_device,
                                                                        *kspw_psi,
                                                                        etatom.data(),
                                                                        nbands_start);
                }
            }
            else // dav, bpcg
            {
                if (psi_device->get_pointer() != kspw_psi->get_pointer())
                {
                    syncmem_complex_op()(kspw_psi->get_pointer(), psi_device->get_pointer(), nbands_l * nbasis);
                }
            }
        }
#ifdef __MPI
        if (PARAM.inp.ks_solver == "bpcg" && PARAM.inp.bndpar > 1)
        {
            std::vector<int> sendcounts(PARAM.inp.bndpar);
            std::vector<int> displs(PARAM.inp.bndpar);
            MPI_Allgather(&nbands_l, 1, MPI_INT, sendcounts.data(), 1, MPI_INT, BP_WORLD);
            displs[0] = 0;
            sendcounts[0] *= nbasis;
            for (int i = 1; i < PARAM.inp.bndpar; i++)
            {
                sendcounts[i] *= nbasis;
                displs[i] = displs[i - 1] + sendcounts[i - 1];
            }
            if (GlobalV::MY_BNDGROUP == 0)
            {
                for (int ip = 1; ip < PARAM.inp.bndpar; ++ip)
                {
                    Parallel_Common::send_data(psi_cpu->get_pointer() + displs[ip], sendcounts[ip], ip, 0, BP_WORLD);
                }
            }
            else
            {
                MPI_Status status;
                Parallel_Common::recv_dev<T, Device>(kspw_psi->get_pointer(), nbands_l * nbasis, 0, 0, BP_WORLD, &status);
            }
        }
#endif
    } // end k-point loop

    if (fill)
    {
        if (not_equal)
        {
            if (!gpu_resident_init)
            {
                delete psi_cpu;
            }
            if (PARAM.inp.device == "gpu")
            {
                delete psi_device;
            }
        }
        else if (PARAM.inp.precision == "single" && PARAM.inp.device == "gpu" && !gpu_resident_init)
        {
            delete psi_cpu;
        }
    }

    ModuleBase::timer::end("PSIPrepare", "initialize_psi");
}

template <typename T, typename Device>
void PSIPrepare<T, Device>::initialize_lcao_in_pw(Psi<T>* psi_local, std::ofstream& ofs_running)
{
    ofs_running << " START WAVEFUNCTION: LCAO_IN_PW, psi initialization skipped " << std::endl;
    assert(this->psi_initer->method() == "nao");
    for (int ik = 0; ik < this->pw_wfc.nks; ik++)
    {
        psi_local->fix_k(ik);
        this->psi_initer->init_psig(psi_local->get_pointer(), ik);
    }
}

void allocate_psi(Psi<std::complex<double>>*& psi,
                  const int& nks,
                  const std::vector<int>& ngk,
                  const int& nbands,
                  const int& npwx)
{
    assert(npwx > 0);
    assert(nks > 0);
    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "npwx", npwx);

    delete psi;
    int nks2 = nks;
    if (PARAM.inp.calculation == "nscf" && PARAM.inp.mem_saver == 1)
    {
        nks2 = 1;
    }
    psi = new psi::Psi<std::complex<double>>(nks2, nbands, npwx * PARAM.globalv.npol, ngk, true);
    const size_t memory_cost = sizeof(std::complex<double>) * nks2 * nbands * (PARAM.globalv.npol * npwx);
    std::cout << " MEMORY FOR PSI (MB)  : " << static_cast<double>(memory_cost) / 1024.0 / 1024.0 << std::endl;
    ModuleBase::Memory::record("Psi_PW", memory_cost);
}

template class PSIPrepare<std::complex<float>, base_device::DEVICE_CPU>;
template class PSIPrepare<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class PSIPrepare<std::complex<float>, base_device::DEVICE_GPU>;
template class PSIPrepare<std::complex<double>, base_device::DEVICE_GPU>;
#endif
} // namespace psi
