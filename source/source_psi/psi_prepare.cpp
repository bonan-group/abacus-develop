#include "psi_prepare.h"

#include "source_base/macros.h"
#include "source_base/memory_recorder.h"
#include "source_base/parallel_device.h"
#include "source_base/parallel_global.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_hsolver/diago_iter_assist.h"
#include "source_io/module_parameter/input_parameter.h"
#include "source_io/module_parameter/system_parameter.h"
#include "source_psi/psi_init_atomic.h"
#include "source_psi/psi_init_atomic_random.h"
#include "source_psi/psi_init_file.h"
#include "source_psi/psi_init_nao.h"
#include "source_psi/psi_init_nao_random.h"
#include "source_psi/psi_init_random.h"
namespace psi
{

template <typename T, typename Device>
PSIPrepare<T, Device>::PSIPrepare(const std::string& init_wfc_in,
                            const std::string& ks_solver_in,
                            const std::string& basis_type_in,
                            const int& rank_in,
                            const UnitCell& ucell_in,
                            const Structure_Factor& sf_in,
                            const K_Vectors& kv_in,
                            const pseudopot_cell_vnl& nlpp_in,
                            const ModulePW::PW_Basis_K& pw_wfc_in,
                            const Input_para& inp_in,
                            const System_para& sys_in,
                            const int my_bndgroup_in,
                            std::ofstream& running_log_in)
    : ucell(ucell_in),
      sf(sf_in),
      nlpp(nlpp_in),
      inp(inp_in),
      sys(sys_in),
      my_bndgroup(my_bndgroup_in),
      running_log(running_log_in),
      kv(kv_in),
      pw_wfc(pw_wfc_in),
      rank(rank_in)
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
        this->running_log << "\n Using RANDOM starting wave functions for all " << this->inp.nbands << " bands\n";
    }
    else if (this->init_wfc == "file")
    {
        this->psi_initer = std::unique_ptr<psi_initializer<T>>(new psi_init_file<T>());
        this->running_log << "\n Using FILE starting wave functions\n";
    }
    else if ((this->init_wfc.substr(0, 6) == "atomic") && (this->ucell.natomwfc == 0))
    {
        std::cout << " WARNING: init_wfc = " + this->init_wfc +
            " requires atomic pseudo wavefunctions(PP_PSWFC),\n but none available."
            " Automatically switch to random initialization." << std::endl;
        this->running_log << "\n Using RANDOM starting wave functions for all " << this->inp.nbands << " bands\n";
        this->running_log << "\n WARNING:\n init_wfc = " + this->init_wfc + " requires atomic pseudo wavefunctions(PP_PSWFC), but none available. \n"
            " Automatically switch to random initialization.\n"
            " Note: Random starting wavefunctions may slow down convergence.\n"
            "      For faster convergence, consider using:\n"
            "      1) A pseudopotential file that includes atomic wavefunctions (with PP_PSWFC), or\n"
            "      2) Numerical atomic orbitals with 'init_wfc = nao' or 'nao+random' if available.\n"
            << std::endl;
        this->psi_initer = std::unique_ptr<psi_initializer<T>>(new psi_init_random<T>());
    }
    else if (this->init_wfc == "atomic"
             || (this->init_wfc == "atomic+random" && this->ucell.natomwfc < this->inp.nbands))
    {
        if (this->ucell.natomwfc < this->inp.nbands)
        {
            int nrandom = this->inp.nbands - this->ucell.natomwfc;
            this->running_log << "\n Using ATOMIC starting wave functions with " << this->ucell.natomwfc << " atomic orbitals"
            << " + " << nrandom << " random orbitals"
            << " (total " << this->inp.nbands << " bands)\n";
        }
        else
        {
            this->running_log << "\n Using ATOMIC starting wave functions for all " << this->ucell.natomwfc << " atomic orbitals"
                << " (covers " << this->inp.nbands << " bands)\n";
        }
        this->psi_initer = std::unique_ptr<psi_initializer<T>>(new psi_init_atomic<T>());
    }
    else if (this->init_wfc == "atomic+random")
    {
        this->psi_initer = std::unique_ptr<psi_initializer<T>>(new psi_init_atomic_random<T>());
        this->running_log << "\n Using ATOMIC+RANDOM starting wave functions with "
                             << this->ucell.natomwfc << " atomic orbitals\n";
    }
    else if (this->init_wfc == "nao")
    {
        this->psi_initer = std::unique_ptr<psi_initializer<T>>(new psi_init_nao<T>());
        this->running_log << "\n Using NAO starting wave functions\n";
    }
    else if (this->init_wfc == "nao+random")
    {
        this->psi_initer = std::unique_ptr<psi_initializer<T>>(new psi_init_nao_random<T>());
        this->running_log << "\n Using NAO+RANDOM starting wave functions\n";
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
    if (kspw_psi->get_nbands() == 0 || (!this->sys.ks_run))
    {
        return;
    }
    if (this->basis_type == "lcao_in_pw")
    {
        return;
    }
    ModuleBase::timer::start("PSIPrepare", "initialize_psi");

    const int nbands_start = this->psi_initer->nbands_start();
    const int nbands_l = psi->get_nbands();
    const int nbasis = psi->get_nbasis();
    const bool not_equal = (nbands_start != nbands_l);

    Psi<T>* psi_cpu = reinterpret_cast<psi::Psi<T>*>(psi);
    Psi<T, Device>* psi_device = kspw_psi;

    bool fill = this->ks_solver != "bpcg" || this->my_bndgroup == 0;
    if (fill)
    {
        if (not_equal)
        {
            psi_cpu = new Psi<T>(1, nbands_start, nbasis, nbasis, true);
            psi_device = this->inp.device == "gpu" ? new psi::Psi<T, Device>(psi_cpu[0])
                                                   : reinterpret_cast<psi::Psi<T, Device>*>(psi_cpu);
        }
        else if (this->inp.precision == "single")
        {
            if (this->inp.device == "cpu")
            {
                psi_cpu = reinterpret_cast<psi::Psi<T>*>(kspw_psi);
                psi_device = kspw_psi;
            }
            else
            {
                psi_cpu = new Psi<T>(1, nbands_start, nbasis, nbasis, true);
                psi_device = kspw_psi;
            }
        }
    }

    // loop over kpoints, make it possible to only allocate memory for psig at the only one kpt
    // like (1, nbands, npwx), in which npwx is the maximal npw of all kpoints
    for (int ik = 0; ik < this->pw_wfc.nks; ik++)
    {
        if(this->inp.use_k_continuity && ik > 0) continue;
        //! Fix the wavefunction to initialize at given kpoint
        psi->fix_k(ik);
        kspw_psi->fix_k(ik);

        //! Update Hamiltonian from other kpoint to the given one
        p_hamilt->updateHk(ik);
        if (fill)
        {
            //! initialize psi_cpu
            this->psi_initer->init_psig(psi_cpu->get_pointer(), ik);
            if (psi_device->get_pointer() != psi_cpu->get_pointer())
            {
                syncmem_h2d_op()(psi_device->get_pointer(), psi_cpu->get_pointer(), nbands_start * nbasis);
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
        if (this->ks_solver == "bpcg" && this->inp.bndpar > 1)
        {
            std::vector<int> sendcounts(this->inp.bndpar);
            std::vector<int> displs(this->inp.bndpar);
            MPI_Allgather(&nbands_l, 1, MPI_INT, sendcounts.data(), 1, MPI_INT, BP_WORLD);
            displs[0] = 0;
            sendcounts[0] *= nbasis;
            for (int i = 1; i < this->inp.bndpar; i++)
            {
                sendcounts[i] *= nbasis;
                displs[i] = displs[i - 1] + sendcounts[i - 1];
            }
            if (this->my_bndgroup == 0)
            {
                for (int ip = 1; ip < this->inp.bndpar; ++ip)
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
            delete psi_cpu;
            if (this->inp.device == "gpu")
            {
                delete psi_device;
            }
        }
        else if (this->inp.precision == "single" && this->inp.device == "gpu")
        {
            delete psi_cpu;
        }
    }

    ModuleBase::timer::end("PSIPrepare", "initialize_psi");
}

template <typename T, typename Device>
void PSIPrepare<T, Device>::initialize_psi_ik(Psi<std::complex<double>>* psi,
                                             psi::Psi<T, Device>* kspw_psi,
                                             hamilt::Hamilt<T, Device>* p_hamilt,
                                             std::ofstream& ofs_running,
                                             const int ik)
{
    if (kspw_psi->get_nbands() == 0 || (!this->sys.ks_run))
    {
        return;
    }
    if (this->basis_type == "lcao_in_pw")
    {
        return;
    }
    ModuleBase::timer::start("PSIPrepare", "initialize_psi");

    const int nbands_start = this->psi_initer->nbands_start();
    const int nbands_l = psi->get_nbands();
    const int nbasis = psi->get_nbasis();
    const bool not_equal = (nbands_start != nbands_l);

    Psi<T>* psi_cpu = reinterpret_cast<psi::Psi<T>*>(psi);
    Psi<T, Device>* psi_device = kspw_psi;

    bool fill = this->ks_solver != "bpcg" || this->my_bndgroup == 0;
    if (fill)
    {
        if (not_equal)
        {
            psi_cpu = new Psi<T>(1, nbands_start, nbasis, nbasis, true);
            psi_device = this->inp.device == "gpu" ? new psi::Psi<T, Device>(psi_cpu[0])
                                                   : reinterpret_cast<psi::Psi<T, Device>*>(psi_cpu);
        }
        else if (this->inp.precision == "single")
        {
            if (this->inp.device == "cpu")
            {
                psi_cpu = reinterpret_cast<psi::Psi<T>*>(kspw_psi);
                psi_device = kspw_psi;
            }
            else
            {
                psi_cpu = new Psi<T>(1, nbands_start, nbasis, nbasis, true);
                psi_device = kspw_psi;
            }
        }
    }

    psi->fix_k(ik);
    kspw_psi->fix_k(ik);
    p_hamilt->updateHk(ik);
    if (fill)
    {
        this->psi_initer->init_psig(psi_cpu->get_pointer(), ik);
        if (psi_device->get_pointer() != psi_cpu->get_pointer())
        {
            syncmem_h2d_op()(psi_device->get_pointer(), psi_cpu->get_pointer(), nbands_start * nbasis);
        }

        if (this->ks_solver == "cg")
        {
            std::vector<typename GetTypeReal<T>::type> etatom(nbands_start, 0.0);
            if (not_equal)
            {
                hsolver::DiagoIterAssist<T, Device>::diag_subspace_init(p_hamilt,
                                                                         psi_device->get_pointer(),
                                                                         nbands_start,
                                                                         nbasis,
                                                                         *(kspw_psi),
                                                                         etatom.data());
            }
            else
            {
                hsolver::DiagoIterAssist<T, Device>::diag_subspace(p_hamilt,
                                                                    *psi_device,
                                                                    *kspw_psi,
                                                                    etatom.data(),
                                                                    nbands_start);
            }
        }
        else
        {
            if (psi_device->get_pointer() != kspw_psi->get_pointer())
            {
                syncmem_complex_op()(kspw_psi->get_pointer(), psi_device->get_pointer(), nbands_l * nbasis);
            }
        }
    }
#ifdef __MPI
    if (this->ks_solver == "bpcg" && this->inp.bndpar > 1)
    {
        std::vector<int> sendcounts(this->inp.bndpar);
        std::vector<int> displs(this->inp.bndpar);
        MPI_Allgather(&nbands_l, 1, MPI_INT, sendcounts.data(), 1, MPI_INT, BP_WORLD);
        displs[0] = 0;
        sendcounts[0] *= nbasis;
        for (int i = 1; i < this->inp.bndpar; i++)
        {
            sendcounts[i] *= nbasis;
            displs[i] = displs[i - 1] + sendcounts[i - 1];
        }
        if (this->my_bndgroup == 0)
        {
            for (int ip = 1; ip < this->inp.bndpar; ++ip)
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

    if (fill)
    {
        if (not_equal)
        {
            delete psi_cpu;
            if (this->inp.device == "gpu")
            {
                delete psi_device;
            }
        }
        else if (this->inp.precision == "single" && this->inp.device == "gpu")
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
                  const int& npwx,
                  const Input_para& inp,
                  const System_para& sys,
                  std::ofstream& running_log,
                  const bool save_memory)
{
    assert(npwx > 0);
    assert(nks > 0);
    ModuleBase::GlobalFunc::OUT(running_log, "npwx", npwx);

    delete psi;
    int nks2 = nks;
    if ((inp.calculation == "nscf" && inp.mem_saver == 1) || save_memory)
    {
        nks2 = 1;
    }
    psi = new psi::Psi<std::complex<double>>(nks2, nbands, npwx * sys.npol, ngk, true);
    const size_t memory_cost = sizeof(std::complex<double>) * nks2 * nbands * (sys.npol * npwx);
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
