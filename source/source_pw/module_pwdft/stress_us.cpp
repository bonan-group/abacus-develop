#include "source_base/libm/libm.h"
#include "source_base/parallel_reduce.h"
#include "source_io/module_parameter/parameter.h"
#include "source_base/math_ylmreal.h"
#include "source_base/timer.h"
#include "source_estate/elecstate_pw.h"
#include "source_pw/module_pwdft/nonlocal_maths.hpp"
#include "source_pw/module_pwdft/kernels/nonlocal_op.h"
#include "stress_pw.h"
#include "source_base/module_device/memory_op.h"

#include <type_traits>

// computes the part of the crystal stress which is due
// to the dependence of the Q function on the atomic position in PW base
template <typename FPTYPE, typename Device>
void Stress_PW<FPTYPE, Device>::stress_us(ModuleBase::matrix& sigma,
                                          ModulePW::PW_Basis* rho_basis,
                                          const pseudopot_cell_vnl& nlpp,
                                          const UnitCell& ucell)
{
    ModuleBase::TITLE("Stress", "stress_us");
    ModuleBase::timer::start("Stress", "stress_us");

    const int npw = rho_basis->npw;
    const int nh_tot = nlpp.nhm * (nlpp.nhm + 1) / 2;
    const std::complex<double> ci_tpi = ModuleBase::IMAG_UNIT * ModuleBase::TWO_PI;
    double* becsum = static_cast<const elecstate::ElecStatePW<std::complex<FPTYPE>, Device>*>(this->pelec)->becsum;
    ModuleBase::matrix stressus(3, 3);

#if defined(__CUDA) || defined(__ROCM)
    if (std::is_same<Device, base_device::DEVICE_GPU>::value
        && this->pelec->pot->get_eff_v_device_data() != nullptr
        && nlpp.get_dqgm_data() != nullptr)
    {
        const int nspin = this->pelec->pot->get_nspin();
        std::complex<double>* vg_device = nullptr;
        double* stress_device = nullptr;
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            vg_device,
            nspin * npw,
            "Stress::uspp_vg");
        base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>()(
            stress_device,
            9,
            "Stress::uspp_stress");
        base_device::memory::set_memory_op<double, base_device::DEVICE_GPU>()(stress_device, 0, 9);
        const double* veff_device = this->pelec->pot->get_eff_v_device_data();
        for (int is = 0; is < nspin; ++is)
        {
            rho_basis->real2recip_gpu<double>(veff_device + is * rho_basis->nrxx,
                                              vg_device + is * npw);
        }
        const std::complex<double>* phase = nlpp.get_qgm_phase_data<double>();
        const std::complex<double>* dqgm = nlpp.get_dqgm_data();
        for (int ipol = 0; ipol < 3; ++ipol)
        {
            for (int it = 0; it < ucell.ntype; ++it)
            {
                const Atom* atom = &ucell.atoms[it];
                if (!atom->ncpp.tvanp)
                {
                    continue;
                }
                const int nij = atom->ncpp.nh * (atom->ncpp.nh + 1) / 2;
                const int first_iat = ucell.itia2iat(it, 0);
                hamilt::uspp_stress_op<double, base_device::DEVICE_GPU>()(
                    nullptr,
                    nspin,
                    atom->na,
                    nij,
                    npw,
                    first_iat,
                    ucell.nat,
                    nh_tot,
                    ipol,
                    ucell.tpiba,
                    vg_device,
                    dqgm + (ipol * ucell.ntype + it) * nh_tot * npw,
                    phase + first_iat * npw,
                    nlpp.get_qgm_gcar_data(),
                    becsum,
                    stress_device);
            }
        }
        base_device::memory::synchronize_memory_op<double,
                                                   base_device::DEVICE_CPU,
                                                   base_device::DEVICE_GPU>()(
            stressus.c,
            stress_device,
            9);
        base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(vg_device);
        base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>()(stress_device);
        Parallel_Reduce::reduce_all(stressus.c, stressus.nr * stressus.nc);
        for (int l = 0; l < 3; ++l)
        {
            for (int m = l; m < 3; ++m)
            {
                stressus(m, l) = stressus(l, m);
            }
        }
        sigma += stressus;
        ModuleBase::timer::end("Stress", "stress_us");
        return;
    }
#endif

    std::vector<double> becsum_host;
#if defined(__CUDA) || defined(__ROCM)
    if (std::is_same<Device, base_device::DEVICE_GPU>::value)
    {
        becsum_host.resize(this->pelec->pot->get_nspin() * ucell.nat * nh_tot);
        base_device::memory::synchronize_memory_op<double,
                                                   base_device::DEVICE_CPU,
                                                   base_device::DEVICE_GPU>()(
            becsum_host.data(),
            becsum,
            becsum_host.size());
        becsum = becsum_host.data();
    }
#endif

    ModuleBase::matrix veff = this->pelec->pot->get_eff_v();
    ModuleBase::ComplexMatrix vg(PARAM.inp.nspin, npw);
    // fourier transform of the total effective potential
    for (int is = 0; is < PARAM.inp.nspin; is++)
    {
        rho_basis->real2recip(&veff.c[is * veff.nc], &vg(is, 0));
    }

    ModuleBase::matrix ylmk0(nlpp.lmaxq * nlpp.lmaxq, npw);
    ModuleBase::YlmReal::Ylm_Real(nlpp.lmaxq * nlpp.lmaxq, npw, rho_basis->gcar, ylmk0);

    // double* qnorm = new double[npw];
    std::vector<double> qnorm_vec(npw);
    double* qnorm = qnorm_vec.data();
    for (int ig = 0; ig < npw; ig++)
    {
        qnorm[ig] = rho_basis->gcar[ig].norm() * ucell.tpiba;
    }

    // here we compute the integral Q*V for each atom,
    //      I = sum_G G_a exp(-iR.G) Q_nm v^*
    // (no contribution from G=0)
    ModuleBase::matrix dylmk0(nlpp.lmaxq * nlpp.lmaxq, npw);
    for (int ipol = 0; ipol < 3; ipol++)
    {
        double* gcar_ptr = reinterpret_cast<double*>(rho_basis->gcar);
        hamilt::Nonlocal_maths<FPTYPE, base_device::DEVICE_CPU>::dylmr2(nlpp.lmaxq * nlpp.lmaxq,
                                                                        npw,
                                                                        gcar_ptr,
                                                                        dylmk0.c,
                                                                        ipol);
        for (int it = 0; it < ucell.ntype; it++)
        {
            Atom* atom = &ucell.atoms[it];
            if (atom->ncpp.tvanp)
            {
                // nij = max number of (ih,jh) pairs per atom type nt
                // qgm contains derivatives of the Fourier transform of the Q function
                const int nij = atom->ncpp.nh * (atom->ncpp.nh + 1) / 2;
                ModuleBase::ComplexMatrix qgm(nij, npw);
                ModuleBase::matrix tbecsum(PARAM.inp.nspin, nij);

                // Compute and store derivatives of Q(G) for this atomic species
                // (without structure factor)
                int ijh = 0;
                for (int ih = 0; ih < atom->ncpp.nh; ih++)
                {
                    for (int jh = ih; jh < atom->ncpp.nh; jh++)
                    {
                        nlpp.radial_fft_dq(npw,
                                          ih,
                                          jh,
                                          it,
                                          ipol,
                                          rho_basis->gcar,
                                          qnorm,
                                          ucell.tpiba,
                                          ylmk0,
                                          dylmk0,
                                          &qgm(ijh, 0));
                        ijh++;
                    }
                }
                double* qgm_data = reinterpret_cast<double*>(qgm.c);

                for (int ia = 0; ia < atom->na; ia++)
                {
                    const int iat = ucell.itia2iat(it, ia);
                    for (int is = 0; is < PARAM.inp.nspin; is++)
                    {
                        for (int ij = 0; ij < nij; ij++)
                        {
                            tbecsum(is, ij) = becsum[is * ucell.nat * nh_tot + iat * nh_tot + ij];
                        }
                    }

                    ModuleBase::ComplexMatrix aux2(PARAM.inp.nspin, npw);
                    double* aux2_data = reinterpret_cast<double*>(aux2.c);

                    const char transa = 'N';
                    const char transb = 'N';
                    const int dim = 2 * npw;
                    const double one = 1;
                    const double zero = 0;
                    BlasConnector::gemm(transb,
                           transa,
                           PARAM.inp.nspin,
                           dim,
                           nij,
                           one,
                           tbecsum.c,
                           nij,
                           qgm_data,
                           dim,
                           zero,
                           aux2_data,
                           dim);

                    for (int is = 0; is < PARAM.inp.nspin; is++)
                    {
                        for (int ig = 0; ig < npw; ig++)
                        {
                            aux2(is, ig) *= conj(vg(is, ig));
                        }
                    }

                    ModuleBase::ComplexMatrix aux1(3, npw);
                    double* aux1_data = reinterpret_cast<double*>(aux1.c);
                    for (int ig = 0; ig < npw; ig++)
                    {
                        double arg = rho_basis->gcar[ig] * atom->tau[ia];
                        std::complex<double> cfac = ucell.tpiba * ModuleBase::libm::exp(ci_tpi * arg);
                        for (int ipol = 0; ipol < 3; ipol++)
                        {
                            aux1(ipol, ig) = cfac * rho_basis->gcar[ig][ipol];
                        }
                    }

                    ModuleBase::matrix fac(PARAM.inp.nspin, 3);
                    const char transc = 'T';
                    const int three = 3;
                    BlasConnector::gemm_cm(transc,
                           transb,
                           three,
                           PARAM.inp.nspin,
                           dim,
                           one,
                           aux1_data,
                           dim,
                           aux2_data,
                           dim,
                           zero,
                           fac.c,
                           three);

                    for (int is = 0; is < PARAM.inp.nspin; is++)
                    {
                        for (int jpol = 0; jpol < 3; jpol++)
                        {
                            stressus(jpol, ipol) += fac(is, jpol);
                        }
                    }
                }
            }
        }
    }

    Parallel_Reduce::reduce_all(stressus.c, stressus.nr * stressus.nc);
    for (int l = 0; l < 3; l++)
    {
        for (int m = l; m < 3; m++)
        {
            stressus(m, l) = stressus(l, m);
        }
    }
    sigma += stressus;

    ModuleBase::timer::end("Stress", "stress_us");
    return;
}

template class Stress_PW<double, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class Stress_PW<double, base_device::DEVICE_GPU>;
#endif
