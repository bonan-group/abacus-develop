#include "forces.h"
#include "source_base/parallel_reduce.h"
#include "source_pw/module_pwdft/vnl_pw.h"
#include "source_base/libm/libm.h"
#include "source_io/module_parameter/parameter.h"
#include "source_base/math_ylmreal.h"
#include "source_base/timer.h"
#include "source_estate/elecstate_pw.h"
#include "source_pw/module_pwdft/kernels/nonlocal_op.h"
#include "source_base/module_device/memory_op.h"

#include <type_traits>

// This routine computes the contribution to atomic forces due
// to the dependence of the Q function on the atomic position.
// \[ F_{j,\text{at}} = \sum_G \sum_{lm} iG_j\ \text{exp}(-iG*R_\text{at})
//    V^*(G)\ Q_{lm}(G)\ \text{becsum}(lm,\text{at}) \]
// where:
// \[ \text{becsum}(lm,\text{at}) = \sum_i \langle \psi_i|\beta_l\rangle
//    w_i\langle \beta_m|\psi_i\rangle \]
// On output: the contribution is added to \(\text{forcenl}\).
template <typename FPTYPE, typename Device>
void Forces<FPTYPE, Device>::cal_force_us(ModuleBase::matrix& forcenl,
                                          const ModulePW::PW_Basis* const rho_basis,
                                          const pseudopot_cell_vnl& nlpp,
                                          const elecstate::ElecState& elec,
                                          const UnitCell& ucell)
{
    ModuleBase::TITLE("Forces", "cal_force_us");
    ModuleBase::timer::start("Forces", "cal_force_us");

    const int npw = rho_basis->npw;
    const int nh_tot = nlpp.nhm * (nlpp.nhm + 1) / 2;
    const std::complex<double> fac = ModuleBase::NEG_IMAG_UNIT * ucell.tpiba;
    const std::complex<double> ci_tpi = ModuleBase::IMAG_UNIT * ModuleBase::TWO_PI;
    double* becsum = static_cast<const elecstate::ElecStatePW<std::complex<FPTYPE>, Device>&>(elec).becsum;

    ModuleBase::matrix forceq(ucell.nat, 3);
    const int nspin = elec.pot->get_nspin();

#if defined(__CUDA) || defined(__ROCM)
    if (std::is_same<Device, base_device::DEVICE_GPU>::value
        && elec.pot->get_eff_v_device_data() != nullptr)
    {
        std::complex<double>* vg_device = nullptr;
        double* force_device = nullptr;
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            vg_device,
            nspin * npw,
            "Forces::uspp_vg");
        base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>()(
            force_device,
            ucell.nat * 3,
            "Forces::uspp_force");
        base_device::memory::set_memory_op<double, base_device::DEVICE_GPU>()(
            force_device,
            0,
            ucell.nat * 3);
        const double* veff_device = elec.pot->get_eff_v_device_data();
        for (int is = 0; is < nspin; ++is)
        {
            rho_basis->real2recip_gpu<double>(veff_device + is * rho_basis->nrxx,
                                              vg_device + is * npw);
        }
        const std::complex<double>* qgm = nlpp.get_qgm_data<double>();
        const std::complex<double>* phase = nlpp.get_qgm_phase_data<double>();
        for (int it = 0; it < ucell.ntype; ++it)
        {
            const Atom* atom = &ucell.atoms[it];
            if (!atom->ncpp.tvanp)
            {
                continue;
            }
            const int nij = atom->ncpp.nh * (atom->ncpp.nh + 1) / 2;
            const int first_iat = ucell.itia2iat(it, 0);
            hamilt::uspp_force_op<double, base_device::DEVICE_GPU>()(
                nullptr,
                nspin,
                atom->na,
                nij,
                npw,
                first_iat,
                ucell.nat,
                nh_tot,
                ucell.omega,
                ucell.tpiba,
                vg_device,
                qgm + it * nh_tot * npw,
                phase + first_iat * npw,
                nlpp.get_qgm_gcar_data(),
                becsum,
                force_device);
        }
        base_device::memory::synchronize_memory_op<double,
                                                   base_device::DEVICE_CPU,
                                                   base_device::DEVICE_GPU>()(
            forceq.c,
            force_device,
            ucell.nat * 3);
        base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(vg_device);
        base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>()(force_device);
        Parallel_Reduce::reduce_all(forceq.c, forceq.nr * forceq.nc);
        forcenl += forceq;
        ModuleBase::timer::end("Forces", "cal_force_us");
        return;
    }
#endif

    ModuleBase::matrix veff = elec.pot->get_eff_v();
    ModuleBase::ComplexMatrix vg(PARAM.inp.nspin, npw);
    // fourier transform of the total effective potential
    for (int is = 0; is < PARAM.inp.nspin; is++)
    {
        rho_basis->real2recip(&veff.c[is * veff.nc], &vg(is, 0));
    }

    ModuleBase::matrix ylmk0(nlpp.lmaxq * nlpp.lmaxq, npw);
    ModuleBase::YlmReal::Ylm_Real(nlpp.lmaxq * nlpp.lmaxq, npw, rho_basis->gcar, ylmk0);

    double* qnorm = new double[npw];
    for (int ig = 0; ig < npw; ig++)
    {
        qnorm[ig] = rho_basis->gcar[ig].norm() * ucell.tpiba;
    }

    for (int it = 0; it < ucell.ntype; it++)
    {
        Atom* atom = &ucell.atoms[it];
        if (atom->ncpp.tvanp)
        {
            // nij = max number of (ih,jh) pairs per atom type nt
            // qgm contains the Q functions in G space
            const int nij = atom->ncpp.nh * (atom->ncpp.nh + 1) / 2;
            ModuleBase::ComplexMatrix qgm(nij, npw);

            // Compute and store Q(G) for this atomic species
            // (without structure factor)
            int ijh = 0;
            for (int ih = 0; ih < atom->ncpp.nh; ih++)
            {
                for (int jh = ih; jh < atom->ncpp.nh; jh++)
                {
                    nlpp.radial_fft_q(npw, ih, jh, it, qnorm, ylmk0, &qgm(ijh, 0));
                    ijh++;
                }
            }
            double* qgm_data = reinterpret_cast<double*>(qgm.c);

            ModuleBase::ComplexArray aux1(3, atom->na, npw);
            ModuleBase::realArray ddeeq(PARAM.inp.nspin, 3, atom->na, nij);
            for (int is = 0; is < PARAM.inp.nspin; is++)
            {
                for (int ia = 0; ia < atom->na; ia++)
                {
                    // aux1 = product of potential, structure factor and iG
                    for (int ig = 0; ig < npw; ig++)
                    {
                        double arg = rho_basis->gcar[ig] * atom->tau[ia];
                        std::complex<double> cfac = fac * vg(is, ig) * ModuleBase::libm::exp(ci_tpi * arg);
                        for (int ipol = 0; ipol < 3; ipol++)
                        {
                            aux1(ipol, ia, ig) = cfac * rho_basis->gcar[ig][ipol];
                        }
                    }
                }
                double* aux1_data = reinterpret_cast<double*>(aux1.ptr);

                // ddeeq = dot product of aux1 with the Q functions
                // No need for special treatment of the G=0 term (is zero)
                const char transa = 'C';
                const char transb = 'N';
                const int dim = 2 * npw;
                const double zero = 0;
                for (int ipol = 0; ipol < 3; ipol++)
                {
                    BlasConnector::gemm(transb,
                           transa,
                           atom->na,
                           nij,
                           dim,
                           ucell.omega,
                           &aux1_data[ipol * dim * atom->na],
                           dim,
                           qgm_data,
                           dim,
                           zero,
                           &ddeeq(is, ipol, 0, 0),
                           nij);
                }
            }

            for (int is = 0; is < PARAM.inp.nspin; is++)
            {
                for (int ia = 0; ia < atom->na; ia++)
                {
                    const int iat = ucell.itia2iat(it, ia);
                    const int index = is * ucell.nat * nh_tot + iat * nh_tot;
                    for (int ipol = 0; ipol < 3; ipol++)
                    {
                        for (int ijh = 0; ijh < nij; ijh++)
                        {
                            forceq(iat, ipol) += ddeeq(is, ipol, ia, ijh) * becsum[index + ijh];
                        }
                    }
                }
            }
        }
    }

    Parallel_Reduce::reduce_all(forceq.c, forceq.nr * forceq.nc);
    forcenl += forceq;

    delete[] qnorm;

    ModuleBase::timer::end("Forces", "cal_force_us");
}

template class Forces<double, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class Forces<double, base_device::DEVICE_GPU>;
#endif
