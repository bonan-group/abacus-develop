#include "charge_mixing.h"
#include "source_io/module_parameter/parameter.h"
#include "source_base/timer.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_base/module_device/types.h"

void Charge_Mixing::mix_rho_recip(Charge* chr)
{
    ModuleBase::TITLE("Charge_Mixing", "mix_rho_recip");
    ModuleBase::timer::start("Charge_Mixing", "mix_rho_recip");

    const int nspin = PARAM.inp.nspin;
    assert(nspin==1 || nspin==2 || nspin==4);
    const bool double_grid = PARAM.globalv.double_grid;

// Full GPU-resident mixing path
// Re-enabled after fixing the complex vector_axpy_op aliasing bug
#if __CUDA
    // Full GPU-resident mixing path for supported reciprocal mixing cases.
    // This path keeps all mixing history and operations on GPU
    if (this->can_use_gpu_resident_mixing(chr))
    {
        validate_gpu_fft_poolnproc(chr->rhopw, "Charge_Mixing::mix_rho_recip");
        this->log_gpu_charge_mixing_active();
        // Resolve the legacy SCF magnetic policy at this control boundary and pass it explicitly to GPU helpers.
        const bool include_magnetism = (nspin != 4 || PARAM.globalv.domag || PARAM.globalv.domag_z);

        // Restore timer context for GPU path
        ModuleBase::timer::end("Charge_Mixing", "mix_rho_recip");

        mix_rho_recip_gpu(chr, include_magnetism);
        return;
    }
    if (device_ == "gpu")
    {
        if (!this->mixing_gpu_enabled)
        {
            this->log_gpu_charge_mixing_fallback("mixing_gpu is false");
        }
        else if (chr->get_device() != "gpu")
        {
            this->log_gpu_charge_mixing_fallback("charge density is not resident on GPU");
        }
        else if (mixing_mode != "plain" && mixing_mode != "broyden" && mixing_mode != "pulay")
        {
            this->log_gpu_charge_mixing_fallback("only plain, Broyden, and Pulay mixing are supported");
        }
        else if (nspin == 4 && this->mixing_angle > 0.0)
        {
            this->log_gpu_charge_mixing_fallback("mixing_angle > 0 is not supported");
        }
        else if (nspin != 1)
        {
            this->log_gpu_charge_mixing_fallback("this spin configuration is not supported");
        }
    }
#elif defined(__ROCM)
    if (device_ == "gpu")
    {
        this->log_gpu_charge_mixing_fallback("the optimized path is not implemented for ROCm");
    }
#endif

    std::complex<double>* rhog_in = nullptr;
    std::complex<double>* rhog_out = nullptr;
    // for smooth part
    std::complex<double>* rhogs_in = chr->rhog_save[0];
    std::complex<double>* rhogs_out = chr->rhog[0];
    // for high_frequency part
    std::complex<double>* rhoghf_in = nullptr;
    std::complex<double>* rhoghf_out = nullptr;

    if ( PARAM.globalv.double_grid)
    {
        // divide into smooth part and high_frequency part
        divide_data(chr->rhog_save[0], rhogs_in, rhoghf_in);
        divide_data(chr->rhog[0], rhogs_out, rhoghf_out);
    }

    //  inner_product_recip_hartree is a hartree-like sum, unit is Ry
    auto inner_product
        = std::bind(&Charge_Mixing::inner_product_recip_hartree, this, std::placeholders::_1, std::placeholders::_2);

    // DIIS Mixing Only for smooth part, while high_frequency part is mixed by plain mixing method.
    if (nspin == 1)
    {
        rhog_in = rhogs_in;
        rhog_out = rhogs_out;
        auto screen = std::bind(&Charge_Mixing::Kerker_screen_recip, this, std::placeholders::_1);
        this->mixing->push_data(this->rho_mdata, rhog_in, rhog_out, screen, true);
        this->mixing->cal_coef(this->rho_mdata, inner_product);
        this->mixing->mix_data(this->rho_mdata, rhog_out);
    }
    else if (nspin == 2)
    {
        // magnetic density
        std::complex<double> *rhog_mag = nullptr;
        std::complex<double> *rhog_mag_save = nullptr;
        const int npw = this->rhopw->npw;
        // allocate rhog_mag[is*ngmc] and rhog_mag_save[is*ngmc]
        rhog_mag = new std::complex<double>[npw * nspin];
        rhog_mag_save = new std::complex<double>[npw * nspin];
        ModuleBase::GlobalFunc::ZEROS(rhog_mag, npw * nspin);
        ModuleBase::GlobalFunc::ZEROS(rhog_mag_save, npw * nspin);
        // get rhog_mag[is*ngmc] and rhog_mag_save[is*ngmc]
        for (int ig = 0; ig < npw; ig++)
        {
            rhog_mag[ig] = chr->rhog[0][ig] + chr->rhog[1][ig];
            rhog_mag_save[ig] = chr->rhog_save[0][ig] + chr->rhog_save[1][ig];
        }
        for (int ig = 0; ig < npw; ig++)
        {
            rhog_mag[ig + npw] = chr->rhog[0][ig] - chr->rhog[1][ig];
            rhog_mag_save[ig + npw] = chr->rhog_save[0][ig] - chr->rhog_save[1][ig];
        }
        //
        rhog_in = rhog_mag_save;
        rhog_out = rhog_mag;
        //
        auto screen = std::bind(&Charge_Mixing::Kerker_screen_recip, this, std::placeholders::_1);
        auto twobeta_mix
            = [this, npw](std::complex<double>* out, const std::complex<double>* in, const std::complex<double>* sres) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
                  for (int i = 0; i < npw; ++i)
                  {
                      out[i] = in[i] + this->mixing_beta * sres[i];
                  }
            // magnetism
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
                  for (int i = npw; i < 2 * npw; ++i)
                  {
                      out[i] = in[i] + this->mixing_beta_mag * sres[i];
                  }
              };
        this->mixing->push_data(this->rho_mdata, rhog_in, rhog_out, screen, twobeta_mix, true);
        this->mixing->cal_coef(this->rho_mdata, inner_product);
        this->mixing->mix_data(this->rho_mdata, rhog_out);
        // get rhog[is][ngmc] from rhog_mag[is*ngmc]
        for (int is = 0; is < nspin; is++)
        {
            ModuleBase::GlobalFunc::ZEROS(chr->rhog[is], npw);
        }
        for (int ig = 0; ig < npw; ig++)
        {
            chr->rhog[0][ig] = 0.5 * (rhog_mag[ig] + rhog_mag[ig+npw]);
            chr->rhog[1][ig] = 0.5 * (rhog_mag[ig] - rhog_mag[ig+npw]);
        }
        // delete
        delete[] rhog_mag;
        delete[] rhog_mag_save;
        // get rhogs_out for combine_data()
        if ( PARAM.globalv.double_grid)
        {
            for (int ig = 0; ig < npw; ig++)
            {
                rhogs_out[ig] = chr->rhog[0][ig];
                rhogs_out[ig + npw] = chr->rhog[1][ig];
            }
        }
    }
    else if (nspin == 4 && PARAM.inp.mixing_angle <= 0)
    {
        // normal broyden mixing for {rho, mx, my, mz}
        rhog_in = rhogs_in;
        rhog_out = rhogs_out;
        const int npw = this->rhopw->npw;
        auto screen = std::bind(&Charge_Mixing::Kerker_screen_recip, this, std::placeholders::_1); // use old one
        auto twobeta_mix
            = [this, npw](std::complex<double>* out, const std::complex<double>* in, const std::complex<double>* sres) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
                  for (int i = 0; i < npw; ++i)
                  {
                      out[i] = in[i] + this->mixing_beta * sres[i];
                  }
            // magnetism, mx, my, mz
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
                  for (int i = npw; i < 4 * npw; ++i)
                  {
                      out[i] = in[i] + this->mixing_beta_mag * sres[i];
                  }
              };
        this->mixing->push_data(this->rho_mdata, rhog_in, rhog_out, screen, twobeta_mix, true);
        this->mixing->cal_coef(this->rho_mdata, inner_product);
        this->mixing->mix_data(this->rho_mdata, rhog_out);
    }
    else if (nspin == 4 && PARAM.inp.mixing_angle > 0)
    {
        // special broyden mixing for {rho, |m|} proposed by J. Phys. Soc. Jpn. 82 (2013) 114706
        // here only consider the case of mixing_angle = 1, which mean only change |m| and keep angle fixed
        const bool use_double_grid = double_grid;
        const int smooth_npw = this->rhopw->npw;
        const int dense_npw = use_double_grid ? this->rhodpw->npw : this->rhopw->npw;
        const int real_nrxx = use_double_grid ? this->rhodpw->nrxx : this->rhopw->nrxx;
        // allocate memory for rho_magabs and rho_magabs_save
        std::vector<double> rho_magabs(real_nrxx, 0.0);
        std::vector<double> rho_magabs_save(real_nrxx, 0.0);
        // calculate rho_magabs and rho_magabs_save
        for (int ir = 0; ir < real_nrxx; ir++)
        {
            // |m| for rho
            rho_magabs[ir] = std::sqrt(chr->rho[1][ir] * chr->rho[1][ir]
            + chr->rho[2][ir] * chr->rho[2][ir]
            + chr->rho[3][ir] * chr->rho[3][ir]);
            // |m| for rho_save
            rho_magabs_save[ir] = std::sqrt(chr->rho_save[1][ir] * chr->rho_save[1][ir]
            + chr->rho_save[2][ir] * chr->rho_save[2][ir]
            + chr->rho_save[3][ir] * chr->rho_save[3][ir]);
        }
        // allocate memory for rhog_magabs and rhog_magabs_save
        std::vector<std::complex<double>> rhog_magabs_dense(dense_npw);
        std::vector<std::complex<double>> rhog_magabs_save_dense(dense_npw);
        std::vector<std::complex<double>> rhog_magabs(smooth_npw * 2);
        std::vector<std::complex<double>> rhog_magabs_save(smooth_npw * 2);
        // calculate rhog_magabs and rhog_magabs_save
        for (int ig = 0; ig < smooth_npw; ig++)
        {
            rhog_magabs[ig] = use_double_grid ? rhogs_out[ig] : chr->rhog[0][ig]; // rho
            rhog_magabs_save[ig] = use_double_grid ? rhogs_in[ig] : chr->rhog_save[0][ig]; // rho_save
        }
        // FT to get rhog_magabs and rhog_magabs_save
        if (use_double_grid)
        {
            this->rhodpw->real2recip(rho_magabs.data(), rhog_magabs_dense.data());
            this->rhodpw->real2recip(rho_magabs_save.data(), rhog_magabs_save_dense.data());
        }
        else
        {
            this->rhopw->real2recip(rho_magabs.data(), rhog_magabs_dense.data());
            this->rhopw->real2recip(rho_magabs_save.data(), rhog_magabs_save_dense.data());
        }
        for (int ig = 0; ig < smooth_npw; ++ig)
        {
            rhog_magabs[ig + smooth_npw] = rhog_magabs_dense[ig];
            rhog_magabs_save[ig + smooth_npw] = rhog_magabs_save_dense[ig];
        }
        //
        rhog_in = rhog_magabs_save.data();
        rhog_out = rhog_magabs.data();
        auto screen = std::bind(&Charge_Mixing::Kerker_screen_recip, this, std::placeholders::_1); // use old one
        auto twobeta_mix = [this, smooth_npw](std::complex<double>* out,
                                               const std::complex<double>* in,
                                               const std::complex<double>* sres) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
                  for (int i = 0; i < smooth_npw; ++i)
                  {
                      out[i] = in[i] + this->mixing_beta * sres[i];
                  }
            // magnetism, |m|
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
                  for (int i = smooth_npw; i < 2 * smooth_npw; ++i)
                  {
                      out[i] = in[i] + this->mixing_beta_mag * sres[i];
                  }
              };
        this->mixing->push_data(this->rho_mdata, rhog_in, rhog_out, screen, twobeta_mix, true);
        this->mixing->cal_coef(this->rho_mdata, inner_product);
        this->mixing->mix_data(this->rho_mdata, rhog_out);
        if (use_double_grid)
        {
            const int high_frequency_npw = dense_npw - smooth_npw;
            if (high_frequency_npw > 0)
            {
                this->mixing_highf->plain_mix(rhoghf_out,
                                              rhoghf_in,
                                              rhoghf_out,
                                              high_frequency_npw,
                                              nullptr);
                Base_Mixing::Plain_Mixing magnetic_highf_mixing(this->mixing_beta_mag);
                magnetic_highf_mixing.plain_mix(rhog_magabs_dense.data() + smooth_npw,
                                                 rhog_magabs_save_dense.data() + smooth_npw,
                                                 rhog_magabs_dense.data() + smooth_npw,
                                                 high_frequency_npw,
                                                 nullptr);
            }
            for (int ig = 0; ig < smooth_npw; ++ig)
            {
                chr->rhog[0][ig] = rhog_magabs[ig];
                rhog_magabs_dense[ig] = rhog_magabs[ig + smooth_npw];
            }
            for (int ig = 0; ig < high_frequency_npw; ++ig)
            {
                chr->rhog[0][smooth_npw + ig] = rhoghf_out[ig];
            }
            this->rhodpw->recip2real(rhog_magabs_dense.data(), rho_magabs.data());
            clean_data(rhogs_in, rhoghf_in);
            clean_data(rhogs_out, rhoghf_out);
        }
        else
        {
            for (int ig = 0; ig < smooth_npw; ++ig)
            {
                chr->rhog[0][ig] = rhog_magabs[ig];
                rhog_magabs_dense[ig] = rhog_magabs[ig + smooth_npw];
            }
            // get new |m| in real space using FT
            this->rhopw->recip2real(rhog_magabs_dense.data(), rho_magabs.data());
        }
        // use new |m| and angle to update {mx, my, mz}
        for (int ir = 0; ir < real_nrxx; ir++)
        {
            const double norm = std::sqrt(chr->rho[1][ir] * chr->rho[1][ir]
                + chr->rho[2][ir] * chr->rho[2][ir]
                + chr->rho[3][ir] * chr->rho[3][ir]);
            if (std::abs(norm) < 1e-10)
            {
                continue;
            }
            const double rescale_tmp = std::abs(rho_magabs[ir]) / norm;
            chr->rho[1][ir] *= rescale_tmp;
            chr->rho[2][ir] *= rescale_tmp;
            chr->rho[3][ir] *= rescale_tmp;
        }
        ModulePW::PW_Basis* output_basis = use_double_grid ? this->rhodpw : this->rhopw;
        for (int is = 1; is < nspin; ++is)
        {
            output_basis->real2recip(chr->rho[is], chr->rhog[is]);
        }
    }

    if (double_grid && !(nspin == 4 && this->mixing_angle > 0))
    {
        // plain mixing for high_frequencies
        const int ndimhf = (this->rhodpw->npw - this->rhopw->npw) * nspin;
        this->mixing_highf->plain_mix(rhoghf_out, rhoghf_in, rhoghf_out, ndimhf, nullptr);

        // combine smooth part and high_frequency part
        combine_data(chr->rhog[0], rhogs_out, rhoghf_out);
        clean_data(rhogs_in, rhoghf_in);
    }

    // rhog to rho
    if (nspin == 4 && PARAM.inp.mixing_angle > 0)
    {
        // only tranfer rhog[0]
        ModulePW::PW_Basis* output_basis = double_grid ? this->rhodpw : this->rhopw;
        output_basis->recip2real(chr->rhog[0], chr->rho[0]);
#if __CUDA || __ROCM
        if (device_ == "gpu" && chr->get_device() == "gpu")
        {
            chr->sync_rho_to_device();
        }
#endif
    }
    else
    {
#if __CUDA || __ROCM
        // GPU path: use GPU FFT when device="gpu".
        if (device_ == "gpu" && chr->get_device() == "gpu")
        {
            // Sync rhog to GPU (mixing was done on CPU, result in chr->rhog[is])
            chr->sync_rhog_to_device();

            for (int is = 0; is < nspin; is++)
            {
                // GPU FFT: rhog -> rho
                // use rhodpw for double_grid (rhodpw is same as rhopw for ! PARAM.globalv.double_grid)
                this->rhodpw->recip_to_real<std::complex<double>, double, base_device::DEVICE_GPU>(
                    chr->get_rhog_d(is), chr->get_rho_d(is));
            }
            // Sync rho back to CPU for subsequent operations
            chr->sync_rho_to_host();
        }
        else
#endif
        {
            // CPU path (existing code)
            for (int is = 0; is < nspin; is++)
            {
                // use rhodpw for double_grid
                // rhodpw is the same as rhopw for ! PARAM.globalv.double_grid
                this->rhodpw->recip_to_real<std::complex<double>,double,base_device::DEVICE_CPU>(chr->rhog[is], chr->rho[is]);
            }
#if __CUDA || __ROCM
            if (device_ == "gpu" && chr->get_device() == "gpu")
            {
                chr->sync_rho_to_device();
            }
#endif
        }
    }
    // For kinetic energy density
    if ((XC_Functional::get_ked_flag()) && mixing_tau)
    {
        std::vector<std::complex<double>> kin_g(nspin * rhodpw->npw);
        std::vector<std::complex<double>> kin_g_save(nspin * rhodpw->npw);
        // FFT to get kin_g and kin_g_save
        for (int is = 0; is < nspin; ++is)
        {
            rhodpw->real2recip(chr->kin_r[is], &kin_g[is * rhodpw->npw]);
            rhodpw->real2recip(chr->kin_r_save[is], &kin_g_save[is * rhodpw->npw]);
        }
        // for smooth part, for ! PARAM.globalv.double_grid only have this part
        std::complex<double>*taugs_in = kin_g_save.data(), *taugs_out = kin_g.data();
        // for high frequency part
        std::complex<double>*taughf_in = nullptr, *taughf_out = nullptr;
        if ( PARAM.globalv.double_grid)
        {
            // divide into smooth part and high_frequency part
            divide_data(kin_g_save.data(), taugs_in, taughf_in);
            divide_data(kin_g.data(), taugs_out, taughf_out);
        }

        // Note: there is no kerker modification for tau because I'm not sure
        // if we should have it. If necessary we can try it in the future.
        this->mixing->push_data(this->tau_mdata, taugs_in, taugs_out, nullptr, false);

        this->mixing->mix_data(this->tau_mdata, taugs_out);

        if ( PARAM.globalv.double_grid)
        {
            // simple mixing for high_frequencies
            const int ndimhf = (this->rhodpw->npw - this->rhopw->npw) * nspin;
            this->mixing_highf->plain_mix(taughf_out, taughf_in, taughf_out, ndimhf, nullptr);

            // combine smooth part and high_frequency part
            combine_data(kin_g.data(), taugs_out, taughf_out);
            clean_data(taugs_in, taughf_in);
        }

        // kin_g to kin_r
        for (int is = 0; is < nspin; is++)
        {
            rhodpw->recip2real(&kin_g[is * rhodpw->npw], chr->kin_r[is]);
        }
    }

    ModuleBase::timer::end("Charge_Mixing", "mix_rho_recip");
    return;
}

void Charge_Mixing::mix_rho_real(Charge* chr)
{
    ModuleBase::TITLE("Charge_Mixing", "mix_rho_real");
    ModuleBase::timer::start("Charge_Mixing", "mix_rho_real");

    const int nspin = PARAM.inp.nspin;
    assert(nspin==1 || nspin==2 || nspin==4);

    double* rhor_in=nullptr;
    double* rhor_out=nullptr;

    if (nspin == 1)
    {
        rhor_in = chr->rho_save[0];
        rhor_out = chr->rho[0];
        auto screen = std::bind(&Charge_Mixing::Kerker_screen_real, this, std::placeholders::_1);
        this->mixing->push_data(this->rho_mdata, rhor_in, rhor_out, screen, true);    
        auto inner_product
            = std::bind(&Charge_Mixing::inner_product_real, this, std::placeholders::_1, std::placeholders::_2);
        this->mixing->cal_coef(this->rho_mdata, inner_product);
        this->mixing->mix_data(this->rho_mdata, rhor_out);
    }
    else if (nspin == 2)
    {
        // magnetic density
        double *rho_mag = nullptr;
        double *rho_mag_save = nullptr; 
        const int nrxx = this->rhopw->nrxx;
        // allocate rho_mag[is*nnrx] and rho_mag_save[is*nnrx]
        rho_mag = new double[nrxx * nspin];
        rho_mag_save = new double[nrxx * nspin];
        ModuleBase::GlobalFunc::ZEROS(rho_mag, nrxx * nspin);
        ModuleBase::GlobalFunc::ZEROS(rho_mag_save, nrxx * nspin);
        // get rho_mag[is*nnrx] and rho_mag_save[is*nnrx]
        for (int ir = 0; ir < nrxx; ir++)
        {
            rho_mag[ir] = chr->rho[0][ir] + chr->rho[1][ir];
            rho_mag_save[ir] = chr->rho_save[0][ir] + chr->rho_save[1][ir];
        }
        for (int ir = 0; ir < nrxx; ir++)
        {
            rho_mag[ir + nrxx] = chr->rho[0][ir] - chr->rho[1][ir];
            rho_mag_save[ir + nrxx] = chr->rho_save[0][ir] - chr->rho_save[1][ir];
        }
        //
        rhor_in = rho_mag_save;
        rhor_out = rho_mag;
        auto screen = std::bind(&Charge_Mixing::Kerker_screen_real, this, std::placeholders::_1);
        auto twobeta_mix
            = [this, nrxx](double* out, const double* in, const double* sres) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
            for (int i = 0; i < nrxx; ++i)
            {
                out[i] = in[i] + this->mixing_beta * sres[i];
            }
            // magnetism
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
            for (int i = nrxx; i < 2 * nrxx; ++i)
            {
                out[i] = in[i] + this->mixing_beta_mag * sres[i];
            }
        };
        this->mixing->push_data(this->rho_mdata, rhor_in, rhor_out, screen, twobeta_mix, true);
        auto inner_product
            = std::bind(&Charge_Mixing::inner_product_real, this, std::placeholders::_1, std::placeholders::_2);
        this->mixing->cal_coef(this->rho_mdata, inner_product);
        this->mixing->mix_data(this->rho_mdata, rhor_out);
        // get new rho[is][nrxx] from rho_mag[is*nrxx]
        for (int is = 0; is < nspin; is++)
        {
            ModuleBase::GlobalFunc::ZEROS(chr->rho[is], nrxx);
            //ModuleBase::GlobalFunc::ZEROS(rho_save[is], nrxx);
        }
        for (int ir = 0; ir < nrxx; ir++)
        {
            chr->rho[0][ir] = 0.5 * (rho_mag[ir] + rho_mag[ir+nrxx]);
            chr->rho[1][ir] = 0.5 * (rho_mag[ir] - rho_mag[ir+nrxx]);
        }
        // delete
        delete[] rho_mag;
        delete[] rho_mag_save;
    }
    else if (nspin == 4 && PARAM.inp.mixing_angle <= 0)
    {
        // normal broyden mixing for {rho, mx, my, mz}
        rhor_in = chr->rho_save[0];
        rhor_out = chr->rho[0];
        const int nrxx = this->rhopw->nrxx;
        auto screen = std::bind(&Charge_Mixing::Kerker_screen_real, this, std::placeholders::_1);
        auto twobeta_mix
            = [this, nrxx](double* out, const double* in, const double* sres) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
            for (int i = 0; i < nrxx; ++i)
            {
                out[i] = in[i] + this->mixing_beta * sres[i];
            }
            // magnetism, mx, my, mz
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
            for (int i = nrxx; i < 4 * nrxx; ++i)
            {
                out[i] = in[i] + this->mixing_beta_mag * sres[i];
            }
        };
        this->mixing->push_data(this->rho_mdata, rhor_in, rhor_out, screen, twobeta_mix, true);
        auto inner_product
            = std::bind(&Charge_Mixing::inner_product_real, this, std::placeholders::_1, std::placeholders::_2);
        this->mixing->cal_coef(this->rho_mdata, inner_product);
        this->mixing->mix_data(this->rho_mdata, rhor_out);
    }
    else if (nspin == 4 && PARAM.inp.mixing_angle > 0)
    {
        // special broyden mixing for {rho, |m|} proposed by J. Phys. Soc. Jpn. 82 (2013) 114706
        // here only consider the case of mixing_angle = 1, which mean only change |m| and keep angle fixed
        const int nrxx = this->rhopw->nrxx;
        // allocate memory for rho_magabs and rho_magabs_save
        double* rho_magabs = new double[nrxx * 2];
        double* rho_magabs_save = new double[nrxx * 2];
        ModuleBase::GlobalFunc::ZEROS(rho_magabs, nrxx * 2);
        ModuleBase::GlobalFunc::ZEROS(rho_magabs_save, nrxx * 2);
        // calculate rho_magabs and rho_magabs_save
        for (int ir = 0; ir < nrxx; ir++)
        {
            rho_magabs[ir] = chr->rho[0][ir]; // rho
            rho_magabs_save[ir] = chr->rho_save[0][ir]; // rho_save
            // |m| for rho
			rho_magabs[nrxx + ir] = std::sqrt(chr->rho[1][ir] * chr->rho[1][ir] 
					+ chr->rho[2][ir] * chr->rho[2][ir] 
					+ chr->rho[3][ir] * chr->rho[3][ir]);
			// |m| for rho_save
			rho_magabs_save[nrxx + ir] = std::sqrt(chr->rho_save[1][ir] * chr->rho_save[1][ir] 
					+ chr->rho_save[2][ir] * chr->rho_save[2][ir] 
					+ chr->rho_save[3][ir] * chr->rho_save[3][ir]);
		}
        rhor_in = rho_magabs_save;
        rhor_out = rho_magabs;

        auto screen = std::bind(&Charge_Mixing::Kerker_screen_real, this, std::placeholders::_1);
        auto twobeta_mix
            = [this, nrxx](double* out, const double* in, const double* sres) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
            for (int i = 0; i < nrxx; ++i)
            {
                out[i] = in[i] + this->mixing_beta * sres[i];
            }
            // magnetism, |m|
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
            for (int i = nrxx; i < 2 * nrxx; ++i)
            {
                out[i] = in[i] + this->mixing_beta_mag * sres[i];
            }
        };
        this->mixing->push_data(this->rho_mdata, rhor_in, rhor_out, screen, twobeta_mix, true);
        auto inner_product
            = std::bind(&Charge_Mixing::inner_product_real, this, std::placeholders::_1, std::placeholders::_2);
        this->mixing->cal_coef(this->rho_mdata, inner_product);
        this->mixing->mix_data(this->rho_mdata, rhor_out);

        // use new |m| and angle to update {mx, my, mz}
        for (int ir = 0; ir < nrxx; ir++)
        {
            chr->rho[0][ir] = rho_magabs[ir]; // rho
			double norm = std::sqrt(chr->rho[1][ir] * chr->rho[1][ir] 
					+ chr->rho[2][ir] * chr->rho[2][ir] 
					+ chr->rho[3][ir] * chr->rho[3][ir]);

			if (norm < 1e-10) 
			{ 
				continue;
			}
            double rescale_tmp = rho_magabs[nrxx + ir] / norm; 
            chr->rho[1][ir] *= rescale_tmp;
            chr->rho[2][ir] *= rescale_tmp;
            chr->rho[3][ir] *= rescale_tmp;
        }
        // delete
        delete[] rho_magabs;
        delete[] rho_magabs_save;
    }
    
    double *taur_out=nullptr;
    double *taur_in=nullptr;
    if ((XC_Functional::get_ked_flag()) && mixing_tau)
    {
        taur_in = chr->kin_r_save[0];
        taur_out = chr->kin_r[0];
        // Note: there is no kerker modification for tau because I'm not sure
        // if we should have it. If necessary we can try it in the future.
        this->mixing->push_data(this->tau_mdata, taur_in, taur_out, nullptr, false);

        this->mixing->mix_data(this->tau_mdata, taur_out);
    }

    ModuleBase::timer::end("Charge_Mixing", "mix_rho_real");
    return;
}


void Charge_Mixing::mix_rho(Charge* chr)
{
    ModuleBase::TITLE("Charge_Mixing", "mix_rho");
    ModuleBase::timer::start("Charge_Mixing", "mix_rho");

    const int nspin = PARAM.inp.nspin;
    assert(nspin==1 || nspin==2 || nspin==4);

    // the charge before mixing.
    const int nrxx = chr->rhopw->nrxx;
    std::vector<double> rho123(nspin * nrxx);
    for (int is = 0; is < nspin; ++is)
    {
        if (is == 0 || is == 3 || !PARAM.globalv.domag_z)
        {
            double* rho123_is = rho123.data() + is * nrxx;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
            for(int ir = 0 ; ir < nrxx ; ++ir)
            {
                rho123_is[ir] = chr->rho[is][ir];
            }
        }
    }
    std::vector<double> kin_r123;
    if ((XC_Functional::get_ked_flag()) && mixing_tau)
    {
        kin_r123.resize(nspin * nrxx);
        for (int is = 0; is < nspin; ++is)
        {
            double* kin_r123_is = kin_r123.data() + is * nrxx;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
            for(int ir = 0 ; ir < nrxx ; ++ir)
            {
                kin_r123_is[ir] = chr->kin_r[is][ir];
            }
        }
    }
    // --------------------Mixing Body--------------------
    if (PARAM.inp.scf_thr_type == 1)
    {
        mix_rho_recip(chr);
    }
    else if (PARAM.inp.scf_thr_type == 2)
    {
        mix_rho_real(chr);
    }
    // ---------------------------------------------------

    // mohan add 2012-06-05
    // rho_save is the charge before mixing
    for (int is = 0; is < nspin; ++is)
    {
        if (is == 0 || is == 3 || !PARAM.globalv.domag_z)
        {
            double* rho123_is = rho123.data() + is * nrxx;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
            for(int ir = 0 ; ir < nrxx ; ++ir)
            {
                chr->rho_save[is][ir] = rho123_is[ir];
            }
        }
    }

    if ((XC_Functional::get_ked_flag()) && mixing_tau)
    {
        for (int is = 0; is < nspin; ++is)
        {
            double* kin_r123_is = kin_r123.data() + is * nrxx;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
            for(int ir = 0 ; ir < nrxx ; ++ir)
            {
                chr->kin_r_save[is][ir] = kin_r123_is[ir];
            }
        }
    }

	if (new_e_iteration) 
	{
		new_e_iteration = false;
	}

	ModuleBase::timer::end("Charge_Mixing", "mix_rho");
    return;
}
