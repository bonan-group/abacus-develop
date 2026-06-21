#include "stress_func.h"
#include "source_base/parallel_reduce.h"
#include "source_hamilt/module_ewald/H_Ewald_pw.h"
#include "source_base/timer.h"
#include "source_base/tool_threading.h"
#include "source_base/libm/libm.h"
#include "source_io/module_parameter/parameter.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
template <typename FPTYPE>
void finalize_ewa_stress(ModuleBase::matrix& sigma, FPTYPE sdewald)
{
    for (int l = 0; l < 3; l++)
    {
        sigma(l, l) += sdewald;
    }
    for (int l = 0; l < 3; l++)
    {
        for (int m = 0; m < l + 1; m++)
        {
            sigma(l, m) = -sigma(l, m);
            Parallel_Reduce::reduce_pool(sigma(l, m));
        }
    }
    for (int l = 0; l < 3; l++)
    {
        for (int m = 0; m < l + 1; m++)
        {
            sigma(m, l) = sigma(l, m);
        }
    }
}
} // namespace

//calcualte the Ewald stress term in PW and LCAO
template<typename FPTYPE, typename Device>
void Stress_Func<FPTYPE, Device>::stress_ewa(const UnitCell& ucell,
											 ModuleBase::matrix& sigma, 
											 ModulePW::PW_Basis* rho_basis, 
											 const bool is_pw)
{
    ModuleBase::TITLE("Stress","stress_ewa");
    ModuleBase::timer::start("Stress","stress_ewa");

    FPTYPE charge=0;
    for(int it=0; it < ucell.ntype; it++)
	{
		charge = charge + ucell.atoms[it].ncpp.zv * ucell.atoms[it].na;
	}
    //choose alpha in order to have convergence in the sum over G
    //upperbound is a safe upper bound for the error ON THE ENERGY

    FPTYPE alpha=2.9;
    FPTYPE upperbound=0.0;

	do{
		alpha-=0.1;
		if(alpha==0.0)
		{
			ModuleBase::WARNING_QUIT("stres_ew", "optimal alpha not found");
		}
		upperbound =ModuleBase::e2 * pow(charge,2) * 
         sqrt( 2 * alpha / (ModuleBase::TWO_PI)) 
         * erfc(sqrt(ucell.tpiba2 * rho_basis->ggecut / 4.0 / alpha));
	}
    while(upperbound>1e-7);

    //G-space sum here
    //Determine if this processor contains G=0 and set the constant term 
    FPTYPE sdewald=0.0;
	const int ig0 = rho_basis->ig_gge0;
    if( ig0 >= 0)
	{
       sdewald = (ModuleBase::TWO_PI) * ModuleBase::e2 / 4.0 / alpha * pow(charge/ucell.omega,2);
    }
    else 
	{
       sdewald = 0.0;
    }

    //sdewald is the diagonal term 

    FPTYPE fact=1.0;
	if (PARAM.globalv.gamma_only_pw && is_pw) 
	{
		fact=2.0;
	}
//    else fact=1.0;

#if ((defined __CUDA) || (defined __UT_USE_CUDA))
    this->device = base_device::get_device_type(this->ctx);
    if (this->device == base_device::GpuDevice)
    {
        std::vector<FPTYPE> tau(ucell.nat * 3);
        std::vector<FPTYPE> atom_z(ucell.nat);
        int iat = 0;
        for (int it = 0; it < ucell.ntype; ++it)
        {
            const FPTYPE zv = ucell.atoms[it].ncpp.zv;
            for (int ia = 0; ia < ucell.atoms[it].na; ++ia)
            {
                tau[iat * 3] = ucell.atoms[it].tau[ia].x;
                tau[iat * 3 + 1] = ucell.atoms[it].tau[ia].y;
                tau[iat * 3 + 2] = ucell.atoms[it].tau[ia].z;
                atom_z[iat] = zv;
                ++iat;
            }
        }
        std::vector<FPTYPE> gcar(rho_basis->npw * 3);
        std::vector<FPTYPE> gg(rho_basis->npw);
        for (int ig = 0; ig < rho_basis->npw; ++ig)
        {
            gcar[ig * 3] = rho_basis->gcar[ig].x;
            gcar[ig * 3 + 1] = rho_basis->gcar[ig].y;
            gcar[ig * 3 + 2] = rho_basis->gcar[ig].z;
            gg[ig] = rho_basis->gg[ig];
        }
        FPTYPE rmax = 0.0;
        int nm1 = 0;
        int nm2 = 0;
        int nm3 = 0;
        const int do_real_space = ig0 >= 0 ? 1 : 0;
        if (do_real_space)
        {
            const FPTYPE sqa = sqrt(alpha);
            rmax = 4.0 / sqa / ucell.lat0;
            FPTYPE bg1[3] = {ucell.G.e11, ucell.G.e12, ucell.G.e13};
            nm1 = static_cast<int>(sqrt(bg1[0] * bg1[0] + bg1[1] * bg1[1] + bg1[2] * bg1[2]) * rmax + 2);
            bg1[0] = ucell.G.e21;
            bg1[1] = ucell.G.e22;
            bg1[2] = ucell.G.e23;
            nm2 = static_cast<int>(sqrt(bg1[0] * bg1[0] + bg1[1] * bg1[1] + bg1[2] * bg1[2]) * rmax + 2);
            bg1[0] = ucell.G.e31;
            bg1[1] = ucell.G.e32;
            bg1[2] = ucell.G.e33;
            nm3 = static_cast<int>(sqrt(bg1[0] * bg1[0] + bg1[1] * bg1[1] + bg1[2] * bg1[2]) * rmax + 2);
        }
        std::vector<FPTYPE> latvec = {ucell.latvec.e11,
                                      ucell.latvec.e12,
                                      ucell.latvec.e13,
                                      ucell.latvec.e21,
                                      ucell.latvec.e22,
                                      ucell.latvec.e23,
                                      ucell.latvec.e31,
                                      ucell.latvec.e32,
                                      ucell.latvec.e33};

        FPTYPE* tau_d = nullptr;
        FPTYPE* atom_z_d = nullptr;
        FPTYPE* gcar_d = nullptr;
        FPTYPE* gg_d = nullptr;
        FPTYPE* latvec_d = nullptr;
        FPTYPE* stress_d = nullptr;
        resmem_var_op()(tau_d, tau.size());
        resmem_var_op()(atom_z_d, atom_z.size());
        resmem_var_op()(gcar_d, gcar.size());
        resmem_var_op()(gg_d, gg.size());
        resmem_var_op()(latvec_d, latvec.size());
        resmem_var_op()(stress_d, 7);
        syncmem_var_h2d_op()(tau_d, tau.data(), tau.size());
        syncmem_var_h2d_op()(atom_z_d, atom_z.data(), atom_z.size());
        syncmem_var_h2d_op()(gcar_d, gcar.data(), gcar.size());
        syncmem_var_h2d_op()(gg_d, gg.data(), gg.size());
        syncmem_var_h2d_op()(latvec_d, latvec.data(), latvec.size());

        hamilt::cal_stress_ewa_op<FPTYPE, Device>()(this->ctx,
                                                    ucell.nat,
                                                    rho_basis->npw,
                                                    ig0,
                                                    do_real_space,
                                                    nm1,
                                                    nm2,
                                                    nm3,
                                                    alpha,
                                                    ucell.omega,
                                                    ucell.tpiba2,
                                                    ucell.lat0,
                                                    fact,
                                                    rmax,
                                                    charge,
                                                    tau_d,
                                                    atom_z_d,
                                                    gcar_d,
                                                    gg_d,
                                                    latvec_d,
                                                    stress_d);
        std::vector<FPTYPE> stress_h(7, 0.0);
        syncmem_var_d2h_op()(stress_h.data(), stress_d, stress_h.size());
        sigma(0, 0) += stress_h[0];
        sigma(1, 0) += stress_h[1];
        sigma(1, 1) += stress_h[2];
        sigma(2, 0) += stress_h[3];
        sigma(2, 1) += stress_h[4];
        sigma(2, 2) += stress_h[5];
        sdewald = stress_h[6];

        delmem_var_op()(tau_d);
        delmem_var_op()(atom_z_d);
        delmem_var_op()(gcar_d);
        delmem_var_op()(gg_d);
        delmem_var_op()(latvec_d);
        delmem_var_op()(stress_d);

        finalize_ewa_stress(sigma, sdewald);
        ModuleBase::timer::end("Stress", "stress_ewa");
        return;
    }
#endif

#pragma omp parallel
{
	ModuleBase::matrix local_sigma(3, 3);
	FPTYPE local_sdewald = 0;

    FPTYPE g2,g2a;
    FPTYPE arg;
    std::complex<FPTYPE> rhostar;
    FPTYPE sewald;

	#pragma omp for
    for(int ig = 0; ig < rho_basis->npw; ig++)
	{
		if(ig == ig0)  
		{
			continue;
		}
		g2 = rho_basis->gg[ig]* ucell.tpiba2;
		g2a = g2 /4.0/alpha;
		rhostar=std::complex<FPTYPE>(0.0,0.0);

		for(int it=0; it < ucell.ntype; it++)
		{
			for(int i=0; i<ucell.atoms[it].na; i++)
			{
				arg = (rho_basis->gcar[ig] * ucell.atoms[it].tau[i]) * (ModuleBase::TWO_PI);
				FPTYPE sinp, cosp;
                ModuleBase::libm::sincos(arg, &sinp, &cosp);
				rhostar = rhostar + std::complex<FPTYPE>(ucell.atoms[it].ncpp.zv * cosp,ucell.atoms[it].ncpp.zv * sinp);
			}
		}
		rhostar /= ucell.omega;
		sewald = fact* (ModuleBase::TWO_PI) * ModuleBase::e2 * ModuleBase::libm::exp(-g2a) / g2 * pow(std::abs(rhostar),2);
		local_sdewald -= sewald;
		for(int l=0;l<3;l++)
		{
			for(int m=0;m<l+1;m++)
			{
				local_sigma(l, m) += sewald * ucell.tpiba2 * 2.0 
					* rho_basis->gcar[ig][l] * rho_basis->gcar[ig][m] / g2 * (g2a + 1);
			}
		}
	}

    //R-space sum here (only for the processor that contains G=0) 
    int *irr=nullptr;
    ModuleBase::Vector3<FPTYPE> *r;
    FPTYPE *r2=nullptr;
    FPTYPE rr=0.0;
    ModuleBase::Vector3<FPTYPE> d_tau;
    FPTYPE r0[3];
    FPTYPE rmax=0.0;
    int nrm=0;
    FPTYPE fac=0.0;

	if(ig0 >= 0)
	{
		FPTYPE sqa = sqrt(alpha);
		FPTYPE sq8a_2pi = sqrt(8 * alpha / (ModuleBase::TWO_PI));
		rmax = 4.0/sqa/ucell.lat0;
		const int mxr = H_Ewald_pw::estimate_mxr(rmax, ucell.G);

		std::vector<ModuleBase::Vector3<FPTYPE>> r(mxr);
		std::vector<FPTYPE> r2(mxr);
		std::vector<int> irr(mxr);

		#pragma omp for
		for(long long ijat = 0; ijat < ucell.nat * ucell.nat; ijat++)
		{
			int it=0;
			int i=0;
			int jt=0;
			int j=0;
			ucell.ijat2iaitjajt(ijat, &i, &it, &j, &jt);
			if (ucell.atoms[it].na != 0 && ucell.atoms[jt].na != 0)
			{
				//calculate tau[na]-tau[nb]
				d_tau = ucell.atoms[it].tau[i] - ucell.atoms[jt].tau[j];
				//generates nearest-neighbors shells 
				H_Ewald_pw::rgen(d_tau, rmax, irr.data(), ucell.latvec, ucell.G, r.data(), r2.data(), mxr, nrm);
				for(int nr=0; nr<nrm; nr++)
				{
					rr=sqrt(r2[nr]) * ucell.lat0;
					fac = -ModuleBase::e2/2.0/ucell.omega*
                          pow(ucell.lat0,2)*ucell.atoms[it].ncpp.zv * ucell.atoms[jt].ncpp.zv 
                          / pow(rr,3) * (erfc(sqa*rr)+rr * sq8a_2pi *  ModuleBase::libm::exp(-alpha * pow(rr,2)));

					for(int l=0; l<3; l++)
					{
						for(int m=0; m<l+1; m++)
						{
							r0[0] = r[nr].x;
							r0[1] = r[nr].y;
							r0[2] = r[nr].z;
							local_sigma(l,m) += fac * r0[l] * r0[m];
						}//end m
					}//end l
				}//end nr
			}
		}
	}//end if

	#pragma omp critical(stress_ewa_reduce)
	{
		sdewald += local_sdewald;
		for(int l=0;l<3;l++)
		{
			for(int m=0;m<l+1;m++)
			{
				sigma(l,m) += local_sigma(l,m);
			}
		}
	}
}

    finalize_ewa_stress(sigma, sdewald);

	ModuleBase::timer::end("Stress","stress_ewa");

	return;
}

template class Stress_Func<double, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class Stress_Func<double, base_device::DEVICE_GPU>;
#endif
