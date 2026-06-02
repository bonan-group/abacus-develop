#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_io/module_parameter/parameter.h"
#include "structure_factor.h"
#include "source_base/constants.h"
#include "source_base/math_bspline.h"
#include "source_base/memory_recorder.h"
#include "source_base/timer.h"
#include "source_base/libm/libm.h"
#if defined(__CUDA) || defined(__UT_USE_CUDA)
#include "source_pw/module_pwdft/kernels/structure_factor_op.h"
#endif

#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

Structure_Factor::Structure_Factor()
{
    // LCAO basis doesn't support GPU acceleration on this function currently.
    if(PARAM.inp.basis_type == "pw")
    {
        this->device = PARAM.inp.device;
    }
}

Structure_Factor::~Structure_Factor()
{
#if defined(__CUDA) || defined(__UT_USE_CUDA)
    this->free_gpu_sf_memory();
#endif
    if (device == "gpu")
    {
        delmem_cd_op()(this->c_eigts1);
        delmem_cd_op()(this->c_eigts2);
        delmem_cd_op()(this->c_eigts3);
        delmem_zd_op()(this->z_eigts1);
        delmem_zd_op()(this->z_eigts2);
        delmem_zd_op()(this->z_eigts3);
    }
    else
    {
        delmem_ch_op()(this->c_eigts1);
        delmem_ch_op()(this->c_eigts2);
        delmem_ch_op()(this->c_eigts3);
        // There's no need to delete double precision pointers while in a CPU environment.
    }
}

// called in input.cpp
void Structure_Factor::set(const ModulePW::PW_Basis* rho_basis_in, const int& nbspline_in)
{
    ModuleBase::TITLE("Structure_Factor","set");
    this->rho_basis = rho_basis_in;
    this->nbspline = nbspline_in;
    return;
}

// Peize Lin optimize and add OpenMP 2021.04.01
//  Calculate structure factor
void Structure_Factor::setup(const UnitCell* Ucell, const Parallel_Grid& pgrid, const ModulePW::PW_Basis* rho_basis)
{
    ModuleBase::TITLE("Structure_Factor","setup");
    ModuleBase::timer::start("Structure_Factor","setup");

    this->ucell = Ucell;
    this->strucFac.create(Ucell->ntype, rho_basis->npw);
    ModuleBase::Memory::record("SF::strucFac", sizeof(std::complex<double>) * Ucell->ntype*rho_basis->npw);

//	std::string outstr;
//	outstr = PARAM.globalv.global_out_dir + "strucFac.dat"; 
//	std::ofstream ofs( outstr.c_str() ) ;
	bool usebspline;
	if(nbspline > 0) 
	{   
		usebspline = true;
	} 
	else 
	{    
		usebspline = false;
	}
    
    if(usebspline)
    {
        nbspline = int((nbspline+1)/2)*2; // nbspline must be a positive even number.
        this->bspline_sf(nbspline, Ucell, pgrid, rho_basis);
    }
    else
    {
#if defined(__CUDA) || defined(__UT_USE_CUDA)
	    if (device == "gpu")
        {
            this->compute_struc_fac_gpu(Ucell, rho_basis);
        }
        else
#endif
        {
            this->compute_struc_fac_cpu(Ucell, rho_basis);
        }
    }

//	ofs.close();

    this->eigts1.create(Ucell->nat, 2*rho_basis->nx + 1);
    this->eigts2.create(Ucell->nat, 2*rho_basis->ny + 1);
    this->eigts3.create(Ucell->nat, 2*rho_basis->nz + 1);

    ModuleBase::Memory::record("SF::eigts123",sizeof(std::complex<double>) 
    * (Ucell->nat*2 * (rho_basis->nx + rho_basis->ny + rho_basis->nz) + 3));

#if defined(__CUDA) || defined(__UT_USE_CUDA)
    if (device == "gpu")
    {
        this->compute_eigts_gpu(Ucell, rho_basis);
    }
    else
#endif
    {
        this->compute_eigts_cpu(Ucell, rho_basis);
    }
    
    if (device == "gpu") {
        if (PARAM.globalv.has_float_data) {
            resmem_cd_op()(this->c_eigts1, Ucell->nat * (2 * rho_basis->nx + 1));
            resmem_cd_op()(this->c_eigts2, Ucell->nat * (2 * rho_basis->ny + 1));
            resmem_cd_op()(this->c_eigts3, Ucell->nat * (2 * rho_basis->nz + 1));
            castmem_z2c_h2d_op()(this->c_eigts1, this->eigts1.c, Ucell->nat * (2 * rho_basis->nx + 1));
            castmem_z2c_h2d_op()(this->c_eigts2, this->eigts2.c, Ucell->nat * (2 * rho_basis->ny + 1));
            castmem_z2c_h2d_op()(this->c_eigts3, this->eigts3.c, Ucell->nat * (2 * rho_basis->nz + 1));
        }
        if (this->z_eigts1 == nullptr)
        {
            resmem_zd_op()(this->z_eigts1, Ucell->nat * (2 * rho_basis->nx + 1));
            resmem_zd_op()(this->z_eigts2, Ucell->nat * (2 * rho_basis->ny + 1));
            resmem_zd_op()(this->z_eigts3, Ucell->nat * (2 * rho_basis->nz + 1));
            syncmem_z2z_h2d_op()(this->z_eigts1, this->eigts1.c, Ucell->nat * (2 * rho_basis->nx + 1));
            syncmem_z2z_h2d_op()(this->z_eigts2, this->eigts2.c, Ucell->nat * (2 * rho_basis->ny + 1));
            syncmem_z2z_h2d_op()(this->z_eigts3, this->eigts3.c, Ucell->nat * (2 * rho_basis->nz + 1));
        }
    }
    else {
        if (PARAM.globalv.has_float_data) {
            resmem_ch_op()(this->c_eigts1, Ucell->nat * (2 * rho_basis->nx + 1));
            resmem_ch_op()(this->c_eigts2, Ucell->nat * (2 * rho_basis->ny + 1));
            resmem_ch_op()(this->c_eigts3, Ucell->nat * (2 * rho_basis->nz + 1));
            castmem_z2c_h2h_op()(this->c_eigts1, this->eigts1.c, Ucell->nat * (2 * rho_basis->nx + 1));
            castmem_z2c_h2h_op()(this->c_eigts2, this->eigts2.c, Ucell->nat * (2 * rho_basis->ny + 1));
            castmem_z2c_h2h_op()(this->c_eigts3, this->eigts3.c, Ucell->nat * (2 * rho_basis->nz + 1));
        }
        this->z_eigts1 = this->eigts1.c;
        this->z_eigts2 = this->eigts2.c;
        this->z_eigts3 = this->eigts3.c;
        // There's no need to delete double precision pointers while in a CPU environment.
    }
    ModuleBase::timer::end("Structure_Factor","setup");
    return;
}

//
//DESCRIPTION:
//    Calculate structure factor with Cardinal B-spline interpolation
//    Ref: J. Chem. Phys. 103, 8577 (1995)
//    qianrui create 2021-9-17
//INPUT LIST:
//    norder: the order of Cardinal B-spline base functions
//FURTHER OPTIMIZATION:
//    1. Use "r2c" fft
//    2. Add parallel algorithm for fftw or na loop
//
void Structure_Factor::bspline_sf(const int norder,
                                  const UnitCell* Ucell,
                                  const Parallel_Grid& pgrid,
                                  const ModulePW::PW_Basis* rho_basis)
{
    double *r = new double [rho_basis->nxyz]; 
    double *tmpr = new double[rho_basis->nrxx];
    double *zpiece = new double[rho_basis->nxy];
    std::complex<double> *b1 = new std::complex<double> [rho_basis->nx];
    std::complex<double> *b2 = new std::complex<double> [rho_basis->ny];
    std::complex<double> *b3 = new std::complex<double> [rho_basis->nz];

    for (int it=0; it<Ucell->ntype; it++)
    {
		const int na = Ucell->atoms[it].na;
		const ModuleBase::Vector3<double> * const taud = Ucell->atoms[it].taud.data();
        ModuleBase::GlobalFunc::ZEROS(r,rho_basis->nxyz);

        //A parallel algorithm can be added in the future.
#ifdef _OPENMP
		#pragma omp parallel for
#endif
        for(int ia = 0 ; ia < na ; ++ia)
        {
            double gridx = taud[ia].x * rho_basis->nx;
            double gridy = taud[ia].y * rho_basis->ny;
            double gridz = taud[ia].z * rho_basis->nz;
            double dx = gridx - floor(gridx);
            double dy = gridy - floor(gridy);
            double dz = gridz - floor(gridz);
            //I'm not sure if there is a mod function for double data

            ModuleBase::Bspline bsx, bsy, bsz;
            bsx.init(norder, 1, 0);
            bsy.init(norder, 1, 0);
            bsz.init(norder, 1, 0);
            bsx.getbspline(dx);
            bsy.getbspline(dy);
            bsz.getbspline(dz);

            for(int iz = 0 ; iz <= norder ; ++iz)
            {
                int icz = int(rho_basis->nz*10-iz+floor(gridz))%rho_basis->nz;
                for(int iy = 0 ; iy <= norder ; ++iy)
                {
                    int icy = int(rho_basis->ny*10-iy+floor(gridy))%rho_basis->ny;
                    for(int ix = 0 ; ix <= norder ; ++ix )
                    {
                        int icx = int(rho_basis->nx*10-ix+floor(gridx))%rho_basis->nx;
#ifdef _OPENMP
		                #pragma omp atomic
#endif
                        r[icz*rho_basis->ny*rho_basis->nx + icx*rho_basis->ny + icy] += bsz.bezier_ele(iz) 
                                                 * bsy.bezier_ele(iy) 
                                                 * bsx.bezier_ele(ix); 
                    }
                }
            }
        }
        
        //distribute data to different processors for UFFT
        //---------------------------------------------------
        for(int iz = 0; iz < rho_basis->nz; iz++)
	    {
	    	if(GlobalV::MY_RANK==0)
	    	{
#ifdef _OPENMP
		    #pragma omp parallel for schedule(static, 512)
#endif
	    		for(int ir = 0; ir < rho_basis->nxy; ir++)
	    		{
	    			zpiece[ir] = r[iz*rho_basis->nxy + ir];
	    		}
	    	}
        
        #ifdef __MPI
	    	pgrid.zpiece_to_all(zpiece, iz, tmpr);
        #else
	    	// Serial build: the whole real-space grid is local, so there is no
	    	// pool to scatter to. zpiece_to_all() is MPI-only, which otherwise
	    	// leaves tmpr uninitialized -> garbage structure factor and a wrong
	    	// total energy. Fill tmpr directly, using the SAME real-space layout
	    	// as zpiece_to_all's serial path: rho[ir*nczp + znow], i.e. xy index
	    	// outer and z innermost (nczp == nz, znow == iz when serial).
	    	for(int ir = 0; ir < rho_basis->nxy; ir++)
	    	{
	    		tmpr[ir*rho_basis->nz + iz] = zpiece[ir];
	    	}
        #endif
        
	    }
        //---------------------------------------------------

        //It should be optimized with r2c
        rho_basis->real2recip(tmpr, &strucFac(it,0));
        this->bsplinecoef(b1,b2,b3,rho_basis->nx, rho_basis->ny, rho_basis->nz, norder);
#ifdef _OPENMP
		#pragma omp parallel for schedule(static, 128)
#endif
        for(int ig = 0 ; ig < rho_basis->npw ; ++ig)
        {
           int idx = int(rho_basis->gdirect[ig].x+0.1+rho_basis->nx)%rho_basis->nx;
           int idy = int(rho_basis->gdirect[ig].y+0.1+rho_basis->ny)%rho_basis->ny;
           int idz = int(rho_basis->gdirect[ig].z+0.1+rho_basis->nz)%rho_basis->nz;
           strucFac(it,ig) *= ( b1[idx] * b2[idy] * b3[idz] * double(rho_basis->nxyz) );
        }
    }   
    delete[] r;
    delete[] tmpr;
    delete[] zpiece; 
    delete[] b1;
    delete[] b2;
    delete[] b3;

    return;
}

void Structure_Factor::bsplinecoef(std::complex<double> *b1, std::complex<double> *b2, std::complex<double> *b3, 
                        const int nx, const int ny, const int nz, const int norder)
{
    const std::complex<double> ci_tpi = ModuleBase::NEG_IMAG_UNIT * ModuleBase::TWO_PI;
    ModuleBase::Bspline bsp;
    bsp.init(norder, 1, 0);
    bsp.getbspline(1.0);
#ifdef _OPENMP
#pragma omp parallel
{
	#pragma omp for schedule(static, 16)
#endif
    for(int ix = 0 ; ix < nx ; ++ix)
    {
        std::complex<double> fracx=0;
        for(int io = 0 ; io < norder - 1 ; ++io)
        {
            fracx += bsp.bezier_ele(io)*ModuleBase::libm::exp(ci_tpi*double(ix)/double(nx)*double(io));
        }
        b1[ix] = ModuleBase::libm::exp(ci_tpi*double(norder*ix)/double(nx))/fracx;
    }
#ifdef _OPENMP
	#pragma omp for schedule(static, 16)
#endif
    for(int iy = 0 ; iy < ny ; ++iy)
    {
        std::complex<double> fracy=0;
        for(int io = 0 ; io < norder - 1 ; ++io)
        {
            fracy += bsp.bezier_ele(io)*ModuleBase::libm::exp(ci_tpi*double(iy)/double(ny)*double(io));
        }
        b2[iy] = ModuleBase::libm::exp(ci_tpi*double(norder*iy)/double(ny))/fracy;
    }
#ifdef _OPENMP
	#pragma omp for schedule(static, 16)
#endif
    for(int iz = 0 ; iz < nz ; ++iz)
    {
        std::complex<double> fracz=0;
        for(int io = 0 ; io < norder - 1 ; ++io)
        {
            fracz += bsp.bezier_ele(io)*ModuleBase::libm::exp(ci_tpi*double(iz)/double(nz)*double(io));
        }
        b3[iz] = ModuleBase::libm::exp(ci_tpi*double(norder*iz)/double(nz))/fracz;
    }
#ifdef _OPENMP
}
#endif
}

void Structure_Factor::compute_struc_fac_cpu(const UnitCell* Ucell, const ModulePW::PW_Basis* rho_basis)
{
    const std::complex<double> ci_tpi = ModuleBase::NEG_IMAG_UNIT * ModuleBase::TWO_PI;
    for (int it = 0; it < Ucell->ntype; ++it)
    {
        const int na = Ucell->atoms[it].na;
        const ModuleBase::Vector3<double>* const tau = Ucell->atoms[it].tau.data();
        const int npw = rho_basis->npw;
        const ModuleBase::Vector3<double>* const gcar = rho_basis->gcar;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int ig = 0; ig < npw; ++ig)
        {
            const ModuleBase::Vector3<double> gcar_ig = gcar[ig];
            std::complex<double> sum_phase = ModuleBase::ZERO;
            for (int ia = 0; ia < na; ++ia)
            {
                sum_phase += ModuleBase::libm::exp(ci_tpi * (gcar_ig * tau[ia]));
            }
            this->strucFac(it, ig) = sum_phase;
        }
    }
}

void Structure_Factor::compute_eigts_cpu(const UnitCell* Ucell, const ModulePW::PW_Basis* rho_basis)
{
    const std::complex<double> ci_tpi = ModuleBase::NEG_IMAG_UNIT * ModuleBase::TWO_PI;
    int inat = 0;
    for (int it = 0; it < Ucell->ntype; ++it)
    {
        for (int ia = 0; ia < Ucell->atoms[it].na; ++ia)
        {
            const ModuleBase::Vector3<double> gtau = Ucell->G * Ucell->atoms[it].tau[ia];
#ifdef _OPENMP
#pragma omp parallel
            {
#pragma omp for schedule(static, 16)
#endif
                for (int n1 = -rho_basis->nx; n1 <= rho_basis->nx; ++n1)
                {
                    this->eigts1(inat, n1 + rho_basis->nx) = ModuleBase::libm::exp(ci_tpi * (n1 * gtau.x));
                }
#ifdef _OPENMP
#pragma omp for schedule(static, 16)
#endif
                for (int n2 = -rho_basis->ny; n2 <= rho_basis->ny; ++n2)
                {
                    this->eigts2(inat, n2 + rho_basis->ny) = ModuleBase::libm::exp(ci_tpi * (n2 * gtau.y));
                }
#ifdef _OPENMP
#pragma omp for schedule(static, 16)
#endif
                for (int n3 = -rho_basis->nz; n3 <= rho_basis->nz; ++n3)
                {
                    this->eigts3(inat, n3 + rho_basis->nz) = ModuleBase::libm::exp(ci_tpi * (n3 * gtau.z));
                }
#ifdef _OPENMP
            }
#endif
            ++inat;
        }
    }
}

#if defined(__CUDA) || defined(__UT_USE_CUDA)
void Structure_Factor::allocate_gpu_sf_memory(const UnitCell* Ucell, const ModulePW::PW_Basis* rho_basis)
{
    using resmem_int_op = base_device::memory::resize_memory_op<int, base_device::DEVICE_GPU>;
    using syncmem_int_h2d_op
        = base_device::memory::synchronize_memory_op<int, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;

    if (this->tau_d != nullptr
        && (this->gpu_nat != Ucell->nat || this->gpu_ntype != Ucell->ntype || this->gpu_npw != rho_basis->npw))
    {
        this->free_gpu_sf_memory();
    }

    std::vector<int> atom_index_h(Ucell->ntype + 1, 0);
    for (int it = 0; it < Ucell->ntype; ++it)
    {
        atom_index_h[it + 1] = atom_index_h[it] + Ucell->atoms[it].na;
    }

    std::vector<double> tau_h(Ucell->nat * 3);
    std::vector<double> gtau_h(Ucell->nat * 3);
    int iat = 0;
    for (int it = 0; it < Ucell->ntype; ++it)
    {
        for (int ia = 0; ia < Ucell->atoms[it].na; ++ia)
        {
            const ModuleBase::Vector3<double>& tau = Ucell->atoms[it].tau[ia];
            tau_h[iat * 3 + 0] = tau.x;
            tau_h[iat * 3 + 1] = tau.y;
            tau_h[iat * 3 + 2] = tau.z;

            const ModuleBase::Vector3<double> gtau = Ucell->G * tau;
            gtau_h[iat * 3 + 0] = gtau.x;
            gtau_h[iat * 3 + 1] = gtau.y;
            gtau_h[iat * 3 + 2] = gtau.z;
            ++iat;
        }
    }

    std::vector<double> gcar_h(rho_basis->npw * 3);
    for (int ig = 0; ig < rho_basis->npw; ++ig)
    {
        gcar_h[ig * 3 + 0] = rho_basis->gcar[ig].x;
        gcar_h[ig * 3 + 1] = rho_basis->gcar[ig].y;
        gcar_h[ig * 3 + 2] = rho_basis->gcar[ig].z;
    }

    if (this->tau_d == nullptr)
    {
        resmem_dd_op()(this->tau_d, Ucell->nat * 3, "SF::tau_d");
        resmem_int_op()(this->atom_index_d, Ucell->ntype + 1, "SF::atom_index_d");
        resmem_dd_op()(this->gcar_d, rho_basis->npw * 3, "SF::gcar_d");
        resmem_dd_op()(this->gtau_d, Ucell->nat * 3, "SF::gtau_d");
        resmem_zd_op()(this->strucFac_d, Ucell->ntype * rho_basis->npw, "SF::strucFac_d");
        this->gpu_nat = Ucell->nat;
        this->gpu_ntype = Ucell->ntype;
        this->gpu_npw = rho_basis->npw;
    }

    syncmem_d2d_h2d_op()(this->tau_d, tau_h.data(), Ucell->nat * 3);
    syncmem_int_h2d_op()(this->atom_index_d, atom_index_h.data(), Ucell->ntype + 1);
    syncmem_d2d_h2d_op()(this->gcar_d, gcar_h.data(), rho_basis->npw * 3);
    syncmem_d2d_h2d_op()(this->gtau_d, gtau_h.data(), Ucell->nat * 3);
}

void Structure_Factor::free_gpu_sf_memory()
{
    using delmem_int_op = base_device::memory::delete_memory_op<int, base_device::DEVICE_GPU>;
    delmem_dd_op()(this->tau_d);
    delmem_int_op()(this->atom_index_d);
    delmem_dd_op()(this->gcar_d);
    delmem_dd_op()(this->gtau_d);
    delmem_zd_op()(this->strucFac_d);
    this->tau_d = nullptr;
    this->atom_index_d = nullptr;
    this->gcar_d = nullptr;
    this->gtau_d = nullptr;
    this->strucFac_d = nullptr;
    this->gpu_nat = 0;
    this->gpu_ntype = 0;
    this->gpu_npw = 0;
}

void Structure_Factor::compute_struc_fac_gpu(const UnitCell* Ucell, const ModulePW::PW_Basis* rho_basis)
{
    this->allocate_gpu_sf_memory(Ucell, rho_basis);
    structure_factor_op::compute_struc_fac_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                                 Ucell->ntype,
                                                                                 this->tau_d,
                                                                                 this->atom_index_d,
                                                                                 rho_basis->npw,
                                                                                 this->gcar_d,
                                                                                 ModuleBase::TWO_PI,
                                                                                 this->strucFac_d);
    syncmem_z2z_d2h_op()(this->strucFac.c, this->strucFac_d, Ucell->ntype * rho_basis->npw);
}

void Structure_Factor::compute_eigts_gpu(const UnitCell* Ucell, const ModulePW::PW_Basis* rho_basis)
{
    this->allocate_gpu_sf_memory(Ucell, rho_basis);
    resmem_zd_op()(this->z_eigts1, Ucell->nat * (2 * rho_basis->nx + 1), "SF::z_eigts1");
    resmem_zd_op()(this->z_eigts2, Ucell->nat * (2 * rho_basis->ny + 1), "SF::z_eigts2");
    resmem_zd_op()(this->z_eigts3, Ucell->nat * (2 * rho_basis->nz + 1), "SF::z_eigts3");
    structure_factor_op::compute_eigts_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                             Ucell->nat,
                                                                             this->gtau_d,
                                                                             rho_basis->nx,
                                                                             rho_basis->ny,
                                                                             rho_basis->nz,
                                                                             ModuleBase::TWO_PI,
                                                                             this->z_eigts1,
                                                                             this->z_eigts2,
                                                                             this->z_eigts3);
    syncmem_z2z_d2h_op()(this->eigts1.c, this->z_eigts1, Ucell->nat * (2 * rho_basis->nx + 1));
    syncmem_z2z_d2h_op()(this->eigts2.c, this->z_eigts2, Ucell->nat * (2 * rho_basis->ny + 1));
    syncmem_z2z_d2h_op()(this->eigts3.c, this->z_eigts3, Ucell->nat * (2 * rho_basis->nz + 1));
}
#endif

template <>
std::complex<float> * Structure_Factor::get_eigts1_data() const
{
    return this->c_eigts1;
}
template <>
std::complex<double> * Structure_Factor::get_eigts1_data() const
{
    return this->z_eigts1;
}

template <>
std::complex<float> * Structure_Factor::get_eigts2_data() const
{
    return this->c_eigts2;
}
template <>
std::complex<double> * Structure_Factor::get_eigts2_data() const
{
    return this->z_eigts2;
}

template <>
std::complex<float> * Structure_Factor::get_eigts3_data() const
{
    return this->c_eigts3;
}
template <>
std::complex<double> * Structure_Factor::get_eigts3_data() const
{
    return this->z_eigts3;
}
