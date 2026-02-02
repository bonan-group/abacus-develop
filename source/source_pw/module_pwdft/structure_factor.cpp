#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_io/module_parameter/parameter.h"
#include "structure_factor.h"
#include "source_base/constants.h"
#include "source_base/math_bspline.h"
#include "source_base/memory.h"
#include "source_base/timer.h"
#include "source_base/libm/libm.h"
#include "source_base/module_device/memory_op.h"
#include "source_pw/module_pwdft/kernels/structure_factor_op.h"


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
    // Free GPU memory for structure factor computation
    this->free_gpu_memory();

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
    ModuleBase::timer::tick("Structure_Factor","setup");

    const std::complex<double> ci_tpi = ModuleBase::NEG_IMAG_UNIT * ModuleBase::TWO_PI;
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
        // Choose GPU or CPU path for structure factor computation
        if (device == "gpu") {
#if defined(__CUDA) || defined(__ROCM)
            this->compute_struc_fac_gpu(Ucell, rho_basis);
#else
            // Fallback to CPU if GPU not available
            ModuleBase::GlobalFunc::OUT(GlobalV::ofs_warning, "GPU requested but not available, using CPU");
            this->compute_struc_fac_cpu(Ucell, rho_basis);
#endif
        } else {
            // CPU computation
            this->compute_struc_fac_cpu(Ucell, rho_basis);
        }
    }

//	ofs.close();

    // Allocate eigts arrays
    this->eigts1.create(Ucell->nat, 2*rho_basis->nx + 1);
    this->eigts2.create(Ucell->nat, 2*rho_basis->ny + 1);
    this->eigts3.create(Ucell->nat, 2*rho_basis->nz + 1);

    ModuleBase::Memory::record("SF::eigts123",sizeof(std::complex<double>)
    * (Ucell->nat*2 * (rho_basis->nx + rho_basis->ny + rho_basis->nz) + 3));

    // Compute eigts arrays (CPU or GPU)
    if (device == "gpu") {
#if defined(__CUDA) || defined(__ROCM)
        this->compute_eigts_gpu(Ucell, rho_basis);
#else
        // Fallback to CPU if GPU not available
        this->compute_eigts_cpu(Ucell, rho_basis);
#endif
    } else {
        this->compute_eigts_cpu(Ucell, rho_basis);
    }

    // Setup device pointers for eigts (after computation)
    if (device == "gpu") {
        if (PARAM.globalv.has_float_data) {
            // Already copied to CPU eigts1/2/3 in compute_eigts_gpu(), so just allocate and cast
            resmem_cd_op()(this->c_eigts1, Ucell->nat * (2 * rho_basis->nx + 1));
            resmem_cd_op()(this->c_eigts2, Ucell->nat * (2 * rho_basis->ny + 1));
            resmem_cd_op()(this->c_eigts3, Ucell->nat * (2 * rho_basis->nz + 1));
            castmem_z2c_h2d_op()(this->c_eigts1, this->eigts1.c, Ucell->nat * (2 * rho_basis->nx + 1));
            castmem_z2c_h2d_op()(this->c_eigts2, this->eigts2.c, Ucell->nat * (2 * rho_basis->ny + 1));
            castmem_z2c_h2d_op()(this->c_eigts3, this->eigts3.c, Ucell->nat * (2 * rho_basis->nz + 1));
        }
        // Note: z_eigts1/2/3 are already on GPU from compute_eigts_gpu()
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
    ModuleBase::timer::tick("Structure_Factor","setup");
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

// CPU implementation of structure factor computation
void Structure_Factor::compute_struc_fac_cpu(const UnitCell* Ucell, const ModulePW::PW_Basis* rho_basis)
{
    const std::complex<double> ci_tpi = ModuleBase::NEG_IMAG_UNIT * ModuleBase::TWO_PI;

    for (int it=0; it<Ucell->ntype; it++)
    {
        const int na = Ucell->atoms[it].na;
        const ModuleBase::Vector3<double> * const tau = Ucell->atoms[it].tau.data();
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int ig=0; ig<rho_basis->npw; ig++)
        {
            const ModuleBase::Vector3<double> gcar_ig = rho_basis->gcar[ig];
            std::complex<double> sum_phase = ModuleBase::ZERO;
            for (int ia=0; ia<na; ia++)
            {
                // e^{-i G*tau}
                sum_phase += ModuleBase::libm::exp( ci_tpi * (gcar_ig * tau[ia]) );
            }
            this->strucFac(it,ig) = sum_phase;
        }
    }
}

// CPU implementation of eigts computation
void Structure_Factor::compute_eigts_cpu(const UnitCell* Ucell, const ModulePW::PW_Basis* rho_basis)
{
    const std::complex<double> ci_tpi = ModuleBase::NEG_IMAG_UNIT * ModuleBase::TWO_PI;

    ModuleBase::Vector3<double> gtau;
    int inat = 0;
    for (int i = 0; i < Ucell->ntype; i++)
    {
        for (int j = 0; j < Ucell->atoms[i].na; j++)
        {
            gtau = Ucell->G * Ucell->atoms[i].tau[j];  // G^T · τ
#ifdef _OPENMP
#pragma omp parallel
{
            #pragma omp for schedule(static, 16)
#endif
            for (int n1 = -rho_basis->nx; n1 <= rho_basis->nx; n1++)
            {
                double arg = n1 * gtau.x;
                this->eigts1(inat, n1 + rho_basis->nx) = ModuleBase::libm::exp(ci_tpi * arg);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static, 16)
#endif
            for (int n2 = -rho_basis->ny; n2 <= rho_basis->ny; n2++)
            {
                double arg = n2 * gtau.y;
                this->eigts2(inat, n2 + rho_basis->ny) = ModuleBase::libm::exp(ci_tpi * arg);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static, 16)
#endif
            for (int n3 = -rho_basis->nz; n3 <= rho_basis->nz; n3++)
            {
                double arg = n3 * gtau.z;
                this->eigts3(inat, n3 + rho_basis->nz) = ModuleBase::libm::exp(ci_tpi * arg);
            }
#ifdef _OPENMP
}
#endif
            inat++;
        }
    }
}

// GPU implementation of structure factor computation
void Structure_Factor::compute_struc_fac_gpu(const UnitCell* Ucell, const ModulePW::PW_Basis* rho_basis)
{
#if defined(__CUDA) || defined(__ROCM)
    // Allocate and upload data to GPU
    this->allocate_gpu_memory(Ucell, rho_basis);

    // Call GPU kernel
    structure_factor_op::compute_struc_fac_op<double, base_device::DEVICE_GPU>()(
        nullptr,  // ctx not used in current implementation
        Ucell->ntype,
        this->tau_d,
        this->atom_index_d,
        rho_basis->npw,
        this->gcar_d,
        ModuleBase::TWO_PI,
        this->strucFac_d
    );

    // Copy result back to CPU
    syncmem_z2z_d2h_op()(
        this->strucFac.c,
        this->strucFac_d,
        Ucell->ntype * rho_basis->npw
    );
#endif
}

// GPU implementation of eigts computation
void Structure_Factor::compute_eigts_gpu(const UnitCell* Ucell, const ModulePW::PW_Basis* rho_basis)
{
#if defined(__CUDA) || defined(__ROCM)
    // Allocate GPU memory for eigts arrays
    resmem_zd_op()(this->z_eigts1, Ucell->nat * (2 * rho_basis->nx + 1));
    resmem_zd_op()(this->z_eigts2, Ucell->nat * (2 * rho_basis->ny + 1));
    resmem_zd_op()(this->z_eigts3, Ucell->nat * (2 * rho_basis->nz + 1));

    // Call GPU kernel (gtau_d was prepared in allocate_gpu_memory)
    structure_factor_op::compute_eigts_op<double, base_device::DEVICE_GPU>()(
        nullptr,  // ctx not used
        Ucell->nat,
        this->gtau_d,
        rho_basis->nx,
        rho_basis->ny,
        rho_basis->nz,
        ModuleBase::TWO_PI,
        this->z_eigts1,
        this->z_eigts2,
        this->z_eigts3
    );

    // Copy result back to CPU (for eigts1/2/3 ComplexMatrix)
    syncmem_z2z_d2h_op()(
        this->eigts1.c,
        this->z_eigts1,
        Ucell->nat * (2 * rho_basis->nx + 1)
    );
    syncmem_z2z_d2h_op()(
        this->eigts2.c,
        this->z_eigts2,
        Ucell->nat * (2 * rho_basis->ny + 1)
    );
    syncmem_z2z_d2h_op()(
        this->eigts3.c,
        this->z_eigts3,
        Ucell->nat * (2 * rho_basis->nz + 1)
    );
#endif
}

// Allocate and upload data to GPU memory
void Structure_Factor::allocate_gpu_memory(const UnitCell* Ucell, const ModulePW::PW_Basis* rho_basis)
{
#if defined(__CUDA) || defined(__ROCM)
    using resmem_int_op = base_device::memory::resize_memory_op<int, base_device::DEVICE_GPU>;
    using syncmem_int_op = base_device::memory::synchronize_memory_op<int, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;

    // Build atom_index array (cumulative atom counts per type)
    std::vector<int> atom_index_h(Ucell->ntype + 1);
    atom_index_h[0] = 0;
    for (int it = 0; it < Ucell->ntype; it++) {
        atom_index_h[it + 1] = atom_index_h[it] + Ucell->atoms[it].na;
    }

    // Flatten tau array (convert Vector3<double>[] to double[nat*3])
    std::vector<double> tau_h(Ucell->nat * 3);
    int iat = 0;
    for (int it = 0; it < Ucell->ntype; it++) {
        for (int ia = 0; ia < Ucell->atoms[it].na; ia++) {
            const ModuleBase::Vector3<double>& pos = Ucell->atoms[it].tau[ia];
            tau_h[iat * 3 + 0] = pos.x;
            tau_h[iat * 3 + 1] = pos.y;
            tau_h[iat * 3 + 2] = pos.z;
            iat++;
        }
    }

    // Flatten gcar array (convert Vector3<double>[] to double[ngm*3])
    std::vector<double> gcar_h(rho_basis->npw * 3);
    for (int ig = 0; ig < rho_basis->npw; ig++) {
        const ModuleBase::Vector3<double>& g = rho_basis->gcar[ig];
        gcar_h[ig * 3 + 0] = g.x;
        gcar_h[ig * 3 + 1] = g.y;
        gcar_h[ig * 3 + 2] = g.z;
    }

    // Prepare gtau array (G^T · τ for each atom)
    std::vector<double> gtau_h(Ucell->nat * 3);
    iat = 0;
    for (int it = 0; it < Ucell->ntype; it++) {
        for (int ia = 0; ia < Ucell->atoms[it].na; ia++) {
            ModuleBase::Vector3<double> gtau = Ucell->G * Ucell->atoms[it].tau[ia];
            gtau_h[iat * 3 + 0] = gtau.x;
            gtau_h[iat * 3 + 1] = gtau.y;
            gtau_h[iat * 3 + 2] = gtau.z;
            iat++;
        }
    }

    // Allocate GPU memory
    resmem_dd_op()(this->tau_d, Ucell->nat * 3);
    resmem_int_op()(this->atom_index_d, Ucell->ntype + 1);
    resmem_dd_op()(this->gcar_d, rho_basis->npw * 3);
    resmem_zd_op()(this->strucFac_d, Ucell->ntype * rho_basis->npw);
    resmem_dd_op()(this->gtau_d, Ucell->nat * 3);

    // Upload data to GPU
    syncmem_d2d_h2d_op()(this->tau_d, tau_h.data(), Ucell->nat * 3);
    syncmem_int_op()(this->atom_index_d, atom_index_h.data(), Ucell->ntype + 1);
    syncmem_d2d_h2d_op()(this->gcar_d, gcar_h.data(), rho_basis->npw * 3);
    syncmem_d2d_h2d_op()(this->gtau_d, gtau_h.data(), Ucell->nat * 3);
#endif
}

// Free GPU device memory
void Structure_Factor::free_gpu_memory()
{
#if defined(__CUDA) || defined(__ROCM)
    using delmem_int_op = base_device::memory::delete_memory_op<int, base_device::DEVICE_GPU>;

    if (this->tau_d != nullptr) {
        delmem_dd_op()(this->tau_d);
        this->tau_d = nullptr;
    }
    if (this->atom_index_d != nullptr) {
        delmem_int_op()(this->atom_index_d);
        this->atom_index_d = nullptr;
    }
    if (this->gcar_d != nullptr) {
        delmem_dd_op()(this->gcar_d);
        this->gcar_d = nullptr;
    }
    if (this->strucFac_d != nullptr) {
        delmem_zd_op()(this->strucFac_d);
        this->strucFac_d = nullptr;
    }
    if (this->gtau_d != nullptr) {
        delmem_dd_op()(this->gtau_d);
        this->gtau_d = nullptr;
    }
#endif
}

