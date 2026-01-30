#ifndef STRUCTURE_FACTOR_H 
#define STRUCTURE_FACTOR_H

#include "source_base/complexmatrix.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/unitcell.h"
#include "source_pw/module_pwdft/parallel_grid.h"
#include "source_psi/psi.h"

class Structure_Factor
{

public:
    Structure_Factor();
    ~Structure_Factor();
    void set(const ModulePW::PW_Basis* rho_basis_in, const int& nbspline_in);

    //===============================================
    // Part 4: G vectors in reciprocal FFT box
    //===============================================
  public:
    int nbspline=0;

	// structure factor (ntype, ngmc)
    ModuleBase::ComplexMatrix strucFac;

	void setup(const UnitCell* Ucell,
			const Parallel_Grid& pgrid,
			const ModulePW::PW_Basis* rho_basis); // Calculate structure factors

    /// calculate structure factors through Cardinal B-spline interpolation
    void bspline_sf(
        const int,
        const UnitCell* Ucell,
        const Parallel_Grid& pgrid,
        const ModulePW::PW_Basis* rho_basis); 

    void bsplinecoef(std::complex<double> *b1, std::complex<double> *b2, std::complex<double> *b3, 
                    const int nx, const int ny, const int nz, const int norder);


public:
	// phase of e^{-iG*tau_s}
    ModuleBase::ComplexMatrix eigts1; // dimension: [Ucell->nat, 2*this->ncx + 1] 
    ModuleBase::ComplexMatrix eigts2; // dimension: [Ucell->nat, 2*this->ncy + 1] 
    ModuleBase::ComplexMatrix eigts3; // dimension: [Ucell->nat, 2*this->ncz + 1]

    template <typename FPTYPE> std::complex<FPTYPE> * get_eigts1_data() const;
    template <typename FPTYPE> std::complex<FPTYPE> * get_eigts2_data() const;
    template <typename FPTYPE> std::complex<FPTYPE> * get_eigts3_data() const;

  public:
    // sf with k points
    std::complex<double>* get_sk(const int ik, const int it, const int ia, const ModulePW::PW_Basis_K* wfc_basis) const;
    template <typename FPTYPE, typename Device>

    void get_sk(Device* ctx, const int ik, const ModulePW::PW_Basis_K* wfc_basis, std::complex<FPTYPE>* sk) const;

    /// @brief Compute structure factor for a single atom (GPU-compatible version)
    /// @param ctx Device context
    /// @param ik k-point index
    /// @param iat Global atom index (0 to nat-1)
    /// @param wfc_basis PW basis
    /// @param sk Output buffer (size: npw)
    template <typename FPTYPE, typename Device>
    void get_sk(Device* ctx, const int ik, const int iat, const ModulePW::PW_Basis_K* wfc_basis, std::complex<FPTYPE>* sk) const;

    std::complex<double>* get_skq(int ik,
                                  int it,
                                  int ia,
                                  const ModulePW::PW_Basis_K* wfc_basis,
                                  ModuleBase::Vector3<double> q);

  private:

    const UnitCell* ucell=nullptr;
    std::complex<float> * c_eigts1 = nullptr;
    std::complex<float> * c_eigts2 = nullptr;
    std::complex<float> * c_eigts3 = nullptr;

    std::complex<double> * z_eigts1 = nullptr;
    std::complex<double> * z_eigts2 = nullptr;
    std::complex<double> * z_eigts3 = nullptr;

    const ModulePW::PW_Basis* rho_basis = nullptr;
    std::string device = "cpu";

    // GPU device memory for structure factor computation
    double* tau_d = nullptr;           // Atom positions [nat * 3] (flattened)
    int* atom_index_d = nullptr;       // Cumulative atom indices [ntype+1]
    double* gcar_d = nullptr;          // G-vector Cartesian coords [ngm * 3]
    std::complex<double>* strucFac_d = nullptr;  // Structure factors [ntype * ngm]

    // GPU device memory for eigts computation
    double* gtau_d = nullptr;          // G^T · τ for each atom [nat * 3] (gtau.x, gtau.y, gtau.z)

    // Helper methods for GPU computation
    void allocate_gpu_memory(const UnitCell* Ucell, const ModulePW::PW_Basis* rho_basis);
    void free_gpu_memory();
    void compute_struc_fac_cpu(const UnitCell* Ucell, const ModulePW::PW_Basis* rho_basis);
    void compute_struc_fac_gpu(const UnitCell* Ucell, const ModulePW::PW_Basis* rho_basis);
    void compute_eigts_cpu(const UnitCell* Ucell, const ModulePW::PW_Basis* rho_basis);
    void compute_eigts_gpu(const UnitCell* Ucell, const ModulePW::PW_Basis* rho_basis);
};
#endif //PlaneWave class
