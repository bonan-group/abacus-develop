#ifndef CHARGE_H
#define CHARGE_H

#include "source_base/complexmatrix.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/module_symmetry/symmetry.h"
// #include "source_estate/fp_energy.h"
#include "source_pw/module_pwdft/parallel_grid.h"
#include "source_base/module_device/types.h"
#include <string>
#include <type_traits>

//a forward declaration of UnitCell
class UnitCell;

// Electron Charge Density
class Charge
{

  public:

    Charge();
    ~Charge();

    //==========================================================
    // MEMBER VARIABLES :
    // init_chg : "atomic" or "file"
    // NAME : total number of electrons
    // NAME : rho (nspin,ncxyz), the charge density in real space
    // NAME : rho_save (nspin,ncxyz), for charge mixing
    // NAME : rhog, charge density in G space
    // NAME : rhog_save, chage density in G space
    // NAME : rho_core [nrxx], the core charge in real space
    // NAME : rhog_core [ngm], the core charge in reciprocal space
    //==========================================================

    double **rho = nullptr;
    double **rho_save = nullptr;

    std::complex<double> **rhog = nullptr;
    std::complex<double> **rhog_save = nullptr;

    double **kin_r = nullptr; // kinetic energy density in real space, for meta-GGA
    double **kin_r_save = nullptr; // kinetic energy density in real space, for meta-GGA
    const Parallel_Grid* pgrid = nullptr;

  private:

    //temporary
    double *_space_rho = nullptr; 
    double *_space_rho_save = nullptr;
    std::complex<double> *_space_rhog = nullptr;
    std::complex<double> *_space_rhog_save = nullptr;
    double *_space_kin_r = nullptr;
    double *_space_kin_r_save = nullptr;

  public:

    double **nhat = nullptr; //compensation charge for PAW
    double **nhat_save = nullptr; //compensation charge for PAW
                                 // wenfei 2023-09-05

    double *rho_core = nullptr;
    std::complex<double> *rhog_core = nullptr;

    int prenspin = 1;

    void set_rhopw(ModulePW::PW_Basis* rhopw_in);

    /**
     * @brief Init charge density from file or atomic pseudo-wave-functions
     *
     * @param eferm_iout [out] fermi energy to be initialized
     * @param ucell [in] unit cell
     * @param strucFac [in] structure factor
     * @param symm [in] symmetry
     * @param klist [in] k points list if needed
     * @param wfcpw [in] PW basis for wave function if needed
     */
    void init_rho(const UnitCell& ucell,
                  const Parallel_Grid& pgrid,
                  const ModuleBase::ComplexMatrix& strucFac,
                  ModuleSymmetry::Symmetry& symm,
                  const void* klist = nullptr,
                  const void* wfcpw = nullptr);

    // mohan add 2025-12-02
    bool kin_density();

    void allocate(const int &nspin_in, const bool kin_den);

    void atomic_rho(const int spin_number_need,
                    const double& omega,
                    double** rho_in,
                    const ModuleBase::ComplexMatrix& strucFac,
                    const UnitCell& ucell) const;

    void set_rho_core(const UnitCell& ucell,
                      const ModuleBase::ComplexMatrix& structure_factor, 
                      const bool* numeric);

    void renormalize_rho();

    double sum_rho() const;

    void save_rho_before_sum_band();

	// for non-linear core correction
    void non_linear_core_correction
    (
        const bool &numeric,
        const double omega,
        const double tpiba2,
        const int mesh,
        const double *r,
        const double *rab,
        const double *rhoc,
        double *rhocg
    ) const;

	double cal_rho2ne(const double *rho_in) const;

    void check_rho(); // to check whether the charge density is normal

    void init_final_scf(); //LiuXh add 20180619

	public:
    /**
     * @brief init some arrays for mpi_inter_pools, rho_mpi
     */
    void init_chgmpi();

    /**
     * @brief Sum rho at different pools (k-point parallelism).
     *        Only used when GlobalV::KPAR > 1
     */
    void rho_mpi();

    /**
     * @brief Sum kin_r at different pools (k-point/band parallelism).
     *        Only used when GlobalV::KPAR * bndpar > 1
     */
    void kin_r_mpi();

	/**
	 * @brief 	Reduce among different pools 
     *          If NPROC_IN_POOLs are all the same, use GlobalV::KP_WORLD
     *          else, gather rho in a POOL, and then reduce among different POOLs
	 * 
	 * @param array_rho f(rho): an array [nrxx]
	 */
	void reduce_diff_pools(double* array_rho) const;

    void set_omega(double* omega_in){this->omega_ = omega_in;};

    // mohan add 2021-02-20
    int nrxx=0; // number of r vectors in this processor
    int nxyz = 0; // total number of r vectors
    int ngmc=0; // number of g vectors in this processor
    int nspin=0; // number of spins
    ModulePW::PW_Basis* rhopw = nullptr;// When double_grid is used, rhopw = rhodpw (dense grid)
    bool cal_elf = false; // whether to calculate electron localization function (ELF)

    //==========================================================
    // Device support (hybrid pattern like PW_Basis)
    //==========================================================

    /// @brief Set device type for charge density storage
    /// @param device_in "cpu" or "gpu"
    void set_device(const std::string& device_in);

    /// @brief Get current device type
    std::string get_device() const { return device_; }

    /// @brief Set precision type
    /// @param precision_in "single" or "double"
    void set_precision(const std::string& precision_in) { precision_ = precision_in; }

    /// @brief Get current precision type
    std::string get_precision() const { return precision_; }

    /// @brief Get device pointer for rho (nullptr if device != "gpu")
    /// @param is spin index
    double* get_rho_d(int is = 0) const;

    /// @brief Get device pointer for rho_save (nullptr if device != "gpu")
    double* get_rho_save_d(int is = 0) const;

    /// @brief Get device pointer for rhog (nullptr if device != "gpu")
    std::complex<double>* get_rhog_d(int is = 0) const;

    /// @brief Get device pointer for rhog_save (nullptr if device != "gpu")
    std::complex<double>* get_rhog_save_d(int is = 0) const;

    /// @brief Get device pointer for kin_r (nullptr if device != "gpu")
    double* get_kin_r_d(int is = 0) const;

    /// @brief Get device pointer for kin_r_save (nullptr if device != "gpu")
    double* get_kin_r_save_d(int is = 0) const;

    /// @brief Sync rho from host to device (GPU specialization)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_GPU>::value, int>::type = 0>
    void sync_rho_to_device();

    /// @brief Sync rho from host to device (CPU no-op)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_CPU>::value, int>::type = 0>
    void sync_rho_to_device() {} // No-op for CPU

    /// @brief Sync rho from device to host (GPU specialization)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_GPU>::value, int>::type = 0>
    void sync_rho_to_host();

    /// @brief Sync rho from device to host (CPU no-op)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_CPU>::value, int>::type = 0>
    void sync_rho_to_host() {} // No-op for CPU

    /// @brief Sync rhog from host to device (GPU specialization)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_GPU>::value, int>::type = 0>
    void sync_rhog_to_device();

    /// @brief Sync rhog from host to device (CPU no-op)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_CPU>::value, int>::type = 0>
    void sync_rhog_to_device() {} // No-op for CPU

    /// @brief Sync rhog from device to host (GPU specialization)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_GPU>::value, int>::type = 0>
    void sync_rhog_to_host();

    /// @brief Sync rhog from device to host (CPU no-op)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_CPU>::value, int>::type = 0>
    void sync_rhog_to_host() {} // No-op for CPU

    /// @brief Sync kin_r from host to device (GPU specialization)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_GPU>::value, int>::type = 0>
    void sync_kin_r_to_device();

    /// @brief Sync kin_r from host to device (CPU no-op)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_CPU>::value, int>::type = 0>
    void sync_kin_r_to_device() {} // No-op for CPU

    /// @brief Sync kin_r from device to host (GPU specialization)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_GPU>::value, int>::type = 0>
    void sync_kin_r_to_host();

    /// @brief Sync kin_r from device to host (CPU no-op)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_CPU>::value, int>::type = 0>
    void sync_kin_r_to_host() {} // No-op for CPU

    /// @brief Sync kin_r_save from host to device (GPU specialization)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_GPU>::value, int>::type = 0>
    void sync_kin_r_save_to_device();

    /// @brief Sync kin_r_save from host to device (CPU no-op)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_CPU>::value, int>::type = 0>
    void sync_kin_r_save_to_device() {} // No-op for CPU

    /// @brief Sync rho_save from host to device (GPU specialization)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_GPU>::value, int>::type = 0>
    void sync_rho_save_to_device();

    /// @brief Sync rho_save from host to device (CPU no-op)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_CPU>::value, int>::type = 0>
    void sync_rho_save_to_device() {} // No-op for CPU

    /// @brief Sync rhog_save from host to device (GPU specialization)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_GPU>::value, int>::type = 0>
    void sync_rhog_save_to_device();

    /// @brief Sync rhog_save from host to device (CPU no-op)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_CPU>::value, int>::type = 0>
    void sync_rhog_save_to_device() {} // No-op for CPU

    /// @brief Sync rhog_save from device to host (GPU specialization)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_GPU>::value, int>::type = 0>
    void sync_rhog_save_to_host();

    /// @brief Sync rhog_save from device to host (CPU no-op)
    template <typename Device,
              typename std::enable_if<std::is_same<Device, base_device::DEVICE_CPU>::value, int>::type = 0>
    void sync_rhog_save_to_host() {} // No-op for CPU

  private:

    void destroy();    // free arrays  liuyu 2023-03-12

    double* omega_ = nullptr; // omega for non-linear core correction

    bool allocate_rho;

    bool allocate_rho_final_scf; // LiuXh add 20180606

#ifdef __MPI
    int *rec = nullptr; //The number of elements each process should receive into the receive buffer.
    int *dis = nullptr; //The displacement (relative to recvbuf) for each process in the receive buffer.
#endif

    //==========================================================
    // Device memory management (hybrid pattern like PW_Basis)
    //==========================================================

    /// Runtime device selection: "cpu" or "gpu"
    std::string device_ = "cpu";

    /// Runtime precision selection: "single" or "double"
    std::string precision_ = "double";

    /// Device memory for rho [nspin*nrxx] (allocated when device_ == "gpu")
    double* rho_d_ = nullptr;

    /// Device memory for rho_save [nspin*nrxx]
    double* rho_save_d_ = nullptr;

    /// Device memory for rhog [nspin*ngmc]
    std::complex<double>* rhog_d_ = nullptr;

    /// Device memory for rhog_save [nspin*ngmc]
    std::complex<double>* rhog_save_d_ = nullptr;

    /// Device memory for kin_r [nspin*nrxx]
    double* kin_r_d_ = nullptr;

    /// Device memory for kin_r_save [nspin*nrxx]
    double* kin_r_save_d_ = nullptr;

    /// @brief Allocate device memory when switching to GPU mode
    void allocate_device_memory();

    /// @brief Free device memory when switching to CPU mode or destroying
    void free_device_memory();

};

#endif // charge
