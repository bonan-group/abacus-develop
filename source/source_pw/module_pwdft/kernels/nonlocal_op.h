#ifndef MODULE_HAMILT_NONLOCAL_H
#define MODULE_HAMILT_NONLOCAL_H

#include "source_base/module_device/types.h"
#include <complex>

namespace hamilt {
template <typename FPTYPE, typename Device>
struct uspp_qgm_build_op
{
    void operator()(const Device* dev,
                    int ntype,
                    int nh_tot,
                    int npw,
                    int lmaxq,
                    int radial_pair_count,
                    int nqxq,
                    int max_terms,
                    FPTYPE dq,
                    FPTYPE tpiba,
                    const FPTYPE* gcar,
                    const int* pair_term_count,
                    const int* pair_radial_index,
                    const int* pair_l,
                    const int* pair_lm,
                    const std::complex<FPTYPE>* pair_coefficient,
                    const FPTYPE* qrad,
                    const FPTYPE* ylm,
                    std::complex<FPTYPE>* qgm);
};

template <typename FPTYPE, typename Device>
struct uspp_overlap_op
{
    void operator()(const Device* dev,
                    int atom_count,
                    int nbands,
                    int nh,
                    int nhm,
                    int nkb,
                    int projector_offset,
                    bool projector_major,
                    const FPTYPE* qq,
                    std::complex<FPTYPE>* ps,
                    const std::complex<FPTYPE>* becp);
};

template <typename FPTYPE, typename Device>
struct uspp_deeq_op
{
    void operator()(const Device* dev,
                    int nspin,
                    int atom_count,
                    int nh,
                    int nhm,
                    int npw,
                    int atom_offset,
                    int nat,
                    FPTYPE omega,
                    bool gamma_only,
                    int g0_index,
                    const std::complex<FPTYPE>* vaux,
                    const std::complex<FPTYPE>* qgm,
                    const std::complex<FPTYPE>* phase,
                    const FPTYPE* dvan,
                    FPTYPE* deeq);
};

template <typename FPTYPE, typename Device>
struct uspp_force_op;

template <typename FPTYPE, typename Device>
struct uspp_stress_op
{
    void operator()(const Device* dev,
                    int nspin,
                    int atom_count,
                    int nij,
                    int npw,
                    int atom_offset,
                    int nat,
                    int nh_tot,
                    int ipol,
                    FPTYPE tpiba,
                    const std::complex<FPTYPE>* vaux,
                    const std::complex<FPTYPE>* dqgm,
                    const std::complex<FPTYPE>* phase,
                    const FPTYPE* gcar,
                    const FPTYPE* becsum,
                    FPTYPE* stress);
};

template <typename FPTYPE, typename Device> 
struct nonlocal_pw_op {
  /// @brief Compute the nonlocal potential of hPsi
  ///
  /// Input Parameters
  /// \param dev : the type of computing device
  /// \param l1 : ucell->atoms[it].na
  /// \param l2 : nbands
  /// \param l3 : ucell->atoms[it].ncpp.nh
  /// \param sum : intermediate value
  /// \param iat : intermediate value
  /// \param spin : current spin
  /// \param nkb : ppcell->nkb, number of kpoints
  /// \param deeq_x : second dimension of deeq
  /// \param deeq_y : third dimension of deeq
  /// \param deeq_z : forth dimension of deeq
  /// \param deeq : ppcell->deeq
  /// \param becp : intermediate array
  ///
  /// Output Parameters
  /// \param ps : output array
  void operator() (
      const Device* dev,
      const int& l1,
      const int& l2,
      const int& l3,
      int& sum,
      int& iat,
      const int& spin,
      const int& nkb,
      const int& deeq_x,
      const int& deeq_y,
      const int& deeq_z,
      const FPTYPE* deeq,
      std::complex<FPTYPE>* ps,
      const std::complex<FPTYPE>* becp);

  /// @brief Compute the nonlocal potential of hPsi, with NSPIN > 2
  ///
  /// Input Parameters
  /// \param dev : the type of computing device
  /// \param l1 : ucell->atoms[it].na
  /// \param l2 : nbands
  /// \param l3 : ucell->atoms[it].ncpp.nh
  /// \param sum : intermediate value
  /// \param iat : intermediate value
  /// \param nkb : ppcell->nkb, number of kpoints
  /// \param deeq_x : second dimension of deeq
  /// \param deeq_y : third dimension of deeq
  /// \param deeq_z : forth dimension of deeq
  /// \param deeq_nc : ppcell->deeq_nc
  /// \param becp : intermediate array
  ///
  /// Output Parameters
  /// \param ps : output array
  void operator() (
      const Device* dev,
      const int& l1,
      const int& l2,
      const int& l3,
      int& sum,
      int& iat,
      const int& nkb,
      const int& deeq_x,
      const int& deeq_y,
      const int& deeq_z,
      const std::complex<FPTYPE>* deeq_nc,
      std::complex<FPTYPE>* ps,
      const std::complex<FPTYPE>* becp);
};
                      
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
template <typename FPTYPE>
struct uspp_qgm_build_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* dev,
                    int ntype,
                    int nh_tot,
                    int npw,
                    int lmaxq,
                    int radial_pair_count,
                    int nqxq,
                    int max_terms,
                    FPTYPE dq,
                    FPTYPE tpiba,
                    const FPTYPE* gcar,
                    const int* pair_term_count,
                    const int* pair_radial_index,
                    const int* pair_l,
                    const int* pair_lm,
                    const std::complex<FPTYPE>* pair_coefficient,
                    const FPTYPE* qrad,
                    const FPTYPE* ylm,
                    std::complex<FPTYPE>* qgm);
};

template <typename FPTYPE>
struct uspp_overlap_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* dev,
                    int atom_count,
                    int nbands,
                    int nh,
                    int nhm,
                    int nkb,
                    int projector_offset,
                    bool projector_major,
                    const FPTYPE* qq,
                    std::complex<FPTYPE>* ps,
                    const std::complex<FPTYPE>* becp);
};

template <typename FPTYPE>
struct uspp_deeq_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* dev,
                    int nspin,
                    int atom_count,
                    int nh,
                    int nhm,
                    int npw,
                    int atom_offset,
                    int nat,
                    FPTYPE omega,
                    bool gamma_only,
                    int g0_index,
                    const std::complex<FPTYPE>* vaux,
                    const std::complex<FPTYPE>* qgm,
                    const std::complex<FPTYPE>* phase,
                    const FPTYPE* dvan,
                    FPTYPE* deeq);
};

template <typename FPTYPE>
struct uspp_force_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* dev,
                    int nspin,
                    int atom_count,
                    int nij,
                    int npw,
                    int atom_offset,
                    int nat,
                    int nh_tot,
                    FPTYPE omega,
                    FPTYPE tpiba,
                    const std::complex<FPTYPE>* vaux,
                    const std::complex<FPTYPE>* qgm,
                    const std::complex<FPTYPE>* phase,
                    const FPTYPE* gcar,
                    const FPTYPE* becsum,
                    FPTYPE* force);
};

template <typename FPTYPE>
struct uspp_stress_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* dev,
                    int nspin,
                    int atom_count,
                    int nij,
                    int npw,
                    int atom_offset,
                    int nat,
                    int nh_tot,
                    int ipol,
                    FPTYPE tpiba,
                    const std::complex<FPTYPE>* vaux,
                    const std::complex<FPTYPE>* dqgm,
                    const std::complex<FPTYPE>* phase,
                    const FPTYPE* gcar,
                    const FPTYPE* becsum,
                    FPTYPE* stress);
};

// Partially specialize functor for base_device::GpuDevice.
template <typename FPTYPE>
struct nonlocal_pw_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* dev,
                    const int& l1,
                    const int& l2,
                    const int& l3,
                    int& sum,
                    int& iat,
                    const int& spin,
                    const int& nkb,
                    const int& deeq_x,
                    const int& deeq_y,
                    const int& deeq_z,
                    const FPTYPE* deeq,
                    std::complex<FPTYPE>* ps,
                    const std::complex<FPTYPE>* becp);

    void operator()(const base_device::DEVICE_GPU* dev,
                    const int& l1,
                    const int& l2,
                    const int& l3,
                    int& sum,
                    int& iat,
                    const int& nkb,
                    const int& deeq_x,
                    const int& deeq_y,
                    const int& deeq_z,
                    const std::complex<FPTYPE>* deeq_nc,
                    std::complex<FPTYPE>* ps,
                    const std::complex<FPTYPE>* becp);
};
#endif // __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
} // namespace hamilt
#endif //MODULE_HAMILT_NONLOCAL_H
