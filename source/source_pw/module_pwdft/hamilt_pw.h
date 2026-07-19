#ifndef HAMILTPW_H
#define HAMILTPW_H

#include "source_base/macros.h"
#include "source_cell/klist.h"
#include "source_estate/module_pot/potential_new.h"
#include "source_hamilt/hamilt.h"
#include "source_lcao/module_dftu/dftu.h" // mohan add 2025-11-06
#include "source_pw/module_pwdft/exx_helper.h"
#include "source_pw/module_pwdft/vnl_pw.h"

namespace hamilt
{

template <typename T, typename Device = base_device::DEVICE_CPU>
class HamiltPW : public Hamilt<T, Device>
{
  private:
    // Note GetTypeReal<T>::type will
    // return T if T is real type(float, double),
    // otherwise return the real type of T(complex<float>, std::complex<double>)
    using Real = typename GetTypeReal<T>::type;

  public:
    HamiltPW(elecstate::Potential* pot_in,
             ModulePW::PW_Basis_K* wfc_basis,
             K_Vectors* p_kv,
             pseudopot_cell_vnl* nlpp,
             Plus_U* p_dftu, // mohan add 2025-11-06
             const UnitCell* ucell);

    ~HamiltPW();

    // for target K point, update consequence of hPsi() and matrix()
    void updateHk(const int ik) override;

    void sPsi(const T* psi_in, // psi
              T* spsi,         // spsi
              const int nrow,  // dimension of spsi: nbands * nrow
              const int npw,   // number of plane waves
              const int nbands // number of bands
    ) const override;

    void set_exx_helper(Exx_Helper<T, Device>& exx_helper_in);

  protected:
    // used in sPhi, which are calculated in hPsi or sPhi
    const pseudopot_cell_vnl* ppcell = nullptr;
    Operator<T, Device>* nonlocal_op = nullptr;
    bool nonlocal_op_in_chain = false;

    Device* ctx = {};
    using syncmem_op = base_device::memory::synchronize_memory_op<T, Device, Device>;
};

} // namespace hamilt

#endif
