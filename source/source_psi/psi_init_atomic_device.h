#ifndef SOURCE_PSI_PSI_INIT_ATOMIC_DEVICE_H
#define SOURCE_PSI_PSI_INIT_ATOMIC_DEVICE_H

#include <memory>

namespace ModulePW
{
class PW_Basis_K;
}

class Structure_Factor;
class UnitCell;

template <typename T>
class psi_init_atomic;

namespace psi
{

template <typename T>
class AtomicGpuInitializer
{
  public:
    AtomicGpuInitializer(const ::psi_init_atomic<T>* atomic_initializer,
                         const ModulePW::PW_Basis_K& pw_wfc,
                         const UnitCell& ucell,
                         const Structure_Factor& structure_factor,
                         bool enabled,
                         int random_seed);
    ~AtomicGpuInitializer();

    AtomicGpuInitializer(const AtomicGpuInitializer&) = delete;
    AtomicGpuInitializer& operator=(const AtomicGpuInitializer&) = delete;

    bool available() const;
    void initialize_k(T* psi, int nbands_start, int ik, int ik_tot);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace psi

#endif
