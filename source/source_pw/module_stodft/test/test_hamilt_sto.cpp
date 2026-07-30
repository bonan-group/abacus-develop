#include "../hamilt_sdft_pw.h"
#include "source_hamilt/operator.h"

#include "gtest/gtest.h"
#include <vector>

elecstate::Potential::~Potential(){}
void elecstate::Potential::cal_v_eff(Charge const*, UnitCell const*, ModuleBase::matrix&){}
void elecstate::Potential::cal_fixed_v(double*){}

template <typename T, typename Device>
hamilt::HamiltPW<T, Device>::HamiltPW(
		elecstate::Potential* pot_in, 
		ModulePW::PW_Basis_K* wfc_basis, 
		K_Vectors* p_kv, 
		pseudopot_cell_vnl* ppcell,
        Plus_U* p_dftu, // mohan add 20251108
		const UnitCell* ucell){}

template <typename T, typename Device>
hamilt::HamiltPW<T, Device>::HamiltPW(elecstate::Potential* pot_in,
                                      ModulePW::PW_Basis_K* wfc_basis,
                                      K_Vectors* p_kv,
                                      pseudopot_cell_vnl* ppcell,
                                      Plus_U* p_dftu,
                                      const UnitCell* ucell,
                                      const hamilt::ExxOperatorOptions& exx_options,
                                      const hamilt::ExxExecutionContext& exx_execution_context)
    : exx_options_(exx_options), exx_execution_context_(exx_execution_context)
{
}

template <typename T, typename Device>
hamilt::HamiltPW<T, Device>::~HamiltPW(){
    delete this->ops;
};
template <typename T, typename Device>
void hamilt::HamiltPW<T, Device>::updateHk(int){}
template <typename T, typename Device>
void hamilt::HamiltPW<T, Device>::sPsi(T const*, T*, const int, const int, const int) const{}

template class hamilt::HamiltPW<std::complex<double>, base_device::DEVICE_CPU>;
template class hamilt::HamiltPW<std::complex<float>, base_device::DEVICE_CPU>;
template class hamilt::HamiltPW<std::complex<double>, base_device::DEVICE_GPU>;
template class hamilt::HamiltPW<std::complex<float>, base_device::DEVICE_GPU>;

/************************************************
 *  unit test of hamilt_sto_pw.cpp
 * - Tested Functions:
 *  - void hPsi(const T* psi_in, T* hpsi, const int& nbands)
 *  - void hPsi_norm(const T* psi_in, T* hpsi, const int& nbands)
 ***********************************************/

template <typename T, typename Device>
class TestOp : public hamilt::Operator<T, Device>
{
  public:
    virtual void act(const int nbands,
                     const int nbasis,
                     const int npol,
                     const T* tmpsi_in,
                     T* tmhpsi,
                     const int ngk_ik = 0,
                     const bool is_first_node = false) const override
    {
        for (int i = 0; i < nbands; i++)
        {
            for (int j = 0; j < nbasis; j++)
            {
                tmhpsi[i * nbasis + j] = tmpsi_in[i * nbasis + j];
            }
        }
    }
};

class TestHamiltSto : public ::testing::Test
{
  public:
    TestHamiltSto()
    {
        const int nbands = 1;
        const int nbasis = 2;
        const int npol = 1;
        // Initialize the hamilt_sto_pw
        pot = new elecstate::Potential();
        wfc_basis = new ModulePW::PW_Basis_K();
        wfc_basis->npwk_max = 2;
        p_kv = new K_Vectors();
        std::vector<int> ngk = {2};
        p_kv->ngk = ngk;
        exx_options.q_tile_size = 7;
        exx_execution_context.my_pool = 2;
        hamilt_sto = new hamilt::HamiltSdftPW<std::complex<double>, base_device::DEVICE_CPU>(pot,
                                                                                             wfc_basis,
                                                                                             p_kv,
                                                                                             nullptr,
                                                                                             nullptr,
                                                                                             npol,
                                                                                             &emin,
                                                                                             &emax,
                                                                                             exx_options,
                                                                                             exx_execution_context);
        hamilt_sto->ops = new TestOp<std::complex<double>, base_device::DEVICE_CPU>();
    }

    ~TestHamiltSto()
    {
        delete pot;
        delete wfc_basis;
        delete p_kv;
        delete hamilt_sto;
    }

    elecstate::Potential* pot;
    ModulePW::PW_Basis_K* wfc_basis;
    K_Vectors* p_kv;
    hamilt::HamiltSdftPW<std::complex<double>, base_device::DEVICE_CPU>* hamilt_sto;
    hamilt::ExxOperatorOptions exx_options;
    hamilt::ExxExecutionContext exx_execution_context;
    double emin = -2.0;
    double emax = 2.0;
};

TEST_F(TestHamiltSto, ExplicitExxConfigurationIsPreserved)
{
    EXPECT_EQ(hamilt_sto->exx_options().q_tile_size, 7);
    EXPECT_EQ(hamilt_sto->exx_execution_context().my_pool, 2);
}

TEST_F(TestHamiltSto, hPsi)
{
    const int nbands = 1;
    const int nbasis = 2;
    // Prepare the input psi
    std::vector<std::complex<double>> psi_in(nbands * nbasis);
    std::vector<std::complex<double>> hpsi(nbands * nbasis);

    for (int i = 0; i < nbands; i++)
    {
        for (int j = 0; j < nbasis; j++)
        {
            psi_in[i * nbasis + j] = i + j;
        }
    }
    hamilt_sto->hPsi(psi_in.data(), hpsi.data(), nbands);
    // Check the result
    for (int i = 0; i < nbands; i++)
    {
        for (int j = 0; j < nbasis; j++)
        {
            EXPECT_EQ(hpsi[i * nbasis + j], psi_in[i * nbasis + j]);
        }
    }
}

TEST_F(TestHamiltSto, hPsi_norm)
{
    int nbands = 1;
    int nbasis = 2;
    std::vector<std::complex<double>> psi_in(nbands * nbasis);
    std::vector<std::complex<double>> hpsi(nbands * nbasis);

    for (int i = 0; i < nbands; i++)
    {
        for (int j = 0; j < nbasis; j++)
        {
            psi_in[i * nbasis + j] = i + j;
        }
    }

    hamilt_sto->hPsi_norm(psi_in.data(), hpsi.data(), nbands);

    // Check the result
    for (int i = 0; i < nbands; i++)
    {
        for (int j = 0; j < nbasis; j++)
        {
            EXPECT_EQ(hpsi[i * nbasis + j], psi_in[i * nbasis + j] * 0.5);
        }
    }
}
