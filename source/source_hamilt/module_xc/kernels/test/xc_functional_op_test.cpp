#include "source_hamilt/module_xc/kernels/xc_functional_op.h"

#include <base/utils/gtest.h>
#include <ATen/core/tensor.h>

#include <complex>
#include <tuple>

namespace hamilt
{

template <typename T>
class XCFunctionalOpTest : public testing::Test
{
};

TYPED_TEST_SUITE(XCFunctionalOpTest, base::utils::ComplexTypes);

TYPED_TEST(XCFunctionalOpTest, GradientWavefunctionOperation)
{
    using Type = typename std::tuple_element<0, decltype(TypeParam())>::type;
    using Device = typename std::tuple_element<1, decltype(TypeParam())>::type;
    using Real = typename GetTypeReal<Type>::type;

    const int ik = 0;
    const int ipol = 0;
    const int npw = 4;
    const int npwx = 4;
    const int nrxx = 4;
    const Real tpiba = 3.3249187157734292;

    ct::Tensor gcar = std::move(ct::Tensor(
        {static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(-0.46354790605367302),
         static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(-0.30903193736911533),
         static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(-0.15451596868455766),
         static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(0)}).to_device<Device>());
    ct::Tensor kvec_c = std::move(ct::Tensor({static_cast<Real>(0)}).to_device<Device>());
    ct::Tensor rhog = std::move(ct::Tensor(
        {static_cast<Type>(-3.0017116103913875e-05, 7.1056289872527673e-08),
         static_cast<Type>(-7.5546991753311607e-05, -2.3428181621642512e-07),
         static_cast<Type>(-0.00092404657246742273, 1.604946087155834e-06),
         static_cast<Type>(0.98692408978015989, 1.8132214954316574e-09)}).to_device<Device>());
    ct::Tensor expected_porter = std::move(ct::Tensor(
        {static_cast<Type>(0), static_cast<Type>(0), static_cast<Type>(0), static_cast<Type>(0)}).to_device<Device>());
    ct::Tensor porter = expected_porter;
    ct::Tensor porter_after = std::move(ct::Tensor(
        {static_cast<Type>(1.909712182465086e-07, 3.6064630015014698e-16),
         static_cast<Type>(-1.6872327080003407e-07, -3.4645680421774294e-16),
         static_cast<Type>(-2.3926040075122384e-07, -5.37330596683816e-16),
         static_cast<Type>(1.5465312289181674e-08, -8.7169854667834556e-17)}).to_device<Device>());
    ct::Tensor expected_grad = porter;
    ct::Tensor grad = expected_grad;
    grad.zero();

    xc_functional_grad_wfc_op<Type, typename ct::ContainerToPsi<Device>::type> operation;
    operation(ik,
              ipol,
              npw,
              npwx,
              tpiba,
              gcar.data<Real>(),
              kvec_c.data<Real>(),
              rhog.data<Type>(),
              porter.data<Type>());
    EXPECT_EQ(porter, expected_porter);

    operation(ipol, nrxx, porter_after.data<Type>(), grad.data<Type>());
    EXPECT_EQ(grad, expected_grad);
}

} // namespace hamilt
