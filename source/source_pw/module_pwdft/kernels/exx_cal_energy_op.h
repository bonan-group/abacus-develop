#ifndef CAL_VEC_NORM_OP_H
#define CAL_VEC_NORM_OP_H
#include "source_base/macros.h"
namespace hamilt{
template <typename T, typename Device>
struct exx_cal_energy_op
{

    using FPTYPE = typename GetTypeReal<T>::type;
    FPTYPE operator()(const T *den, const FPTYPE *pot, FPTYPE scala, int npw);
};


// Operator to calculate element-wise norm squared of a complex vector
template <typename T, typename Device>
struct exx_vector_elementwise_norm_squared_op
{

    using FPTYPE = typename GetTypeReal<T>::type;
    FPTYPE operator()(const T *vector_in,
                     FPTYPE *vector_buffer,
                     const FPTYPE *pot,
                    FPTYPE *vec_temp,
                    FPTYPE *weight,
                     int npw,
                     int batch_idx);
} ;



} // namespace hamilt
#endif //CAL_VEC_NORM_OP_H
