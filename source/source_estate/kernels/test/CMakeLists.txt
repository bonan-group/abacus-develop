abacus_disable_feature_definitions(__MPI)
abacus_disable_feature_definitions(__EXX)
abacus_disable_feature_definitions(__CUDA)
abacus_disable_feature_definitions(__ROCM)
abacus_disable_feature_definitions(__MLALGO)

AddTest(
    TARGET MODULE_ESTATE_elecstate_kernels_gpu
    LIBS parameter psi base device
    SOURCES elecstate_op_test.cpp
)
