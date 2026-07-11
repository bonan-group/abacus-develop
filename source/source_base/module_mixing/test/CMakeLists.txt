abacus_disable_feature_definitions(__MPI)
AddTest(
  TARGET MODULE_BASE_MIXING_unittests
  LIBS parameter base device
  SOURCES mixing_test.cpp
)
if (USE_CUDA)
  target_compile_definitions(MODULE_BASE_MIXING_unittests PRIVATE __UT_USE_CUDA)
endif()
