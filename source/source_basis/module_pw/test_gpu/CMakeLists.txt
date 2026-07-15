abacus_add_local_feature_definitions(__NORMAL)
if (USE_CUDA AND ENABLE_FLOAT_FFTW)
  AddTest(
    TARGET pw_test_gpu
    LIBS parameter base planewave device FFTW3::FFTW3_FLOAT
    SOURCES pw_test.cpp pw_basis_C2R.cpp pw_basis_C2C.cpp pw_basis_k_C2C.cpp pw_basis_sup_map.cpp
  )
  add_test(
    NAME MODULE_PW_pw_basis_sup_gpu_map
    COMMAND pw_test_gpu --gtest_filter=PWBasisSupGpuTest.GammaYReducedIndexMapMatchesFftLayout
  )
endif()
