add_definitions(-D__NORMAL)
if (USE_CUDA)
AddTest(
  TARGET pw_test_gpu
  LIBS parameter ${math_libs} base planewave device FFTW3::FFTW3_FLOAT
  SOURCES pw_test.cpp  pw_basis_C2R.cpp pw_basis_C2C.cpp  pw_basis_k_C2C.cpp
)

# Batch FFT test
AddTest(
  TARGET pw_batch_fft_test
  LIBS parameter ${math_libs} base planewave device
  SOURCES pw_basis_k_batch_test.cpp
)
endif()

