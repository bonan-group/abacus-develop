remove_definitions(-D__MPI)
AddTest(
  TARGET MODULE_BASE_MIXING_unittests
  LIBS parameter base device ${math_libs} 
  SOURCES mixing_test.cpp
)
if (USE_CUDA)
  target_compile_definitions(MODULE_BASE_MIXING_unittests PRIVATE __UT_USE_CUDA)
endif()
