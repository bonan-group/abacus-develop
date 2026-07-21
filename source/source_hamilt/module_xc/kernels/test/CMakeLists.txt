AddTest(
  TARGET MODULE_HAMILT_XC_Functional_UTs
  LIBS device base container
  SOURCES xc_functional_op_test.cpp
)

AddTest(
  TARGET MODULE_HAMILT_XC_Formula_UTs
  LIBS device base
  SOURCES xc_formula_test.cpp xc_selector_test.cpp
)

set(xc_kernel_test_libs xc_ parameter device base container planewave)
if(ENABLE_LIBXC)
  list(APPEND xc_kernel_test_libs Libxc::xc)
endif()

AddTest(
  TARGET MODULE_HAMILT_XC_Public_UTs
  LIBS ${xc_kernel_test_libs}
  SOURCES xc_public_test.cpp
    xc_vxc_mock_unitcell_deps.cpp
    ../../../../source_estate/module_charge/charge.cpp
    ../../../../source_estate/module_charge/charge_device.cpp
    ../../../../source_cell/test/support/mock_unitcell.cpp
)

if(USE_CUDA)
  AddTest(
    TARGET MODULE_HAMILT_XC_Resident_UTs
    LIBS ${xc_kernel_test_libs}
    SOURCES xc_resident_gpu_test.cpp
      xc_resident_op_test.cpp
      xc_resident_test_main.cpp
      xc_vxc_mock_unitcell_deps.cpp
      ../../../../source_estate/module_charge/charge.cpp
      ../../../../source_estate/module_charge/charge_device.cpp
      ../../../../source_cell/test/support/mock_unitcell.cpp
  )
endif()
