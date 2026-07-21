AddTest(
  TARGET MODULE_HAMILT_XC_Functional_UTs
  LIBS device base container
  SOURCES xc_functional_op_test.cpp
)

AddTest(
  TARGET MODULE_HAMILT_XC_Formula_UTs
  LIBS device base
  SOURCES xc_formula_test.cpp
)

if(USE_CUDA)
  AddTest(
    TARGET MODULE_HAMILT_XC_Resident_UTs
    LIBS xc_ parameter device base container planewave
    SOURCES xc_resident_gpu_test.cpp
      xc_resident_op_test.cpp
      xc_vxc_mock_unitcell_deps.cpp
      ../../../../source_estate/module_charge/charge.cpp
      ../../../../source_estate/module_charge/charge_device.cpp
      ../../../../source_cell/test/support/mock_unitcell.cpp
  )
endif()
