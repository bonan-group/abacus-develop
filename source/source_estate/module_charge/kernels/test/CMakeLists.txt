AddTest(
  TARGET MODULE_ESTATE_charge_mixing_kernels_gpu
  LIBS base device
  SOURCES charge_mixing_op_test.cpp
)

AddTest(
  TARGET MODULE_ESTATE_charge_device
  LIBS xc_ parameter device base container planewave
  SOURCES charge_device_test.cpp
    ../../charge.cpp
    ../../charge_device.cpp
    ../../../../source_cell/test/support/mock_unitcell.cpp
    ../../../../source_hamilt/module_xc/kernels/test/xc_vxc_mock_unitcell_deps.cpp
)
