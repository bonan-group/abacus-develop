AddTest(
    TARGET MODULE_HAMILT_XC_Functional_UTs
    LIBS parameter ${math_libs} device base container planewave
    SOURCES xc_formula_test.cpp
      xc_functional_op_test.cpp
      xc_vxc_mock_unitcell_deps.cpp
      ../../xc_functional.cpp
      ../../xc_pot.cpp
      ../../xc_grad.cpp
      ../../xc_lda_wrap.cpp
      ../../xc_gga_wrap.cpp
      ../../xc_lda_exch.cpp
      ../../xc_lda_corr.cpp
      ../../xc_gga_exch.cpp
      ../../xc_gga_corr.cpp
      ../../xc_hcth.cpp
      ../../libxc_setup.cpp
      ../../libxc_tools.cpp
      ../../libxc_pot.cpp
      ../../libxc_lda_wrap.cpp
      ../../libxc_gga_wrap.cpp
      ../../libxc_mgga_wrap.cpp
      ../../exx_info.cpp
      ../../../../source_estate/module_charge/charge.cpp
      ../../../../source_estate/module_charge/charge_device.cpp
      ../../../../source_cell/test/support/mock_unitcell.cpp
)

if(ENABLE_LIBXC)
  target_link_libraries(MODULE_HAMILT_XC_Functional_UTs PRIVATE Libxc::xc)
endif()
