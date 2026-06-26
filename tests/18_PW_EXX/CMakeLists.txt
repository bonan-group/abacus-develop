enable_testing()

find_program(BASH bash)
if(ENABLE_ASAN)
    add_test(
        NAME 18_PW_EXX_CPU_test_with_asan
        COMMAND ${BASH} ../integrate/Autotest.sh -f CASES_CPU.txt -a ${ABACUS_BIN_PATH} -n 2 -s true -o 1
        WORKING_DIRECTORY ${ABACUS_TEST_DIR}/18_PW_EXX
    )
else()
    add_test(
        NAME 18_PW_EXX_CPU
        COMMAND ${BASH} ../integrate/Autotest.sh -f CASES_CPU.txt -a ${ABACUS_BIN_PATH} -n 4 -o 1
        WORKING_DIRECTORY ${ABACUS_TEST_DIR}/18_PW_EXX
    )
    add_test(
        NAME 18_PW_EXX_CPU_SERIAL
        COMMAND ${BASH} ../integrate/Autotest.sh -f CASES_CPU_SERIAL.txt -a ${ABACUS_BIN_PATH} -n 1 -o 1
        WORKING_DIRECTORY ${ABACUS_TEST_DIR}/18_PW_EXX
    )
endif()

if(USE_CUDA)
    add_test(
        NAME 18_PW_EXX_GPU
        COMMAND ${BASH} ../integrate/Autotest.sh -f CASES_GPU.txt -a ${ABACUS_BIN_PATH} -n 4 -o 1
        WORKING_DIRECTORY ${ABACUS_TEST_DIR}/18_PW_EXX
    )
endif()
