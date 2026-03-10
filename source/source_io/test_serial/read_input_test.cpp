#include "source_io/module_parameter/read_input.h"

#include "source_base/tool_quit.h"
#include "source_io/module_parameter/parameter.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <fstream>

// mock
namespace GlobalV
{
int NPROC = 1;
int MY_RANK = 0;
std::ofstream ofs_running;
std::ofstream ofs_warning;
} // namespace GlobalV
namespace ModuleBase
{
void TITLE(const std::string& class_name, const std::string& function_name, const bool disable)
{
}
namespace GlobalFunc
{
bool SCAN_BEGIN(std::ifstream& ifs, const std::string& TargetName, const bool restart, const bool ifwarn)
{
    return false;
}
} // namespace GlobalFunc
namespace Global_File
{
void make_dir_out(const std::string& suffix,
                  const std::string& calculation,
                  const bool& out_dir,
                  const bool& out_wfc_dir,
                  const int rank,
                  const bool& restart,
                  const bool out_alllog)
{
}
} // namespace Global_File
} // namespace ModuleBase

/************************************************
 *  unit test of read_input_test.cpp
 ***********************************************/

/**
 * - Tested Functions:
 *   - Selfconsistent_Read:
 *     - read empty INPUT file and write INPUT.ref back
 *     - read INPUT.ref file again and write INPUT
 *   - Check:
 *     - check_mode = true
 */

class InputTest : public testing::Test
{
  protected:
    bool compare_two_files(const std::string& filename1, const std::string& filename2)
    {
        std::ifstream file1(filename1.c_str());
        std::ifstream file2(filename2.c_str());
        EXPECT_TRUE(file1.is_open());
        EXPECT_TRUE(file2.is_open());

        std::string line1, line2;
        int lineNumber = 1;
        bool allpass = true;
        while (std::getline(file1, line1) && std::getline(file2, line2))
        {
            std::istringstream iss1(line1);
            std::istringstream iss2(line2);

            std::string col1_file1, col2_file1;
            std::string col1_file2, col2_file2;

            // read two columns from each file
            iss1 >> col1_file1 >> col2_file1;
            iss2 >> col1_file2 >> col2_file2;

            // compare two columns
            // compare two columns
            if (col1_file1 != col1_file2 || col2_file1 != col2_file2)
            {
                std::cout << "Mismatch found at line " << lineNumber << " in files " << filename1 << " and "
                          << filename2 << std::endl;
                std::cout << "File1: " << col1_file1 << " " << col2_file1 << std::endl;
                std::cout << "File2: " << col1_file2 << " " << col2_file2 << std::endl;
                allpass = false;
            }

            lineNumber++;
        }

        file1.close();
        file2.close();
        return allpass;
    }
};

TEST_F(InputTest, Selfconsistent_Read)
{
    ModuleIO::ReadInput readinput(0);
    readinput.check_ntype_flag = false;
    { // PW
        std::ofstream emptyfile("empty_INPUT");
        emptyfile << "INPUT_PARAMETERS";
        emptyfile.close();
        Parameter param;
        // readinput.read_parameters(param, "./empty_INPUT");
        EXPECT_NO_THROW(readinput.read_parameters(param, "./empty_INPUT"));
        readinput.write_parameters(param, "./my_INPUT1");
        readinput.clear();
        // readinput.read_parameters(param, "./my_INPUT1");
        EXPECT_NO_THROW(readinput.read_parameters(param, "./my_INPUT1"));
        readinput.write_parameters(param, "./my_INPUT2");
        EXPECT_TRUE(compare_two_files("./my_INPUT1", "./my_INPUT2"));
        EXPECT_TRUE(std::remove("./empty_INPUT") == 0);
        EXPECT_TRUE(std::remove("./my_INPUT1") == 0);
        EXPECT_TRUE(std::remove("./my_INPUT2") == 0);
        readinput.clear();
    }
    { // LCAO
        std::ofstream emptyfile("empty_INPUT");
        emptyfile << "INPUT_PARAMETERS\n";
        emptyfile << "basis_type           lcao";
        emptyfile.close();
        Parameter param;
        // readinput.read_parameters(param, "./empty_INPUT");
        EXPECT_NO_THROW(readinput.read_parameters(param, "./empty_INPUT"));
        readinput.write_parameters(param, "./my_INPUT1");
        readinput.clear();
        // readinput.read_parameters(param, "./my_INPUT1");
        EXPECT_NO_THROW(readinput.read_parameters(param, "./my_INPUT1"));
        readinput.write_parameters(param, "./my_INPUT2");
        EXPECT_TRUE(compare_two_files("./my_INPUT1", "./my_INPUT2"));
        EXPECT_TRUE(std::remove("./empty_INPUT") == 0);
        EXPECT_TRUE(std::remove("./my_INPUT1") == 0);
        EXPECT_TRUE(std::remove("./my_INPUT2") == 0);
        readinput.clear();
    }
}

TEST_F(InputTest, DftuMixingAutoSet)
{
    ModuleIO::ReadInput readinput(0);
    readinput.check_ntype_flag = false;

    {
        std::ofstream input_file("input_dftu_auto_all_missing");
        input_file << "INPUT_PARAMETERS\n";
        input_file << "dft_plus_u 1\n";
        input_file.close();

        Parameter param;
        testing::internal::CaptureStdout();
        EXPECT_NO_THROW(readinput.read_parameters(param, "./input_dftu_auto_all_missing"));
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_DOUBLE_EQ(param.inp.mixing_restart, 0.0);
        EXPECT_TRUE(param.inp.mixing_dmr);
        EXPECT_EQ(param.inp.mixing_dmr_start, 10);
        EXPECT_THAT(output, testing::HasSubstr("automatically set mixing_dmr = 1"));
        EXPECT_THAT(output, testing::HasSubstr("automatically set mixing_dmr_start = 10"));
        readinput.clear();
        EXPECT_TRUE(std::remove("./input_dftu_auto_all_missing") == 0);
    }

    {
        std::ofstream input_file("input_dftu_auto_partial_missing");
        input_file << "INPUT_PARAMETERS\n";
        input_file << "dft_plus_u 1\n";
        input_file << "mixing_dmr_start 20\n";
        input_file.close();

        Parameter param;
        testing::internal::CaptureStdout();
        EXPECT_NO_THROW(readinput.read_parameters(param, "./input_dftu_auto_partial_missing"));
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_DOUBLE_EQ(param.inp.mixing_restart, 0.0);
        EXPECT_TRUE(param.inp.mixing_dmr);
        EXPECT_EQ(param.inp.mixing_dmr_start, 20);
        EXPECT_THAT(output, testing::HasSubstr("automatically set mixing_dmr = 1"));
        EXPECT_THAT(output, testing::Not(testing::HasSubstr("automatically set mixing_dmr_start = 10")));
        readinput.clear();
        EXPECT_TRUE(std::remove("./input_dftu_auto_partial_missing") == 0);
    }

    {
        std::ofstream input_file("input_dftu_auto_disabled");
        input_file << "INPUT_PARAMETERS\n";
        input_file << "dft_plus_u 1\n";
        input_file << "mixing_dmr 0\n";
        input_file.close();

        Parameter param;
        testing::internal::CaptureStdout();
        EXPECT_NO_THROW(readinput.read_parameters(param, "./input_dftu_auto_disabled"));
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_DOUBLE_EQ(param.inp.mixing_restart, 0.0);
        EXPECT_FALSE(param.inp.mixing_dmr);
        EXPECT_EQ(param.inp.mixing_dmr_start, 10);
        EXPECT_THAT(output, testing::Not(testing::HasSubstr("automatically set mixing_dmr = 1")));
        EXPECT_THAT(output, testing::HasSubstr("automatically set mixing_dmr_start = 10"));
        readinput.clear();
        EXPECT_TRUE(std::remove("./input_dftu_auto_disabled") == 0);
    }

    {
        std::ofstream input_file("input_dftu_auto_off");
        input_file << "INPUT_PARAMETERS\n";
        input_file << "dft_plus_u 0\n";
        input_file.close();

        Parameter param;
        testing::internal::CaptureStdout();
        EXPECT_NO_THROW(readinput.read_parameters(param, "./input_dftu_auto_off"));
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_DOUBLE_EQ(param.inp.mixing_restart, 0.0);
        EXPECT_FALSE(param.inp.mixing_dmr);
        EXPECT_EQ(param.inp.mixing_dmr_start, 10);
        EXPECT_THAT(output, testing::Not(testing::HasSubstr("automatically set mixing_dmr = 1")));
        EXPECT_THAT(output, testing::Not(testing::HasSubstr("automatically set mixing_dmr_start = 10")));
        readinput.clear();
        EXPECT_TRUE(std::remove("./input_dftu_auto_off") == 0);
    }
}

TEST_F(InputTest, Check)
{
    ModuleIO::ReadInput readinput(0);
    readinput.check_ntype_flag = false;
    {
        std::ofstream emptyfile("empty_INPUT");
        emptyfile << "INPUT_PARAMETERS";
        emptyfile.close();

        Parameter param;
        readinput.read_parameters(param, "./empty_INPUT");
        readinput.write_parameters(param, "./INPUT.ref");
        EXPECT_TRUE(std::remove("./empty_INPUT") == 0);
        readinput.clear();
    }

    ModuleIO::ReadInput::check_mode = true;
    Parameter param;
    testing::internal::CaptureStdout();
    EXPECT_EXIT(readinput.read_parameters(param, "./INPUT.ref"), ::testing::ExitedWithCode(0), "");
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(output, testing::HasSubstr("INPUT parameters have been successfully checked!"));
    EXPECT_TRUE(std::remove("./INPUT.ref") == 0);
}
