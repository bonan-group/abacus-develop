#include <cassert>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <ctime>
#include "memory_recorder.h"
#include "global_variable.h"
#include "source_base/parallel_reduce.h"

namespace ModuleBase
{
//    8 bit  = 1 Byte
// 1024 Byte = 1 KB
// 1024 KB   = 1 MB
// 1024 MB   = 1 GB

const double memory_warning_threshold_mb = 20.0;

double Memory::total = 0.0;
int Memory::complex_matrix_memory = 2*sizeof(double); // 16 byte
int Memory::double_memory = sizeof(double); // 8 byte
int Memory::int_memory = sizeof(int); // 4.0 Byte
int Memory::bool_memory = sizeof(bool); // 1.0 Byte
int Memory::float_memory = sizeof(float); // 4.0 Byte
int Memory::short_memory = sizeof(short); // 2.0 Byte

int Memory::n_memory = 1000;
int Memory::n_now = 0;
bool Memory::init_flag = false;

#if defined(__CUDA) || defined(__ROCM)

namespace
{
std::string json_escape(const std::string& input)
{
    std::string output;
    output.reserve(input.size());
    for (const char ch : input)
    {
        switch (ch)
        {
        case '"':
            output += "\\\"";
            break;
        case '\\':
            output += "\\\\";
            break;
        case '\b':
            output += "\\b";
            break;
        case '\f':
            output += "\\f";
            break;
        case '\n':
            output += "\\n";
            break;
        case '\r':
            output += "\\r";
            break;
        case '\t':
            output += "\\t";
            break;
        default:
            output += std::iscntrl(static_cast<unsigned char>(ch)) ? '?' : ch;
            break;
        }
    }
    return output;
}
}

double Memory::total_gpu = 0.0;
double Memory::current_gpu = 0.0;
double Memory::peak_gpu = 0.0;
int Memory::n_now_gpu = 0;
bool Memory::init_flag_gpu = false;
bool Memory::mem_stream_enabled = false;
std::string Memory::mem_stream_path;
std::ofstream Memory::mem_stream;

std::string *Memory::name_gpu;
std::string *Memory::class_name_gpu;
double *Memory::consume_gpu;

#endif


std::string *Memory::name;
std::string *Memory::class_name;
double *Memory::consume;

Memory::Memory()
{
}

Memory::~Memory()
{
}

double Memory::calculate_mem(const long &n_in,const std::string &type)
{
	double n = static_cast<double>(n_in);
	double mem = 0.0;
	
	double factor = 1.0/1024.0/1024.0;
	double complex_matrix_mem = complex_matrix_memory * factor;
	double double_mem = double_memory * factor;
	double int_mem = int_memory * factor;
	double bool_mem = bool_memory * factor;
	double float_mem = float_memory * factor;
	double short_mem = short_memory * factor;

	if(type=="ModuleBase::ComplexMatrix" || type=="complexmatrix" || type=="cdouble")
	{
		mem = complex_matrix_mem;
	}
	else if(type=="real" || type=="double")
	{
		mem = double_mem;
	}
	else if(type=="int")
	{
		mem = int_mem;
	}
	else if(type=="bool")
	{
		mem = bool_mem;
	}
	else if(type=="short")
	{
		mem = short_mem;
	}
	else if(type=="float")
	{
		mem = float_mem;
	}
	else if(type=="AtomLink")
	{
		mem =  int_mem * 2 + double_mem * 3;
	}
	else if(type=="ModuleBase::Vector3<double>")
	{
		mem = 3 * double_mem;
	}
	else
	{
		std::cout<<"not this type in memory storage : "<<type << std::endl;
	}
	total += n * mem;	
	return n*mem;
}
	

double Memory::record
(
 	const std::string &class_name_in,
	const std::string &name_in,
	const long &n_in,
	const std::string &type,
	const bool accumulate
)
{
	if(!Memory::init_flag)
	{
		name = new std::string[n_memory];
		class_name = new std::string[n_memory];
		consume = new double[n_memory];
		for(int i=0;i<n_memory;i++)
		{
			consume[i] = 0.0;
		}
		Memory::init_flag = true;
	}

	int find = 0;
	for(find = 0; find < n_now; find++)
	{
		if( name_in == name[find] )
		{
			break;
		}
	}

	// find == n_now : found a new record.	
	if(find == n_now)
	{
		n_now++;
		name[find] = name_in;
		class_name[find] = class_name_in;
	}
	if(n_now >= n_memory)
	{
		std::cout<<" Error! Too many memories required.";
		return 0.0;
	}

	consume[find] = Memory::calculate_mem(n_in,type);

	if(consume[find] > memory_warning_threshold_mb)
	{
		print(name[find], consume[find]);
	}
	return consume[find];
}

void Memory::record
(
	const std::string &name_in,
	const long &n_in,
	const bool accumulate
)
{
	if(!Memory::init_flag)
	{
		name = new std::string[n_memory];
		class_name = new std::string[n_memory];
		consume = new double[n_memory];
		for(int i=0;i<n_memory;i++)
		{
			consume[i] = 0.0;
		}
		Memory::init_flag = true;
	}

	int find = 0;
	for(find = 0; find < n_now; find++)
	{
		if( name_in == name[find] )
		{
			break;
		}
	}

	// find == n_now : found a new record.	
	if(find == n_now)
	{
		n_now++;
		name[find] = name_in;
		class_name[find] = "";
	}
	if(n_now >= n_memory)
	{
		std::cout<<" Error! Too many memories has been recorded.";
		return;
	}

	const double factor = 1.0/1024.0/1024.0;
	double size_mb = n_in * factor;

	if(accumulate)
	{
		consume[find] += size_mb;
		Memory::total += size_mb;
	}
	else
	{
		if(consume[find] < size_mb)
		{
			Memory::total += size_mb - consume[find];
			consume[find] = size_mb;
			if(consume[find] > memory_warning_threshold_mb)
			{
				print(name[find], consume[find]);
			}
		}
	}

	return;
}

#if defined(__CUDA) || defined(__ROCM)

double Memory::record_gpu
(
 	const std::string &class_name_in,
	const std::string &name_in,
	const long &n_in,
	const std::string &type,
	const bool accumulate
)
{
	if(!Memory::init_flag_gpu)
	{
		init_gpu_records();
	}

	int find = 0;
	for(find = 0; find < n_now_gpu; find++)
	{
		if( name_in == name_gpu[find] )
		{
			break;
		}
	}

	// find == n_now : found a new record.	
	if(find == n_now_gpu)
	{
		n_now_gpu++;
		name_gpu[find] = name_in;
		class_name_gpu[find] = class_name_in;
	}
	if(n_now_gpu >= n_memory)
	{
		std::cout<<" Error! Too many gpu memories required.";
		return 0.0;
	}

	consume_gpu[find] = Memory::calculate_mem(n_in,type);

	if(consume_gpu[find] > memory_warning_threshold_mb)
	{
		print(name_gpu[find], consume_gpu[find]);
	}
	return consume_gpu[find];
}

void Memory::record_gpu(const std::string &name_in, const size_t &n_in)
{
    record_gpu(name_in, n_in, false);
}

void Memory::record_gpu
(
	const std::string &name_in,
	const size_t &n_in,
	const bool accumulate
)
{
	if(!Memory::init_flag_gpu)
	{
		init_gpu_records();
	}

	int find = 0;
	for(find = 0; find < n_now_gpu; find++)
	{
		if( name_in == name_gpu[find] )
		{
			break;
		}
	}

	// find == n_now : found a new record.	
	if(find == n_now_gpu)
	{
		n_now_gpu++;
		name_gpu[find] = name_in;
		class_name_gpu[find] = "";
	}
	if(n_now_gpu >= n_memory)
	{
		std::cout<<" Error! Too many gpu memories has been recorded.";
		return;
	}

	const double factor = 1.0/1024.0/1024.0;
	double size_mb = n_in * factor;

	if(accumulate)
	{
		consume_gpu[find] += size_mb;
		Memory::total_gpu += size_mb;
	}
	else
	{
		if(consume_gpu[find] < size_mb)
		{
			Memory::total_gpu += size_mb - consume_gpu[find];
			consume_gpu[find] = size_mb;
			if(consume_gpu[find] > memory_warning_threshold_mb)
			{
				print(name_gpu[find], consume_gpu[find]);
			}
		}
	}
	return;
}

void Memory::set_stream_enabled(const bool enabled)
{
    set_stream_enabled(enabled, "");
}

void Memory::set_stream_enabled(const bool enabled, const std::string& path)
{
    mem_stream_enabled = enabled;
    if (mem_stream.is_open())
    {
        mem_stream.close();
    }
    if (!enabled)
    {
        return;
    }

    mem_stream_path = path.empty() ? "memory_stream.jsonl" : path;
    mem_stream.open(mem_stream_path, std::ios::out | std::ios::trunc);
}

bool Memory::stream_enabled()
{
    return mem_stream_enabled;
}

double Memory::get_gpu_peak_mb()
{
    return peak_gpu;
}

void Memory::init_gpu_records()
{
    if (Memory::init_flag_gpu)
    {
        return;
    }
    name_gpu = new std::string[n_memory];
    class_name_gpu = new std::string[n_memory];
    consume_gpu = new double[n_memory];
    for (int i = 0; i < n_memory; ++i)
    {
        consume_gpu[i] = 0.0;
    }
    Memory::init_flag_gpu = true;
}

void Memory::record_gpu_alloc(const std::string& name_in, const size_t n_in)
{
    const double factor = 1.0 / 1024.0 / 1024.0;
    const double size_mb = static_cast<double>(n_in) * factor;
    current_gpu += size_mb;
    peak_gpu = std::max(peak_gpu, current_gpu);
    if (name_in != "no_record")
    {
        record_gpu(name_in, n_in);
    }
    else
    {
        total_gpu = std::max(total_gpu, peak_gpu);
        init_gpu_records();
    }

    if (mem_stream_enabled && mem_stream.is_open())
    {
        mem_stream << "{\"event\":\"alloc\",\"name\":\"" << json_escape(name_in) << "\",\"bytes\":" << n_in
                   << ",\"current_mb\":" << std::setprecision(12) << current_gpu
                   << ",\"peak_mb\":" << peak_gpu << "}" << std::endl;
    }
}

void Memory::reset_gpu()
{
    current_gpu = 0.0;
    peak_gpu = 0.0;
    total_gpu = 0.0;
    n_now_gpu = 0;
}

void Memory::record_gpu_free(const size_t n_in)
{
    const double factor = 1.0 / 1024.0 / 1024.0;
    const double size_mb = static_cast<double>(n_in) * factor;
    current_gpu = std::max(0.0, current_gpu - size_mb);

    if (mem_stream_enabled && mem_stream.is_open())
    {
        mem_stream << "{\"event\":\"free\",\"bytes\":" << n_in
                   << ",\"current_mb\":" << std::setprecision(12) << current_gpu
                   << ",\"peak_mb\":" << peak_gpu << "}" << std::endl;
    }
}

#endif

void Memory::print(const std::string& mem_name, double size_mb)
{
	// Check if output stream is open before writing (may not be initialized during early setup)
	if (GlobalV::ofs_running.is_open())
	{
		GlobalV::ofs_running <<"\n *** Memory Allocation Warning *** "
		<<" "<< mem_name <<" "<< size_mb <<" MB" << std::endl;
	}
	return;
}


void Memory::finish(std::ofstream &ofs)
{
	print_all(ofs);
	if(init_flag)
	{
		delete[] name;
		delete[] class_name;
		delete[] consume;
		init_flag = false;
	}
#if defined(__CUDA) || defined(__ROCM)
	if(init_flag_gpu)
	{
		delete[] name_gpu;
		delete[] class_name_gpu;
		delete[] consume_gpu;
		init_flag_gpu = false;
	}
		if(mem_stream.is_open())
		{
			mem_stream.close();
		}
		reset_gpu();
#endif
	return;
}

void Memory::print_all(std::ofstream &ofs)
{
	if(!init_flag) 
	{
#if defined(__CUDA) || defined(__ROCM)
		if(!init_flag_gpu)
		{
			return;
		}
#else
		return;
#endif
	}

	const double small = 1.0; // unit is MB 
#ifdef __MPI
	Parallel_Reduce::reduce_all(Memory::total);
#if defined(__CUDA) || defined(__ROCM)
	Parallel_Reduce::reduce_all(Memory::total_gpu);
#endif
#endif
	bool *print_flag = nullptr;
	if(init_flag)
	{
		ofs <<"\n NAME-------------------------|MEMORY(MB)------------------" << std::endl;
		ofs << std::right;
		ofs << std::setw(30)<< "total" << std::setw(15) <<std::setprecision(4)<< Memory::total << std::endl;

		assert(n_memory>0);

		print_flag = new bool[n_memory];

		for(int i=0; i<n_memory; i++)
		{
			print_flag[i] = false;
		}
	}

	for (int i=0; init_flag && i<n_memory; i++)
    {
#ifdef __MPI
		Parallel_Reduce::reduce_all(consume[i]);
#endif
	}

	for (int i=0; init_flag && i<n_memory; i++) // Xiaoyang fix memory record sum bug 2023/10/25
	{
		int k = 0;
		double tmp = -1.0;
		for(int j=0; j<n_memory; j++)
		{
			if(print_flag[j])
			{
				continue;
			}
			else if(tmp < consume[j])
			{
				k = j;
				tmp = consume[j];
			}
		}
		print_flag[k] = true;
		if ( consume[k] < small )
        {
			continue;
		}
		else
		{
			ofs << std::setw(30) << name[k]
            << std::setw(15) << consume[k] << std::endl;
		}

	}

#if defined(__CUDA) || defined(__ROCM)
	if(!init_flag_gpu) 
	{
		delete[] print_flag;
		return;
	}

	ofs <<"\n NAME-------------------------|GPU MEMORY(MB)--------------" << std::endl;
	ofs <<std::setw(30)<< "total" << std::setw(15) <<std::setprecision(4)<< Memory::total_gpu << std::endl;
	ofs <<std::setw(30)<< "current" << std::setw(15) <<std::setprecision(4)<< Memory::current_gpu << std::endl;
	ofs <<std::setw(30)<< "peak" << std::setw(15) <<std::setprecision(4)<< Memory::peak_gpu << std::endl;
    
    assert(n_memory>0);

	bool *print_flag_gpu = new bool[n_memory];

	for(int i=0; i<n_memory; i++) 
	{
		print_flag_gpu[i] = false;
	}	

	for (int i=0; i<n_memory; i++)
    {
#ifdef __MPI
		Parallel_Reduce::reduce_all(consume_gpu[i]);
#endif
	}

	for (int i=0; i<n_memory; i++) // Xiaoyang fix memory record sum bug 2023/10/25
	{
		int k = 0;
		double tmp = -1.0;
		for(int j=0; j<n_memory; j++)
		{
			if(print_flag_gpu[j])
			{
				continue;
			}
			else if(tmp < consume_gpu[j])
			{
				k = j;
				tmp = consume_gpu[j];
			}
		}
		print_flag_gpu[k] = true;
		if ( consume_gpu[k] < small )
        {
			continue;
		}
		else
		{
			ofs << std::setw(30) << name_gpu[k]
            << std::setw(15) << consume_gpu[k] << std::endl;
		}

	}

	delete[] print_flag_gpu;
#endif

	ofs<<" -------------   < 1.0 MB has been ignored ----------------"<<std::endl;
    ofs<<" ----------------------------------------------------------"<<std::endl;

	delete[] print_flag; //mohan fix by valgrind at 2012-04-02
	return;
}

}
