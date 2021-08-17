#pragma once

#include "location_enums.h"



class SolverResults{

private:

    int npages = 0;
    double time_taken = 0;
    double abs_tol = 0;

    CPU_EXISTENCE cpu_exists = CPU_EXISTENCE::NON_EXISTENT; 
	GPU_EXISTENCE gpu_exists = GPU_EXISTENCE::NON_EXISTENT;

	float* iter_count_cpu = nullptr;
	int* conv_flag_cpu = nullptr;
	double* true_residual_norm_cpu = nullptr;
	double* iter_residual_norm_cpu = nullptr;

    float* iter_count_gpu = nullptr;
	int* conv_flag_gpu = nullptr;
	double* true_residual_norm_gpu = nullptr;
	double* iter_residual_norm_gpu = nullptr;

public:

	bool ExistsCPU() const
	{
		return this->cpu_exists == CPU_EXISTENCE::EXISTENT;
	}

	bool ExistsGPU() const
	{
		return this->gpu_exists == GPU_EXISTENCE::EXISTENT;
	}

	void SetNumPages(const int pages)
	{
		this->npages = pages;
	}

	int GetNumPages() const
	{
		return this->npages;
	}

    void SetATOL(const double tol) 
    {
        this->abs_tol = tol;
    }

    double GetATOL() const
    {
        return this->abs_tol;
    }

	void SetTimeTaken(const double time) 
    {
        this->time_taken = time;
    }

    double GetTimeTaken() const
    {
        return this->time_taken;
    }

	void AllocateMemory(const LOCATION loc);
	
	void DeallocateMemory(const LOCATION loc);
	

	float* GetPtrToCpuIterCount() const
	{
		return this->iter_count_cpu;
	}

	double* GetPtrToCpuTrueResNorm() const
	{
		return this->true_residual_norm_cpu;
	}

	double* GetPtrToCpuIterResNorm() const
	{
		return this->iter_residual_norm_cpu;
	}

	int* GetPtrToCpuConvFlag() const{
		return this->conv_flag_cpu;
	}

	float* GetPtrToGpuIterCount() const
	{
		return this->iter_count_gpu;
	}

	int* GetPtrToGpuConvFlag() const
	{
		return this->conv_flag_gpu;
	}

	double* GetPtrToGpuTrueResNorm() const{
		return this->true_residual_norm_gpu;
	}

	double* GetPtrToGpuIterResNorm() const{
		return this->iter_residual_norm_gpu;
	}

	
	~SolverResults();

	SolverResults();

    SolverResults(const int num_pages, const double atol,  const CPU_EXISTENCE to_allocate_on_cpu , const GPU_EXISTENCE to_allocate_on_gpu);

	void CopyFromCpuToGpu();

	void CopyFromGpuToCpu();

	SolverResults(const SolverResults& sol) = delete;
	SolverResults(const SolverResults&& sol) = delete;
	SolverResults& operator = (const SolverResults& sol) = delete;
	SolverResults& operator = (const SolverResults&& sol) = delete;
	
};






