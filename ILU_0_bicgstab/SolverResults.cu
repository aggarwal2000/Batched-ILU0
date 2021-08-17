#include<iostream>
#include<vector>

#include "location_enums.h"
#include "SolverResults.h"
	
    
void SolverResults::AllocateMemory(const LOCATION loc)
{
    //fill this
    if(loc == LOCATION::CPU)
    {	
        if(this->ExistsCPU() == true)
            this->DeallocateMemory(LOCATION::CPU);

        this->true_residual_norm_cpu = new double[this->GetNumPages()];
        this->iter_residual_norm_cpu = new double[this->GetNumPages()];
        this->iter_count_cpu = new float[this->GetNumPages()];
        this->conv_flag_cpu = new int[this->GetNumPages()];
        this->cpu_exists = CPU_EXISTENCE::EXISTENT;
        
    }
    else
    {
        if(this->ExistsGPU() == true)
            this->DeallocateMemory(LOCATION::GPU);

        cudaError_t err;
        err = cudaMalloc((void**)&(this->true_residual_norm_gpu), sizeof(double)*this->GetNumPages());
        if(err != cudaSuccess)
            printf("\n%s", cudaGetErrorString(err));

        err = cudaMalloc((void**)&(this->iter_residual_norm_gpu), sizeof(double)*this->GetNumPages());
        if(err != cudaSuccess)
            printf("\n%s", cudaGetErrorString(err));

        err = cudaMalloc((void**)&(this->iter_count_gpu),sizeof(float)*this->GetNumPages() );
        if(err != cudaSuccess)
            printf("\n%s", cudaGetErrorString(err));

        err = cudaMalloc((void**)&(this->conv_flag_gpu), sizeof(int)*this->GetNumPages());
        if(err != cudaSuccess)
            printf("\n%s", cudaGetErrorString(err));
        this->gpu_exists = GPU_EXISTENCE::EXISTENT;
            
    }
    
}


void SolverResults::DeallocateMemory(const LOCATION loc)
{
    if(loc == LOCATION::CPU)
    {
        if(this->ExistsCPU() == true)
        {
            delete[] this->iter_count_cpu;
            delete[] this->conv_flag_cpu;
            delete[] this->true_residual_norm_cpu;
            this->cpu_exists = CPU_EXISTENCE::NON_EXISTENT;
            this->iter_count_cpu = nullptr;
            this->conv_flag_cpu = nullptr;
            this->true_residual_norm_cpu = nullptr;
            this->iter_residual_norm_cpu = nullptr;
        }
    }
    else
    {
        if(this->ExistsGPU() == true)
        {
            cudaFree(this->iter_count_gpu);
            cudaFree(this->conv_flag_gpu);
            cudaFree(this->true_residual_norm_gpu);
            this->gpu_exists = GPU_EXISTENCE::NON_EXISTENT;
            this->iter_count_gpu = nullptr;
            this->conv_flag_gpu = nullptr;
            this->true_residual_norm_gpu = nullptr;
            this->iter_residual_norm_gpu = nullptr;
        }	
    }
    
}



SolverResults:: ~SolverResults()
{
    if(this->ExistsCPU() == true)
        this->DeallocateMemory(LOCATION::CPU);

    if(this->ExistsGPU() == true)
        this->DeallocateMemory(LOCATION::GPU);
}

SolverResults::SolverResults()
{

}

SolverResults::SolverResults(const int num_pages, const double atol ,  const CPU_EXISTENCE to_allocate_on_cpu , const GPU_EXISTENCE to_allocate_on_gpu )
{
    this->SetNumPages(num_pages);
    this->SetATOL(atol);

    if(to_allocate_on_cpu == CPU_EXISTENCE::EXISTENT)
        this->AllocateMemory(LOCATION::CPU);

    if(to_allocate_on_gpu == GPU_EXISTENCE::EXISTENT)
        this->AllocateMemory(LOCATION::GPU);
}



void SolverResults::CopyFromCpuToGpu()
{
    if(this->ExistsCPU() == false)
    {	
        std::cout << " Not present on CPU " << std::endl;
        exit(1);

    }
    
    if(this->ExistsGPU() == true)
        this->DeallocateMemory(LOCATION::GPU);
    this->AllocateMemory(LOCATION::GPU);

    cudaError_t err;
    err = cudaMemcpy(this->iter_count_gpu ,this->iter_count_cpu, sizeof(float)*this->GetNumPages(), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
            printf("\n%s", cudaGetErrorString(err));
    err = cudaMemcpy(this->conv_flag_gpu, this->conv_flag_cpu, sizeof(int)*this->GetNumPages() , cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
            printf("\n%s", cudaGetErrorString(err));
    err = cudaMemcpy(this->true_residual_norm_gpu, this->true_residual_norm_cpu, sizeof(double)*this->GetNumPages(), cudaMemcpyHostToDevice );
    if(err != cudaSuccess)
            printf("\n%s", cudaGetErrorString(err));
    err = cudaMemcpy(this->iter_residual_norm_gpu, this->iter_residual_norm_cpu, sizeof(double)*this->GetNumPages(), cudaMemcpyHostToDevice );
        if(err != cudaSuccess)
            printf("\n%s", cudaGetErrorString(err));

}


void SolverResults::CopyFromGpuToCpu()
{
    if(this->ExistsGPU() == false)
    {
        std::cout << " Not present on GPU " << std::endl;
        exit(1);
    }

    if(this->ExistsCPU() == true)
        this->DeallocateMemory(LOCATION::CPU);

    this->AllocateMemory(LOCATION::CPU);

    cudaError_t err;
    err = cudaMemcpy(this->iter_count_cpu,this->iter_count_gpu, sizeof(float)*this->GetNumPages(), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
            printf("\n%s", cudaGetErrorString(err));
    err = cudaMemcpy(this->conv_flag_cpu,this-> conv_flag_gpu, sizeof(int)*this->GetNumPages(), cudaMemcpyDeviceToHost );
    if(err != cudaSuccess)
            printf("\n%s", cudaGetErrorString(err));
    err = cudaMemcpy(this->true_residual_norm_cpu, this->true_residual_norm_gpu,sizeof(double)*this->GetNumPages(), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
            printf("\n%s", cudaGetErrorString(err));
    err = cudaMemcpy(this->iter_residual_norm_cpu, this->iter_residual_norm_gpu,sizeof(double)*this->GetNumPages(), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
        printf("\n%s", cudaGetErrorString(err));
}



