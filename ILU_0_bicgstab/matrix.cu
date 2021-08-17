#include<iostream>
#include<vector>

#include "location_enums.h"
#include "matrix.h"


	void PagedCSRMatrices::AllocateMemory(const LOCATION loc)
	{
		//fill this
		if(loc == LOCATION::CPU)
		{	
			if(this->ExistsCPU() == true)
				this->DeallocateMemory(LOCATION::CPU);

			this->values_cpu = new double[this->GetNumPages()* this->GetNumNz()];
			this->row_ptrs_cpu = new int[this->GetNumRows() + 1];
			this->col_ind_cpu = new int[this->GetNumNz()];
			this->cpu_exists = CPU_EXISTENCE::EXISTENT;
			
		}
		else
		{
			if(this->ExistsGPU() == true)
				this->DeallocateMemory(LOCATION::GPU);

			cudaError_t err;
			err = cudaMalloc((void**)&(this->values_gpu), sizeof(double)*this->GetNumPages()* this->GetNumNz());
			if(err != cudaSuccess)
				printf("\n%s", cudaGetErrorString(err));

			err = cudaMalloc((void**)&(this->row_ptrs_gpu),sizeof(int)*(this->GetNumRows() + 1) );
			if(err != cudaSuccess)
				printf("\n%s", cudaGetErrorString(err));

			err = cudaMalloc((void**)&(this->col_ind_gpu), sizeof(int)*this->GetNumNz());
			if(err != cudaSuccess)
				printf("\n%s", cudaGetErrorString(err));
			this->gpu_exists = GPU_EXISTENCE::EXISTENT;
				
		}
		
	}

	void PagedCSRMatrices::DeallocateMemory(const LOCATION loc)
	{
		if(loc == LOCATION::CPU)
		{
			if(this->ExistsCPU() == true)
			{
				delete[] this->values_cpu;
				delete[] this->row_ptrs_cpu;
				delete[] this->col_ind_cpu;
				this->cpu_exists = CPU_EXISTENCE::NON_EXISTENT;
				this->values_cpu = nullptr;
				this->row_ptrs_cpu = nullptr;
				this->col_ind_cpu = nullptr;
			}
		}
		else
		{
			if(this->ExistsGPU() == true)
			{
				cudaFree(this->values_gpu);
				cudaFree(this->row_ptrs_gpu);
				cudaFree(this->col_ind_gpu);
				this->gpu_exists = GPU_EXISTENCE::NON_EXISTENT;
				this->values_gpu = nullptr;
				this->row_ptrs_gpu = nullptr;
				this->col_ind_gpu = nullptr;
			}	
		}
		
	}

	

	PagedCSRMatrices:: ~PagedCSRMatrices()
	{
		if(this->ExistsCPU() == true)
			this->DeallocateMemory(LOCATION::CPU);

		if(this->ExistsGPU() == true)
			this->DeallocateMemory(LOCATION::GPU);
	}

	PagedCSRMatrices::PagedCSRMatrices()
	{

	}

	void PagedCSRMatrices::CopyFromCpuToGpu()
	{
		if(this->ExistsCPU() == false)
		{	
			std::cout << " Not present on CPU " << std::endl;
			exit(1);

		}
		
		//Only thing is some vals.. in nrows etc..,cpu mem already -ok,  but user changed this afterwards, cpu thing not latest.
		if(this->ExistsGPU() == true)
			this->DeallocateMemory(LOCATION::GPU);
		this->AllocateMemory(LOCATION::GPU);

		cudaError_t err;
		err = cudaMemcpy(this->values_gpu,this->values_cpu, sizeof(double)*this->GetNumPages()* this->GetNumNz(), cudaMemcpyHostToDevice);
		if(err != cudaSuccess)
				printf("\n%s", cudaGetErrorString(err));
		err = cudaMemcpy(this->row_ptrs_gpu, row_ptrs_cpu, sizeof(int)*(this->GetNumRows() + 1), cudaMemcpyHostToDevice);
		if(err != cudaSuccess)
				printf("\n%s", cudaGetErrorString(err));
		err = cudaMemcpy(this->col_ind_gpu, this->col_ind_cpu, sizeof(int)*this->GetNumNz(), cudaMemcpyHostToDevice );
		if(err != cudaSuccess)
				printf("\n%s", cudaGetErrorString(err));

	}


	void PagedCSRMatrices::CopyFromGpuToCpu()
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
		err = cudaMemcpy(this->values_cpu,this->values_gpu, sizeof(double)*this->GetNumPages()* this->GetNumNz(), cudaMemcpyDeviceToHost);
		if(err != cudaSuccess)
				printf("\n%s", cudaGetErrorString(err));
		err = cudaMemcpy(this->row_ptrs_cpu,this-> row_ptrs_gpu, sizeof(int)*(this->GetNumRows() + 1), cudaMemcpyDeviceToHost );
		if(err != cudaSuccess)
				printf("\n%s", cudaGetErrorString(err));
		err = cudaMemcpy(this->col_ind_cpu, this->col_ind_gpu,sizeof(int)*this->GetNumNz(), cudaMemcpyDeviceToHost);
		if(err != cudaSuccess)
				printf("\n%s", cudaGetErrorString(err));
	}


//-----------------------------------------------------------------------



void PagedCOOMatrices::AllocateMemory(const LOCATION loc)
{
	//fill this
	if(loc == LOCATION::CPU)
	{	
		if(this->ExistsCPU() == true)
			this->DeallocateMemory(LOCATION::CPU);

		this->values_cpu = new double[this->GetNumPages()* this->GetNumNz()];
		this->row_ind_cpu = new int[this->GetNumNz()];
		this->col_ind_cpu = new int[this->GetNumNz()];
		this->cpu_exists = CPU_EXISTENCE::EXISTENT;
		
	}
	else
	{
		if(this->ExistsGPU() == true)
			this->DeallocateMemory(LOCATION::GPU);

		cudaError_t err;
		err = cudaMalloc((void**)&(this->values_gpu), sizeof(double)*this->GetNumPages()* this->GetNumNz());
		if(err != cudaSuccess)
			printf("\n%s", cudaGetErrorString(err));

		err = cudaMalloc((void**)&(this->row_ind_gpu),sizeof(int)*this->GetNumNz() );
		if(err != cudaSuccess)
			printf("\n%s", cudaGetErrorString(err));

		err = cudaMalloc((void**)&(this->col_ind_gpu), sizeof(int)*this->GetNumNz());
		if(err != cudaSuccess)
			printf("\n%s", cudaGetErrorString(err));
		this->gpu_exists = GPU_EXISTENCE::EXISTENT;
			
	}
	
}

void PagedCOOMatrices::DeallocateMemory(const LOCATION loc)
{
	if(loc == LOCATION::CPU)
	{
		if(this->ExistsCPU() == true)
		{
			delete[] this->values_cpu;
			delete[] this->row_ind_cpu;
			delete[] this->col_ind_cpu;
			this->cpu_exists = CPU_EXISTENCE::NON_EXISTENT;
			this->values_cpu = nullptr;
			this->row_ind_cpu = nullptr;
			this->col_ind_cpu = nullptr;
		}
	}
	else
	{
		if(this->ExistsGPU() == true)
		{
			cudaFree(this->values_gpu);
			cudaFree(this->row_ind_gpu);
			cudaFree(this->col_ind_gpu);
			this->gpu_exists = GPU_EXISTENCE::NON_EXISTENT;
			this->values_gpu = nullptr;
			this->row_ind_gpu = nullptr;
			this->col_ind_gpu = nullptr;
		}	
	}
	
}



PagedCOOMatrices:: ~PagedCOOMatrices()
{
	if(this->ExistsCPU() == true)
		this->DeallocateMemory(LOCATION::CPU);

	if(this->ExistsGPU() == true)
		this->DeallocateMemory(LOCATION::GPU);
}

PagedCOOMatrices::PagedCOOMatrices()
{

}


void PagedCOOMatrices::CopyFromCpuToGpu()
{
	if(this->ExistsCPU() == false)
	{	
		std::cout << " Not present on CPU " << std::endl;
		exit(1);

	}
	
	//Only thing is some vals.. in nrows etc..,cpu mem already -ok,  but user changed this afterwards, cpu thing not latest.
	if(this->ExistsGPU() == true)
		this->DeallocateMemory(LOCATION::GPU);
	this->AllocateMemory(LOCATION::GPU);

	cudaError_t err;
	err = cudaMemcpy(this->values_gpu,this->values_cpu, sizeof(double)*this->GetNumPages()* this->GetNumNz(), cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
			printf("\n%s", cudaGetErrorString(err));
	err = cudaMemcpy(this->row_ind_gpu, row_ind_cpu, sizeof(int)* this->GetNumNz(), cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
			printf("\n%s", cudaGetErrorString(err));
	err = cudaMemcpy(this->col_ind_gpu, this->col_ind_cpu, sizeof(int)*this->GetNumNz(), cudaMemcpyHostToDevice );
	if(err != cudaSuccess)
			printf("\n%s", cudaGetErrorString(err));

}


void PagedCOOMatrices::CopyFromGpuToCpu()
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
	err = cudaMemcpy(this->values_cpu,this->values_gpu, sizeof(double)*this->GetNumPages()* this->GetNumNz(), cudaMemcpyDeviceToHost);
	if(err != cudaSuccess)
			printf("\n%s", cudaGetErrorString(err));
	err = cudaMemcpy(this->row_ind_cpu,this-> row_ind_gpu, sizeof(int)*this->GetNumNz(), cudaMemcpyDeviceToHost );
	if(err != cudaSuccess)
			printf("\n%s", cudaGetErrorString(err));
	err = cudaMemcpy(this->col_ind_cpu, this->col_ind_gpu,sizeof(int)*this->GetNumNz(), cudaMemcpyDeviceToHost);
	if(err != cudaSuccess)
			printf("\n%s", cudaGetErrorString(err));
}




//------------------------------------------------------------------------



	void PagedVectors::AllocateMemory(const LOCATION loc)
	{
		//fill this
		if(loc == LOCATION::CPU)
		{
			if(this->cpu_exists == CPU_EXISTENCE::EXISTENT)
			{	
				this->DeallocateMemory(LOCATION::CPU);
			}

			this->values_cpu = new double[this->GetNumPages() * this->GetNumElements()];
			this->cpu_exists = CPU_EXISTENCE::EXISTENT;
		}
		else
		{
			if(this->gpu_exists == GPU_EXISTENCE::EXISTENT)
			{
				this->DeallocateMemory(LOCATION::GPU);
			}

			cudaError_t err;
			err = cudaMalloc((void**)&(this->values_gpu), this->GetNumPages() * this->GetNumElements()*sizeof(double));
			if(err != cudaSuccess)
				printf("\n%s", cudaGetErrorString(err));
			this->gpu_exists = GPU_EXISTENCE::EXISTENT;


		}
		
	}

	void PagedVectors::DeallocateMemory(const LOCATION loc)
	{
		if(loc == LOCATION::CPU)
		{
			if(this->ExistsCPU() == true)
			{
				delete[] this->values_cpu;
				this->cpu_exists = CPU_EXISTENCE::NON_EXISTENT;
				this->values_cpu = nullptr;
				
			}
		}
		else
		{
			if(this->ExistsGPU() == true)
			{
				cudaFree(this->values_gpu);
				this->gpu_exists = GPU_EXISTENCE::NON_EXISTENT;
				this->values_gpu = nullptr;
			
			}	
		}	
	}

	PagedVectors::~PagedVectors()
	{
		if(this->ExistsCPU() == true)
			this->DeallocateMemory(LOCATION::CPU);

		if(this->ExistsGPU() == true)
			this->DeallocateMemory(LOCATION::GPU);
	}

	PagedVectors::PagedVectors()
	{

	}

	

	void PagedVectors::CopyFromCpuToGpu()
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
		err = cudaMemcpy(this->values_gpu,this->values_cpu, sizeof(double)*this->GetNumPages()* this->GetNumElements(), cudaMemcpyHostToDevice);
		if(err != cudaSuccess)
				printf("\n%s", cudaGetErrorString(err));
	}


	void PagedVectors::CopyFromGpuToCpu()
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
		err = cudaMemcpy(this->values_cpu,this->values_gpu, sizeof(double)*this->GetNumPages()* this->GetNumElements(), cudaMemcpyDeviceToHost);
		if(err != cudaSuccess)
				printf("\n%s", cudaGetErrorString(err));
	}


	




//----------------------------------------------
