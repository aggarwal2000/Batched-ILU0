#pragma once

#include "location_enums.h"


class PagedCSRMatrices{

private:
	int nrows = 0;
	int ncols = 0;
	int nnz = 0;
	int npages = 0;

	CPU_EXISTENCE cpu_exists = CPU_EXISTENCE::NON_EXISTENT; 
	GPU_EXISTENCE gpu_exists = GPU_EXISTENCE::NON_EXISTENT;

	int* col_ind_cpu = nullptr;
	int* row_ptrs_cpu = nullptr;
	double* values_cpu = nullptr;

	int* col_ind_gpu = nullptr;
	int* row_ptrs_gpu = nullptr;
	double* values_gpu = nullptr;

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

	void SetNumRows(const int rows)
	{
		this->nrows = rows;
	}

	int GetNumRows() const{
		return this->nrows;
	}

	void SetNumCols(const int cols)
	{
		this->ncols = cols;
	}

	int GetNumCols()const{
		return this->ncols;
	}

	void SetNumNz(const int nz)
	{
		this->nnz = nz;
	}

	int GetNumNz()const{
		return this->nnz;
	}

	void AllocateMemory(const LOCATION loc);
	
	void DeallocateMemory(const LOCATION loc);
	

	double* GetPtrToCpuValues() const
	{
		return this->values_cpu;
	}

	int* GetPtrToCpuColInd() const
	{
		return this->col_ind_cpu;
	}

	int* GetPtrToCpuRowPtrs() const{
		return this->row_ptrs_cpu;
	}

	double* GetPtrToGpuValues() const
	{
		return this->values_gpu;
	}

	int* GetPtrToGpuColInd() const
	{
		return this->col_ind_gpu;
	}

	int* GetPtrToGpuRowPtrs() const{
		return this->row_ptrs_gpu;
	}

	
	~PagedCSRMatrices();

	PagedCSRMatrices();

	void CopyFromCpuToGpu();

	void CopyFromGpuToCpu();

	PagedCSRMatrices(const PagedCSRMatrices& mat_pages) = delete;
	PagedCSRMatrices(const PagedCSRMatrices&& mat_pages) = delete;
	PagedCSRMatrices& operator = (const PagedCSRMatrices& mat_pages) = delete;
	PagedCSRMatrices& operator = (const PagedCSRMatrices&& mat_pages) = delete;
	
};





class PagedCOOMatrices{

private:
	int nrows = 0;
	int ncols = 0;
	int nnz = 0;
	int npages = 0;

	CPU_EXISTENCE cpu_exists = CPU_EXISTENCE::NON_EXISTENT; 
	GPU_EXISTENCE gpu_exists = GPU_EXISTENCE::NON_EXISTENT;

	int* col_ind_cpu = nullptr;
	int* row_ind_cpu = nullptr;
	double* values_cpu = nullptr;

	int* col_ind_gpu = nullptr;
	int* row_ind_gpu = nullptr;
	double* values_gpu = nullptr;

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

	void SetNumRows(const int rows)
	{
		this->nrows = rows;
	}

	int GetNumRows() const{
		return this->nrows;
	}

	void SetNumCols(const int cols)
	{
		this->ncols = cols;
	}

	int GetNumCols()const{
		return this->ncols;
	}

	void SetNumNz(const int nz)
	{
		this->nnz = nz;
	}

	int GetNumNz()const{
		return this->nnz;
	}

	void AllocateMemory(const LOCATION loc);
	
	void DeallocateMemory(const LOCATION loc);
	

	double* GetPtrToCpuValues() const
	{
		return this->values_cpu;
	}

	int* GetPtrToCpuColInd() const
	{
		return this->col_ind_cpu;
	}

	int* GetPtrToCpuRowInd() const{
		return this->row_ind_cpu;
	}

	double* GetPtrToGpuValues() const
	{
		return this->values_gpu;
	}

	int* GetPtrToGpuColInd() const
	{
		return this->col_ind_gpu;
	}

	int* GetPtrToGpuRowInd() const{
		return this->row_ind_gpu;
	}

	
	~PagedCOOMatrices();

	PagedCOOMatrices();

	void CopyFromCpuToGpu();

	void CopyFromGpuToCpu();

	PagedCOOMatrices(const PagedCOOMatrices& mat_pages) = delete;
	PagedCOOMatrices(const PagedCOOMatrices&& mat_pages) = delete;
	PagedCOOMatrices& operator = (const PagedCOOMatrices& mat_pages) = delete;
	PagedCOOMatrices& operator = (const PagedCOOMatrices&& mat_pages) = delete;
	
};








class PagedVectors{

private:
	int npages = 0;
	int nelements = 0;

	CPU_EXISTENCE cpu_exists = CPU_EXISTENCE::NON_EXISTENT; 
	GPU_EXISTENCE gpu_exists = GPU_EXISTENCE::NON_EXISTENT;

	double* values_cpu = nullptr;
	double* values_gpu = nullptr;

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

	void SetNumElements(const int eles)
	{
		this->nelements = eles;
	}

	int GetNumElements() const
	{
		return this->nelements;
	}

	void AllocateMemory(const LOCATION loc);

	void DeallocateMemory(const LOCATION loc);

	~PagedVectors();

	PagedVectors();

	double* GetPtrToCpuValues() const
	{
		return this->values_cpu;
	}

	double* GetPtrToGpuValues() const
	{
		return this->values_gpu;
	}

	void CopyFromCpuToGpu();


	void CopyFromGpuToCpu();


	PagedVectors(const PagedVectors& mat_pages) = delete;
	PagedVectors(const PagedVectors&& mat_pages) = delete;
	PagedVectors& operator = (const PagedVectors& mat_pages) = delete;
	PagedVectors& operator = (const PagedVectors&& mat_pages) = delete;

};



