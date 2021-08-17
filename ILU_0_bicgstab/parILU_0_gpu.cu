#include<iostream>
#include<stdio.h>
#include<vector>
#include<array>
#include<cassert>
#include<chrono>
#include<cmath>
#include "cuda_profiler_api.h"
#include "matrix.h"
#include "ReadWriteData.h"
#include "header.h"
#include "factorization.h"
#include "parILU_0.h"

namespace{


__device__ __forceinline__ void  parilu0_sweep_for_batch_entry_approach2(const int nnz, const double* const A_vals , double* const L_vals , double* const U_vals , 
   const int* const dependencies , const int* const nz_ptrs)
{

    for(int el = threadIdx.x ; el < nnz; el+= blockDim.x) //non-coalesced access but some data locality
    {
        double diag_val = 1;

        int st = nz_ptrs[el];
        int end = nz_ptrs[el + 1] - 1;

        bool has_diag_dependency = (end + 1- st )%2 == 0 ? true : false;

        double sum = 0;

        for(int idx = st + 1; idx <= end - 1; idx += 2)
        {
            sum += L_vals[dependencies[idx]] * U_vals[dependencies[idx + 1]];
        }

        if(has_diag_dependency == true)
        {
            diag_val = U_vals[dependencies[end]];
        }

        double to_write = (A_vals[el] - sum)/diag_val;

        if(has_diag_dependency == true)
        {
            L_vals[dependencies[st]] = to_write;
        }
        else
        {
            U_vals[dependencies[st]] = to_write;
        }

        
    }

}




__global__ void  compute_parilu_0_approach2_kernel(const int npages , const int nrows, const int nnz, 
const double* const values, const int L_nnz, double* const L_values, 
const int U_nnz, double* const U_values , const int num_iterations , const int* const dependencies, const int* const nz_ptrs)
{
    for(int page_id = blockIdx.x; page_id < npages; page_id += gridDim.x)
    {   
        extern __shared__ double shared_mem[];
        double* L_values_sh = shared_mem;
        double* U_values_sh = L_values_sh + L_nnz;

        // __shared__ double L_values_sh[MAX_NUM_NZ];
        // __shared__ double U_values_sh[MAX_NUM_NZ]; or MAX_NUM_NZ + MAX_NUM_ROWS to account for diagonal addition in case there are some missing diagonal elements.

        for(int i = threadIdx.x ; i < L_nnz ; i += blockDim.x)
        {
            L_values_sh[i] = *(L_values + page_id * L_nnz + i);
        }

        for(int i = threadIdx.x ; i < U_nnz ; i += blockDim.x)
        {
            U_values_sh[i] = *(U_values + page_id * U_nnz + i);
        }

       

        __syncthreads();


        for(int iter = 0; iter < num_iterations; iter++)
        {
            // parilu0_sweep_for_batch_entry_approach2(nnz, values + page_id * nnz , L_values + page_id * L_nnz , U_values + page_id * U_nnz , 
            //  dependencies , nz_ptrs);

            parilu0_sweep_for_batch_entry_approach2(nnz, values + page_id * nnz , L_values_sh , U_values_sh, 
                dependencies , nz_ptrs);

            __syncthreads();
        }

        for(int i = threadIdx.x ; i < L_nnz ; i += blockDim.x)
        {
            *(L_values + page_id * L_nnz + i) =  L_values_sh[i];
        }

        for(int i = threadIdx.x ; i < U_nnz ; i += blockDim.x)
        {
            *(U_values + page_id * U_nnz + i) =  U_values_sh[i];
        }


    }
}




void create_dependency_graph_parilu(const PagedCSRMatrices & A_pages, std::vector<int> & dependencies , std::vector<int> & nz_ptrs, 
    const PagedCSRMatrices & L_pages , const PagedCSRMatrices & U_pages)
{   
    const int nrows = A_pages.GetNumRows();
   
    int* const row_ptrs = new int[A_pages.GetNumRows() + 1];
    int* const col_idxs = new int[A_pages.GetNumNz()];
    int* const L_row_ptrs =  new int[L_pages.GetNumRows() + 1];
    int* const L_col_idxs = new int[L_pages.GetNumNz()];
    int* const U_row_ptrs = new int[U_pages.GetNumRows() + 1];
    int* const U_col_idxs = new int[U_pages.GetNumNz()];
    cudaMemcpy(row_ptrs , A_pages.GetPtrToGpuRowPtrs(), sizeof(int) * (A_pages.GetNumRows() + 1) , cudaMemcpyDeviceToHost);
    cudaMemcpy(col_idxs , A_pages.GetPtrToGpuColInd() , sizeof(int) * A_pages.GetNumNz(), cudaMemcpyDeviceToHost );
    cudaMemcpy(L_row_ptrs , L_pages.GetPtrToGpuRowPtrs(), sizeof(int) * (L_pages.GetNumRows() + 1) , cudaMemcpyDeviceToHost );
    cudaMemcpy(L_col_idxs , L_pages.GetPtrToGpuColInd() , sizeof(int) * L_pages.GetNumNz() , cudaMemcpyDeviceToHost );
    cudaMemcpy(U_row_ptrs , U_pages.GetPtrToGpuRowPtrs(), sizeof(int) * (U_pages.GetNumRows() + 1) , cudaMemcpyDeviceToHost );
    cudaMemcpy(U_col_idxs , U_pages.GetPtrToGpuColInd() , sizeof(int) * U_pages.GetNumNz() , cudaMemcpyDeviceToHost );
    
    nz_ptrs[0] = 0;

    for(int row_index = 0; row_index < nrows ; row_index++ )
    {
        const int row_start = row_ptrs[row_index];
        const int row_end = row_ptrs[row_index + 1];

        for(int loc = row_start; loc < row_end; loc++)
        {   

            const int col_index = col_idxs[loc];

            if(row_index > col_index)
            {
                //find corr. index in L

                const int L_idx = loc - row_start  + L_row_ptrs[row_index];

                dependencies.push_back(L_idx);

               // printf("\n write in L: %d \n", L_idx);
            }
            else
            {
                //find corr. index in U

                const int U_idx =  ( U_row_ptrs[row_index + 1] - 1)  -  (row_end -1 - loc ); 

                dependencies.push_back(U_idx);

               // printf("\n write in U: %d , U_row_ptrs[row_index + 1] : %d , row_end -1 : %d , loc: %d \n", U_idx, U_row_ptrs[row_index + 1] , row_end -1 , loc );

            }

            const int k_max = std::min(row_index , col_index) - 1;

            int num_dependencies = 0;

            for(int l_idx = L_row_ptrs[row_index]; l_idx < L_row_ptrs[row_index + 1]; l_idx++)
            {
                const int k = L_col_idxs[l_idx];

                if(k > k_max)
                {
                    continue;
                }

                //find corresponding u at position k,col_index

                for(int u_idx = U_row_ptrs[k]; u_idx < U_row_ptrs[k + 1]; u_idx++)
                {
                    if(U_col_idxs[u_idx] == col_index)
                    {
                        dependencies.push_back(l_idx);
                        dependencies.push_back(u_idx);

                        num_dependencies += 2;
                    }
                }
            }

            
            if(row_index > col_index)
            {
                const int diag_loc = U_row_ptrs[col_index]; 
                //std::cout << "line 346: " << col_index << std::endl;
                dependencies.push_back(diag_loc);

                num_dependencies++;
            }


            nz_ptrs[loc + 1] = nz_ptrs[loc] + num_dependencies + 1;


        }
    }

    delete[] row_ptrs;
    delete[] col_idxs;
    delete[] L_row_ptrs;
    delete[] L_col_idxs;
    delete[] U_row_ptrs;
    delete[] U_col_idxs;
}




void Print_Parilu_Dep_Graph(const std::vector<int> & dependencies_cpu , const std::vector<int> & nz_ptrs_cpu)
{
    for(int loc = 0; loc < nz_ptrs_cpu.size() - 1 ; loc++)
    {
        const int start = nz_ptrs_cpu[loc];
        const int end = nz_ptrs_cpu[loc + 1];

        printf("\n\n Dependencies for element at loc = %d are: ", loc);

        if( (end - start)%2 == 0)
        {
            printf("\nwrite in L\n");
        } 
        else
        {
            printf("\n write in U \n");
        }

        printf("\n To write at idx: %d \n", dependencies_cpu[start]);

        for(int i = start + 1; i < end; i++)
        {
            printf("\n %d ", dependencies_cpu[i]);
        }
    }
}




void ParILU0_Approach2(const PagedCSRMatrices & A_sorted_Pages, const PagedCSRMatrices & L_pages, const PagedCSRMatrices & U_pages, const int num_iterations )
{
   
    std::vector<int> dependencies_cpu;
    std::vector<int > nz_ptrs_cpu(A_sorted_Pages.GetNumNz() + 1);
    
    
    create_dependency_graph_parilu(A_sorted_Pages,  dependencies_cpu, nz_ptrs_cpu, L_pages , U_pages);
    
    int* dependencies = nullptr;
    int* nz_ptrs = nullptr;
    cudaMalloc((void**)&dependencies , dependencies_cpu.size() * sizeof(int));
    cudaMemcpy(dependencies , dependencies_cpu.data() , dependencies_cpu.size() * sizeof(int) , cudaMemcpyHostToDevice );
    cudaMalloc((void**)&nz_ptrs , nz_ptrs_cpu.size() * sizeof(int) );
    cudaMemcpy( nz_ptrs , nz_ptrs_cpu.data() , nz_ptrs_cpu.size() * sizeof(int) , cudaMemcpyHostToDevice );
    

    //Print_Parilu_Dep_Graph(dependencies_cpu , nz_ptrs_cpu);

    dim3 block(THREADS_PER_BLOCK);
    int grid_dim =  A_sorted_Pages.GetNumPages();
    dim3 grid( grid_dim );

    const int dynamic_shared_mem_size = sizeof(double) * ( L_pages.GetNumNz() + U_pages.GetNumNz());

    compute_parilu_0_approach2_kernel <<< grid , block, dynamic_shared_mem_size >>>(A_sorted_Pages.GetNumPages(), A_sorted_Pages.GetNumRows(), A_sorted_Pages.GetNumNz(), 
    A_sorted_Pages.GetPtrToGpuValues(),  L_pages.GetNumNz(), L_pages.GetPtrToGpuValues(), 
    U_pages.GetNumNz(), U_pages.GetPtrToGpuValues() , num_iterations , dependencies, nz_ptrs );   


    cudaFree(dependencies);
    cudaFree(nz_ptrs);
}

} //unnamed namespace

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

void ParILU_0_Factorization_Gpu(const PagedCSRMatrices & A_pages , PagedCSRMatrices & L_pages, PagedCSRMatrices & U_pages, const int num_iterations)
{
    //first assert matrix is square
    assert(A_pages.GetNumCols() == A_pages.GetNumRows());


    PagedCSRMatrices A_sorted_Pages; 
    //We would want to use copy assignment here... or even a copy constructor.  implement it later...
    //copy A to A_sorted
    Copy_Gpu_PagedCSRMatrices(A_pages , A_sorted_Pages); //TODO: avoid an extra copy here... if matrix is already sorted.

    //SortCSRMatrix(A_sorted_Pages); if unsorted, pls sort the paged matrix before proceeding. (All these matrices are already sorted.(sorted while storing))
    
    int* diag_info = nullptr;
    cudaMalloc((void**)&diag_info, sizeof(int) * A_sorted_Pages.GetNumRows());

    int num_missing_diagonal_eles = Count_Missing_Diagonal_Elements(A_sorted_Pages , diag_info);

    if(num_missing_diagonal_eles > 0)
    {
        PagedCSRMatrices New_A_sorted_Pages;

        Add_Missing_Diagonal_Elements(New_A_sorted_Pages, A_sorted_Pages, diag_info , num_missing_diagonal_eles);

        Copy_Gpu_PagedCSRMatrices(New_A_sorted_Pages , A_sorted_Pages); //TODO: avoid an extra copy here

    }

    // std::cout << "\n\nMATRIX AFTER ADDITION OF DIAGONAL ELEMENTS: " << std::endl;
    // PrintPagedCSRMatrix(A_sorted_Pages);

    //continue to use A_sorted here...

    Find_locations_of_diagonal_elements(A_sorted_Pages, diag_info);
    //std::cout << "\n\nLocn of diagonal elements:" << std::endl;
    //print_kernel<<< 1, 1 >>>(A_sorted_Pages.GetNumRows(), diag_info);
    //cudaDeviceSynchronize();

    Update_row_pointers_L_and_U_and_Allocate_Memory(A_sorted_Pages , diag_info, L_pages, U_pages);

    Fill_L_and_U_col_idxs_and_vals(A_sorted_Pages, L_pages, U_pages);

    //Now L_pages and U_pages are initialized... (Initial guess is ready)

    cudaProfilerStart();

   
    //approach 2
    ParILU0_Approach2(A_sorted_Pages , L_pages , U_pages , num_iterations);
  

    cudaProfilerStop();

    // std::cout << "\n\nMATRIX L: " << std::endl;
    // PrintPagedCSRMatrix(L_pages);
 
 
    // std::cout << "\n\nMATRIX U: " << std::endl;
    // PrintPagedCSRMatrix(U_pages);
 
    
    cudaFree(diag_info);

    cudaDeviceSynchronize(); //for timing purpose

}


