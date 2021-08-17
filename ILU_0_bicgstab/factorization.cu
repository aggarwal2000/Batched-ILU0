#include <vector>
#include <cassert>
#include "header.h"
#include "matrix.h"
#include "factorization.h"


//TODO: Move kernels like: Print, Copy, Prefix Sum to some common operations file.


namespace{

const int max_possible_grid_dim = 65536;

template< typename T>
__global__ void copy_array(const T* const src, T* const dst, const int length);


__global__ void initialize_diag_info(const int nrows, int* const diag_info)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    for(int i = gid; i < nrows ; i += gridDim.x * blockDim.x)
    {
        diag_info[i] = 1;
    }
}



__global__ void missing_diagonal_elements(const int nrows, const int nnz, int* const diag_info, const int* const row_ptrs, const int* const col_idxs, 
    const double* const values)
{
    int gid = blockDim.x * blockIdx.x  + threadIdx.x;

    int warp_id = gid/WARP_SIZE;

    int id_within_warp = gid % WARP_SIZE;

    int total_num_warps = (gridDim.x * blockDim.x)/WARP_SIZE;

    for(int row_index = warp_id ; row_index < nrows ; row_index += total_num_warps)
    {
        for(int i = row_ptrs[row_index] + id_within_warp ; i < row_ptrs[row_index + 1]; i+= WARP_SIZE)
        {
            if(col_idxs[i] == row_index)
            {
                diag_info[row_index] = 0;
            }
        }
    }
}


//TODO: Parallelize Prefix Sum kernel
__global__ void prefix_sum_kernel(const int length , int* const array)
{
    if(threadIdx.x == 0)
    {
        for(int i = 0; i <= length - 2 ; i++)
        {
            array[i + 1] += array[i];
        }
    }
}




__global__ void  generate_row_pointers_common_diagonal_add(const int nrows, int* const new_row_ptrs, const int* const old_row_ptrs, const int* const diag_info)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    for(int i = gid; i < nrows; i += gridDim.x * blockDim.x)
    {
        new_row_ptrs[i + 1] = old_row_ptrs[i + 1] + diag_info[i];
    }

}


__global__ void  generate_col_holders_common_pattern_diagonal_add(const int nrows,  const int* const old_row_ptrs, 
    const int* const old_col_idxs , const int* const new_row_ptrs, int* const new_col_holders)
{
    int gid = blockDim.x * blockIdx.x  + threadIdx.x;

    int warp_id = gid/WARP_SIZE;

    int id_within_warp = gid % WARP_SIZE;

    int total_num_warps = (gridDim.x * blockDim.x)/WARP_SIZE;

    for(int row_index = warp_id ; row_index < nrows; row_index += total_num_warps)
    {
        int old_row_start = old_row_ptrs[row_index];
        int old_row_end = old_row_ptrs[row_index + 1];
    
        int new_row_start = new_row_ptrs[row_index];
        int new_row_end = new_row_ptrs[row_index + 1];
    
        if(new_row_end - new_row_start == old_row_end - old_row_start)
        {   
            //no missing diagonal elements
            
            int disp = new_row_start - old_row_start;
    
            for(int i = old_row_start + id_within_warp; i < old_row_end; i += WARP_SIZE)
            {   
                new_col_holders[i + disp ] = i;         
            }
        }
        else
        {   
            // a missing diagonal element in the row
            int disp = new_row_start - old_row_start;
    
            for(int i = old_row_start + id_within_warp; i < old_row_end; i += WARP_SIZE)
            {   
                int col_idx = old_col_idxs[i];
    
                int flag = col_idx > row_index; //false for lower threads say 0,1 ....; true for higher index threads 
                                                // where change is from false to true, for a certain thread : false; for next: true , 
                                                //the thread where its true, find that i + disp
                if(col_idx < row_index)
                {
                    new_col_holders[i + disp ] = i;  
                }
                else if(col_idx > row_index)
                {
                    new_col_holders[i + disp + 1 ] = i;
                }
    
                //one of the threads has to do an extra work of adding the diagonal element.
                if(id_within_warp == 0 && flag == 1)
                {
                    new_col_holders[i + disp ] =  -1 * row_index  +  -1;
                }
                else if( id_within_warp > 0 && flag == 1 )// 0--> false ; 1 ---> false;  2---> true ; 3---> true, THEN 2 will do the work of adding the diagonal element.
                {
                    //for a thread, get the flag value of a previous thread in warp
                    int flag_previous_thread_in_warp  = __shfl_up_sync( __activemask() , flag , 1);
                    if( flag_previous_thread_in_warp == 0)
                    {
                        new_col_holders[i + disp ] = -1 * row_index  +  -1;
    
                    }
                }   //NOTE: For this logic to work, matrix: sorted assumption is important.
    
    
                if(i == old_row_end - 1  && flag == 0) //when all elements at and after the diagonal were zero
                {   
                   
                    new_col_holders[i + disp + 1] = -1 * row_index  +  -1;
                }
                         
            }
    
            if(old_row_start == old_row_end  && id_within_warp == 0) //case when whole row was zero
            {   
                new_col_holders[new_row_start] = -1 * row_index  +  -1;
            }
    
        }
    }

}


__global__ void update_col_idxs_and_values_diagonal_add(const int npages, const int nrows, const int* const new_col_holders ,
     const int old_nnz, const int new_nnz, const int* const old_col_idxs,
const double* const old_values, int* const new_col_idxs, double* const new_values)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    for(int i = gid; i < new_nnz * npages; i += gridDim.x * blockDim.x)
    {   
            int index = new_col_holders[i % new_nnz];

            if(i < new_nnz)
                new_col_idxs[i] = index < 0 ?  - 1 * (index + 1) : old_col_idxs[index];

            int page_id = i/new_nnz;

            //new_values[(page_id * new_nnz) + (i % new_nnz)] = index < 0 ? 0 : old_values[index +  page_id * old_nnz];
            new_values[i] = index < 0 ? 0 : old_values[index +  page_id * old_nnz];
       
    }
}



__global__ void find_locn_of_diag_elements_kernel(const int nrows, int* const diag_ptrs , const int* const row_ptrs , const int* const col_idxs)
{   
    int gid = blockDim.x * blockIdx.x  + threadIdx.x;

    int warp_id = gid/WARP_SIZE;

    int id_within_warp = gid % WARP_SIZE;

    int total_num_warps = (gridDim.x * blockDim.x)/WARP_SIZE;

    for(int row_index = warp_id ; row_index < nrows; row_index += total_num_warps)
    {
        for(int i = row_ptrs[row_index] + id_within_warp; i < row_ptrs[row_index + 1]; i +=  WARP_SIZE)
        {
            if( col_idxs[i] == row_index ) //this will be executed for one and only thread in the warp
            {
                diag_ptrs[row_index] = i;
            }
        }
    }
}



__global__ void  count_nnz_per_row_L_and_U(const int nrows, const int* const row_ptrs, const int* const diag_ptrs , int* const row_ptrs_L, 
    int* const row_ptrs_U)
    {
        int gid = threadIdx.x + blockIdx.x * blockDim.x;
    
        if(gid == 0)
        {
            row_ptrs_U[0] = 0;
            row_ptrs_L[0] = 0;
        }
    
        for(int i = gid ; i < nrows; i += gridDim.x * blockDim.x)
        {   
            const int diag_ele_loc = diag_ptrs[i];
    
            row_ptrs_L[i + 1] = diag_ele_loc -  row_ptrs[i]  +  1;
            row_ptrs_U[i + 1] = row_ptrs[i + 1] - diag_ele_loc;
        }
    }
    




__global__ void generate_common_pattern_to_fill_L_and_U(const int nrows, const int* const row_ptrs, const int* const col_idxs, const int* const row_ptrs_L ,
const int* const row_ptrs_U, int* const L_col_holders , int* const U_col_holders)
{
        int gid = blockDim.x * blockIdx.x  + threadIdx.x;

    int warp_id = gid/WARP_SIZE;

    int id_within_warp = gid % WARP_SIZE;

    int total_num_warps = (gridDim.x * blockDim.x)/WARP_SIZE;

    for(int row_index = warp_id ; row_index < nrows; row_index += total_num_warps)
    {   
        int L_row_start = row_ptrs_L[row_index];
        int U_row_start = row_ptrs_U[row_index];
        int row_start = row_ptrs[row_index];
        int row_end = row_ptrs[row_index + 1];

        //int diag_ele_loc = diag_ptrs[row_index];

        int nnz_per_row_L = row_ptrs_L[row_index + 1] - row_ptrs_L[row_index];
        int diag_ele_loc = row_start + nnz_per_row_L - 1;
        
        for(int i = row_start + id_within_warp; i < row_end ; i += WARP_SIZE)
        {
            if(i < diag_ele_loc) //or col_idxs[i] < row_index
            {   
                const int corresponding_l_index = L_row_start + (i - row_start );
                L_col_holders[corresponding_l_index] = i;
            }
            else
            {
                if(i == diag_ele_loc)  //or col_idxs[i] == row_index
                {    
                    const int corresponding_l_index = L_row_start + (i - row_start);
                    L_col_holders[corresponding_l_index] = (-1 * row_index ) - 1;
                }

                const int corresponding_u_index = U_row_start + (i - diag_ele_loc);
                U_col_holders[ corresponding_u_index ] = i;
            }
        }
    }
}




__global__ void fill_L_and_U(const int npages, const int nrows, const int nnz, const int* const col_idxs, const double* const vals, const int L_nnz, int* const L_col_idxs, 
double* const L_vals, const int* const L_col_holders , const int U_nnz, int* const U_col_idxs, double* const U_vals, const int* const U_col_holders)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int greater_nnz = L_nnz > U_nnz ? L_nnz : U_nnz;

    for(int i = gid ; i < greater_nnz * npages; i += gridDim.x * blockDim.x)
    {   
        int page_id;
        int col;
        double val;

        if(i < L_nnz * npages)
        {
            int l_A_index = L_col_holders[i % L_nnz];
        
            if(l_A_index >= 0 )
            {
                col = col_idxs[l_A_index] ;
                page_id = i/L_nnz;
                val = vals[l_A_index  + page_id * nnz];

            }
            else
            {
                col = -1 * (l_A_index + 1);
                val = 1;
            }
            
            if(i < L_nnz)
            {
                L_col_idxs[i] = col;
            }
    
            L_vals[i] = val;
        }

        if(i < U_nnz * npages)
        {
            int u_A_index = U_col_holders[i % U_nnz];
            page_id = i/U_nnz;

            col = col_idxs[u_A_index];
            val = vals[u_A_index + page_id * nnz];

            if(i < U_nnz)
            {
                U_col_idxs[i] = col;
            }

            U_vals[i] = val;
        }
        
        
            
    }
}




__global__ void print_kernel(const int length, const int * const arr)
{
    if(threadIdx.x == 0)
    {   
        for(int i = 0; i < length; i++)
        {
            printf("\n %d ", arr[i]);
        }
        
    }
}

__global__ void print_kernel(const int length, const double * const arr)
{
    if(threadIdx.x == 0)
    {   
        for(int i = 0; i < length; i++)
        {
            printf("\n %g ", arr[i]);
        }
        
    }
}



template< typename T>
__global__ void copy_array(const T* const src, T* const dst, const int length)
{
    const int gid = threadIdx.x   +   blockIdx.x * blockDim.x;

    for(int i = gid ; i < length ; i += gridDim.x * blockDim.x)
    {
        dst[i] = src[i];
    }
}



}//unnamed namespace


void PrefixSum(const int length, int* const array)
{
    prefix_sum_kernel <<< 1, 1 >>>(length, array);
}




int Count_Missing_Diagonal_Elements(PagedCSRMatrices & Factored_Pages, int* const diag_info)
{   
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid_diag_info(ceil( (double)Factored_Pages.GetNumRows()/ (double)THREADS_PER_BLOCK ));

    initialize_diag_info <<<  grid_diag_info , block >>>( Factored_Pages.GetNumRows(), diag_info );

    const int total_num_warps = Factored_Pages.GetNumRows();
    const int num_warps_in_block = THREADS_PER_BLOCK/WARP_SIZE;
    const int grid_size = ceil((double)total_num_warps / (double)num_warps_in_block);

    dim3 grid_missing_eles(grid_size);

    missing_diagonal_elements<<<  grid_missing_eles, block  >>> ( Factored_Pages.GetNumRows(), Factored_Pages.GetNumNz(), diag_info, Factored_Pages.GetPtrToGpuRowPtrs(),
Factored_Pages.GetPtrToGpuColInd(), Factored_Pages.GetPtrToGpuValues() );

    PrefixSum(Factored_Pages.GetNumRows(), diag_info);

    int sum;
    cudaMemcpy( &sum, diag_info + (Factored_Pages.GetNumRows() - 1), sizeof(int), cudaMemcpyDeviceToHost );

    return sum;
}



void Add_Missing_Diagonal_Elements(PagedCSRMatrices & New_Factored_Pages , const PagedCSRMatrices & Factored_Pages, const int* const diag_info, 
    const int num_missing_diagonal_eles)
{
    const int new_nnz = Factored_Pages.GetNumNz() + num_missing_diagonal_eles;
    New_Factored_Pages.SetNumRows(Factored_Pages.GetNumRows());
    New_Factored_Pages.SetNumPages(Factored_Pages.GetNumPages());
    New_Factored_Pages.SetNumCols(Factored_Pages.GetNumCols());
    New_Factored_Pages.SetNumNz(new_nnz);
    New_Factored_Pages.AllocateMemory(LOCATION::GPU);

    dim3 block(THREADS_PER_BLOCK);
    
    dim3 grid_row_ptrs( ceil( (double)New_Factored_Pages.GetNumRows() / (double)THREADS_PER_BLOCK ) );

    generate_row_pointers_common_diagonal_add<<<  grid_row_ptrs , block >>>(Factored_Pages.GetNumRows(), New_Factored_Pages.GetPtrToGpuRowPtrs(), Factored_Pages.GetPtrToGpuRowPtrs(), diag_info );

    const int total_num_warps = Factored_Pages.GetNumRows();
    const int num_warps_in_block = THREADS_PER_BLOCK/WARP_SIZE;
    const int grid_size = ceil((double)total_num_warps / (double)num_warps_in_block);

    int* new_col_holders = nullptr;
    cudaMalloc( (void**)&new_col_holders , sizeof(int) * New_Factored_Pages.GetNumNz() );

    generate_col_holders_common_pattern_diagonal_add <<<  dim3(grid_size) , block  >>> ( Factored_Pages.GetNumRows(), Factored_Pages.GetPtrToGpuRowPtrs() , 
Factored_Pages.GetPtrToGpuColInd()  , New_Factored_Pages.GetPtrToGpuRowPtrs() , new_col_holders );

    int grid_size_col_and_vals_updation = ceil((double)(New_Factored_Pages.GetNumPages() * New_Factored_Pages.GetNumNz())/(double)THREADS_PER_BLOCK);

    

    if(grid_size_col_and_vals_updation > max_possible_grid_dim)
    {   
       // std::cout << "\n Using max possible grid dim at line:"  << __LINE__ << "\n";

        grid_size_col_and_vals_updation = max_possible_grid_dim;
    }

    update_col_idxs_and_values_diagonal_add <<<  dim3(grid_size_col_and_vals_updation) , block >>>(Factored_Pages.GetNumPages(), Factored_Pages.GetNumRows(), new_col_holders ,  Factored_Pages.GetNumNz() ,
     New_Factored_Pages.GetNumNz() , Factored_Pages.GetPtrToGpuColInd() , Factored_Pages.GetPtrToGpuValues() , New_Factored_Pages.GetPtrToGpuColInd() , 
    New_Factored_Pages.GetPtrToGpuValues() );

    cudaFree(new_col_holders);
}




void Find_locations_of_diagonal_elements(const PagedCSRMatrices & Factored_Pages , int* const diag_info)
{
    const int total_num_warps = Factored_Pages.GetNumRows();
    const int num_warps_in_block = THREADS_PER_BLOCK/WARP_SIZE;
    const int grid_size = ceil((double)total_num_warps / (double)num_warps_in_block);

    find_locn_of_diag_elements_kernel <<<  dim3(grid_size)  , dim3(THREADS_PER_BLOCK)  >>>(Factored_Pages.GetNumRows(), diag_info, Factored_Pages.GetPtrToGpuRowPtrs(),
     Factored_Pages.GetPtrToGpuColInd());
}






void Update_row_pointers_L_and_U_and_Allocate_Memory(const PagedCSRMatrices & Factored_Pages , const int* const diag_ptrs, PagedCSRMatrices & L_pages, PagedCSRMatrices & U_pages)
{   
    const int nrows = Factored_Pages.GetNumRows();
    int* row_ptrs_L = nullptr;
    int* row_ptrs_U = nullptr;
    cudaMalloc( (void**)&row_ptrs_L , sizeof(int) * (nrows + 1) );
    cudaMalloc( (void**)&row_ptrs_U, sizeof(int) * (nrows + 1) );

    //call kernel for updation
    dim3 block(THREADS_PER_BLOCK);

    dim3 grid(ceil(  (double)nrows + 1 / (double) THREADS_PER_BLOCK  ));

    count_nnz_per_row_L_and_U<<< grid , block >>>(nrows, Factored_Pages.GetPtrToGpuRowPtrs(), diag_ptrs ,  row_ptrs_L, row_ptrs_U);

    PrefixSum(nrows + 1, row_ptrs_L);
    PrefixSum(nrows + 1, row_ptrs_U);

    int nnz_L;
    int nnz_U;

    cudaMemcpy( &nnz_L , row_ptrs_L + nrows , sizeof(int), cudaMemcpyDeviceToHost );
    cudaMemcpy( &nnz_U , row_ptrs_U + nrows , sizeof(int), cudaMemcpyDeviceToHost );    


    L_pages.SetNumRows(nrows);
    L_pages.SetNumCols(nrows);
    L_pages.SetNumNz(nnz_L);
    L_pages.SetNumPages(Factored_Pages.GetNumPages());
    L_pages.AllocateMemory(LOCATION::GPU);

  

    U_pages.SetNumRows(nrows);
    U_pages.SetNumCols(nrows);
    U_pages.SetNumNz(nnz_U);
    U_pages.SetNumPages(Factored_Pages.GetNumPages());
    U_pages.AllocateMemory(LOCATION::GPU);

  
   
    dim3 grid_for_copy( ceil( (double) nrows + 1/ (double)THREADS_PER_BLOCK )  );

    copy_array <<<   grid_for_copy , block >>>(row_ptrs_L , L_pages.GetPtrToGpuRowPtrs(), nrows + 1); //find ways to avoid this extra copy
    copy_array <<<   grid_for_copy , block >>>(row_ptrs_U , U_pages.GetPtrToGpuRowPtrs(), nrows + 1); //find ways to avoid this extra copy
    cudaFree(row_ptrs_L);
    cudaFree(row_ptrs_U);
}





void Fill_L_and_U_col_idxs_and_vals(const PagedCSRMatrices & Factored_Pages, PagedCSRMatrices & L_pages, PagedCSRMatrices & U_pages)
{
   
    int* L_col_holders = nullptr;
    int* U_col_holders = nullptr;

    cudaMalloc( (void**)&L_col_holders , L_pages.GetNumNz() * sizeof(int) );
    cudaMalloc( (void**)&U_col_holders , U_pages.GetNumNz() * sizeof(int));

    const int total_num_warps = Factored_Pages.GetNumRows();
    const int num_warps_in_block = THREADS_PER_BLOCK/WARP_SIZE;
    const int grid_size = ceil((double)total_num_warps / (double)num_warps_in_block);

    //use one warp per row
    generate_common_pattern_to_fill_L_and_U <<< dim3(grid_size) , dim3(THREADS_PER_BLOCK) >>>(Factored_Pages.GetNumRows(), Factored_Pages.GetPtrToGpuRowPtrs(), Factored_Pages.GetPtrToGpuColInd(), 
L_pages.GetPtrToGpuRowPtrs(), U_pages.GetPtrToGpuRowPtrs(), L_col_holders , U_col_holders);

    const int greater_nnz = L_pages.GetNumNz() > U_pages.GetNumNz() ? L_pages.GetNumNz() : U_pages.GetNumNz();
    int grid_fill_LU = ceil( (double) greater_nnz * Factored_Pages.GetNumPages()  / (double)THREADS_PER_BLOCK );

   

    if(grid_fill_LU > max_possible_grid_dim)
    {   
        //std::cout << "\n Using max possible grid dim at line:"  << __LINE__ << "\n";
        grid_fill_LU = max_possible_grid_dim;
    }

    fill_L_and_U <<<  dim3(grid_fill_LU) ,  dim3(THREADS_PER_BLOCK) >>>(Factored_Pages.GetNumPages() ,Factored_Pages.GetNumRows(), Factored_Pages.GetNumNz() , Factored_Pages.GetPtrToGpuColInd(), Factored_Pages.GetPtrToGpuValues(), L_pages.GetNumNz() ,
     L_pages.GetPtrToGpuColInd(), L_pages.GetPtrToGpuValues(), L_col_holders,  U_pages.GetNumNz() , U_pages.GetPtrToGpuColInd(),
      U_pages.GetPtrToGpuValues(), U_col_holders);

    cudaFree(L_col_holders);
    cudaFree(U_col_holders);
  
}





void PrintPagedCSRMatrix(const PagedCSRMatrices & CSRMatrix_pages)
{   
    printf("\n\n row pointers: \n");
    print_kernel<<< 1, 1 >>>(CSRMatrix_pages.GetNumRows() + 1, CSRMatrix_pages.GetPtrToGpuRowPtrs());
    cudaDeviceSynchronize();

    printf("\n\n column indices: \n");
    print_kernel<<< 1, 1 >>>( CSRMatrix_pages.GetNumNz(), CSRMatrix_pages.GetPtrToGpuColInd() );
    cudaDeviceSynchronize();

    for(int page_id = 0; page_id < CSRMatrix_pages.GetNumPages() ; page_id++)
    {
        printf("\n\n values of page id: %d  are \n", page_id);
        print_kernel <<< 1, 1 >>>( CSRMatrix_pages.GetNumNz(), CSRMatrix_pages.GetPtrToGpuValues() + CSRMatrix_pages.GetNumNz() * page_id);
        cudaDeviceSynchronize();
    }
  
}





void Copy_Gpu_PagedCSRMatrices(const PagedCSRMatrices & Src_pages, PagedCSRMatrices & Dst_pages)
{   
    assert(Src_pages.ExistsGPU() == true);

    Dst_pages.SetNumPages(Src_pages.GetNumPages());
    Dst_pages.SetNumCols(Src_pages.GetNumCols());
    Dst_pages.SetNumRows(Src_pages.GetNumRows());
    Dst_pages.SetNumNz(Src_pages.GetNumNz());
    Dst_pages.AllocateMemory(LOCATION::GPU);  
    
    const int* src_row_ptrs = Src_pages.GetPtrToGpuRowPtrs();
    const int* src_col_idxs = Src_pages.GetPtrToGpuColInd();
    const double* src_vals = Src_pages.GetPtrToGpuValues();

    int* dst_row_ptrs = Dst_pages.GetPtrToGpuRowPtrs();
    int* dst_col_idxs = Dst_pages.GetPtrToGpuColInd();
    double* dst_vals = Dst_pages.GetPtrToGpuValues();

    dim3 block(THREADS_PER_BLOCK,1,1);

    copy_array <<<  dim3(    ceil(      (double)Dst_pages.GetNumRows() + 1/ (double)THREADS_PER_BLOCK    )    ) , block >>> (src_row_ptrs , dst_row_ptrs, Dst_pages.GetNumRows() + 1);
    copy_array<<< dim3( ceil( (double)Dst_pages.GetNumNz()/ (double)THREADS_PER_BLOCK )) , block  >>>(src_col_idxs, dst_col_idxs, Dst_pages.GetNumNz() );
    copy_array<<< dim3( ceil( (double)Dst_pages.GetNumNz() * Dst_pages.GetNumPages()/ (double)THREADS_PER_BLOCK )) , block >>>(src_vals, dst_vals, Dst_pages.GetNumNz() * Dst_pages.GetNumPages());

}