#include<iostream>
#include<stdio.h>
#include<vector>
#include<cassert>
#include<chrono>
#include<cmath>
#include "cuda_profiler_api.h"
#include "matrix.h"
#include "ReadWriteData.h"
#include "header.h"
#include "PrecondBiCGSTAB.h"
#include "SolverResults.h"

#include "parILU_0.h"
#include "ILU_0.h"


//TODO: Move kernels like: Norm, Innerproduct, SpMV, sparse trsv to some other file.
//TODO: parallel reductions(norm, inner product)

namespace {

__device__ void ComputeResidualVec(const int num_rows,const int* const A_row_ptrs_shared,const int* const A_col_inds_shared,
    const double* const A_vals_shared,const double* const b_shared,const double* const x_shared, double* const res_shared)
{
    
    int num_warps_in_block = blockDim.x/WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE -1);
    int local_warp_index = threadIdx.x/WARP_SIZE; //local warp index in a block

    for(int i = local_warp_index; i < num_rows ; i = i + num_warps_in_block)
    {
        int start_ind_for_row = A_row_ptrs_shared[i];
        int end_ind_for_row = A_row_ptrs_shared[i + 1];

        double temp = 0;

        for(int k = start_ind_for_row + lane; k < end_ind_for_row; k = k + WARP_SIZE)
        {
            temp += A_vals_shared[k]*x_shared[A_col_inds_shared[k]];
        }

        double val = temp;

        //warp level reduction
        for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    
        if(lane == 0)
        {   
            res_shared[i] = b_shared[i] - val;
        }

       

    }

}



__device__ void block_reduce(double* data)
{
    int nt = blockDim.x;
    int tid = threadIdx.x;

    for (int k = nt / 2; k > 0; k = k / 2)
    {
        __syncthreads();
        if (tid < k)
        {
            data[tid] += data[tid + k];
        }
    }


}

__device__ double inner_product1(const int num_rows, const double* const vec1_shared, const double* const vec2_shared, double* const temp_shared)
{   
    double tmp = 0;

    for(int i = threadIdx.x; i < num_rows; i = i + blockDim.x)
    {
        tmp = tmp + vec1_shared[i]*vec2_shared[i];
    }

    temp_shared[threadIdx.x] = tmp;

    __syncthreads();

    block_reduce(temp_shared);

    __syncthreads();

    return temp_shared[0];

}

__device__ double inner_product(const int num_rows, const double* const vec1_shared, const double* const vec2_shared)
{   
    double tmp = 0;

    for(int i=0; i < num_rows; i++)
        tmp = tmp + vec1_shared[i]*vec2_shared[i];
    

    __syncthreads();

   
   return tmp;

}



__device__ double L2Norm(const int num_rows,const double* const vec_shared)
{
    return sqrt(inner_product(num_rows,vec_shared,vec_shared));
}




__device__ void initialization(const int num_rows, const int num_nz,const int* const row_ptrs,const int* const col_inds,
    const double* const vals_mat,const double* const vals_rhs ,double* const  x_shared,double* const v_shared,double* const p_shared,
double* const r_shared,double* const r_hat_shared)
{
    int num_warps_in_block = blockDim.x/WARP_SIZE;
    int local_thread_id = threadIdx.x; //local thread id in block
    int local_warp_index = threadIdx.x/WARP_SIZE; //local warp index in a block
    int page_id = blockIdx.x;
    int lane  = threadIdx.x & (WARP_SIZE -1);

    
    // x:initialize with 0s {Later on, have a provision for user's choice. So, may be x_pages: initialize--> with something n copy that to here}
    // r = b - A*x
    // r_hat = r
    // rho, alpha, omega
    // v with 0s
    // p with 0s

   

    for(int i = local_thread_id ; i < num_rows; i = i + blockDim.x)
    {   
        x_shared[i] = 0.00;
        v_shared[i] = 0.00;
        p_shared[i] = 0.00;
        
    }

    __syncthreads();

    //initialize r
    ComputeResidualVec(num_rows, row_ptrs, col_inds, vals_mat + page_id*num_nz, vals_rhs + page_id*num_rows, x_shared,r_shared);
    __syncthreads();

    
    for(int i = local_warp_index*WARP_SIZE  + lane ; i < num_rows ; i = i + num_warps_in_block*WARP_SIZE)
    {   
        r_hat_shared[i] = r_shared[i];
    }

    
}


__device__ void Update_p(const int num_rows,double* const p_shared,const double* const r_shared,const double* const v_shared,
    const double beta,const double omega_old)
{
    
    for(int i = threadIdx.x ; i < num_rows; i = i + blockDim.x)
    {   
        double val = r_shared[i] + beta*(p_shared[i] - omega_old*v_shared[i]);
        p_shared[i] = val;
        
    }

} 


__device__ void Update_s(const int num_rows,double* const s_shared,const double* const r_shared,const double alpha,const double* const v_shared)
{
    for(int i = threadIdx.x; i < num_rows; i = i + blockDim.x)
    {
        s_shared[i] = r_shared[i] - alpha*v_shared[i];
    }
}

__device__ void Update_x(const int num_rows,double* const x_shared,const double* const p_shared,const double* const s_shared,const double alpha,
    const double omega_new)
{
    for(int i = threadIdx.x; i < num_rows; i = i + blockDim.x)
    {
        x_shared[i] = x_shared[i] + alpha*p_shared[i] + omega_new*s_shared[i];
    }
}


__device__ void Update_x_middle(const int num_rows, double* const x_shared,const double* const p_shared, const double alpha)
{   
    for(int i = threadIdx.x; i < num_rows; i = i + blockDim.x)
    {
        x_shared[i] = x_shared[i] + alpha*p_shared[i] ;
    }

}



__device__ void Update_r(const int num_rows,double* const r_shared,const double* const s_shared,const double* const t_shared,const double omega_new)
{
    
    for(int i = threadIdx.x; i < num_rows; i = i + blockDim.x)
    {
        r_shared[i] = s_shared[i] - omega_new*t_shared[i];
    }

    
}





__device__ void SpMV(const int num_rows,const int* const mat_row_ptrs_shared,const int* const mat_col_inds_shared,
    const double* const mat_vals_shared,const double* const vec_shared,double* const ans_shared)
{
  
    int num_warps_in_block = blockDim.x/WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE -1);
    int local_warp_index = threadIdx.x/WARP_SIZE; //local warp index in a block

    for(int i = local_warp_index; i < num_rows ; i = i + num_warps_in_block)
    {
        int start_ind_for_row = mat_row_ptrs_shared[i];
        int end_ind_for_row = mat_row_ptrs_shared[i + 1];

        double temp = 0;

        for(int k = start_ind_for_row + lane; k < end_ind_for_row; k = k + WARP_SIZE)
        {
            temp += mat_vals_shared[k]*vec_shared[mat_col_inds_shared[k]];
        }

        double val = temp;

        //warp level reduction
        for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    

        if(lane == 0)
        {   
            ans_shared[i] = val;
        }

        

    }

}







__global__ void KernelFillTrueResNorms(const int num_rows, const int num_nz, const int num_pages, const int* const row_ptrs, 
    const int* const col_inds, const double* const vals_mat, const double* const vals_rhs, const double* const vals_ans, double* const true_residual_norms)
{
    __shared__ int A_row_ptrs_shared[MAX_NUM_ROWS + 1];
    __shared__ int A_col_inds_shared[MAX_NUM_NZ];
    __shared__ double A_vals_shared[MAX_NUM_NZ];
    __shared__ double b_shared[MAX_NUM_ROWS];
    __shared__ double x_shared[MAX_NUM_ROWS];

    __shared__ double r_true_shared[MAX_NUM_ROWS];

    int page_id = blockIdx.x;
    
    if(page_id < num_pages)
    {

            for(int i = threadIdx.x; i < num_rows + 1; i = i + blockDim.x)
            {   
                A_row_ptrs_shared[i] = row_ptrs[i];

            }


            for(int i = threadIdx.x ; i < num_nz; i = i + blockDim.x)
            {   
                A_col_inds_shared[i] = col_inds[i];
                A_vals_shared[i] = vals_mat[i + page_id*num_nz];

            }

            for(int i = threadIdx.x ; i < num_rows; i = i + blockDim.x)
            {   
                b_shared[i] = vals_rhs[i + page_id*num_rows];
                x_shared[i] = vals_ans[i + page_id*num_rows];   
            }

            __syncthreads();

            ComputeResidualVec(num_rows, A_row_ptrs_shared, A_col_inds_shared, A_vals_shared, b_shared, x_shared,r_true_shared);
            __syncthreads();


            double true_resi_norm = L2Norm(num_rows,r_true_shared);

            if(threadIdx.x == 0)
                true_residual_norms[page_id] = true_resi_norm;
    }



}    


__device__ void legacy_sparse_lower_triangular_solve(const int num_rows, const int* const L_row_ptrs, const int* const L_col_idxs, 
const double* const L_values, const double* const vec_shared, volatile double* const temp_vec_shared)
{   
   
        const int row_index = threadIdx.x;

        if(row_index >= num_rows)
        {
            return;
        }

        double sum = 0;

        const int start = L_row_ptrs[row_index];
        const int end = L_row_ptrs[row_index + 1] - 1;
        int i = start;
        
        
        bool completed = false;

        while(!completed)
        {   
            

            const int col_index = L_col_idxs[i];

            if( i < end  &&  isfinite(temp_vec_shared[col_index]))
            {
                sum += L_values[i] * temp_vec_shared[col_index];
                i++;
            }

           
            if(i == end)
            {   
                temp_vec_shared[row_index] = (vec_shared[row_index] - sum)/L_values[end];
               
                completed = true;
               
            }

          
        }

       
        
}


__device__ void legacy_sparse_upper_triangular_solve(const int num_rows,  const int* const U_row_ptrs, const int* const U_col_idxs, 
const double* const U_values, volatile const double* const temp_vec_shared, volatile double* const vec_hat_shared)
{
    const int row_index = threadIdx.x;

    if(row_index >= num_rows)
    {
        return;
    }

    double sum = 0;

    const int start = U_row_ptrs[row_index];
    const int end = U_row_ptrs[row_index + 1]  - 1;
    int i = end;

    bool completed = false;

    while(!completed )
    {   
       

        const int col_index = U_col_idxs[i];

        if( i > start && isfinite(vec_hat_shared[col_index]))
        {
            sum += U_values[i] * vec_hat_shared[col_index];
            i--;
        }

      
        if(i == start)
        {
            vec_hat_shared[row_index] = (temp_vec_shared[row_index] - sum)/U_values[start];
           
            completed = true;
        }

      
    }

}

__device__ void ApplyPreconditionerILU(const int num_rows , const int* const L_row_ptrs, 
    const int* const L_col_idxs , const double* const L_values,  const int* const U_row_ptrs,
    const int* const U_col_idxs, const double* const U_values,  const double* const vec_shared,volatile double* const vec_hat_shared)
{

   
    // vec_hat = precond * vec
    // => L * U  * vec_hat = vec
    // => L * y = vec , find y , and then U * vec_hat = y, find vec_hat

    // we need sparse triangular solves for that!
    //if we want to use the busy waiting while loop approach, then the num_rows should be <= threadblock size, else there is possibility of a deadlock !

    //TODO: For upper trsv, use thread 0 for the bottommost row, this way we could avoid :  assert(num_rows <= blockDim.x), as there won't be a possibility of deadlock then!

    
    assert(num_rows <= blockDim.x);

    __shared__  volatile double temp_vec_shared[MAX_NUM_ROWS];

    for(int i = threadIdx.x ; i < num_rows; i += blockDim.x)
    {
        temp_vec_shared[i] = 1.8/0; //TODO: find a better way to deal with this!
        vec_hat_shared[i] = 1.3/0;

    }

    __syncthreads();
    
    

    legacy_sparse_lower_triangular_solve(num_rows,  L_row_ptrs, L_col_idxs, L_values, vec_shared, temp_vec_shared);

    __syncthreads();

    

    legacy_sparse_upper_triangular_solve(num_rows,  U_row_ptrs, U_col_idxs, U_values, temp_vec_shared, vec_hat_shared);

    

}




__global__ void KernelBatchedPreconditionedBiCGSTAB(const int num_rows, const int num_nz, const int num_pages, const int* const row_ptrs, 
    const int* const col_inds, const double* const vals_mat, const double* const vals_rhs, double* const vals_ans,
    const int L_nnz , const int* const L_row_ptrs, const int* const L_col_idxs, const double* const L_vals ,
    const int U_nnz, const int* const U_row_ptrs, const int* const U_col_idxs, const double* const U_vals,
    float* const iter_counts , int* const conv_flags, double* const iter_residual_norms)
{
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~shared memory ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

   

    //This won't work for isoocatne
    //Except for isooctane, the nrows is atmost 54, and nnz: 2560 ... 
    
    //TODO: Use dynamic shared memory
    /*
        --> Now it is easy to use dynamic shared memory as we don't need to store ints !
        --> But with isooctane, there is no warning/error about the shared memory limits; kernel is simply not launched (if dynamic shared mem greater than what is available is used), this leads to wrong results! Pending: Check with cuda get last error...
    */

    // __shared__ int row_ptrs_shared[MAX_NUM_ROWS + 1];
    // __shared__ int col_idxs_shared[MAX_NUM_NZ];
    // __shared double values_shared[MAX_NUM_NZ];

    __shared__ double x_shared[MAX_NUM_ROWS];
    __shared__ double r_shared[MAX_NUM_ROWS];
    __shared__ double r_hat_shared[MAX_NUM_ROWS];
    __shared__ double p_shared[MAX_NUM_ROWS];
    __shared__ double v_shared[MAX_NUM_ROWS];
    __shared__ double s_shared[MAX_NUM_ROWS];
    __shared__ double t_shared[MAX_NUM_ROWS];
   // __shared__ double r_true_shared[MAX_NUM_ROWS];
    

    __shared__ double s_hat_shared[MAX_NUM_ROWS];
    __shared__ double p_hat_shared[MAX_NUM_ROWS];
 


    int page_id = blockIdx.x;


    if(page_id < num_pages)
    {   


        
        /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ shared memory initialization/assigments~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
        initialization(num_rows, num_nz, row_ptrs, col_inds, vals_mat, vals_rhs, x_shared, v_shared, p_shared, r_shared, r_hat_shared);
        
        __syncthreads();


        /*--------------------------------------------------- Preconditioner already generated ----------------------------------------------------*/


        double res_initial = L2Norm(num_rows, r_shared); 
          
        double iter_residual_norm = res_initial;

        double rho_old = 1;
        double rho_new = 1;
        double omega_old = 1;
        double omega_new = 1;
        double alpha = 1;
        double beta = 1; 

        double b_norm = L2Norm(num_rows, vals_rhs + page_id*num_rows);
        
        int conv_flag = -1;



        if(b_norm == 0)
        {   
            for(int i = threadIdx.x; i < num_rows ; i += blockDim.x)
                x_shared[i] = 0;


            if(threadIdx.x == 0 )
            {   
                printf(" RHS for problem id: %d is 0. x = 0 is the solution. ",page_id);

                iter_counts[page_id] = 0;
                conv_flags[page_id] = 1;
                iter_residual_norms[page_id] = 0;
            }    

            __syncthreads();
        
        }
        else
        {
            if(res_initial < ATOL )
            {   
                if(threadIdx.x == 0 )
                {   
                    printf("\n Initial guess for problem id: %d is good enough. No need of iterations. \n", page_id);


                    iter_counts[page_id] = 0;
                    conv_flags[page_id] = 1;
                    iter_residual_norms[page_id] = res_initial;
                }	    
            }
            else
            {
                 /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Then can start iterating ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
                    float iter = 0;
                
                    while(iter < MAX_ITER)
                    {
                        rho_new = inner_product(num_rows, r_shared, r_hat_shared);

                        if(rho_new == 0)
                        {
                            if(threadIdx.x == 0)
                            {
                                printf("\n Method failed for problem id: %d\n",page_id);
                            }

                            break;
                        }
                        
                        beta = (rho_new/rho_old)*(alpha/omega_old);
                    
                        
                        Update_p(num_rows,p_shared,r_shared ,v_shared,beta,omega_old);
                        __syncthreads();
                        

                        ApplyPreconditionerILU(num_rows, L_row_ptrs, L_col_idxs, L_vals + page_id * L_nnz, U_row_ptrs, U_col_idxs , U_vals + page_id * U_nnz, p_shared, p_hat_shared);

                        __syncthreads();

                        SpMV(num_rows, row_ptrs,col_inds, vals_mat + page_id*num_nz, p_hat_shared, v_shared);
                        __syncthreads(); 
                        
                        
                        double r_hat_and_v_inner_prod = inner_product(num_rows,r_hat_shared,v_shared);
                        alpha = rho_new/r_hat_and_v_inner_prod;        
                    

                        Update_s(num_rows,s_shared,r_shared,alpha,v_shared);
                        __syncthreads();
                        

                        iter_residual_norm = L2Norm(num_rows, s_shared); //an estimate
                        
                        iter = iter + 0.5;

                        if( iter_residual_norm < ATOL)
                        {
                            Update_x_middle(num_rows,x_shared,p_hat_shared,alpha);
                            __syncthreads();

                            conv_flag = 1;

                            
                            break;
    
                        }

                     
                        ApplyPreconditionerILU(num_rows, L_row_ptrs, L_col_idxs, L_vals + page_id * L_nnz, U_row_ptrs, U_col_idxs , U_vals + page_id * U_nnz, s_shared, s_hat_shared);
                        __syncthreads();


                        SpMV( num_rows, row_ptrs , col_inds, vals_mat + page_id*num_nz , s_hat_shared, t_shared);
                        __syncthreads();
                    


                        double t_and_s_inner_prod = inner_product(num_rows,t_shared,s_shared);
                        double t_and_t_inner_prod = inner_product(num_rows,t_shared,t_shared);
                        omega_new = t_and_s_inner_prod/t_and_t_inner_prod;
                        

                        Update_x(num_rows,x_shared,p_hat_shared,s_hat_shared,alpha,omega_new);
                        __syncthreads();
                        
                        
                        iter = iter + 0.5;


                        Update_r(num_rows,r_shared,s_shared,t_shared,omega_new);
                        __syncthreads();

                        iter_residual_norm = L2Norm(num_rows,r_shared);
                        rho_old = rho_new;
                        omega_old = omega_new;

                        if( iter_residual_norm < ATOL)
                        {   
                            conv_flag = 1;
                            break;
                        }

                        
                    }

                    __syncthreads();

                   /*  ComputeResidualVec(num_rows, row_ptrs , col_inds, vals_mat + page_id*num_nz, vals_rhs + page_id*num_rows, x_shared,r_true_shared);
                    __syncthreads();
                
                    
                    double true_resi_norm = L2Norm(num_rows,r_true_shared); */
                    
                    if(threadIdx.x == 0 )
                    {   
                      // printf("\nConv flag for problem_id: %d is %d , iter resi norm : %0.17lg, true resi norm: %0.17lg, iter:%f ",page_id,conv_flag, iter_residual_norm, true_resi_norm, iter );
                        iter_counts[page_id] = iter;
                        conv_flags[page_id] = conv_flag;
                        iter_residual_norms[page_id] = iter_residual_norm;
                    }

            }

        }

       
       // At the end,copy x_shared to global memory.
        for(int i = threadIdx.x; i < num_rows; i += blockDim.x)
            vals_ans[i + page_id*num_rows] = x_shared[i];

    
    }

}




int Batched_BiCGSTAB_Gpu_helper(const PagedCSRMatrices & A_pages,const PagedVectors& b_pages,PagedVectors & x_pages, SolverResults & solver_results,const bool is_parilu , const int num_iter_par_ilu )
{
    std::cout << "\n\n-------------------------------------------------------------------------------\n Batched_Preconditioned BiCGSTAB_Gpu_helper " << std::endl;
    
   
    auto start = std::chrono::high_resolution_clock::now();
    
     //generate ILU preconditioner
    PagedCSRMatrices L_pages;
    PagedCSRMatrices U_pages;


    if(is_parilu)
    {	
    	//std::cout << " \npar ilu with num iter: " << num_iter_par_ilu << std::endl; 
        ParILU_0_Factorization_Gpu(A_pages , L_pages, U_pages, num_iter_par_ilu);
    }
    else
    {
        const int approach_num = 1;
        //Note: For pele matrices, approach 1 works better as compared to the depenedency graph approach as the matrices are not that sparse. For other cases, approach 3 is exepected to be faster than others.
        //std::cout << " \nilu " << std::endl;
        ILU_0_Factorization_Gpu(A_pages , L_pages, U_pages, approach_num);
    }
    	
	
    dim3 block(THREADS_PER_BLOCK,1,1);
    dim3 grid_solver(A_pages.GetNumPages(),1,1 );

    //------------------------------------------------------------------------------- Call main solver kernel-------------------------------------------------//

    KernelBatchedPreconditionedBiCGSTAB<<< grid_solver, block , 0  >>>(A_pages.GetNumRows(), A_pages.GetNumNz(), A_pages.GetNumPages(),
    A_pages.GetPtrToGpuRowPtrs(),A_pages.GetPtrToGpuColInd(), A_pages.GetPtrToGpuValues(), b_pages.GetPtrToGpuValues(), x_pages.GetPtrToGpuValues(),
    L_pages.GetNumNz(), L_pages.GetPtrToGpuRowPtrs(), L_pages.GetPtrToGpuColInd(), L_pages.GetPtrToGpuValues(), 
    U_pages.GetNumNz() ,U_pages.GetPtrToGpuRowPtrs(), U_pages.GetPtrToGpuColInd(), U_pages.GetPtrToGpuValues(),
    solver_results.GetPtrToGpuIterCount(), solver_results.GetPtrToGpuConvFlag() , solver_results.GetPtrToGpuIterResNorm());

    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    

    std::cout << "\n Batched Preconditioned BiCGSTAB on GPU is done!!!" << std::endl;
    std::cout << " Time taken is: "  << (double)duration.count() << " microseconds\n\n ";  

    solver_results.SetTimeTaken((double)duration.count()/ 1000);


    //fill it with true residual norms
    KernelFillTrueResNorms<<< grid_solver , block , 0  >>>(A_pages.GetNumRows(), A_pages.GetNumNz(), A_pages.GetNumPages(), A_pages.GetPtrToGpuRowPtrs(),
    A_pages.GetPtrToGpuColInd(), A_pages.GetPtrToGpuValues(), b_pages.GetPtrToGpuValues(), x_pages.GetPtrToGpuValues(), solver_results.GetPtrToGpuTrueResNorm());

    cudaDeviceSynchronize();

    return 1;
}



} //unnamed namespace


//----------------------------------------------------------------------------------------------------------------------------------------------------------------






// A*x = b
void Batched_ILU_Preconditioned_BiCGSTAB_Gpu(const std::vector<std::string> & subdir, const PagedCSRMatrices & A_pages,const PagedVectors& b_pages,PagedVectors & x_pages,const bool is_scaled,  SolverResults & solver_results , const bool is_parilu , const int num_iter_par_ilu  )
{
    assert(A_pages.ExistsGPU() == true);
    assert(b_pages.ExistsGPU() == true);
    assert(x_pages.ExistsGPU() == true);

    const int num_pages = A_pages.GetNumPages();
    assert(num_pages == b_pages.GetNumPages());
    assert(num_pages == x_pages.GetNumPages());

    const int num_rows = A_pages.GetNumRows();
    const int num_cols = A_pages.GetNumCols();
    
    assert(num_rows == num_cols);
    assert(num_cols == x_pages.GetNumElements());
    assert(num_rows == b_pages.GetNumElements());

   
    int success_code = 0;


    success_code = Batched_BiCGSTAB_Gpu_helper(A_pages,b_pages,x_pages, solver_results, is_parilu, num_iter_par_ilu);

    std::string solution_file;

    if(is_scaled == true)
        solution_file = "x_scaled_gpu_ilu_bicgstab.mtx";
    else
        solution_file = "x_gpu_ilu_bicgstab.mtx";

    if(success_code == 1)
    {
        x_pages.CopyFromGpuToCpu();
        Print_ans(subdir,x_pages, solution_file);
        std::cout << "files containing soluation: x  are produced...  ( " <<  solution_file <<  " ) in their respective directories " << std::endl;

    }


}
