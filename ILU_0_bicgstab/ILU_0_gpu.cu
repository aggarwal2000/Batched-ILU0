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
#include "ILU_0.h"


namespace{

const int max_possible_grid_dim = 65536;

//---------------------------------------------------------------------------------------------------------------------------------------------------------
     //APPROACH-1


__device__ void fill_partial_current_row_array(const int nrows, const int curr_row_index, double* const current_row_elements_arr, const int* const row_ptrs, 
const int* const col_idxs, const double* const page_values, const int* const diag_ptrs)
{
    const int diag_ele_loc = diag_ptrs[curr_row_index];
    const int row_end_loc = row_ptrs[curr_row_index + 1];


    for(int i = threadIdx.x + curr_row_index; i < nrows ; i += blockDim.x)
    {
        current_row_elements_arr[i] = 0;
    }

    __syncthreads();

    for(int loc = threadIdx.x + diag_ele_loc ; loc < row_end_loc ; loc += blockDim.x)
    {
        current_row_elements_arr[ col_idxs[loc] ] = page_values[ loc ];
  
    }

}



__device__ void modify_rows_below_curr_row(const int nrows, const int curr_row_index,const double* const column_elements_array_for_current_row, const int* const row_ptrs, 
    const int* const col_idxs, double* const page_values, const int* const diag_ptrs)
{       
    const int warp_id = threadIdx.x / WARP_SIZE;

    const int id_within_warp = threadIdx.x % WARP_SIZE;

    const int total_num_warps_in_block = blockDim.x / WARP_SIZE;

    __shared__ double row_ele_arr[MAX_NUM_ROWS];
    //initilaize it with zeroes

    for(int i = threadIdx.x + curr_row_index + 1; i < nrows ; i += blockDim.x)
    {
        row_ele_arr[i] = 0;
    }

    __syncthreads();
 
    //one warp per row
    for(int row_below_index = warp_id + curr_row_index + 1; row_below_index < nrows ; row_below_index += total_num_warps_in_block )
    {
        for(int i = id_within_warp + row_ptrs[row_below_index] ; i < row_ptrs[row_below_index + 1]; i += WARP_SIZE)
        {   
            const int col_index = col_idxs[i];
 
            if(col_index == curr_row_index)
            {   
                double diag_ele = page_values[diag_ptrs[curr_row_index]];
                assert(diag_ele != 0);
                double row_ele = page_values[i] / diag_ele;
                row_ele_arr[row_below_index] = row_ele;
                page_values[i] = row_ele;
            }
            
            __syncwarp(__activemask()); //else a warning

            if(col_index > curr_row_index)
            {
                double col_ele = column_elements_array_for_current_row[col_index];
                page_values[i] -= row_ele_arr[row_below_index] * col_ele; 

            }
            

        }

    }
    

}




__global__ void compute_ilu_0_approach1_kernel(const int npages, const int nrows, const int nnz, const int* const row_ptrs, const int* const col_idxs, 
    double* const values, const int* const diag_ptrs)
{
    for(int page_id = blockIdx.x ; page_id < npages; page_id += gridDim.x)
    {
        //Tried out ---> Having stuff in shared memory slows down the kernel, so don't copy global arrays to shared memory.

        __shared__ double current_row_elements_arr[MAX_NUM_ROWS];

        
        // __shared__ int row_ptrs_sh[MAX_NUM_ROWS + 1];
        // __shared__ int col_idxs_sh[MAX_NUM_NZ];
        // __shared__ double page_vals_sh[MAX_NUM_NZ];
        // __shared__ int diag_ptrs_sh[MAX_NUM_ROWS];

        // for(int i = threadIdx.x; i < nrows + 1; i += blockDim.x)
        // {
        //     row_ptrs_sh[i] = row_ptrs[i];
        // }

        // for(int i = threadIdx.x ; i < nnz ; i += blockDim.x)
        // {
        //     col_idxs_sh[i] = col_idxs[i];
        //     page_vals_sh[i] = values[i + page_id * nnz ];
        // }

        // for(int i = threadIdx.x; i < nrows; i += blockDim.x)
        // {
        //     diag_ptrs_sh[i] = diag_ptrs[i];
        // }

        // __syncthreads();


        for(int curr_row_index = 0; curr_row_index < nrows; curr_row_index ++)
        {   
           
            fill_partial_current_row_array(nrows, curr_row_index , current_row_elements_arr, row_ptrs, col_idxs , values +  nnz * page_id, diag_ptrs);

            //fill_partial_current_row_array(nrows, curr_row_index , current_row_elements_arr, row_ptrs_sh, col_idxs_sh , page_vals_sh, diag_ptrs_sh);

            //If we plan to use shared memory for values, just send in page_vals_sh

            __syncthreads();

            modify_rows_below_curr_row(nrows, curr_row_index, current_row_elements_arr, row_ptrs, col_idxs, values + nnz * page_id, diag_ptrs);

            //modify_rows_below_curr_row(nrows, curr_row_index, current_row_elements_arr, row_ptrs_sh, col_idxs_sh, page_vals_sh, diag_ptrs_sh);

            __syncthreads();

        }


        // for(int i = threadIdx.x ; i < nnz; i += blockDim.x)
        // {
        //     values[i + page_id * nnz ] = page_vals_sh[i];
        // }

    }
}




void ComputeILU0Approach1(PagedCSRMatrices & Factored_Pages , const int* const diag_ptrs)
{   
    dim3 block(THREADS_PER_BLOCK);

    dim3 grid(Factored_Pages.GetNumPages());

    compute_ilu_0_approach1_kernel<<< grid, block >>>(Factored_Pages.GetNumPages(), Factored_Pages.GetNumRows(), Factored_Pages.GetNumNz() , Factored_Pages.GetPtrToGpuRowPtrs(), 
Factored_Pages.GetPtrToGpuColInd(), Factored_Pages.GetPtrToGpuValues(), diag_ptrs); // one thread block per small matrix in batch

}



//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//

     //APPROACH-2


__global__ void single_matrix_case_legacy_kernel(const int nrows, const int nnz, const int* const row_ptrs, const int* const col_idxs, volatile double* const values, 
    const int* const diag_ptrs, volatile bool* const ready)
{
    const int gid = blockDim.x * blockIdx.x  +  threadIdx.x;

    const int loc = gid;

    if(loc >= nnz)
        return;


    //Note: I am not using one warp per row as we need a unique thread for each element.

    const int col_index = col_idxs[loc];

    int row_index;
    //find the row_index of this element

    for(int a = 0; a < nrows + 1; a++) //uncoalesced memory accesses
    {
        if(row_ptrs[a] > loc)
        {
            row_index = a - 1;
            break;
        }
    }

    // for(int a = 0; a < nrows + 1; a++)
    // {
    //     if(row_ptrs[a] >= loc + 1)
    //     {
    //         row = a- 1;
    //     }
    // }   //eqvt. to the code above

    //F_i_j is the element of interest here, L_i_k,U_k_j pairs for k = 0 to min(i,j)-1  and diagonal U_j_j if i > j 

    double sum = 0;
    double diag_val = 1;
    bool finished = false;

    const int k_max = min(row_index , col_index) - 1;

    int maybe_l_loc = row_ptrs[row_index];

    bool diag_flag = row_index > col_index ? false : true;
    
    
    bool sum_flag = false;
    
    
    int current_corr_u_loc;
    bool current_corr_u_flag = false;
  
    //all memory accesses are uncoalesced here
   // int tmp_counter = 0;

    while(!finished )
    {   
        //tmp_counter++;

       // printf("\n line: %d, thread: %d, counter: %d , diag_flag: %d , sum_flag: %d , maybe_l_loc : %d , row_ptrs[row_index + 1]: %d \n", __LINE__ , threadIdx.x, tmp_counter, diag_flag, sum_flag, maybe_l_loc, row_ptrs[row_index + 1]);

        if(maybe_l_loc < row_ptrs[row_index + 1])
        {   
            const int col = col_idxs[maybe_l_loc]; ////uncoalesced memory accesses when accessing col_idxs[]
          
            if(col > k_max)
            {
                maybe_l_loc++;

               // printf("\n line: %d, thread: %d, counter: %d , diag_flag: %d , sum_flag: %d , maybe_l_loc : %d , row_ptrs[row_index + 1]: %d \n", __LINE__ , threadIdx.x, tmp_counter, diag_flag, sum_flag, maybe_l_loc, row_ptrs[row_index + 1]);
                 continue;

            }
            
           
            if(current_corr_u_flag == true)
            {   

                if(ready[maybe_l_loc] == true && ready[current_corr_u_loc] == true)
                {
                    sum += values[maybe_l_loc] * values[current_corr_u_loc];

                    maybe_l_loc++;

                    current_corr_u_flag = false;
                }
            }
            else
            {   
              

                int maybe_u_loc = row_ptrs[col];
                for(; maybe_u_loc < row_ptrs[col + 1]; maybe_u_loc++) //uncoalesced memory accesses when accessing col_idxs[]
                {
                    if(col_idxs[maybe_u_loc] == col_index)
                    {   
                        current_corr_u_flag = true;
                        current_corr_u_loc = maybe_u_loc;

                        if(ready[maybe_l_loc] == true && ready[current_corr_u_loc] == true)
                        {
                            sum += values[maybe_l_loc] * values[current_corr_u_loc];
                            maybe_l_loc++;
                            current_corr_u_flag = false;
                        }

                        break;
                    }
                }

                if(maybe_u_loc == row_ptrs[col + 1]) //that means no corr. u entry is there
                {
                    maybe_l_loc++;
                }

            }
        


        }
        else
        {   
            sum_flag = true;
        }

        
    
        if(diag_flag == false)
        {   
            const int diag_loc = diag_ptrs[col_index];
            if(ready[diag_loc] == true)
            {
                diag_val = values[diag_loc];
                diag_flag = true;

            }
            
        }

       // printf("\n line: %d, thread: %d, counter: %d , diag_flag: %d , sum_flag: %d , maybe_l_loc : %d , row_ptrs[row_index + 1]: %d \n", __LINE__ , threadIdx.x, tmp_counter, diag_flag, sum_flag, maybe_l_loc, row_ptrs[row_index + 1]);
        if(diag_flag == true && sum_flag == true )
        {
            values[loc] = (values[loc] - sum)/diag_val;

            __threadfence();

            ready[loc] = true;

            finished = true;
           // printf("\n line: %d, thread: %d, counter: %d , now loc: %d is ready!\n", __LINE__ , threadIdx.x, tmp_counter, loc);
            
        }
    
    }

}


void ComputeILU0Approach2_SingleMatrix(PagedCSRMatrices & Factored_Pages , const int* const diag_ptrs)
{   
    dim3 block(THREADS_PER_BLOCK);

    int grid_dim =  ceil( (double)Factored_Pages.GetNumNz() /(double)THREADS_PER_BLOCK ) ;
    dim3 grid( grid_dim );

    const int nnz = Factored_Pages.GetNumNz();

    bool* ready = nullptr;

    cudaMalloc((void**)&ready , nnz * sizeof(bool) );
    cudaMemset( ready , false,  nnz * sizeof(bool) );

    single_matrix_case_legacy_kernel<<< grid , block >>>(Factored_Pages.GetNumRows() , Factored_Pages.GetNumNz(), Factored_Pages.GetPtrToGpuRowPtrs(), 
    Factored_Pages.GetPtrToGpuColInd(), Factored_Pages.GetPtrToGpuValues() , diag_ptrs, ready);
    cudaDeviceSynchronize();

    cudaFree(ready);    
}

struct dependency{

    int location;
    bool is_diagonal;
    struct dependency* next; 
};

typedef struct dependency dependency;


__device__ void insert_dependency(dependency** graph_element , dependency* new_dependency)
{
    dependency* address_currently_stored = *graph_element;
    new_dependency->next = address_currently_stored;
    *graph_element = new_dependency;
}



__global__ void create_dependency_graph_for_ilu0_computation(const int nrows, const int* const row_ptrs, const int* const col_idxs, 
const int* const diag_ptrs, dependency ** graph)
{
     //we use one warp per row
     const int gid = blockDim.x * blockIdx.x  + threadIdx.x;

     const int warp_id = gid/WARP_SIZE;
 
     const int id_within_warp = gid % WARP_SIZE;
 
     const int total_num_warps = (gridDim.x * blockDim.x)/WARP_SIZE;
 
     for(int row_index = warp_id ; row_index < nrows; row_index += total_num_warps)
     {
         const int row_start = row_ptrs[row_index];
         const int row_end = row_ptrs[row_index + 1];
 
         for(int loc = row_start + id_within_warp; loc < row_end ; loc += WARP_SIZE)
         {   
             graph[loc] = nullptr;
            
             const int col_index = col_idxs[loc];
 
             const int k_max = min(row_index, col_index) - 1;
 
             //the thread concerned for the particular element at: row_index, col_index does all this.--> but this is inefficent as that thread first searches for possible L_val, for each L_val, 
             //again searches for U_val. (Lot of uncoalesced accesses)
 
             for(int maybe_l_loc = row_start ; maybe_l_loc < loc; maybe_l_loc++) //use loc instead of row_end as the matrix is sorted
             {
                 const int k = col_idxs[maybe_l_loc]; //this should definitely be less than col_index, but we want to make sure it is less than or equal to k_max
 
                 if(k > k_max)
                 {
                     continue;
                 }
 
 
                 //find corresponfing  U at position:  k,col_index
                 for(int maybe_u_loc = row_ptrs[k]; maybe_u_loc < row_ptrs[k+1]; maybe_u_loc++)
                 {
                     if(col_idxs[maybe_u_loc] == col_index )
                     {   
                         dependency* dep_node_l =  (dependency*)malloc(sizeof(dependency));   
                         assert(dep_node_l != nullptr);
                         dep_node_l->location = maybe_l_loc;
                         dep_node_l->is_diagonal = false;
                         dep_node_l->next = nullptr;
                         insert_dependency( &graph[loc] , dep_node_l  );

                         dependency* dep_node_u =  (dependency*)malloc(sizeof(dependency));   
                         assert(dep_node_u != nullptr);
                         dep_node_u->location = maybe_u_loc;
                         dep_node_u->is_diagonal = false;
                         dep_node_u->next = nullptr;
                         insert_dependency( &graph[loc] , dep_node_u );

                     }
                 }
 
             }
 
 
             if(row_index > col_index)
             {   
                 const int diag_loc = diag_ptrs[col_index];
                 dependency* dep_node_diag = (dependency*)malloc(sizeof(dependency));
                 assert(dep_node_diag != nullptr);
                 dep_node_diag->location = diag_loc;
                 dep_node_diag->is_diagonal = true;
                 dep_node_diag->next = nullptr;
                 insert_dependency( &graph[loc] , dep_node_diag );
                 
             }
 
         }
 
     }
}




__global__ void deallocate_graph_mem(dependency** graph, const int nnz)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;

    for(int loc = gid; loc < nnz; loc +=  gridDim.x * blockDim.x)
    {
        dependency* address_of_next = graph[loc];
        while(address_of_next != nullptr)
        {   
            dependency* tmp = address_of_next->next;
            free(address_of_next);
            address_of_next = tmp;
        }

    }

}



__global__ void print_dependency_graph(dependency** graph, const int nnz)
{
    if(threadIdx.x == 0)
    {
        for(int loc  = 0; loc < nnz; loc++)
        {   
            printf("\n\n\n Dependencies for element at location: %d are as follows: ", loc);
            dependency* address =  graph[loc];

            while(address != nullptr)
            {
                printf("\n%d" , address->location);
                address = address->next;
            }
        }
    }
}

void PrintGraph(dependency** graph, const PagedCSRMatrices & Factored_Pages)
{
    print_dependency_graph<<< 1, 1>>>(graph, Factored_Pages.GetNumNz());
    cudaDeviceSynchronize();
}




__global__ void compute_ilu_0_approach2_legacy_kernel(const int npages, const int nrows, const int nnz, volatile double* const values, dependency** graph, volatile bool* const ready)
{   

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;

    const int num_threads = gridDim.x * blockDim.x;
    if(num_threads < nnz)
    {   
        printf("\n\n Sorry, we need atleast %d number of threads for this kernel to work\n\n", nnz); //This guarantees for nnz elements belonging to one small matrix, we have different threads(different gids) doing the compuatation
        return;
    }

    for(int id = gid; id < nnz * npages ; id += gridDim.x * blockDim.x) //uncoalesced accesses when a thread accesses dependency locations 
    {   

        const int page_id = id/nnz;

        const int loc = id % nnz;    

        dependency* address =  graph[loc];
       

        double diag_value = 1;
        double u_val;
        double l_val;

        double sum = 0;

        bool u_flag = false;
        bool l_flag = false;
        bool finished = false;

        int diag_loc;
        int u_loc;
        int l_loc;

        while(!finished)
        {
            
            if(address != nullptr && address->is_diagonal == true)
            {
                diag_loc = address->location;
                if(ready[diag_loc + page_id * nnz] == true)
                {
                    diag_value =  values[diag_loc  + page_id * nnz];
                    assert(diag_value != 0);
                    address = address->next;

                }
            } 

            if(address != nullptr  && address->is_diagonal == false )
            {
                u_loc = address->location;
                l_loc = (address->next)->location;

                
                
                // if(ready[u_loc + page_id * nnz] == true && ready[l_loc + page_id * nnz] == true)
                // {   
                   
                //     double u_val = values[u_loc +  page_id * nnz];
                //     double l_val = values[l_loc + page_id * nnz];
                //     sum += u_val * l_val;
    
                //     address = (address->next)->next;

                // }  
                
                
                if(u_flag == false && ready[u_loc + page_id * nnz] == true)
                {
                    u_val = values[u_loc +  page_id * nnz];
                    u_flag = true;
                }

                if(l_flag == false && ready[l_loc + page_id * nnz] == true)
                {
                    l_val = values[l_loc + page_id * nnz];
                    l_flag = true;
                }

                if(u_flag == true && l_flag == true)
                {
                    sum += u_val * l_val;
                    u_flag = false;
                    l_flag = false;
                    address = (address->next)->next;
                }
            }

          
            if(address == nullptr)
            {  
                values[loc + page_id * nnz] = (values[loc + page_id * nnz] - sum)/diag_value;
                __threadfence();
                ready[loc + page_id * nnz] = true;
                finished = true;
                //printf("\n Now loc: %d is ready", loc);
            }

        }

    }    

}





void ComputeILU0Approach2(PagedCSRMatrices & Factored_Pages , const int* const diag_ptrs)
{
    dim3 block(THREADS_PER_BLOCK);
    const int total_num_warps = Factored_Pages.GetNumRows();
    const int num_warps_in_block = THREADS_PER_BLOCK/WARP_SIZE;
    const int grid_size = ceil((double)total_num_warps / (double)num_warps_in_block);
    dim3 grid( grid_size );


    void* GRAPH;
    cudaMalloc( (void**)& GRAPH, sizeof(dependency*) * Factored_Pages.GetNumNz()); //array of pointers of type dependency
    dependency** graph = (dependency**)GRAPH;

   

    //Create a dependency graph for the ilu0 computation on the device memory. (The graph is stored using adjacency list datastrucure)
    create_dependency_graph_for_ilu0_computation<<< grid, block >>>(Factored_Pages.GetNumRows(), Factored_Pages.GetPtrToGpuRowPtrs(), Factored_Pages.GetPtrToGpuColInd(), 
 diag_ptrs, graph);

  // PrintGraph(graph, Factored_Pages);
  


   int grid_dim =  ceil( (double)Factored_Pages.GetNumNz() * (double)Factored_Pages.GetNumPages() /(double)THREADS_PER_BLOCK ) ;
   
   

    if(grid_dim > max_possible_grid_dim)
    {   
       // std::cout << "\n Using max possible grid dim at line:"  << __LINE__ << "\n";
        grid_dim = max_possible_grid_dim;
    }
   dim3 grid_1( grid_dim );

    bool* ready = nullptr;
    cudaMalloc((void**)&ready , Factored_Pages.GetNumPages() * Factored_Pages.GetNumNz()* sizeof(bool) );
    cudaMemset( ready , false,  Factored_Pages.GetNumPages() * Factored_Pages.GetNumNz()* sizeof(bool) );
    compute_ilu_0_approach2_legacy_kernel <<< grid_1 , block >>>(Factored_Pages.GetNumPages(), Factored_Pages.GetNumRows(), Factored_Pages.GetNumNz(), 
    Factored_Pages.GetPtrToGpuValues(), graph, ready);   

   

    dim3 grid_2(ceil( (double)Factored_Pages.GetNumNz()/(double)THREADS_PER_BLOCK ));
    deallocate_graph_mem<<< grid_2, block >>>(graph, Factored_Pages.GetNumNz());

    

    cudaFree(GRAPH);
    cudaFree(ready);
}


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//

    //APPROACH-3


void create_dependency_graph(PagedCSRMatrices & Factored_Pages, std::vector<int> & dependencies , std::vector<int> & nz_ptrs, std::vector<int> & diag_ptrs)
{   
    const int nrows = Factored_Pages.GetNumRows();
    const int* const row_ptrs = Factored_Pages.GetPtrToCpuRowPtrs();
    const int* const col_idxs = Factored_Pages.GetPtrToCpuColInd();
    
    nz_ptrs[0] = 0;

    for(int row_index = 0; row_index < nrows ; row_index++ )
    {
        const int row_start = row_ptrs[row_index];
        const int row_end = row_ptrs[row_index + 1];

        for(int loc = row_start; loc < row_end; loc++)
        {   
            int num_dependencies = 0;

            const int col_index = col_idxs[loc];

            if(row_index == col_index)
            {
                diag_ptrs[row_index] = loc;
            }

            const int k_max = std::min(row_index , col_index) - 1;

            for(int maybe_l_loc = row_start; maybe_l_loc < loc; maybe_l_loc++) //use loc instead of row_end as the matrix is sorted
            {
                const int k = col_idxs[maybe_l_loc];

                if(k > k_max)
                {
                    continue;
                }

                //find corresponding u at position k,col_index

                for(int maybe_u_loc = row_ptrs[k]; maybe_u_loc < row_ptrs[k + 1]; maybe_u_loc++)
                {
                    if(col_idxs[maybe_u_loc] == col_index)
                    {
                        dependencies.push_back(maybe_l_loc);
                        dependencies.push_back(maybe_u_loc);

                        num_dependencies += 2;
                    }
                }
            }


            if(row_index > col_index)
            {
                const int diag_loc = diag_ptrs[col_index]; //diag_ptrs[col_index] has correct value as it has been found when doing stuff for previous rows as col_index < row_index here
                dependencies.push_back(diag_loc);

                num_dependencies++;
            }

            nz_ptrs[loc + 1] = nz_ptrs[loc] + num_dependencies;


        }
    }
}



__global__ void compute_ilu_0_approach3_legacy_kernel(const int npages, const int nrows, const int nnz, volatile double* const values, const int dep_length, 
    const int* const dependencies, const int* const nz_ptrs, volatile bool* const ready)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;

    const int num_threads = gridDim.x * blockDim.x;
    if(num_threads < nnz)
    {   
        printf("\n\n Sorry, we need atleast %d number of threads for this kernel to work\n\n", nnz); //This guarantees for nnz elements belonging to one small matrix, we have different threads(different gids) doing the compuatation
        return;
    }

    for(int id = gid; id < nnz * npages ; id += gridDim.x * blockDim.x) //uncoalesced accesses when a thread accesses dependencies array
    {
        const int page_id = id/nnz;

        const int loc = id % nnz;  

        const int start =  nz_ptrs[loc];
        
        const int end = nz_ptrs[loc + 1] - 1;

        const bool has_diag_dependency = (end + 1 - start)% 2  == 1 ? true : false;

        int current = start;

        double diag_value = 1;
        double u_val;
        double l_val;

        double sum = 0;

        bool u_flag = false;
        bool l_flag = false;
        bool finished = false;

       
        while(!finished)
        {
           

            if( (has_diag_dependency == true && current <= end - 2) || (has_diag_dependency == false && current <= end - 1) )
            {
                const int l_loc = dependencies[current] + page_id * nnz;
                const int u_loc = dependencies[current + 1] + page_id * nnz;

                // if(ready[l_loc] == true && ready[u_loc] == true)
                // {
                //     l_val = values[l_loc];
                //     u_val = values[u_loc];

                //     sum += l_val * u_val;

                //     current += 2;
                // }


                if(l_flag == false && ready[l_loc] == true)
                {   
                    l_val = values[l_loc];
                    l_flag = true;
                }

                if(u_flag == false && ready[u_loc] == true)
                {   
                    u_val = values[u_loc];
                    u_flag = true;
                }

                if(l_flag == true && u_flag == true)
                {
                    sum += l_val * u_val;

                    current += 2;

                    l_flag = false;
                    u_flag = false;
                }
            }



            if(has_diag_dependency == true && current == end )
            {
                const int diag_loc = dependencies[end] + page_id * nnz;

                if(ready[diag_loc] == true)
                {   
                    diag_value = values[diag_loc];
                    assert(diag_value != 0);
                    current++;
                }
            }


            if(current == end + 1)
            {
                values[loc + page_id * nnz] = (values[loc + page_id * nnz] - sum)/diag_value;
                __threadfence(); 
                ready[loc + page_id * nnz] = true;
                finished = true;
                //printf("\n Now loc: %d is ready", loc);
            }



        }   

    }
}


void Print_Dep_Graph(const std::vector<int> & dependencies_cpu , const std::vector<int> & nz_ptrs_cpu)
{
    for(int loc = 0; loc < nz_ptrs_cpu.size() - 1 ; loc++)
    {
        const int start = nz_ptrs_cpu[loc];
        const int end = nz_ptrs_cpu[loc + 1];

        printf("\n\n Dependencies for element at loc = %d are: ", loc);

        for(int i = start; i < end; i++)
        {
            printf("\n %d ", dependencies_cpu[i]);
        }
    }
}

void ComputeILU0Approach3(PagedCSRMatrices & Factored_Pages , int* const diag_ptrs)
{
    //Here the representation of dependency graph is a bit different
    std::vector<int> dependencies_cpu;
    std::vector<int > nz_ptrs_cpu(Factored_Pages.GetNumNz() + 1);
    std::vector<int > diag_ptrs_cpu( Factored_Pages.GetNumRows());
    

    Factored_Pages.AllocateMemory(LOCATION::CPU);
    Factored_Pages.CopyFromGpuToCpu();
    create_dependency_graph(Factored_Pages, dependencies_cpu, nz_ptrs_cpu, diag_ptrs_cpu);

    int* dependencies = nullptr;
    int* nz_ptrs = nullptr;
    cudaMalloc((void**)&dependencies , dependencies_cpu.size() * sizeof(int));
    cudaMemcpy(dependencies , dependencies_cpu.data() , dependencies_cpu.size() * sizeof(int) , cudaMemcpyHostToDevice );
    cudaMalloc((void**)&nz_ptrs , nz_ptrs_cpu.size() * sizeof(int) );
    cudaMemcpy( nz_ptrs , nz_ptrs_cpu.data() , nz_ptrs_cpu.size() * sizeof(int) , cudaMemcpyHostToDevice );
    cudaMemcpy( diag_ptrs , diag_ptrs_cpu.data() , diag_ptrs_cpu.size() * sizeof(int) , cudaMemcpyHostToDevice);

    //Print_Dep_Graph(dependencies_cpu , nz_ptrs_cpu);

    dim3 block(THREADS_PER_BLOCK);
    int grid_dim =  ceil( (double)Factored_Pages.GetNumNz() * (double)Factored_Pages.GetNumPages() /(double)THREADS_PER_BLOCK ) ;
   
    

    if(grid_dim > max_possible_grid_dim)
    {   
        //std::cout << "\n Using max possible grid dim at line:"  << __LINE__ << "\n";
        grid_dim = max_possible_grid_dim;
    }

    dim3 grid( grid_dim );

    bool* ready = nullptr;
    cudaMalloc((void**)&ready , Factored_Pages.GetNumPages() * Factored_Pages.GetNumNz()* sizeof(bool) );
    cudaMemset( ready , false,  Factored_Pages.GetNumPages() * Factored_Pages.GetNumNz()* sizeof(bool) );

    compute_ilu_0_approach3_legacy_kernel <<< grid , block >>>(Factored_Pages.GetNumPages(), Factored_Pages.GetNumRows(), Factored_Pages.GetNumNz(), 
    Factored_Pages.GetPtrToGpuValues(), dependencies_cpu.size() ,dependencies, nz_ptrs , ready);   


    cudaFree(dependencies);
    cudaFree(nz_ptrs);
    cudaFree(ready);

}




//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//

     //APPROACH-4


      

void create_dependency_list(PagedCSRMatrices & Factored_Pages , std::vector<int> & dependencies, std::vector<int> & diag_starters, std::vector<int> & new_era,  std::vector<int> & diag_ptrs)
{
    const int nrows = Factored_Pages.GetNumRows();
    const int* const row_ptrs = Factored_Pages.GetPtrToCpuRowPtrs();
    const int* const col_idxs = Factored_Pages.GetPtrToCpuColInd();

    diag_ptrs = std::vector<int>(nrows);
    
    int d_start_ptr_to_dependencies_arr = 0;

    int era_start_ptr_to_diag_starters = 0;

    new_era.push_back(0);

    for(int current_row = 0; current_row < nrows ; current_row++)
    {   
        int temp_arr_curr_row[MAX_NUM_ROWS];

        for(int i = 0; i < nrows; i++)
        {
            temp_arr_curr_row[i] = -1;
        }

        for(int j = row_ptrs[current_row]; j < row_ptrs[current_row + 1]; j++)
        {   
            if(col_idxs[j] == current_row)
            {
                diag_ptrs[current_row] = j;
            }
            temp_arr_curr_row[col_idxs[j]] = j;
        }



        for(int row_below = current_row + 1; row_below < nrows; row_below++)
        {
            const int start = row_ptrs[row_below];
            const int end = row_ptrs[row_below + 1];

            int loc_row_ele = -1;

            for(int loc = start; loc < end; loc++)
            {
                int col = col_idxs[loc];

                if(col < current_row)
                {
                    continue;
                }
                else if(col == current_row)
                {
                    //Now there's only one dependency here--> that is divison by the diag element in the current row
                    //find loc_diag_ele
                    const int loc_diag_ele = temp_arr_curr_row[current_row];

                    dependencies.push_back(loc);
                    dependencies.push_back(loc_diag_ele);

                    diag_starters.push_back(d_start_ptr_to_dependencies_arr);

                    d_start_ptr_to_dependencies_arr += 2;
                    era_start_ptr_to_diag_starters++;
                   
                    loc_row_ele = loc;
                }
                else
                {
                    if(loc_row_ele == -1) //So if that row_ele is missing, then the whole row is not modified
                    {
                        break;
                    }
                    else
                    {
                        // find loc_col_ele
                        const int loc_col_ele = temp_arr_curr_row[col];

                        if(loc_col_ele == -1)
                        {
                            continue;
                        }

                        dependencies.push_back(loc);
                        dependencies.push_back(loc_col_ele);

                        d_start_ptr_to_dependencies_arr += 2;
                    }

                }


            }

           
        }


        if(new_era[new_era.size() - 1] < era_start_ptr_to_diag_starters) //If both are equal, then that means there were no dependencies at all for the current row.
        {
            new_era.push_back(era_start_ptr_to_diag_starters);
        }
  
    }

    diag_starters.push_back(d_start_ptr_to_dependencies_arr);


}

void Print_Dep_List(std::vector<int> & dependencies_cpu, std::vector<int> & diag_starters_cpu, std::vector<int> & new_era_cpu)
{   
    std::cout << "\n\n dependencies:  " << std::endl;
    for(int i = 0; i < dependencies_cpu.size(); i++)
    {
        std::cout << dependencies_cpu[i] << "    ";
    }

    std::cout << "\n\n diag starters(ptrs to dependencies array): " << std::endl;
    for(int i = 0; i < diag_starters_cpu.size(); i++)
    {
        std::cout << diag_starters_cpu[i] << "    ";
    }

    std::cout << "\n\n new era(ptrs to diag starters array): " << std::endl;
    for(int i = 0; i < new_era_cpu.size(); i++)
    {
        std::cout << new_era_cpu[i] << "    ";
    }
}


__device__ void modify_elements_in_an_era(double* const values, const int* const dependencies, const int* const diag_starters, 
    const int* const era_array ,const int era_idx )
{  

    int start_idx_in_diag_starters = era_array[era_idx];
    int end_idx_in_diag_starters = era_array[era_idx + 1];

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int id_within_warp = threadIdx.x % WARP_SIZE;
    const int total_num_warps_in_block = blockDim.x / WARP_SIZE;

    for(int idx_in_diag_starters = start_idx_in_diag_starters + warp_id; idx_in_diag_starters < end_idx_in_diag_starters ; idx_in_diag_starters += total_num_warps_in_block)
    {   
        int start_idx_in_dep_arr =  diag_starters[idx_in_diag_starters];
        int end_idx_in_dep_arr =  diag_starters[idx_in_diag_starters + 1];
        double row_ele;

        for(int i = start_idx_in_dep_arr + id_within_warp; i < end_idx_in_dep_arr; i += WARP_SIZE)
        {
            const int loc = dependencies[i];
            auto mask = __activemask();
            int loc_1 = __shfl_sync(mask, loc, id_within_warp + 1); //For id_within_warp = 31, the result is undefined! But that is not used, so no worries!
            
            if(i == start_idx_in_dep_arr)
            {
                double diag_val = values[loc_1];
                assert(diag_val != 0);
                values[loc ] /= diag_val;
                row_ele = values[loc ];   
            }

            __syncwarp(mask);

            row_ele = __shfl_sync( mask , row_ele, 0); 

            if(i > start_idx_in_dep_arr + 1  &&  id_within_warp % 2 == 0  && id_within_warp <= 30)
            {   
                //loc_m1 = dependencies[i + 1];
                values[loc] -= values[loc_1] * row_ele;   
            }

        }


        // if(id_within_warp == 0)
        // {   
        //     double diag_val = values[dependencies[start_idx_in_dep_arr + 1] ];
        //     assert(diag_val != 0);
        //     values[dependencies[start_idx_in_dep_arr] ] /= diag_val;
        //     row_ele = values[dependencies[start_idx_in_dep_arr]];

        //     for(int i = start_idx_in_dep_arr + 2 ; i < end_idx_in_dep_arr - 1; i += 2)
        //     {
        //         values[dependencies[i] ] -= values[dependencies[i + 1]] * row_ele;
        //     }
        // }

    }
    
}

__global__ void compute_ilu_0_approach4_kernel(const int npages, const int nnz, double* const values, const int* const dependencies, const int* const diag_starters, 
const int* const era_array , const int era_arr_length , const int dep_length)
{   
    const int page_id = blockIdx.x;

    if(page_id >= npages)
        return;


   
    // extern __shared__ double array[];
    // double* vals_sh  = (double*)array;


    // for(int i = threadIdx.x; i < nnz ; i += blockDim.x)
    // {
    //     vals_sh[i] = values[i + page_id * nnz];
    // }
    // __syncthreads();

    //dependenices array is too large to fit in shared memory
    //storing values in shared memory makes it slower


    const int num_eras = era_arr_length - 1;

    for(int era_idx = 0; era_idx < num_eras ; era_idx ++)
    {
        modify_elements_in_an_era(values + page_id * nnz, dependencies, diag_starters, era_array, era_idx);
       //modify_elements_in_an_era(vals_sh, dependencies, diag_starters, era_array, era_idx);
        
        __syncthreads();
    }


    // for(int i = threadIdx.x; i < nnz ; i += blockDim.x)
    // {
    //     values[i + page_id * nnz] = vals_sh[i];
    // }

}



void ComputeILU0Approach4(PagedCSRMatrices & Factored_Pages, int* const diag_ptrs )
{
    std::vector<int> dependencies_cpu;
    std::vector<int> diag_starters_cpu;
    std::vector<int> new_era_cpu; 
    std::vector<int> diag_ptrs_cpu;


    Factored_Pages.CopyFromGpuToCpu();
    create_dependency_list(Factored_Pages, dependencies_cpu, diag_starters_cpu, new_era_cpu, diag_ptrs_cpu );

    int* dependencies = nullptr;
    int* diag_starters = nullptr;
    int* new_era = nullptr;

    cudaMalloc( (void**)&dependencies , dependencies_cpu.size() * sizeof(int) );
    cudaMemcpy( dependencies , dependencies_cpu.data(),  dependencies_cpu.size() * sizeof(int) , cudaMemcpyHostToDevice );

    cudaMalloc( (void**)&diag_starters,  diag_starters_cpu.size() * sizeof(int) );
    cudaMemcpy( diag_starters , diag_starters_cpu.data(), diag_starters_cpu.size() * sizeof(int) , cudaMemcpyHostToDevice );

    cudaMalloc( (void**)&new_era,  new_era_cpu.size() * sizeof(int) );
    cudaMemcpy( new_era,  new_era_cpu.data() , new_era_cpu.size() * sizeof(int), cudaMemcpyHostToDevice );

    cudaMemcpy( diag_ptrs , diag_ptrs_cpu.data(), diag_ptrs_cpu.size() * sizeof(int), cudaMemcpyHostToDevice);


   // Print_Dep_List(dependencies_cpu, diag_starters_cpu, new_era_cpu);

    dim3 block(THREADS_PER_BLOCK);

    dim3 grid(Factored_Pages.GetNumPages());

   // const int dynamic_shared_mem_size = Factored_Pages.GetNumNz() * sizeof(double);

   // const int dynamic_shared_mem_size = dependencies_cpu.size() * sizeof(int);

    const int dynamic_shared_mem_size = 0;

    compute_ilu_0_approach4_kernel<<< grid, block, dynamic_shared_mem_size >>>(Factored_Pages.GetNumPages(), Factored_Pages.GetNumNz(), Factored_Pages.GetPtrToGpuValues(), 
dependencies, diag_starters, new_era, new_era_cpu.size(), dependencies_cpu.size()); // one thread block per small matrix in batch


    cudaFree(dependencies);
    cudaFree(diag_starters);
    cudaFree(new_era);

}


//-------------------------------------------------------------------------------------------------------------------------------------------------------------------//


} //unnamed namespace

//-------------------------------- calling function for all small pieces ----------------------------------------------------------------------------------------

void ILU_0_Factorization_Gpu(const PagedCSRMatrices & A_pages , PagedCSRMatrices & L_pages, PagedCSRMatrices & U_pages, const int approach_num)
{
    //cudaProfilerStart();

    // std::cout << "\n\nORIGINAL MATRIX: " << std::endl;
    // PrintPagedCSRMatrix(A_pages);


    //first assert matrix is square
    assert(A_pages.GetNumCols() == A_pages.GetNumRows());

    PagedCSRMatrices Factored_Pages; 
    //We would want to use copy assignment here... or even a copy constructor.  implement it later...
    //copy A to F
    Copy_Gpu_PagedCSRMatrices(A_pages , Factored_Pages);

    //SortCSRMatrix(Factored_Pages); if unsorted, pls sort the paged matrix befoe proceeding. (All these matrices are already sorted.(sorted while storing))
    
    int* diag_info = nullptr;
    cudaMalloc((void**)&diag_info, sizeof(int) * Factored_Pages.GetNumRows());

    int num_missing_diagonal_eles = Count_Missing_Diagonal_Elements(Factored_Pages , diag_info);

    if(num_missing_diagonal_eles > 0)
    {
        PagedCSRMatrices New_Factored_Pages;

        Add_Missing_Diagonal_Elements(New_Factored_Pages, Factored_Pages, diag_info , num_missing_diagonal_eles);

        Copy_Gpu_PagedCSRMatrices(New_Factored_Pages , Factored_Pages); //TODO: avoid an extra copy here

    }

    // std::cout << "\n\nMATRIX AFTER ADDITION OF DIAGONAL ELEMENTS: " << std::endl;
    // PrintPagedCSRMatrix(Factored_Pages);

    //continue to use Factored_pages here...

    cudaProfilerStart();
    if(approach_num == 1)
    {
         Find_locations_of_diagonal_elements(Factored_Pages, diag_info);
        //std::cout << "\n\nLocn of diagonal elements:" << std::endl;
        //print_kernel<<< 1, 1 >>>(Factored_Pages.GetNumRows(), diag_info);
        //cudaDeviceSynchronize();

        

        ComputeILU0Approach1(Factored_Pages , diag_info);
    }
    else if(approach_num == 2)
    {
        Find_locations_of_diagonal_elements(Factored_Pages, diag_info);
        //std::cout << "\n\nLocn of diagonal elements:" << std::endl;
        //print_kernel<<< 1, 1 >>>(Factored_Pages.GetNumRows(), diag_info);
        //cudaDeviceSynchronize();
        
        ComputeILU0Approach2(Factored_Pages , diag_info);
        //ComputeILU0Approach2_SingleMatrix(Factored_Pages, diag_info);
    
    }
    else if(approach_num == 3)
    {
        ComputeILU0Approach3(Factored_Pages , diag_info);
          //  std::cout << "\n\nLocn of diagonal elements:" << std::endl; 
        //  print_kernel<<< 1, 1 >>>(Factored_Pages.GetNumRows(), diag_info);
    }
    else if(approach_num == 4)
    {
        ComputeILU0Approach4(Factored_Pages , diag_info);
          //  std::cout << "\n\nLocn of diagonal elements:" << std::endl; 
         //  print_kernel<<< 1, 1 >>>(Factored_Pages.GetNumRows(), diag_info);
    }
    else
    {
        printf("\n NOT IMPLEMENTED\n");
    }
    cudaProfilerStop();

    // std::cout << "\n\nFACTORIZED MATRIX(ILU(0)): " << std::endl;
    // PrintPagedCSRMatrix(Factored_Pages);
    

    Update_row_pointers_L_and_U_and_Allocate_Memory(Factored_Pages , diag_info, L_pages, U_pages);

    Fill_L_and_U_col_idxs_and_vals(Factored_Pages, L_pages, U_pages);

   
//     std::cout << "\n\nMATRIX L: " << std::endl;
//    PrintPagedCSRMatrix(L_pages);


//    std::cout << "\n\nMATRIX U: " << std::endl;
//    PrintPagedCSRMatrix(U_pages);

    cudaFree(diag_info);

    cudaDeviceSynchronize(); //for timing purpose

    //cudaProfilerStop();

    
}



//TODO:
//Parallelize Prefix Sum

