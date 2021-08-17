#pragma once
#include<vector>
#include<string>

class PagedCSRMatrices;
class PagedVectors;
class SolverResults;


#define MAX_ITER 200

#define ATOL 0.00000000001



void Batched_ILU_Preconditioned_BiCGSTAB_Gpu(const std::vector<std::string> & subdir, const PagedCSRMatrices & A_pages,const PagedVectors& b_pages,PagedVectors & x_pages,const bool is_scaled,  SolverResults & solver_results , const bool , const int );
