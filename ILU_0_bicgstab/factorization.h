#pragma once


//TODO: Move kernels like: Print, Copy, Prefix Sum to some common operations file.

class PagedCSRMatrices;
class PagedVectors;


int Count_Missing_Diagonal_Elements(PagedCSRMatrices & Factored_Pages, int* const diag_info);


void Add_Missing_Diagonal_Elements(PagedCSRMatrices & New_Factored_Pages , const PagedCSRMatrices & Factored_Pages, const int* const diag_info, 
    const int num_missing_diagonal_eles);



void Find_locations_of_diagonal_elements(const PagedCSRMatrices & Factored_Pages , int* const diag_info);


void Update_row_pointers_L_and_U_and_Allocate_Memory(const PagedCSRMatrices & Factored_Pages , const int* const diag_ptrs, PagedCSRMatrices & L_pages, PagedCSRMatrices & U_pages);

void Fill_L_and_U_col_idxs_and_vals(const PagedCSRMatrices & Factored_Pages, PagedCSRMatrices & L_pages, PagedCSRMatrices & U_pages);


void Copy_Gpu_PagedCSRMatrices(const PagedCSRMatrices & Src_pages, PagedCSRMatrices & Dst_pages);


void PrintPagedCSRMatrix(const PagedCSRMatrices & CSRMatrix_pages);


