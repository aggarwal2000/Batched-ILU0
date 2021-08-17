#pragma once
#include<vector>
#include<string>

class PagedCSRMatrices;

void ILU_0_Factorization_Cpu(const PagedCSRMatrices & A_pages , PagedCSRMatrices & L_pages , PagedCSRMatrices & U_pages);

void ILU_0_Factorization_Gpu(const PagedCSRMatrices & A_pages , PagedCSRMatrices & L_pages, PagedCSRMatrices & U_pages, const int approach_num);
