#pragma once

class PagedVectors;
class PagedCSRMatrices;
class SolverResults;

void Print_ans(const std::vector<std::string> & subdir,const PagedVectors & Ans_pages, const std::string & ans_name);

void FillSubdir(const std::string& main_dir, std::vector<std::string> &subdir);

void ReadData(const int Problem_Size, const std::vector<std::string>& subdir, PagedCSRMatrices & A_pages, PagedVectors & b_pages, const bool is_scaled);

void PrintSolverResults(const std::vector<std::string> &subdir,const SolverResults &solver_results, const std::string & f_name);