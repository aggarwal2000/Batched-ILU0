#include<iostream>
#include<vector>
#include<string>
#include <fstream>

#include"matrix.h"
#include"ReadWriteData.h"
#include"PrecondBiCGSTAB.h"
#include "SolverResults.h"



int main(int argc, char *argv[])
{	

	if(argc < 6)
	{
		printf("\nFormat: ./a.out [category name] [problemsize] [1 for scaled] [0 for conventional ilu, 1 for par ilu] [ num_iter for par ilu(for conventional ilu, this will not be used) ] [<optional> file to store execution time(cuda implementation)]\n");
		exit(0);
	}
	
	std::string category = argv[1];



	std::string dir = "../../../Pele_Matrices_market/" + category + "/";

	std::vector<std::string> subdir{};

	FillSubdir(dir,subdir);

	std::cout << "\nNumber of small problems in category - " << category << " is: " << subdir.size() << std::endl;

	int Problem_Size = std::stoi(argv[2]);

	std::cout << "\nSo, the number of small problems to be solved: " << Problem_Size*subdir.size() << std::endl;


	bool is_scaled = true;
	
	int scale_info = std::stoi(argv[3]);
	
	if(scale_info == 1)
	  is_scaled = true;
	else
	  is_scaled = false;


	// for(int i=0;i< subdir.size();i++)
	// 	std::cout << subdir[i] << std::endl;

	
	bool is_parilu = std::stoi(argv[4]) == 1;
	
	int num_iter_par_ilu = 0;
	
	if(is_parilu == true)
	{
		num_iter_par_ilu = std::stoi(argv[5]);
	}

	std::cout << "\n\nStart reading the matrices and their rhs..." << std::endl;

	PagedCSRMatrices A_pages;
	PagedVectors b_pages;

	ReadData(Problem_Size ,subdir,A_pages,b_pages, is_scaled); //store all info in cpu arrays.

	PagedVectors x_pages;
	x_pages.SetNumPages(A_pages.GetNumPages());
	x_pages.SetNumElements(A_pages.GetNumCols());
	x_pages.AllocateMemory(LOCATION::CPU);


	SolverResults solver_results(A_pages.GetNumPages(), ATOL, CPU_EXISTENCE::EXISTENT, GPU_EXISTENCE::EXISTENT);

	//solving A*x = b; 
	
	
	//---------------------------------------------------------------------------------------------------------------------------------------------------------


	std::cout << "***************************************************************************************************************************" << std::endl;
	std::cout << "***************************************************************************************************************************" << std::endl;
	std::cout << std::endl << "Welcome to ilu preconditioned batched BiCGSTAB on GPU\n";
	//Copy stuff to gpu
	A_pages.AllocateMemory(LOCATION::GPU);
	A_pages.CopyFromCpuToGpu();
	b_pages.AllocateMemory(LOCATION::GPU);
	b_pages.CopyFromCpuToGpu();
	x_pages.AllocateMemory(LOCATION::GPU);

	double av_time = 0;

	for(int i=0; i< 5; i++)
	{
		Batched_ILU_Preconditioned_BiCGSTAB_Gpu(subdir, A_pages, b_pages, x_pages, is_scaled,  solver_results, is_parilu, num_iter_par_ilu);  
		av_time += solver_results.GetTimeTaken();
	}
	
	av_time = av_time/5;
	solver_results.CopyFromGpuToCpu();


	if(argc == 7)
	{
		std::string f = argv[6];
		std::ofstream timings_file;
  		timings_file.open (f, std::ofstream::app);
  		// timings_file << "category: "  <<  category << "   is_scaled: " << is_scaled << "   problem_size: " << Problem_Size << "  time(cuda) in millisec: " << solver_results.GetTimeTaken() <<  " \n";
  		timings_file  << Problem_Size << "   " <<  av_time <<  " \n"; 
		timings_file.close();
	}
	
	std::string specifics;
	if(is_parilu == true)
	{
	   specifics = std::string("parILU_") + std::to_string(num_iter_par_ilu)  +  std::string("_");
	}
	else
	{
	   specifics = std::string("conv_ILU_");
	}
	
	PrintSolverResults(subdir,solver_results, std::string("Results_bicgstab_")  + specifics + category + std::string(".txt"));




}
