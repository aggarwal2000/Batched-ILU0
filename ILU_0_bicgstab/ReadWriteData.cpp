#include<iostream>
#include<vector>
#include<string>
#include<dirent.h>
#include<cstring>
#include <utility>
#include<algorithm>
#include<cassert>

#include"mmio.h"
#include"matrix.h"
#include"SolverResults.h"



void Print_ans(const std::vector<std::string> & subdir,const PagedVectors & Ans_pages,const std::string & ans_name)
{   
    assert(Ans_pages.ExistsCPU() == true);
    
    for(int page_id =0; page_id < subdir.size(); page_id++)
    {
        std::string file_ans = subdir[page_id] + "/" + ans_name;

        FILE* fp;
        fp = fopen(file_ans.c_str() , "w");

		if(fp == NULL)
		 printf("\n\nCan't open file : %s " , file_ans.c_str());

        MM_typecode matcode;
        mm_initialize_typecode(&matcode); 
        mm_set_array(&matcode);
        mm_set_real(&matcode);
        mm_set_general(&matcode);
        mm_set_matrix(&matcode);
        
        mm_write_banner(fp,matcode);
    
        int M = Ans_pages.GetNumElements();
        int N = 1 ;
        mm_write_mtx_array_size(fp,M,N);

        for(int i=0;i<M;i++)
        {
            fprintf(fp,"%0.17lg\n", Ans_pages.GetPtrToCpuValues()[i +  page_id*M] );
        }
    

        if(fp != stdin)
             fclose(fp);

	}
        
}





void PrintSolverResults(const std::vector<std::string> &subdir,const SolverResults &solver_results, const std::string & f_name)
{
	assert(solver_results.ExistsCPU() == true);

	 
        FILE* fp;
        fp = fopen(f_name.c_str() , "w");

		if(fp == NULL)
		 printf("\n\nCan't open file : %s " , f_name.c_str());


		 // fprintf(fp, " problem , conv_flag , iter_count, iter_residual_norm, true_residual_norm\n\n ");

        for(int i=0;i< subdir.size(); i++)
        {	
			std::string problem = subdir[i];

             		// fprintf(fp, " %s , %d , %f , %0.17lg , %0.17lg \n ", problem.c_str() ,solver_results.GetPtrToCpuConvFlag()[i] , solver_results.GetPtrToCpuIterCount()[i], 
			// solver_results.GetPtrToCpuIterResNorm()[i], solver_results.GetPtrToCpuTrueResNorm()[i] ); 

			fprintf(fp, "%d   %f  \n",solver_results.GetPtrToCpuConvFlag()[i] , solver_results.GetPtrToCpuIterCount()[i] );
        }
    

        if(fp != stdin)
             fclose(fp);

	
}






void FillSubdir(const std::string& main_dir, std::vector<std::string> &subdir)
{
	struct dirent *de;  // Pointer for directory entry 
  
    // opendir() returns a pointer of DIR type.  
    DIR *dr = opendir(main_dir.c_str()); 
  
    if (dr == NULL)  // opendir returns NULL if couldn't open directory 
    { 
        std::cout << "Could not open " << main_dir << " directory." ; 
        exit(0);
    } 
  
    // Refer http://pubs.opengroup.org/onlinepubs/7990989775/xsh/readdir.html 
    // for readdir() 
    while ((de = readdir(dr)) != NULL)
    {    
         if(strcmp(de->d_name,".") == 0 || strcmp(de->d_name,"..") == 0 )   
            continue;

         subdir.push_back(main_dir + std::string(de->d_name));
    }
           
    closedir(dr);   
}



void Fill_general_COO(FILE* f_A,std::vector<int> & row_ind_coo,std::vector<int> & col_ind_coo,
std::vector<double> & val_coo ,const int nz_file, const MM_typecode matcode_A, int & true_nz)
{
	int I;
    int J;
    double val;
    //read matrix A (in COO format)

	true_nz = nz_file;

    if (mm_is_real(matcode_A) || mm_is_integer(matcode_A))
    {
        for (int i = 0; i < nz_file; i++)
        {
            fscanf(f_A, "%d %d %lg \n", &I, &J, &val);
            I--;   // adjust from 1-based to 0-based 
            J--;
			row_ind_coo.push_back(I);
			col_ind_coo.push_back(J);
			val_coo.push_back(val);
        }
    }
    else
    {
        printf("This case is not handled!");
        exit(1);
    }
}




static bool compare_first(
    const std::pair< int, double >& a,
    const std::pair< int, double >& b)
{
    return (a.first < b.first);
}


void Convert_coo_to_csr(const int M,const int nz,const std::vector<int> &row_ind_coo,
const std::vector<int> & col_ind_coo, const std::vector<double> & val_coo,int* row_ptrs_csr,
int* col_ind_csr, double* val_csr)
{	
	std::vector< std::pair< int, double > > rowval;
	// original code from  Nathan Bell and Michael Garland
    for (int i = 0; i < M; i++)
        row_ptrs_csr[i] = 0;

    for (int i = 0; i < nz; i++)
        row_ptrs_csr[row_ind_coo[i]]++;

    // cumulative sum the nnz per row to get row[]
    int cumsum;
    cumsum = 0;
    for (int i = 0; i < M; i++) {
        int temp = row_ptrs_csr[i];
        row_ptrs_csr[i] = cumsum;
        cumsum += temp;
    }
    row_ptrs_csr[M] = nz;


    for (int i = 0; i < nz; i++) {
        int row_ = row_ind_coo[i];
        int dest = row_ptrs_csr[row_];
        col_ind_csr[dest] = col_ind_coo[i];
        val_csr[dest] = val_coo[i];
        row_ptrs_csr[row_]++;
    }

    int last;
    last = 0;
    for (int i = 0; i <= M; i++) {
        int temp = (row_ptrs_csr)[i];
        (row_ptrs_csr)[i] = last;
        last = temp;
    }

    row_ptrs_csr[M] = nz;

    // sort column indices within each row
    // copy into vector of pairs (column index, value), sort by column index
    for (int k = 0; k < M; ++k) {
        int kk = (row_ptrs_csr)[k];
        int len = (row_ptrs_csr)[k + 1] - row_ptrs_csr[k];
        rowval.resize(len);
        for (int i = 0; i < len; ++i) {
            rowval[i] = std::make_pair(col_ind_csr[kk + i], val_csr[kk + i]);
        }
        std::sort(rowval.begin(), rowval.end(), compare_first);
        for (int i = 0; i < len; ++i) {
            col_ind_csr[kk + i] = rowval[i].first;
            val_csr[kk + i] = rowval[i].second;
        }
    }

}


void Read_A_and_b(const int problem_id, const std::string subdir_path,  PagedCSRMatrices & A_pages, PagedVectors & b_pages, const bool is_scaled)
{	

	std::string file_A;
	std::string file_b;
	if(is_scaled == true)
	{
		 file_A = subdir_path + "/A_scaled.mtx";
	     file_b = subdir_path + "/b_scaled.mtx";
	}
	else
	{ 
		 file_A = subdir_path + "/A.mtx";
	     file_b = subdir_path + "/b.mtx";

	}
	

	// std::cout << std::endl << "Problem id: " << problem_id << "     Reading files: " << file_A  << "  and " << file_b;

	int ret_code_A, ret_code_b;
    MM_typecode matcode_A, matcode_b;
    FILE* f_A, * f_b;

    int M, N, nz_file; //for matrix A
    int rows_b, cols_b; //for vector b

	f_A = fopen(file_A.c_str() , "r");

    if(f_A == NULL)
      {
         std::cout << "\nIssue with matrix A file";
         exit(1);
      }

    if (mm_read_banner(f_A, &matcode_A) != 0)
    {
        std::cout << "Could not process Matrix Market banner.\n" << std::endl;
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (!(mm_is_matrix(matcode_A) && mm_is_coordinate(matcode_A)))
    {
        std::cout << "Sorry , this application does not support this type for matrix A!" << std::endl;
        std::cout << "Matrix Market File " << mm_typecode_to_str(matcode_A) << std::endl;
        exit(1);
    }

    /* find out size of sparse matrix A .... */
    ret_code_A = mm_read_mtx_crd_size(f_A, &M, &N, &nz_file);
    if (ret_code_A != 0)
        exit(1);

    if (M != N)
    {
        printf("\nMatrix is not sqaure; not supported by the application");
        exit(1);
    }

	if ((f_b = fopen(file_b.c_str(), "r")) == NULL)
	{
		std::cout << " Issue with vector b file\n" << std::endl;
		exit(1);
	}

	if (mm_read_banner(f_b, &matcode_b) != 0)
	{
		std::cout << "Could not process Matrix Market banner.\n" << std::endl;
		exit(1);
	}

	if (!(mm_is_matrix(matcode_b) && mm_is_array(matcode_b)))
	{
		std::cout << "Sorry , this appication does not support this type for vector b !" << std::endl;
		std::cout << "Matrix Market File " << mm_typecode_to_str(matcode_A) << std::endl;
		exit(1);
	}

	// find out size of vector b 
	ret_code_b = mm_read_mtx_array_size(f_b, &rows_b, &cols_b);
	if (ret_code_b != 0)
		exit(1);


	if ((cols_b != 1) || (rows_b != M))
	{
		std::cout << "Sorry , System Ax = b makes no sense !!\n";
		exit(1);
	}
      //start  storing things
	
	          //--------------------------------------------------------------------------

	//store A
	std::vector<int> row_ind_coo;
	std::vector<int> col_ind_coo;
	std::vector<double> val_coo;

	int true_nz = 0;

    if(mm_is_general(matcode_A))
    	Fill_general_COO(f_A, row_ind_coo, col_ind_coo, val_coo , nz_file, matcode_A,true_nz);
	else
	{
		std::cout << "This case is not handled yet...";
		exit(1);
	}
	
	if(problem_id == 0)
	{	
		A_pages.SetNumRows(M);
		A_pages.SetNumCols(N);
		A_pages.SetNumNz(true_nz);

		A_pages.AllocateMemory(LOCATION::CPU); //(M +1 for row-ptrs; true_nz for col inds, true_nz*num_pages for vals)

		Convert_coo_to_csr(M,true_nz,row_ind_coo,col_ind_coo,val_coo,A_pages.GetPtrToCpuRowPtrs(), A_pages.GetPtrToCpuColInd(),
		A_pages.GetPtrToCpuValues());
	}
	else
	{	
		if(!(M == A_pages.GetNumRows() && N == A_pages.GetNumCols()))
		{
			std::cout << "\n\nrows or ncols is different!!" << std::endl;
			std::cout << "file currently opened: " << file_A << std::endl;
			exit(1);
		}
		
		std::vector<int> row_ptrs_csr(M + 1,0);
		std::vector<int> col_ind_csr(true_nz,0);
		std::vector<double> val_csr(true_nz,0);

		Convert_coo_to_csr(M,true_nz,row_ind_coo,col_ind_coo,val_coo, row_ptrs_csr.data() ,col_ind_csr.data(),val_csr.data());

		if(true_nz != A_pages.GetNumNz())
		{
			std::cout << "\n\nSparsity pattern is not the same -- nnz different !!" << std::endl;
			std::cout << "file currently opened: " << file_A << std::endl;
			exit(0);
		} 

		//after the conversion, store into main paged structure.

		double* val_A = A_pages.GetPtrToCpuValues();
		int* col_ind_A = A_pages.GetPtrToCpuColInd();
		int* row_ptr_A = A_pages.GetPtrToCpuRowPtrs();
		for(int i=0;i < true_nz ;i++ )
		{
			if(col_ind_A[i] != col_ind_csr[i])
			{

				std::cout << "\n\nSparsity pattern is not the same -- In col_ind comparison for csr";
				std::cout << "\nfile currently opened: " << file_A << std::endl;
				exit(1);
			}
		}
		for(int i=0;i< M ; i++)
		{
		
			if(row_ptr_A[i] != row_ptrs_csr[i])
			{
				std::cout << "\n\nSparsity pattern is not the same-- In row_ptrs comaprison for csr";
				std::cout << "\nfile currently opened: " << file_A << std::endl;
				exit(1);
			}
		}

		for(int i=0; i < true_nz; i++)
		{
			val_A[i + problem_id*true_nz] = val_csr[i];
		}
	}
		
	if (f_A != stdin) fclose(f_A);


	//store b
	if(problem_id == 0)
	{
		b_pages.SetNumElements(rows_b);
		b_pages.AllocateMemory(LOCATION::CPU);
	}
	

	for (int i = 0; i < M; i++)
	{
		double val;
		if (mm_is_integer(matcode_b) || mm_is_real(matcode_b))
		{
			fscanf(f_b, "%lg \n", &val);
			b_pages.GetPtrToCpuValues()[i + problem_id*M] = val;
		}
		else
		{
			printf(" This case is not handled ...");
			exit(1);
		}
	}

	if (f_b != stdin) fclose(f_b);
}





void ReadData(const int Problem_Size, const std::vector<std::string>& subdir, PagedCSRMatrices & A_pages, PagedVectors & b_pages,  const bool is_scaled)
{
	const int num_pages =  Problem_Size*subdir.size();

	A_pages.SetNumPages(num_pages);
	b_pages.SetNumPages(num_pages);

	if(num_pages == 0)
	{
		std::cout <<  std::endl << "Empty directory!" << std::endl;
		exit(0);
	}

	
	for(int problem_id=0; problem_id < num_pages ; problem_id++)
	{
		Read_A_and_b(problem_id,subdir[problem_id % subdir.size() ],A_pages,b_pages , is_scaled );
	}

	std::cout << "\n\nAll data read..." << std::endl;
}


