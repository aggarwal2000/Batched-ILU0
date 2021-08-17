#!/bin/bash


nvcc main_bicgstab_ilu.cpp matrix.cu  factorization.cu ILU_0_gpu.cu  parILU_0_gpu.cu PrecondBiCGSTAB_gpu_ilu.cu SolverResults.cu  ReadWriteData.cpp  mmio.cpp 

for category in  drm19 dodecane_lu gri30 gri12
do
	
	file_name="timings_ilu0_"$category".txt"

	for ((problem_size= 1 ; problem_size<=100 ; problem_size++))
	do

		./a.out $category $problem_size  1  0  0  $file_name
		
	done
	
done




for category in  drm19 dodecane_lu gri30 gri12
do
	for num_iter in 1 2 3 4
	do

		file_name="timings_par_ilu0_"$category"_sweeps_"$num_iter".txt"

		for ((problem_size= 1 ; problem_size<=100 ; problem_size++))
		do

			./a.out $category $problem_size 1  1  $num_iter $file_name
			
		done

	done
	
done




