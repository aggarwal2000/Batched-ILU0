## ILU(0) preconditioned bicgstab


``` nvcc main_bicgstab_ilu.cpp matrix.cu  factorization.cu ILU_0_gpu.cu  parILU_0_gpu.cu PrecondBiCGSTAB_gpu_ilu.cu SolverResults.cu  ReadWriteData.cpp  mmio.cpp  ```

### <b><i><u> Entering category name and problem size and option for scaled or unscaled as command line arguments : </b></i></u>


###  Format: ./a.out [category name] [problemsize] [1 for scaled] [0 for conventional ilu, 1 for par ilu ] [num_sweeps for par ilu(will be used only in case of par ilu) ] 


-- Pls enter one of these names for category name:
- gri30
- gri12
- drm19
- isooctane
- dodecane_lu
- lidryer


-- Problem Size:
  Problem size k indicates that total of k*n problems are to be solved (say the entered category has n small matrices).
  But pls note that answer files are produced only for the first n problems. Rest of the answers are not actually written into any file.


-- Enter 1 to use the scaled version of matrices.


Note: I am yet to test if it works for isoocatne category by increasing the allocated static shared memory!
