#include<stdio.h>
#include<stdlib.h>


#pragma once

#define THREADS_PER_BLOCK 512

// size of warp(as provided by CUDA )
#define WARP_SIZE 32

#define FULL_MASK 0xffffffff

#define CHECK( X ) do { cudaError_t err = (X);  if(err != cudaSuccess) { printf(" cuda error - %s:%d Returned:%d \n ", __FILE__, __LINE__, err); }}while(0);


# define MAX_NUM_ROWS 54
# define MAX_NUM_NZ 2560