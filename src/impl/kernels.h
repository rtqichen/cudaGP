/*
 * kernels.h
 *
 *  Created on: Mar 25, 2016
 *      Author: tqichen
 */

#ifndef KERNELS_H_
#define KERNELS_H_

#include "../cudagp.h"

typedef float (*kernelfunc)(float*,float*,int,float*); // vecx, vecy, dim, params

/**
 * Suite of kernel functions
 */
__device__ __host__ float squared_exponential_kernel(float *x, float *y, int d, float *params);

__device__ __host__ float exponential_kernel(float *x, float *y, int d, float *params);

__device__ __host__ float rational_quadratic_kernel(float *x, float *y, int d, float *params);

/**
 * Returns the appropriate kernel function.
 */
__device__ __host__ kernelfunc getKernelFunction(kernelstring_enum whatKernel);

#endif /* KERNELS_H_ */
