#include "kernels.h"
#include <math.h>
#include <stdio.h>

/*
 * Returns ||x-y||_2
 */
__device__ __host__ float diffL2Squared(float *x , float *y, int d) {
    float sum = 0.0f;
    for (int i=0; i<d; i++) {
        sum += (x[i]-y[i])*(x[i]-y[i]);
    }
    return sum;
}

/**
 * Suite of kernel functions
 */
__device__ __host__ float squared_exponential_kernel(float *x, float *y, int d, float *params) {
    float r = diffL2Squared(x,y,d);
    float l = params[0];
    return exp(-r/(l*l*2));
}

__device__ __host__ float exponential_kernel(float *x, float *y, int d, float *params) {
    float r = diffL2Squared(x,y,d);
    float l = params[0];
    return exp(-sqrt(r)/l);
}

__device__ __host__ float rational_quadratic_kernel(float *x, float *y, int d, float *params) {
    float r = diffL2Squared(x,y,d);
    float l = params[0];
    float a = params[1];
    return pow(1 + (r) / (2*a*l*l), -a);
}

/**
 * Returns the appropriate kernel function.
 */
__device__ __host__ kernelfunc getKernelFunction(kernelstring_enum whatKernel) {
    switch (whatKernel) {
    case cudagpSquaredExponentialKernel:
        return squared_exponential_kernel;
    case cudagpExponentialKernel:
        return exponential_kernel;
    case cudagpRationalQuadraticKernel:
        return rational_quadratic_kernel;
    default:
        printf("WARN: Invalid kernel %d. Defaulting to squared exponential kernel.", whatKernel);
        return squared_exponential_kernel;
    }
}
