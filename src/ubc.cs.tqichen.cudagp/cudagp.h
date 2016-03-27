/*
 ============================================================================
 Name        : cudagp
 Author      : tqichen@cs.ubc.ca
 Version     :
 Copyright   :
 Description : CUDA implementation of Gaussian Process regression.
 ============================================================================
 */

#ifndef CUDAGP_H_
#define CUDAGP_H_

#include <stdio.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

struct dataset_t {
    float* X;
    float* y;
    int  n,d;
};

typedef enum {
    cudagpSquaredExponentialKernel,
    cudagpExponentialKernel,
    cudagpRationalQuadraticKernel
} Kernel_t;

struct cudagp_handle_t {
    dataset_t d_dataset;
    Kernel_t kernel;
    float* d_kernel_params;
    int numParams;
    float* d_cov;
    cusolverDnHandle_t cusolverHandle;
    cublasHandle_t cublasHandle;
};

static int numParams(Kernel_t kernel) {
    switch(kernel) {
    case cudagpSquaredExponentialKernel:
        return 1;
    case cudagpExponentialKernel:
        return 1;
    case cudagpRationalQuadraticKernel:
        return 2;
    default:
        fprintf(stderr, "ERROR: Invalid kernel method.");
        exit(EXIT_FAILURE);
        return -1;
    }
}

/**
 * Initializes cudaGP on the GPU, and returns a handle.
 */
cudagp_handle_t initializeCudaGP(float *h_X, float* h_y, int n, int d, Kernel_t kernel); // parameters will be randomized
cudagp_handle_t initializeCudaGP(float *h_X, float* h_y, int n, int d, Kernel_t kernel, float* defaultparams); // parameters will be set to these default ones.

void freeCudaGP(cudagp_handle_t ahandle); // frees up the GPU memory.

#endif /* CUDAGP_H_ */
