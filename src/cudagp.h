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

struct cudagphandle_t {
    int numClusters;
    dataset_t d_dataset;
    Kernel_t kernel;
    float* d_params;
    int numParams;
    cusolverDnHandle_t cusolverHandle;
    cublasHandle_t cublasHandle;
};

struct prediction_t {
    float* mean;
    float* cov;
    int t;
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
cudagphandle_t initializeCudaGP(float *h_X, float* h_y, int n, int d, Kernel_t kernel); // parameters will be randomized
cudagphandle_t initializeCudaGP(float *h_X, float* h_y, int n, int d, Kernel_t kernel, float* defaultparams); // parameters will be set to these default ones.
prediction_t predict(cudagphandle_t cudagphandle, float* h_Xtest, int t);
void freeCudaGP(cudagphandle_t ahandle); // frees up the GPU memory.

/**
 * Initializes "distributed" GP.
 */
cudagphandle_t initializeCudaDGP(float *h_X, float* h_y, int n, int d, Kernel_t kernel); // parameters will be randomized
cudagphandle_t initializeCudaDGP(float *h_X, float* h_y, int n, int d, Kernel_t kernel, float* defaultparams); // parameters will be set to these default ones.

#endif /* CUDAGP_H_ */
