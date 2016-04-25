#include <stdio.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include "../cudagp.h"
#include "cov.h"
#include "utils_cuda.h"

/*
 * Initializes the sparseGP by assigning data points to separate experts.
 *
 * Specifically, the initialization function performs the following tasks:
 *
 * (1) The full covariance matrix is computed.
 * (2) Data points are clustered (?).
 * (3) Each cluster is assigned to an expert.
 */

/*
 * Copies the dataset (X,y) onto the GPU
 */
dataset_t transferDataToDevice(const float *h_X, const float* h_y, const int n, const int d) {

    float *d_X, *d_y;
    checkCudaErrors(cudaMalloc((void**)&d_X,sizeof(float)*n*d));
    checkCudaErrors(cudaMalloc((void**)&d_y,sizeof(float)*n));

    checkCudaErrors(cudaMemcpyAsync(d_X,h_X,sizeof(float)*n*d,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_y,h_y,sizeof(float)*n,cudaMemcpyHostToDevice));

    dataset_t d_data;
    d_data.X = d_X;
    d_data.y = d_y;
    d_data.d = d;
    d_data.n = n;

    return d_data;
}

/**
 * Transfers the full dataset onto the GPU.
 * Then calculates the covariance matrix along with its Cholesky decomposition.
 */
cudagphandle_t initializeCudaGP(
        const float *h_X,
        const float* h_y,
        const int n,
        const int d,
        const Kernel_t kernel,
        float* h_defaultParams,
        bool useDefaultparams) {

    // --- Transfer dataset to GPU
    dataset_t d_ds = transferDataToDevice(h_X,h_y,n,d);

    // --- Set kernel parameters
    int np = numParams(kernel);
    float* h_params;

    if (!h_defaultParams) {
        h_params = (float*) malloc(np*sizeof(float));
        for (int i=0; i<np; i++) {
            h_params[i] = rand() / RAND_MAX; // is there a smarter way to do initialization? this can introduce numerical instability.
        }
    } else {
        h_params = h_defaultParams;
    }

    float* d_params;
    checkCudaErrors(cudaMalloc((void**)&d_params, np*sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_params, h_params, np*sizeof(float), cudaMemcpyHostToDevice));

    // --- CuBLAS initialization
    cublasHandle_t cublashandle;
    cublasCreate(&cublashandle);

    // --- CuSolver initialization
    cusolverDnHandle_t cusolverhandle;
    cusolverDnCreate(&cusolverhandle);

    // --- CudaGP handle
    cudagphandle_t cudagphandle;
    cudagphandle.d_dataset = d_ds;
    cudagphandle.kernel = kernel;
    cudagphandle.numParams = np;
    cudagphandle.d_params = d_params;
    cudagphandle.cusolverHandle = cusolverhandle;
    cudagphandle.cublasHandle = cublashandle;

    return cudagphandle;
}

cudagphandle_t initializeCudaGP(float *h_X, float* h_y, int n, int d, Kernel_t kernel) {
    return initializeCudaGP(h_X, h_y, n, d, kernel, 0, false);
}

cudagphandle_t initializeCudaGP(float *h_X, float* h_y, int n, int d, Kernel_t kernel, float* defaultparams) {
    return initializeCudaGP(h_X, h_y, n, d, kernel, defaultparams, true);
}

void freeCudaGP(cudagphandle_t ahandle) {
    checkCudaErrors(cudaFree(ahandle.d_dataset.X));
    checkCudaErrors(cudaFree(ahandle.d_dataset.y));
    checkCudaErrors(cudaFree(ahandle.d_params));
    checkCusolverErrors(cusolverDnDestroy(ahandle.cusolverHandle));
    cublasDestroy(ahandle.cublasHandle);
}
