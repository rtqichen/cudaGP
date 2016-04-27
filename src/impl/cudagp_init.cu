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

parameters_t initDeviceParams(kernelstring_enum kernel, float *h_defaultParams, bool useDefaultParams) {
    // --- Set kernel parameters
    int np = numParams(kernel);
    float *h_params;

    if (!useDefaultParams) {
        h_params = (float*) malloc(np*sizeof(float));
        for (int i=0; i<np; i++) {
            h_params[i] = rand() / RAND_MAX; // is there a smarter way to do initialization? this can introduce numerical instability.
        }
    } else {
        h_params = h_defaultParams;
    }

    float *d_paramvalues;
    checkCudaErrors(cudaMalloc((void**)&d_paramvalues, np*sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_paramvalues, h_params, np*sizeof(float), cudaMemcpyHostToDevice));

    parameters_t d_parameters;
    d_parameters.kernel = kernel;
    d_parameters.values = d_paramvalues;
    d_parameters.numParams = np;

    return d_parameters;
}

cublasHandle_t initCublas() {
    // --- CuBLAS initialization
    cublasHandle_t cublashandle;
    cublasCreate(&cublashandle);
    return cublashandle;
}

cusolverDnHandle_t initCusolver() {
    // --- CuSolver initialization
    cusolverDnHandle_t cusolverhandle;
    cusolverDnCreate(&cusolverhandle);
    return cusolverhandle;
}

/**
 * Splits the dataset into clusters.
 * For now, just returns an array of the indices of the start of each cluster.
 * TODO: cluster smartly so that information lost is minimized. (Covariance matrix retains the big numbers for examples.)
 */
int* splitDataset(int n, int numClusters) {
    int clusterSize = divUp(n, numClusters);

    int *startIndices = (int*) malloc(numClusters*sizeof(int));

    startIndices[0] = 0;
    for (int i=1; i<numClusters; i++) {
        startIndices[i] = startIndices[i-1]+clusterSize < n ? startIndices[i-1]+clusterSize : n;
    }
    return startIndices;
}

/**
 * Construction calls for cudaGP.
 */
cudagphandle_t initializeCudaGP(float *h_X, float* h_y, int n, int d, kernelstring_enum kernel) {
    cudagphandle_t cudagphandle;

    cudagphandle.numClusters = 1;
    cudagphandle.d_dataset = (dataset_t*) malloc(sizeof(dataset_t));
    cudagphandle.d_dataset[0] = transferDataToDevice(h_X, h_y, n, d);
    cudagphandle.d_parameters = initDeviceParams(kernel, 0, false);
    cudagphandle.cusolverHandle = initCusolver();
    cudagphandle.cublasHandle = initCublas();

    return cudagphandle;
}

cudagphandle_t initializeCudaGP(float *h_X, float* h_y, int n, int d, kernelstring_enum kernel, float* defaultparams) {
    cudagphandle_t cudagphandle;

    cudagphandle.numClusters = 1;
    cudagphandle.d_dataset = (dataset_t*) malloc(sizeof(dataset_t));
    cudagphandle.d_dataset[0] = transferDataToDevice(h_X, h_y, n, d);
    cudagphandle.d_parameters = initDeviceParams(kernel, defaultparams, true);
    cudagphandle.cusolverHandle = initCusolver();
    cudagphandle.cublasHandle = initCublas();

    return cudagphandle;
}

cudagphandle_t initializeCudaDGP(float *h_X, float* h_y, int n, int d, kernelstring_enum kernel, int numClusters) {
    cudagphandle_t cudagphandle;

    cudagphandle.numClusters = numClusters;
    cudagphandle.d_dataset = (dataset_t*) malloc(numClusters*sizeof(dataset_t));

    int *start = splitDataset(n, numClusters);
    for (int i=0; i<numClusters; i++) {
        int size = i == numClusters-1 ? n - start[i] : start[i+1]-start[i];
        cudagphandle.d_dataset[i] = transferDataToDevice(&h_X[start[i]],&h_y[start[i]],size,d);
    }

    cudagphandle.d_parameters = initDeviceParams(kernel, 0, false);
    cudagphandle.cusolverHandle = initCusolver();
    cudagphandle.cublasHandle = initCublas();

    return cudagphandle;
}

cudagphandle_t initializeCudaDGP(float *h_X, float* h_y, int n, int d, kernelstring_enum kernel, int numClusters, float* defaultparams) {
    cudagphandle_t cudagphandle;

    cudagphandle.numClusters = numClusters;
    cudagphandle.d_dataset = (dataset_t*) malloc(numClusters*sizeof(dataset_t));

    int *start = splitDataset(n, numClusters);
    for (int i=0; i<numClusters; i++) {
        int size = i == numClusters-1 ? n - start[i] : start[i+1]-start[i];
        cudagphandle.d_dataset[i] = transferDataToDevice(&h_X[start[i]],&h_y[start[i]],size,d);
    }

    cudagphandle.d_parameters = initDeviceParams(kernel, defaultparams, true);
    cudagphandle.cusolverHandle = initCusolver();
    cudagphandle.cublasHandle = initCublas();

    return cudagphandle;
}

void freeCudaGP(cudagphandle_t ahandle) {
    for (int i=0; i<ahandle.numClusters; i++) {
        checkCudaErrors(cudaFree(ahandle.d_dataset[i].X));
        checkCudaErrors(cudaFree(ahandle.d_dataset[i].y));
    }
    checkCudaErrors(cudaFree(ahandle.d_parameters.values));
    checkCusolverErrors(cusolverDnDestroy(ahandle.cusolverHandle));
    cublasDestroy(ahandle.cublasHandle);
}
