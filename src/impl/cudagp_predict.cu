/*
 * cudagp_predict.cu
 *
 *  Created on: Mar 27, 2016
 *      Author: tqichen
 */

#include "../cudagp.h"
#include "cov.h"
#include "utils_cuda.h"

/**
 * Predicts the mean and covariance matrices given some test points.
 *
 * TODO: calculate parts of the mean and cov calculations can be done in parallel.
 */
prediction_t predict(cudagphandle_t cudagphandle, float* h_Xtest, int t) {

    cusolverDnHandle_t cusolverhandle = cudagphandle.cusolverHandle;
    cublasHandle_t cublashandle = cudagphandle.cublasHandle;
    kernelstring_enum kernel = cudagphandle.d_parameters.kernel;
    float *d_params = cudagphandle.d_parameters.values;

    int d = cudagphandle.d_dataset[0].d;

    // --- transfer Xtest to GPU
    float* d_Xtest;
    checkCudaErrors(cudaMalloc((void**)&d_Xtest, t*d*sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_Xtest, h_Xtest, t*d*sizeof(float), cudaMemcpyHostToDevice));

    // --- Create containers for d_mean and d_tcov
    float *d_mean;
    checkCudaErrors(cudaMalloc((void**)&d_mean, t*sizeof(float)));
    checkCudaErrors(cudaMemset(d_mean, 0, t*sizeof(float)));

    float *d_tcov;
    checkCudaErrors(cudaMalloc((void**)&d_tcov, t*t*sizeof(float)));
    checkCudaErrors(cudaMemset(d_tcov, 0, t*t*sizeof(float)));

    // for each cluster:
    for (int i=0; i<cudagphandle.numClusters; i++) {

        float *d_X = cudagphandle.d_dataset[i].X;
        float *d_y = cudagphandle.d_dataset[i].y;
        int n = cudagphandle.d_dataset[i].n;

        // --- Construct full covariance matrix
        float *d_cov = constructCovMatrix(d_X, n, d, kernel, d_params);

        // --- Calculate Cholesky factorization
        cholFactorizationL(d_cov, n, cusolverhandle);

        // --- Calculate Kfy (t by n)
        float *d_covfy = constructCrossCovMatrix(d_X, n, d, d_Xtest, t, kernel, d_params);

        // --- Calculate test mean (t by 1)
        float *d_clusterMean = conditionalMean(d_y, n, d_cov, d_Xtest, t, d_covfy, cusolverhandle, cublashandle);

        // --- Calculate test covariance (t by t)
        float *d_clusterTCov = conditionalCov(n, d, d_cov, d_Xtest, t, d_covfy, kernel, d_params, cusolverhandle, cublashandle);

        // --- TODO: How to aggregate the means and covariances?
        // For now, just sum elementwise.
        float alpha = 1.0f;
        checkCublasErrors(cublasSaxpy_v2(cublashandle, t, &alpha, d_clusterMean, 1, d_mean, 1));
        checkCublasErrors(cublasSaxpy_v2(cublashandle, t*t, &alpha, d_clusterTCov, 1, d_tcov, 1));

        checkCudaErrors(cudaFree(d_cov));
        checkCudaErrors(cudaFree(d_covfy));
        checkCudaErrors(cudaFree(d_clusterMean));
        checkCudaErrors(cudaFree(d_clusterTCov));
    }

    // --- Transfer data to host
    float *h_mean = (float*) malloc(t*sizeof(float));
    checkCudaErrors(cudaMemcpy(h_mean, d_mean, t*sizeof(float), cudaMemcpyDeviceToHost));
    float* h_tcov = (float*) malloc(t*t*sizeof(float));
    checkCudaErrors(cudaMemcpy(h_tcov, d_tcov, t*t*sizeof(float), cudaMemcpyDeviceToHost));

    struct prediction_t pred;
    pred.mean = h_mean;
    pred.cov = h_tcov;
    pred.t = t;

    // free up memory
    checkCudaErrors(cudaFree(d_Xtest));
    checkCudaErrors(cudaFree(d_mean));
    checkCudaErrors(cudaFree(d_tcov));

    return pred;
}
