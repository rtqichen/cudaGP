/*
 * cudagp_predict.cu
 *
 *  Created on: Mar 27, 2016
 *      Author: tqichen
 */

#include "../cudagp.h"
#include "cov.h"
#include "linalg.h"
#include "utils_cuda.h"

void aggregatePOE(float* d_tmean, float* d_tvar, float* d_clusterMean, float* d_clusterCov, int t, cublasHandle_t cublashandle) {
    float alpha = 1.0f;

    // calculate var_k: marginal variances
    float* d_cvar = diag(d_clusterCov,t);

    // calculate var_k^-1
    elementwisePower(d_cvar, t, -1.0f);

    // update mean: mean += var_k^1 * mean_k
    elementwiseMultiplication(d_clusterMean, d_cvar, t);
    checkCublasErrors(cublasSaxpy_v2(cublashandle, t, &alpha, d_clusterMean, 1, d_tmean, 1));

    // update var: var^-1 += var_k^-1
    checkCublasErrors(cublasSaxpy_v2(cublashandle, t, &alpha, d_cvar, 1, d_tvar, 1));

    checkCudaErrors(cudaFree(d_cvar));
}

void postprocessPOE(float* d_tmean, float* d_tvar, int t) {
    // calculate var
    elementwisePower(d_tvar, t, -1.0f);

    // calculate mean
    elementwiseMultiplication(d_tmean, d_tvar, t);
}

/**
 * Predicts the mean and covariance matrices given some test points.
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
    float *d_tmean;
    checkCudaErrors(cudaMalloc((void**)&d_tmean, t*sizeof(float)));
    checkCudaErrors(cudaMemset(d_tmean, 0, t*sizeof(float)));

    float *d_tvar;
    checkCudaErrors(cudaMalloc((void**)&d_tvar, t*t*sizeof(float)));
    checkCudaErrors(cudaMemset(d_tvar, 0, t*t*sizeof(float)));

    // for each cluster TODO: do this in parallel
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
        float *d_clusterCov = conditionalCov(n, d, d_cov, d_Xtest, t, d_covfy, kernel, d_params, cusolverhandle, cublashandle);

        // --- Aggregate the results
        // (Various aggregation schemes in DGP paper.)
        // TODO: implement the others.
        aggregatePOE(d_tmean, d_tvar, d_clusterMean, d_clusterCov, t, cublashandle);

        // --- Free up memory.
        checkCudaErrors(cudaFree(d_cov));
        checkCudaErrors(cudaFree(d_covfy));
        checkCudaErrors(cudaFree(d_clusterMean));
        checkCudaErrors(cudaFree(d_clusterCov));
    }

    // --- Postprocess the aggregation
    // TODO: implement the others.
    postprocessPOE(d_tmean, d_tvar, t);

    // --- Transfer data to host
    float *h_tmean = (float*) malloc(t*sizeof(float));
    checkCudaErrors(cudaMemcpy(h_tmean, d_tmean, t*sizeof(float), cudaMemcpyDeviceToHost));
    float* h_tvar = (float*) malloc(t*t*sizeof(float));
    checkCudaErrors(cudaMemcpy(h_tvar, d_tvar, t*t*sizeof(float), cudaMemcpyDeviceToHost));

    struct prediction_t pred;
    pred.mean = h_tmean;
    pred.var = h_tvar;
    pred.t = t;

    // free up memory
    checkCudaErrors(cudaFree(d_Xtest));
    checkCudaErrors(cudaFree(d_tmean));
    checkCudaErrors(cudaFree(d_tvar));

    return pred;
}
