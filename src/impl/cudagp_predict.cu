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

    int d = cudagphandle.d_dataset.d;

    // --- transfer Xtest to GPU
    float* d_Xtest;
    checkCudaErrors(cudaMalloc((void**)&d_Xtest, t*d*sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_Xtest, h_Xtest, t*d*sizeof(float), cudaMemcpyHostToDevice));

    // --- Calculate Kfy (t by n)
    float *d_covfy = constructCrossCovMatrix(cudagphandle, d_Xtest, t);

    // --- Calculate test mean (t by 1)
    float *d_mean = conditionalMean(cudagphandle, d_Xtest, t, d_covfy);

    // --- Calculate test covariance (t by t)
    //float* d_tcov = conditionalCov(cudagphandle, d_Xtest, t, d_covfy);

    // --- Transfer data to host

    float *h_mean = (float*) malloc(t*sizeof(float));
    checkCudaErrors(cudaMemcpy(h_mean, d_mean, t*sizeof(float), cudaMemcpyDeviceToHost));

    //float* h_tcov = (float*) malloc(t*t*sizeof(float));
    //checkCudaErrors(cudaMemcpy(h_tcov, d_tcov, t*t*sizeof(float), cudaMemcpyDeviceToHost));

    struct prediction_t pred;
    pred.mean = h_mean;
    //pred.cov = h_tcov;
    pred.t = t;

    return pred;
}
