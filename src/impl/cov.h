/*
 * cov.h
 *
 *  Created on: Mar 24, 2016
 *      Author: tqichen
 */

#ifndef COV_H_
#define COV_H_

/**
 * Constructs the covariance matrix on the device,
 * and returns a device pointer to the matrix.
 */
float* constructCovMatrix(float *d_X, int n, int d, kernelstring_enum kernel, float *d_params);

/**
 * Constructs the cross covariance matrix K(Xtest,X) of size t by n.
 */
float* constructCrossCovMatrix(float *d_X, int n, int d, float *d_Xtest, int t, kernelstring_enum kernel, float *d_params);

/**
 * Calculates the conditional mean of t test data points.
 * Requires the calculation of Kyf (n by t) covariance matrix
 * between the training points and testing points.
 */
float* conditionalMean(float *d_y, int n, float *d_cov, float *d_Xtest, int t, float *d_covfy, cusolverDnHandle_t cusolverhandle, cublasHandle_t cublashandle);

float* conditionalCov(int n, int d, float *d_cov, float *d_Xtest, int t, float *d_covfy, kernelstring_enum kernel, float *d_params, cusolverDnHandle_t cusolverhandle, cublasHandle_t cublashandle);

/**
 * Performs Cholesky factorization on a covariance matrix of n by n.
 * The Cholesky factorization is stored in (overwrites) the lower
 * triangular half of the covariance matrix.
 */
void cholFactorizationL(float* d_cov, int n, cusolverDnHandle_t cusolverhandle);

/**
 * Constructs a dense n by n identity matrix on the device,
 * placing it in the address location specified by d_eye.
 */
float* eye(unsigned int n);

#endif /* COV_H_ */
