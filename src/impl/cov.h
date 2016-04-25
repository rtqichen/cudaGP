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
float* constructCovMatrix(float *d_X, int n, int d, Kernel_t kernel, float *d_params);

/**
 * Constructs the cross covariance matrix K(Xtest,X) of size t by n.
 */
float* constructCrossCovMatrix(cudagphandle_t cudagphandle, float *d_Xtest, int t);

/**
 * Calculates the conditional mean of t test data points.
 * Requires the calculation of Kyf (n by t) covariance matrix
 * between the training points and testing points.
 */
float* conditionalMean(cudagphandle_t cudagphandle, float *d_cov, float *d_Xtest, int t, float *d_covfy);

float* conditionalCov(cudagphandle_t cudagphandle, float *d_cov, float *d_Xtest, int t, float *d_covfy);

/**
 * Performs Cholesky factorization on a covariance matrix of n by n.
 * The Cholesky factorization is stored in (overwrites) the lower
 * triangular half of the covariance matrix.
 */
void cholFactorizationL(cusolverDnHandle_t cusolverhandle, float* d_cov, int n);

/**
 * Constructs a dense n by n identity matrix on the device,
 * placing it in the address location specified by d_eye.
 */
void eye(int n, float* d_eye);

#endif /* COV_H_ */
