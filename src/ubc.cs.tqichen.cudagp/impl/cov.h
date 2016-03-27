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
float* constructCovMatrix(dataset_t d_ds, Kernel_t kernel_string, float* d_kernel_params);

/**
 * Performs Cholesky factorization on a covariance matrix of n by n.
 * The Cholesky factorization is stored in (overwrites) the lower
 * triangular half of the covariance matrix.
 */
void cholFactorizationL(cusolverDnHandle_t cusolverhandle, float* d_cov, int n);

/**
 * Takes the lower triangular Cholesky decomposition of a covariance matrix and inverts it.
 */
float* invertCovMatrix(cublasHandle_t cublashandle, float* d_cov, int n);

/**
 * Constructs a dense n by n identity matrix on the device,
 * placing it in the address location specified by d_eye.
 */
void eye(int n, float* d_eye);

#endif /* COV_H_ */
