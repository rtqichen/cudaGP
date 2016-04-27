/*
 * linalg.h
 *
 * Contains linear algebra routines not available in cuBLAS.
 *
 *  Created on: Apr 25, 2016
 *      Author: tqichen
 */

#ifndef LINALG_H_
#define LINALG_H_

/**
 * Performs elementwise power and *overwrites* the existing vector/matrix.
 *
 * n is size of A
 */
void elementwisePower(float* d_A, int n, float pow);

/**
 * Performs elementwise multiplication and *overwrites* the first vector/matrix.
 *
 * n is size of A and B
 */
void elementwiseMultiplication(float* d_A, float* d_B, int n);

/**
 * Returns the diagonal of the matrix in a new vector.
 *
 * A is a square matrix (n by n) stored in row-major order.
 */
float* diag(float* d_A, int n);

/**
 * Adds alpha to the diagonal elements of a square matrix A.
 */
void diagAdd(float* d_A, int n, float alpha);

#endif /* LINALG_H_ */
