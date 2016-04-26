/*
 * linalg.cu
 *
 *  Created on: Apr 25, 2016
 *      Author: tqichen
 */

#include "utils_cuda.h"

#define BLOCKSIZE 256
#define ELEMWISE_WORK 1

__global__ void elementwisePower_k(float* d_A, int n, float pow) {
    unsigned int idx = threadIdx.x + ELEMWISE_WORK*blockIdx.x*blockDim.x;

    for (int i=0; i<ELEMWISE_WORK; i++) {
        if (idx+ELEMWISE_WORK < n) d_A[idx] = powf(d_A[idx], pow);
    }
}

void elementwisePower(float* d_A, int n, float pow) {
    int nthreads = divUp(n,ELEMWISE_WORK);

    dim3 blocksize(BLOCKSIZE);
    dim3 gridsize = divUp(nthreads, BLOCKSIZE);

    elementwisePower_k<<<gridsize,blocksize>>>(d_A,n,pow);
}

__global__ void elementwiseMultiplication_k(float* d_A, float* d_B, int n) {
    unsigned int idx = threadIdx.x + ELEMWISE_WORK*blockIdx.x*blockDim.x;

    for (int i=0; i<ELEMWISE_WORK; i++) {
        if (idx+ELEMWISE_WORK < n) d_A[idx] = d_A[idx] * d_B[idx];
    }
}

void elementwiseMultiplication(float* d_A, float* d_B, int n) {
    int nthreads = divUp(n,ELEMWISE_WORK);

    dim3 blocksize(BLOCKSIZE);
    dim3 gridsize = divUp(nthreads, BLOCKSIZE);

    elementwiseMultiplication_k<<<gridsize,blocksize>>>(d_A,d_B,n);
}

__global__ void diag_k(float* d_A, int n, float* d_diag) {
    unsigned int idx = threadIdx.x + ELEMWISE_WORK*blockIdx.x*blockDim.x;

    for (int i=0; i<ELEMWISE_WORK; i++) {
        if (idx+ELEMWISE_WORK < n) d_diag[idx] = d_A[idx*n+idx];
    }
}

float* diag(float* d_A, int n) {
    float *d_diag;
    checkCudaErrors(cudaMalloc((void**)&d_diag, n*sizeof(float)));

    int nthreads = divUp(n,ELEMWISE_WORK);

    dim3 blocksize(BLOCKSIZE);
    dim3 gridsize = divUp(nthreads, BLOCKSIZE);

    diag_k<<<gridsize,blocksize>>>(d_A,n,d_diag);

    return d_diag;
}
