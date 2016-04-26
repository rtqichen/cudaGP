#include <stdio.h>

#include "kernels.h"
#include "utils_cuda.h"

/**
 * Computations of the covariance matrix.
 *
 * Assumes that matrices are stored in a row-major flattened array.
 *
 */

#define BLOCKSIZE 1024
#define BLOCKSIZE2D 32

/*
 * Functions that implement this must be tested VERY carefully...
 * So far functions that implement this:
 *  - eye
 */
#define MULTI 5

#define CHOLESKY_FILL_MODE CUBLAS_FILL_MODE_UPPER

/**
 * Computes the full covariance matrix and stores it in d_cov.
 */
__global__ void constructCovMatrix_k(float *d_X, int n, int d, kernelstring_enum kernel, float* d_params, float* d_cov) {

    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y*blockDim.y;

    if (idx < n && idy < n) {
        kernelfunc kernfunc = getKernelFunction(kernel);

        float *vecx = &d_X[idx*d];
        float *vecy = &d_X[idy*d];

        d_cov[idx*n+idy] = kernfunc(vecx,vecy,d,d_params);
    }
}


float* constructCovMatrix(float *d_X, int n, int d, kernelstring_enum kernel, float *d_params) {

    dim3 blocksize(BLOCKSIZE2D,BLOCKSIZE2D);
    dim3 gridsize = divUp(dim3(n,n), blocksize);

    float *d_cov;
    checkCudaErrors(cudaMalloc((void**)&d_cov, n*n*sizeof(float)));

    constructCovMatrix_k<<<gridsize,blocksize>>>(d_X, n, d, kernel, d_params, d_cov);

    return d_cov;
}

/**
 * Kernel for computing the t by n matrix K(Xtest,X)
 */
__global__ void constructCrossCovMatrix_k(float *d_X, int n, float *d_Xtest, int t,
        kernelstring_enum kernel_string, float *d_kernel_params, int d, float *d_covfy)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y*blockDim.y;

    if (idx < t && idy < n) {
        kernelfunc kernfunc = getKernelFunction(kernel_string);

        float *vecx = &d_Xtest[idx*d];
        float *vecy = &d_X[idy*d];

        d_covfy[idx*n+idy] = kernfunc(vecx, vecy, d, d_kernel_params);
    }
}

/**
 * Constructs the t by n covariance matrix between two sets of points.
 */
float* constructCrossCovMatrix(float *d_X, int n, int d, float *d_Xtest, int t, kernelstring_enum kernel, float *d_params) {
    dim3 blocksize(BLOCKSIZE2D, BLOCKSIZE2D);
    dim3 gridsize = divUp(dim3(t, n), blocksize);

    float* d_covfy;
    checkCudaErrors(cudaMalloc((void**)&d_covfy, t*n*sizeof(float)));

    constructCrossCovMatrix_k<<<gridsize,blocksize>>>(d_X, n, d_Xtest, t, kernel, d_params, d, d_covfy);

    return d_covfy;
}

/**
 * Calculates the conditional mean of the test data points given the prior GP.
 * Calculates Kfy * Kyy^-1 * y
 */
float* conditionalMean(float *d_y, int n, float *d_cov, float *d_Xtest, int t, float *d_covfy, cusolverDnHandle_t cusolverhandle, cublasHandle_t cublashandle) {

    // create container for the intermediate value
    float *d_interm;
    checkCudaErrors(cudaMalloc((void**)&d_interm, n*sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_interm, d_y, n*sizeof(float), cudaMemcpyDeviceToDevice));

    // calculate z = Kyy^-1 * y
    int *d_devInfo;
    checkCudaErrors(cudaMalloc((void**)&d_devInfo, sizeof(int)));
    checkCusolverErrors(cusolverDnSpotrs(cusolverhandle, CHOLESKY_FILL_MODE, n, 1, d_cov, n, d_interm, n, d_devInfo));

    int h_devInfo = 0;  checkCudaErrors(cudaMemcpy(&h_devInfo, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_devInfo != 0) {
        fprintf(stderr, "ERROR: Failed to compute Kyf^-1*y with devInfo=%d\n", h_devInfo);
        exit(EXIT_FAILURE);
    }

    // calculate Kfy * z
    float *d_mean;
    checkCudaErrors(cudaMalloc((void**)&d_mean, t*sizeof(float)));

    float alpha = 1.0f; float beta = 0.0f;
    checkCublasErrors(cublasSgemv_v2(cublashandle, CUBLAS_OP_T, n, t, &alpha, d_covfy, n, d_interm, 1, &beta, d_mean, 1));

    cudaFree(d_interm);
    cudaFree(d_devInfo);

    return d_mean;
}

/**
 * Calculates the conditional covariance of the test data points given the prior GP.
 * Calculates Kff - Kfy * Kyy^-1 *Kyf
 */
float* conditionalCov(int n, int d, float *d_cov, float *d_Xtest, int t, float *d_covfy, kernelstring_enum kernel, float *d_params, cusolverDnHandle_t cusolverhandle, cublasHandle_t cublashandle) {

    // create container for the intermediate value
    float *d_interm;
    checkCudaErrors(cudaMalloc((void**)&d_interm, n*t*sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_interm, d_covfy, n*t*sizeof(float), cudaMemcpyDeviceToDevice));

    // calculate Z = Kyy^-1 * Kyf
    int *d_devInfo;
    checkCudaErrors(cudaMalloc((void**)&d_devInfo, sizeof(int)));
    checkCusolverErrors(cusolverDnSpotrs(cusolverhandle, CHOLESKY_FILL_MODE, n, t, d_cov, n, d_interm, n, d_devInfo));

    int h_devInfo = 0;  checkCudaErrors(cudaMemcpy(&h_devInfo, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_devInfo != 0) {
        fprintf(stderr, "ERROR: Failed to compute Kyf^-1*Kyf with devInfo=%d\n", h_devInfo);
        exit(EXIT_FAILURE);
    }

    // calculate Kff
    float* d_covff = constructCovMatrix(d_Xtest, t, d, kernel, d_params);

    // calculate Kff - Kfy * Z
    float alpha = -1.0f; float beta = 1.0f;
    checkCublasErrors(cublasSgemm_v2(cublashandle, CUBLAS_OP_T, CUBLAS_OP_N, t, t, n, &alpha, d_covfy, n, d_interm, n, &beta, d_covff, t));

    cudaFree(d_interm);
    cudaFree(d_devInfo);

    return d_covff;
}

/**
 * Performs Cholesky factorization on a covariance matrix of n by n.
 * The Cholesky factorization is stored in (overwrites) the lower
 * triangular half of the covariance matrix.
 */
void cholFactorizationL(float* d_cov, int n, cusolverDnHandle_t cusolverhandle) {

    // --- Compute Cholesky factorization
    int Lwork = 0; float* d_workspace;
    checkCusolverErrors(cusolverDnSpotrf_bufferSize(cusolverhandle, CHOLESKY_FILL_MODE, n, d_cov, n, &Lwork));
    checkCudaErrors(cudaMalloc((void**)&d_workspace, Lwork * sizeof(float)));

    int *d_devInfo; cudaMalloc(&d_devInfo, sizeof(int));
    checkCusolverErrors(cusolverDnSpotrf(cusolverhandle, CHOLESKY_FILL_MODE, n, d_cov, n, d_workspace, Lwork, d_devInfo));

    int h_devInfo = 0;  checkCudaErrors(cudaMemcpy(&h_devInfo, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_devInfo != 0) {
        fprintf(stderr, "ERROR: Failed to compute Cholesky decomposition of covariance matrix with devInfo=%d\n", h_devInfo);
        exit(EXIT_FAILURE);
    }
}

// this function can be better optimized...
__global__ void eye_k(unsigned int n, float* d_eye) {

    int dimx = blockDim.x;
    int dimy = blockDim.y;

    unsigned int idx = blockIdx.x * (MULTI*dimx) + threadIdx.x;
    unsigned int idy = blockIdx.y * (MULTI*dimy) + threadIdx.y;

    if (idx < n || idy < n)
        return;

    for (int i=0; i<MULTI; i++) {
        for (int j=0; j<MULTI; j++) {
            int x = min(idx+i*dimx, n-1);
            int y = min(idy+j*dimy, n-1);

            d_eye[x*n+y] = x==y;
        }
    }
}

float* eye(unsigned int n) {
    float* d_eye;
    checkCudaErrors(cudaMalloc((void**)&d_eye, n*n*sizeof(float)));

    int bsize = min(n, BLOCKSIZE2D);

    dim3 blocksize(bsize, bsize);
    dim3 gridsize = divUp(dim3(n,n), blocksize, MULTI);

    eye_k<<<gridsize,blocksize>>>(n, d_eye);

    return d_eye;
}
