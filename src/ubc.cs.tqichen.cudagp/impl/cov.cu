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

#define MULTI 5 // need to debug this before changing to >1

#define DEFAULT_FILL_MODE CUBLAS_FILL_MODE_UPPER

void printCov(float* d_cov, int n);

/**
 * Computes the full covariance matrix and stores it in d_cov.
 */
__global__ void constructCovMatrix_k(dataset_t d_ds, Kernel_t kernel_string, float* d_kernel_params, float* d_cov) {

    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y*blockDim.y;

    if (idx < d_ds.n && idy < d_ds.n) {
        kernelfunc kernfunc = getKernelFunction(kernel_string);

        float *vecx = &d_ds.X[idx*d_ds.d];
        float *vecy = &d_ds.X[idy*d_ds.d];

        d_cov[idx*d_ds.n+idy] = kernfunc(vecx,vecy,d_ds.d,d_kernel_params);
    }
}

float* constructCovMatrix(dataset_t d_ds, Kernel_t kernel_string, float* d_kernel_params) {

    dim3 blocksize(BLOCKSIZE2D,BLOCKSIZE2D);
    dim3 gridsize = divUp(dim3(d_ds.n,d_ds.n), blocksize);

    float* d_cov;
    checkCudaErrors(cudaMalloc((void**)&d_cov, d_ds.n*d_ds.n*sizeof(float)));

    constructCovMatrix_k<<<gridsize,blocksize>>>(d_ds, kernel_string, d_kernel_params, d_cov);

    return d_cov;
}

/**
 * Performs Cholesky factorization on a covariance matrix of n by n.
 * The Cholesky factorization is stored in (overwrites) the lower
 * triangular half of the covariance matrix.
 */
void cholFactorizationL(cusolverDnHandle_t cusolverhandle, float* d_cov, int n) {

    // --- Compute Cholesky factorization
    int Lwork = 0; float* d_workspace;
    checkCusolverErrors(cusolverDnSpotrf_bufferSize(cusolverhandle, DEFAULT_FILL_MODE, n, d_cov, n, &Lwork));
    checkCudaErrors(cudaMalloc((void**)&d_workspace, Lwork * sizeof(float)));

    int *d_devInfo; cudaMalloc(&d_devInfo, sizeof(int));
    checkCusolverErrors(cusolverDnSpotrf(cusolverhandle, DEFAULT_FILL_MODE, n, d_cov, n, d_workspace, Lwork, d_devInfo));

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

/**
 * Is this step really neccessary?
 */
float* invertCovMatrixAfterChol(cusolverDnHandle_t cusolverhandle, float* d_cov, int n) {
    float* d_eye = eye(n);

    int *d_devInfo; cudaMalloc(&d_devInfo, sizeof(int));
    checkCusolverErrors(cusolverDnSpotrs(cusolverhandle, DEFAULT_FILL_MODE, n, n, d_cov, n, d_eye, n, d_devInfo));

    int h_devInfo = 0;  checkCudaErrors(cudaMemcpy(&h_devInfo, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_devInfo != 0) {
        fprintf(stderr, "ERROR: Failed to compute inverse of covariance matrix with devInfo=%d\n", h_devInfo);
        exit(EXIT_FAILURE);
    }

    cudaFree(d_devInfo);

    return d_eye;
}

int main(int argc, char **argv) {

    float h_test[] = {
            2.8345,    1.8859,    2.0785,    1.9442,    1.9567,
            1.8859,    2.3340,    2.0461,    2.3164,    2.0875,
            2.0785,    2.0461,    2.8591,    2.4606,    1.9473,
            1.9442,    2.3164,    2.4606,    2.6848,    2.2768,
            1.9567,    2.0875,    1.9473,    2.2768,    2.5853};

    int size = 5;

    float* d_test;
    checkCudaErrors(cudaMalloc((void**)&d_test, size*size*sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_test, h_test, size*size*sizeof(float), cudaMemcpyHostToDevice));

    printDeviceSqrMatrix(d_test,size);

    cublasHandle_t cublashandle;
    cublasCreate_v2(&cublashandle);

    cusolverDnHandle_t cusolverhandle;
    cusolverDnCreate(&cusolverhandle);

    cholFactorizationL(cusolverhandle, d_test, size);

    printf("Chol factorization:\n");
    printDeviceSqrMatrix(d_test, size);

    float* d_inv = invertCovMatrixAfterChol(cusolverhandle, d_test, size);

    printf("Inverse?\n");
    printDeviceSqrMatrix(d_inv,size);

    cudaFree(d_inv);
    cudaFree(d_test);
    cudaDeviceReset();
}

