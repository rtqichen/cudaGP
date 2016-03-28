#ifndef UTILS_CUDA_H_
#define UTILS_CUDA_H_

#include <iostream>

/**************************/
/*  CUDA ERROR CHECKING   */
/**************************/
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
#define checkCusolverErrors(err) __checkCusolverErrors(err, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**************************/
/* CUSOLVE ERROR CHECKING */
/**************************/
static const char *_cusolverGetErrorEnum(cusolverStatus_t error)
{
    switch (error)
    {
    case CUSOLVER_STATUS_SUCCESS:
        return "CUSOLVER_SUCCESS";

    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";

    case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";

    case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";

    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";

    case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";

    case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";

    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

    }

    return "<unknown>";
}

static void __checkCusolverErrors(cusolverStatus_t err, const char *file, const int line)
{
    if (err != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSolver error at: " << file << ":" << line << std::endl;
        std::cerr << _cusolverGetErrorEnum(err) << " " << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**************************/
/*   Returns  ceil(a/b)   */
/**************************/
static int divUp(int a, int b) {
    return (a+(b-1))/b;
}

static dim3 divUp(dim3 a, dim3 b) {
    int x = divUp(a.x,b.x);
    int y = divUp(a.y,b.y);
    int z = divUp(a.z,b.z);
    return dim3(x,y,z);
}

static dim3 divUp(dim3 a, dim3 b, int multi) {
    int x = divUp(a.x,b.x*multi);
    int y = divUp(a.y,b.y*multi);
    int z = divUp(a.z,b.z*multi);
    return dim3(x,y,z);
}

/**************************/
/*        magic(n)        */
/**************************/

static float* magic(int n) {
    float* h_magic = (float*)malloc(n*n*sizeof(float));
    for (int i=0; i<n*n; i++) {
        h_magic[i] = i;
    }

    // Fisher-Yates shuffle
    int j;
    float tmp;
    for (int i=0; i<n*n; i++) {
        j = rand() % ((n*n)-i);
        tmp = h_magic[i];
        h_magic[i] = h_magic[i+j];
        h_magic[i+j] = tmp;
    }

    float* d_magic;
    checkCudaErrors(cudaMalloc((void**)&d_magic, n*n*sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_magic, h_magic, n*n*sizeof(float), cudaMemcpyHostToDevice));

    free(h_magic);

    return d_magic;
}


static void printDeviceSqrMatrix(float* d_cov, int n) {

    float* h_cov = (float*)malloc(n*n*sizeof(float));
    checkCudaErrors((cudaMemcpy(h_cov, d_cov, n*n*sizeof(float), cudaMemcpyDeviceToHost)));

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            printf("%8.2f ", h_cov[i*n+j]);
        }
        printf("\n");
    }

    free(h_cov);
}

#endif
