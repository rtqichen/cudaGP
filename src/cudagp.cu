/*
 * Wraps the calls to kernels here.
 */

#include "cudagp.h"
#include <stdio.h>
#include "impl/kernels.h"

void randomFloats(float* h_matrix, int size) {
    for (int i=0; i<size; i++) {
        h_matrix[i] = rand() / (float)RAND_MAX * 100;
    }
}

void printMatrix(float* matrix, int numRows, int numCols) {
    printf("Printing %d by %d matrix:\n", numRows, numCols);
    for (int i=0; i<numRows; i++) {
        for (int j=0; j<numCols; j++) {
            printf("%.4f ", matrix[i*numCols+j]);
        }
        printf("\n");
    }
}

void printDiagOfMatrix(float* matrix, int numRows, int numCols) {
    printf("Printing diagonals of %d by %d matrix:\n", numRows, numCols);
    for (int i=0; i<min(numRows,numCols); i++) {
        printf("%.4f\n", matrix[i*numCols+i]);
    }
}

int readData(float* X, float* y, int n) {

    FILE *infile = fopen("testdata/grayroos.dat", "r");
    if (!infile) {
        printf("Failed to read file.");
    }

    int i=0;
    char line[100];
    int a,b;
    while(i<n && fgets(line, sizeof(line), infile) != NULL) {
        sscanf(line, "%d\t%d[^\n]", &a, &b);
        X[i] = (float) a;
        y[i] = (float) b;
        i++;
    }

    return n;
}

float* constructCovMatrix_ref(float* X, int n, int d, kernelstring_enum kernel_string, float* h_kernel_params) {

    float* h_cov = (float*)malloc(n*n*sizeof(float));

    kernelfunc kernfunc = getKernelFunction(kernel_string);

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            h_cov[i*n+j] = kernfunc(&X[i*d], &X[j*d], d, h_kernel_params);
        }
    }

    return h_cov;
}

float* linspace(int min, int max, int len) {
    float *x = (float*) malloc(len*sizeof(float));
    for (int i=0; i<len; i++) {
        x[i] = min + (max-min)*(i/(float)(len-1));
    }
    return x;
}

int main(int argc, const char** argv) {

    srand(0);

    int n = 42;
    int d = 1;
    float* X = (float*)malloc(n*d*sizeof(float));
    float* y = (float*)malloc(n*sizeof(float));
    readData(X, y, n);

    int t = 201;
    float* Xtest = linspace(500, 860, t);

    float params[1] = {10.0f};

    // test the full GP
    cudagphandle_t cudagphandle = initializeCudaGP(X,y,n,d, cudagpSquaredExponentialKernel, params);
    prediction_t pred = predict(cudagphandle, Xtest, t);

    printf("Full GP Done!\n");

    // test the clustered GP
    cudagphandle_t cudagphandle2 = initializeCudaDGP(X,y,n,d, cudagpSquaredExponentialKernel, 2, params);
    prediction_t pred2 = predict(cudagphandle2, Xtest, t);

    printf("Clustered GP Done!\n");

    printMatrix(pred2.mean, t, 1);

    free(X);
    free(y);
    free(Xtest);

    freeCudaGP(cudagphandle);
    freeCudaGP(cudagphandle2);

    free(pred.mean);
    free(pred.var);
    free(pred2.mean);
    free(pred2.var);

    printf("Done!\n");

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
