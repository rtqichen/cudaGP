/*
 * Wraps the calls to kernels here.
 */

#include "cudagp.h"
#include <stdio.h>
#include <math.h>
#include "impl/kernels.h"
#include <time.h>
#include "impl/utils_cuda.h"

void randomFloats(float* h_matrix, int size) {
    for (int i=0; i<size; i++) {
        h_matrix[i] = rand() / (float)RAND_MAX * 100;
    }
}

void printMatrix(float* matrix, int numRows, int numCols) {
    printf("Printing %d by %d matrix:\n", numRows, numCols);
    for (int i=0; i<numRows; i++) {
        for (int j=0; j<numCols; j++) {
            printf("%.8f ", matrix[i*numCols+j]);
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

    FILE *infile = fopen("test_data", "r");
    if (!infile) {
        printf("Failed to read file.");
    }

    int i=0;
    char line[100];
    float a,b;
    while(i<n && fgets(line, sizeof(line), infile) != NULL) {
        sscanf(line, "%f\t%f[^\n]", &a, &b);
        X[i] = a;
        y[i] = b;
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

/**
 * Generates random numbers uniformly in [min, max]
 */
float* uniform(int len, float max, float min) {
    float *x = (float*) malloc(len*sizeof(float));
    for (int i=0; i<len; i++) {
        x[i] = ((float)rand()/(float)RAND_MAX) * (max-min+1) + min;
    }
    return x;
}

float* func(float *x, int n) {
    float *y = (float*) malloc(n*sizeof(float));
    for (int i=0; i<n; i++) {
        y[i] = sin(x[i]) + ((float)rand()/(float)RAND_MAX) * 0.4;
    }
    return y;
}

double getMean(double *x, int size) {
    double sum = 0.0;
    for (int i=0; i<size; i++) {
        sum += x[i];
    }
    return sum/size;
}

double getStdev(double *x, int size) {
    double var = 0.0;
    double mean = getMean(x, size);
    for (int i=0; i<size; i++) {
        var += (x[i]-mean)*(x[i]-mean);
    }
    return sqrt(var/(size-1));
}

void warmup() {
    int n = 30;
    int d = 1;

    float *X = uniform(n, -400, 400);
    float *y = func(X, n);

    int t = 201;
    float* Xtest = linspace(-400, 400, t);

    float params[2] = {1.8, 1.15};

    cudagphandle_t cudagphandle = initializeCudaGP(X,y,n,d, cudagpSquaredExponentialKernel, params);
    prediction_t pred = predict(cudagphandle, Xtest, t);

    free(Xtest);
    free(X);
    free(y);
    freeCudaGP(cudagphandle);
    free(pred.mean);
    free(pred.var);
}

#define NTESTS 8

int main(int argc, const char** argv) {

    srand(0);

    int n = 7000;
    int d = 1;
    //float* X = (float*)malloc(n*d*sizeof(float));
    //float* y = (float*)malloc(n*sizeof(float));
    //readData(X, y, n);

    float *X = uniform(n, -400, 400);
    float *y = func(X, n);

    warmup();

    printf("Using N=%d training data.\n", n);

//    printf("Data:\n");
//    printMatrix(X, n, 1);
//    printMatrix(y, n, 1);

    int t = 201;
    float* Xtest = linspace(-400, 400, t);

    float params[2] = {1.8, 1.15};

    int ntrials = 20;

    // time the full GP
//    do {
//
//        printf("Timing the Full GP implementation . . .\n") ;
//        clock_t tic = clock();
//        for (int i=0; i<ntrials; i++) {
//            cudagphandle_t cudagphandle = initializeCudaGP(X,y,n,d, cudagpSquaredExponentialKernel, params);
//            prediction_t pred = predict(cudagphandle, Xtest, t);
//        }
//        clock_t toc = clock();
//        printf("Full GP Prediction - Elapsed time: %f seconds\n\n", (double)(toc - tic) / CLOCKS_PER_SEC / ntrials);
//
//    } while (false);

    // time the clustered GP with k clusters
    do {
        printf("Timing the data-parallel GP implementation . . .\n");
        int numClusters[NTESTS] = {1,10,20,50,100,200,500,1000};
        double times[ntrials];
        for (int k=0; k<NTESTS; k++) {

            clock_t tic;
            clock_t toc;
            for (int i=0; i<ntrials; i++) {
                tic = clock();
                cudagphandle_t cudagphandle2 = initializeCudaDGP(X,y,n,d, cudagpSquaredExponentialKernel, numClusters[k], params);
                prediction_t pred2 = predict(cudagphandle2, Xtest, t);
                toc = clock();
                times[i] = (double)(toc - tic) / CLOCKS_PER_SEC;

                freeCudaGP(cudagphandle2);
                free(pred2.mean);
                free(pred2.var);
            }
            double mean = getMean(times,ntrials);
            double stdev = getStdev(times,ntrials);
            printf("K=%d Sparse GP Prediction - Elapsed time: %f seconds with std %f seconds\n", numClusters[k], mean, stdev);
        }
    } while (false);

//    printf("Xtest:\n"); printMatrix(Xtest, t, 1);
//    printf("Mean:\n"); printMatrix(pred2.mean, t, 1);
//    printf("Marginal Variance:\n"); printMatrix(pred2.var, t, 1);

//    free(X);
//    free(y);
//    free(Xtest);
//
//    freeCudaGP(cudagphandle);
//    freeCudaGP(cudagphandle2);
//
//    free(pred.mean);
//    free(pred.var);
//    free(pred2.mean);
//    free(pred2.var);
//
//    printf("Done!\n");
//
//    cudaDeviceReset();
    return EXIT_SUCCESS;
}
