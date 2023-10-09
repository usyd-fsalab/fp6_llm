#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Performance Benchmark
#define WARM_UP_ITERATION 100
#define BENCHMARK_ITERATION 10000

void __forceinline__ CheckMallocCPU(void* PTR, int line = -1) {
    if (PTR == NULL) {
        printf("Error in CPU Malloc, line %d!\n", line);
        exit(-1);
    }
}

void __forceinline__ CheckMallocCUDA(void* PTR, int line = -1) {
    if (PTR == NULL) {
        printf("Error in cudaMalloc, line %d!\n", line);
        exit(-1);
    }
}

void checkCublasError(cublasStatus_t status, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Cublas Error at line %d, Error Code: %d\n", line, status);
        exit(EXIT_FAILURE);
    }
}

void checkLastCudaError(int line)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Last Cuda Error Detected at line: %d, Error: %s.\n", line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Note: totalAbsSum might overflow if (1)the shape of output matrix are large and (2)the value of each element within the output matrix is large.
// The overflow might result in NaN for the "TotalAbsError/TotalAbsSum", while the fp6_llm is working correctly.
// This problem can be fixed by setting the quantizaiton scales to a smaller value.
double ComputeTotalError(half* CuBlas, half* Other, size_t m, size_t n)
{
    long double totalError = 0.0;
    for (size_t i = 0; i < m * n; i++)
        totalError += fabs(__half2float(CuBlas[i]) - __half2float(Other[i]));
    long double totalAbsSum = 0.0;
    for (size_t i = 0; i < m * n; i++)
        totalAbsSum += fabs(__half2float(CuBlas[i]));    
    return totalError/totalAbsSum;
}

void PrintPerformance(const char* KernelName, float milliseconds, float tflops, double error)
{
    printf("%-10s \t -> \t\t Time/ms: %5.3f \t Performance/TFLOPs: %4.2f \t TotalAbsError/TotalAbsSum: %.8lf\n",
           KernelName,
           milliseconds,
           tflops,
           error);
}


void PrintMismatch(const char* KernelName,
                   size_t      MaxNumMismatch,
                   float       RelativeErrorThreshold,
                   half*       CuBlas,
                   half*       Other,
                   size_t         M_GLOBAL,
                   size_t         N_GLOBAL)
{
    //printf("First %d Mismatches between Cublas and %s:\n", MaxNumMismatch, KernelName);
    size_t count = 0;
    for (size_t i = 0; i < M_GLOBAL; i++) {
        for (size_t j = 0; j < N_GLOBAL; j++) {
            if (fabs(__half2float(CuBlas[i + j * M_GLOBAL]) - __half2float(Other[i + j * M_GLOBAL]))/fabs(__half2float(CuBlas[i + j * M_GLOBAL])) > RelativeErrorThreshold) {
                count++;
                printf("(%d,%d) CuBlas=%f %s=%f\n",
                       i,
                       j,
                       __half2float(CuBlas[i + j * M_GLOBAL]),
                       KernelName,
                       __half2float(Other[i + j * M_GLOBAL]));
            }
            if (count == MaxNumMismatch)
                break;
        }
        if (count == MaxNumMismatch)
            break;
    }
}

