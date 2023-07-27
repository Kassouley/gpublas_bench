#ifndef KERNEL_CUBLAS_H
#define KERNEL_CUBLAS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK(cmd) \
{\
    cudaError_t error  = cmd;\
    if (error != cudaSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", cudaGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

void kernel_cublasDgemm (cublasHandle_t handle, unsigned int m, unsigned int k, const double* a, const double* b, double* c);
void kernel_cublasSgemm (cublasHandle_t handle, unsigned int m, unsigned int k, const float* a, const float* b, float* c);
         
#ifdef SP
    #define kernel_gpublasXgemm(m, k, a, b, c)\
    { \
        kernel_cublasSgemm(m, k, a, b, c); \
    }
#endif
#ifdef DP
    #define kernel_gpublasXgemm(m, k, a, b, c)\
    { \
        kernel_cublasDgemm(m, k, a, b, c); \
    }
#endif 

#endif