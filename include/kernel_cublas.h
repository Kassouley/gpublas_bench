#ifndef KERNEL_CUBLAS_H
#define KERNEL_CUBLAS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

void kernel_cublasDgemm (cublasHandle_t handle, unsigned int m, unsigned int k, const double* a, const double* b, double* c);
void kernel_cublasSgemm (cublasHandle_t handle, unsigned int m, unsigned int k, const float* a, const float* b, float* c);
         
#ifdef SP
    #define kernel_gpublasXgemm(handle, m, k, a, b, c)\
    { \
        kernel_cublasSgemm(handle, m, k, a, b, c); \
    }
#endif
#ifdef DP
    #define kernel_gpublasXgemm(handle, m, k, a, b, c)\
    { \
        kernel_cublasDgemm(handle, m, k, a, b, c); \
    }
#endif 

#endif
