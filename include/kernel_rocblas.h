#ifndef KERNEL_ROCBLAS_H
#define KERNEL_ROCBLAS_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>


void kernel_rocblasDgemm (rocblas_handle handle, unsigned int m, unsigned int k, const double* a, const double* b, double* c);
void kernel_rocblasSgemm (rocblas_handle handle, unsigned int m, unsigned int k, const float* a, const float* b, float* c);

#ifdef SP
    #define kernel_gpublasXgemm(handle, m, k, a, b, c)\
    { \
        kernel_rocblasSgemm(handle, m, k, a, b, c); \
    }
#endif
#ifdef DP
    #define kernel_gpublasXgemm(handle, m, k, a, b, c)\
    { \
        kernel_rocblasDgemm(handle, m, k, a, b, c); \
    }
#endif

#endif