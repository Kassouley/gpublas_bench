#ifndef KERNEL_ROCBLAS_H
#define KERNEL_ROCBLAS_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

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