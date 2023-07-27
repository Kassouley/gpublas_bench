#ifndef KERNEL_ROCBLAS_H
#define KERNEL_ROCBLAS_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

void kernel_rocblasDmm (rocblas_handle handle, unsigned int m, unsigned int k, const double* a, const double* b, double* c);
void kernel_rocblasSmm (rocblas_handle handle, unsigned int m, unsigned int k, const float* a, const float* b, float* c);

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}

#define HANDLE_T rocblas_handle
#define HANDLE_CREATE(handle) \
{\
    rocblas_create_handle(&handle);\
}
#define HANDLE_DESTROY(handle) \
{\
    rocblas_destroy_handle(handle);\
}
#ifdef SP
    #define KERNEL(handle, m, k, a, b, c)\
    { \
        kernel_rocblasSmm(handle, m, k, a, b, c); \
    }
#endif
#ifdef DP
    #define KERNEL(handle, m, k, a, b, c)\
    { \
        kernel_rocblasDmm(handle, m, k, a, b, c); \
    }
#endif

#endif