#ifndef KERNEL_CUBLAS_H
#define KERNEL_CUBLAS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

void kernel_cublasDmm (unsigned int m, unsigned int k, const double* a, const double* b, double* c);
void kernel_cublasSmm (unsigned int m, unsigned int k, const float* a, const float* b, float* c);

#define CHECK(cmd) \
{\
    cudaError_t error  = cmd;\
    if (error != cudaSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", cudaGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}

#define HANDLE_T cublasHandle_t
#define HANDLE_CREATE(handle) \
{\
    cublasCreate(&handle);\
}
#define HANDLE_DESTROY(handle) \
{\
    cublasDestroy(handle);\
}
#ifdef SP
    #define KERNEL(m, k, a, b, c)\
    { \
        kernel_cublasSmm(m, k, a, b, c); \
    }
#endif
#ifdef DP
    #define KERNEL(m, k, a, b, c)\
    { \
        kernel_cublasDmm(m, k, a, b, c); \
    }
#endif


#endif