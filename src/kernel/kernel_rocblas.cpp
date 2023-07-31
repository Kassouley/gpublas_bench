#include <stdio.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "gpublas.h"
#include "kernel_rocblas.h"

#ifdef ROCBLAS_WO_DT
void kernel_rocblasDgemm (rocblas_handle handle, unsigned int m, unsigned int k, const double* a, const double* b, double* c)
{
    const double alpha = 1.0f; 
    const double beta = 0.0f;

    // C[mxm] = A[mxk] * B[kxm]
    rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                m, m, k, &alpha, b, m, a, k, &beta, c, m);
}

void kernel_rocblasSgemm (rocblas_handle handle, unsigned int m, unsigned int k, const float* a, const float* b, float* c)
{
    const float alpha = 1.0f; 
    const float beta = 0.0f;

    // C[mxm] = A[mxk] * B[kxm]
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                m, m, k, &alpha, a, k, b, m, &beta, c, m);
}
#endif
#ifdef ROCBLAS
void kernel_rocblasDgemm (rocblas_handle handle, unsigned int m, unsigned int k, const double* a, const double* b, double* c)
{
    int size_ab = m * k * sizeof(double);
    int size_c  = m * m * sizeof(double);
    
    double* d_a;
    double* d_b;
    double* d_c;

    CHECK(hipMalloc((void**)&d_a, size_ab));
    CHECK(hipMalloc((void**)&d_b, size_ab));
    CHECK(hipMalloc((void**)&d_c, size_c));

    CHECK(hipMemcpy(d_a, a, size_ab, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_b, b, size_ab, hipMemcpyHostToDevice));

    const double alpha = 1.0f; 
    const double beta = 0.0f;

    // C[mxm] = A[mxk] * B[kxm]
    rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                m, m, k, &alpha, d_b, m, d_a, k, &beta, d_c, m);

    CHECK(hipMemcpy(c, d_c, size_c, hipMemcpyDeviceToHost));

    CHECK(hipFree(d_a));
    CHECK(hipFree(d_b));
    CHECK(hipFree(d_c));
}

void kernel_rocblasSgemm (rocblas_handle handle, unsigned int m, unsigned int k, const float* a, const float* b, float* c)
{
    int size_ab = m * k * sizeof(float);
    int size_c  = m * m * sizeof(float);
    
    float* d_a;
    float* d_b;
    float* d_c;

    CHECK(hipMalloc((void**)&d_a, size_ab));
    CHECK(hipMalloc((void**)&d_b, size_ab));
    CHECK(hipMalloc((void**)&d_c, size_c));

    CHECK(hipMemcpy(d_a, a, size_ab, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_b, b, size_ab, hipMemcpyHostToDevice));

    const float alpha = 1.0f; 
    const float beta = 0.0f;

    // C[mxm] = A[mxk] * B[kxm]
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                m, m, k, &alpha, d_a, k, d_b, m, &beta, d_c, m);

    CHECK(hipMemcpy(c, d_c, size_c, hipMemcpyDeviceToHost));

    CHECK(hipFree(d_a));
    CHECK(hipFree(d_b));
    CHECK(hipFree(d_c));
}
#endif