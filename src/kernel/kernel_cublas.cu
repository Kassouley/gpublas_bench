#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernel_cublas.h"

void kernel_cublasDgemm (cublasHandle_t handle, unsigned int m, unsigned int k, const double* a, const double* b, double* c)
{
    int size_ab = m * k * sizeof(double);
    int size_c  = m * m * sizeof(double);
    
    double* d_a;
    double* d_b;
    double* d_c;

	CHECK(cudaMalloc(&d_a, size_ab));
    CHECK(cudaMalloc(&d_b, size_ab));
    CHECK(cudaMalloc(&d_c, size_c));

    CHECK(cudaMemcpy(d_a, a, size_ab, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, size_ab, cudaMemcpyHostToDevice));

    double alpha = 1.0f;
    double beta = 0.0f;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, k, &alpha, d_a, m, d_b, k, &beta, d_c, m);
        
	CHECK(cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
}

void kernel_cublasSgemm (cublasHandle_t handle, unsigned int m, unsigned int k, const float* a, const float* b, float* c)
{
    int size_ab = m * k * sizeof(float);
    int size_c  = m * m * sizeof(float);
    
    float* d_a;
    float* d_b;
    float* d_c;

	CHECK(cudaMalloc(&d_a, size_ab));
    CHECK(cudaMalloc(&d_b, size_ab));
    CHECK(cudaMalloc(&d_c, size_c));

    CHECK(cudaMemcpy(d_a, a, size_ab, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, size_ab, cudaMemcpyHostToDevice));

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, k, &alpha, d_a, m, d_b, k, &beta, d_c, m);
        
	CHECK(cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
}