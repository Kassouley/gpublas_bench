#ifndef GPUBLAS_H
#define GPUBLAS_H

#if defined(ROCBLAS) || defined(ROCBLAS_WO_DT)
    #include <hip/hip_runtime.h>
    #include <rocblas/rocblas.h>
    #include "kernel_rocblas.h"

    #define CHECK(cmd) \
    {\
        hipError_t error  = cmd;\
        if (error != hipSuccess) { \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
            exit(EXIT_FAILURE);\
        }\
    }

    typedef rocblas_handle gpublas_handle_t;

    #define gpublas_handle_create(handle) \
    {\
        rocblas_create_handle(&handle);\
    }

    #define gpublas_handle_destroy(handle) \
    {\
        rocblas_destroy_handle(handle);\
    }

    #define gpublas_malloc(ptr, size) \
    {\
        CHECK(hipMalloc((void**)&ptr, size));\
    }

    #define gpublas_free(ptr) \
    {\
        CHECK(hipFree(ptr));\
    }

    #define gpublas_memcpy_HtD(dst, src, size) \
    {\
        CHECK(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));\
    }

    #define gpublas_memcpy_DtH(dst, src, size) \
    {\
        CHECK(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost));\
    }

    #define gpublas_deviceSynchronize() \
    {\
        hipDeviceSynchronize();\
    }\

#endif
#if defined(CUBLAS) || defined(CUBLAS_WO_DT)
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include "kernel_cublas.h"

    #define CHECK(cmd) \
    {\
        cudaError_t error  = cmd;\
        if (error != cudaSuccess) { \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", cudaGetErrorString(error), error,__FILE__, __LINE__); \
            exit(EXIT_FAILURE);\
        }\
    }

    typedef cublasHandle_t gpublas_handle_t;
  
    #define gpublas_handle_create(handle) \
    {\
        cublasCreate(&handle);\
    }

    #define gpublas_handle_destroy(handle) \
    {\
        cublasDestroy(handle);\
    }

    #define gpublas_malloc(ptr, size) \
    {\
        CHECK(cudaMalloc(&ptr, size));\
    }

    #define gpublas_free(ptr) \
    {\
        CHECK(cudaFree(ptr));\
    }

    #define gpublas_memcpy_HtD(dst, src, size) \
    {\
        CHECK(cudaMemcpy(dst, src, size,cudaMemcpyHostToDevice));\
    }

    #define gpublas_memcpy_DtH(dst, src, size) \
    {\
        CHECK(cudaMemcpy(dst, src, size,cudaMemcpyDeviceToHost));\
    }

    #define gpublas_deviceSynchronize() \
    {\
        cudaDeviceSynchronize();\
    }\


#endif
#endif