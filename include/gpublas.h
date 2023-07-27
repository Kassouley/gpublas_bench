#ifndef GPUBLAS_H
#define GPUBLAS_H

#ifdef ROCBLAS
    #include <hip/hip_runtime.h>
    #include <rocblas/rocblas.h>
    #include "kernel_rocblas.h"

    typedef rocblas_handle gpublas_handle_t;

    #define gpublas_handle_create(handle) \
    {\
        rocblas_create_handle(&handle);\
    }
    #define gpublas_handle_destroy(handle) \
    {\
        rocblas_destroy_handle(handle);\
    }

#endif
#ifdef CUBLAS
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include "kernel_cublas.h"

    typedef cublasHandle_t gpublas_handle_t;
  
    #define gpublas_handle_create(handle) \
    {\
        cublasCreate(&handle);\
    }
    #define gpublas_handle_destroy(handle) \
    {\
        cublasDestroy(handle);\
    }

#endif
#endif