#ifndef DRIVER_H
    #define DRIVER_H

    #ifdef SP
        #define PRECISION_T float
        #define RANDOM_ARRAY_2D(m, k, array) \
        {\
            random_Sarray_2D(m, k, array); \
        }
    #endif
    #ifdef DP
        #define PRECISION_T double
        #define RANDOM_ARRAY_2D(m, k, array) \
        {\
            random_Darray_2D(m, k, array); \
        }
    #endif
    #ifdef ROCBLAS
        #include <hip/hip_runtime.h>
        #include <rocblas/rocblas.h>
        #include "kernel_rocblas.h"
    #endif
    #ifdef CUBLAS
        #include <cuda_runtime.h>
        #include <cublas_v2.h>
        #include "kernel_cublas.h"
    #endif
    #ifdef CBLAS
        #include <cblas.h>
        #include "kernel_cblas.h"
#endif
#endif