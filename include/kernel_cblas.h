#ifndef KERNEL_CBLAS_H
#define KERNEL_CBLAS_H

#include <cblas.h>
#ifdef SP
    #define kernel_cblasXgemm(m, k, a, b, c)\
    { \
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, m, k, 1.0, a, k, b, m, 0.0, c, m); \
    }
#endif
#ifdef DP
    #define kernel_cblasXgemm(m, k, a, b, c)\
    { \
        cblas_dgemm(CblasRowMajor, CblasNoTrans , CblasNoTrans, m, m, k, 1.0, a, k, b, m, 0.0, c, m); \
    }
#endif  

#endif