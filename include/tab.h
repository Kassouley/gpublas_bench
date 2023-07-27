#ifndef TAB_H
#define TAB_H

void random_Darray_2D(unsigned int row, unsigned int col, double* array);
void random_Sarray_2D(unsigned int row, unsigned int col, float* array);

#ifdef SP
    typedef float precision_t;
    #define random_Xarray_2D(m, k, array) \
    {\
        random_Sarray_2D(m, k, array); \
    }
#endif
#ifdef DP
    typedef double precision_t;
    #define random_Xarray_2D(m, k, array) \
    {\
        random_Darray_2D(m, k, array); \
    }
#endif


#endif