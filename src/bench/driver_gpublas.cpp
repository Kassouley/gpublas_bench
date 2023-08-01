#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include "gpublas.h"
extern "C" {
#include "tab.h"
#include "print_measure.h"
}

#define NB_META 31


int main(int argc, char **argv)
{
    unsigned int m, k, nwu, nrep;
    if (argc != 5) 
    {
        fprintf (stderr, "Usage: %s <m> <k> <nb warmup> <nb measure>\n", argv[0]);
        return 1;
    }
    else
    {
        m = atoi(argv[1]);
        k = atoi(argv[2]);
        nwu = atoi(argv[3]);
        nrep = atoi(argv[4]);
    }

    double tdiff[NB_META];
    srand(0);
    
    gpublas_handle_t handle;
    gpublas_handle_create(handle);
    
    int size_ab = m * k * sizeof(precision_t);
    int size_c  = m * m * sizeof(precision_t);

    precision_t *a = (precision_t*)malloc(size_ab);
    precision_t *b = (precision_t*)malloc(size_ab);

    random_Xarray_2D(m, k, a);
    random_Xarray_2D(k, m, b);

    for (unsigned int n = 0; n < NB_META; n++)
    {
        precision_t *c = (precision_t*)malloc(size_c);
        if ( n == 0 )
        {
            for (unsigned int i = 0; i < nwu; i++)
            {
                kernel_gpublasXgemm(handle, m, k, a, b, c);
            }
        }
        else
        {
            kernel_gpublasXgemm(handle, m, k, a, b, c);
        }

        const double t1 = omp_get_wtime();
        for (unsigned int i = 0; i < nrep; i++)
        {
            kernel_gpublasXgemm(handle, m, k, a, b, c);
        }
        const double t2 = omp_get_wtime();
        
        tdiff[n] = t2 - t1;
        free(c);
    }
    
    gpublas_handle_destroy(handle);
    free(a);
    free(b);

    print_measure(m, k, nrep, tdiff);
    
    return EXIT_SUCCESS;
}