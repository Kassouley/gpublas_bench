#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cblas.h>
#include "kernel_cblas.h"
#include "tab.h"
#include "print_measure.h"

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

    uint64_t tdiff[NB_META];
    srand(0);
    
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
                kernel_cblasXgemm(m, k, a, b, c);
            }
        }
        else
        {
            kernel_cblasXgemm(m, k, a, b, c);
        }

        const uint64_t t1 = measure_clock();
        for (unsigned int i = 0; i < nrep; i++)
        {
            kernel_cblasXgemm(m, k, a, b, c);
        }
        const uint64_t t2 = measure_clock();

        tdiff[n] = t2 - t1;
        free(c);
    }
    
    free(a);
    free(b);

    print_measure(m, k, nrep, tdiff);
    
    return EXIT_SUCCESS;
}