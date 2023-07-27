#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "driver.h"
extern "C" {
#include "tab.h"
#include "print_measure.h"
#include "time_measure.h"
}

#define NB_META 31

HANDLE_T handle;

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
    PRECISION_T *a = (PRECISION_T*)malloc(m * k * sizeof(a));
    PRECISION_T *b = (PRECISION_T*)malloc(k * m * sizeof(b));
    RANDOM_ARRAY_2D(m, k, a);
    RANDOM_ARRAY_2D(k, m, b);
    
    HANDLE_CREATE(handle);

    for (unsigned int n = 0; n < NB_META; n++)
    {
        PRECISION_T *c = (PRECISION_T*)malloc(m * m * sizeof(c));
        if ( n == 0 )
        {
            for (unsigned int i = 0; i < nwu; i++)
            {
                KERNEL(handle, m, k, a, b, c);
            }
        }
        else
        {
            KERNEL(handle, m, k, a, b, c);
        }

        const uint64_t t1 = measure_clock();
        for (unsigned int i = 0; i < nrep; i++)
        {
            KERNEL(handle, m, k, a, b, c);
        }
        const uint64_t t2 = measure_clock();

        tdiff[n] = t2 - t1;
        free(c);
    }
    
    HANDLE_DESTROY(handle);
    free(a);
    free(b);

    print_measure(m, k, nrep, tdiff);
    
    return EXIT_SUCCESS;
}