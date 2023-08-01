#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include "print_measure.h"
#ifndef NB_META
#define NB_META 31
#endif

static int cmp_uint64 (const void *a, const void *b)
{
    const uint64_t va = *((uint64_t *) a);
    const uint64_t vb = *((uint64_t *) b);

    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

void print_measure(unsigned int m, unsigned int k, unsigned int nrep, double tdiff[NB_META])
{
    FILE * output = NULL;

    qsort (tdiff, NB_META, sizeof tdiff[0], cmp_uint64);

    const double nb_gflops = (double)((m*m)*(2*k-1)) * 1e-9;
    const double time_min  = (double)tdiff[0]/(double)nrep;
    const double time_med  = (double)tdiff[NB_META/2]/(double)nrep;
    const float stabilite  = (tdiff[NB_META/2] - tdiff[0]) * 100.0f / tdiff[0];

    double rate = 0.0, drate = 0.0;
    for (unsigned int i = 0; i < NB_META; i++)
    {
        rate += nb_gflops / (tdiff[i]/nrep);
        drate += (nb_gflops * nb_gflops) / ((tdiff[i]/nrep) * (tdiff[i]/nrep));
    }
    rate /= (double)(NB_META);
    drate = sqrt(drate / (double)(NB_META) - (rate * rate));
  
    printf("-----------------------------------------------------\n");

    printf("Minimum (time, ms): %13s %10.3f ms\n", "", time_min * 1e3);
    printf("Median (time, ms):  %13s %10.3f ms\n", "", time_med * 1e3);
    
    if (stabilite >= 10)
        printf("Bad Stability: %18s %10.2f %%\n", "", stabilite);
    else if ( stabilite >= 5 )
        printf("Average Stability: %14s %10.2f %%\n", "", stabilite);
    else
        printf("Good Stability: %17s %10.2f %%\n", "", stabilite);

    printf("\033[1m%s %4s \033[42m%10.2lf +- %.2lf GFLOP/s\033[0m\n",
        "Average performance:", "", rate, drate);
    printf("-----------------------------------------------------\n");
    

    output = fopen("./output/measure_tmp.out", "a");
    if (output != NULL) 
    {
        
        fprintf(output, " %5d | %5d | %15f | %15f | %15f | %12f\n", 
                m, k,
                rate, 
                time_min*1e3, 
                time_med*1e3,
                stabilite);
        fclose(output);
    }
    else
    {
        char cwd[1028];
        if (getcwd(cwd, sizeof(cwd)) != NULL) 
        {
            printf("Couldn't open '%s/output/measure_tmp.out' file\n", cwd);
        }
    }
}