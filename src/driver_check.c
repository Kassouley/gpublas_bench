#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include "driver.h"
#include "tab.h"

#define OUTOUT_FILE "output_check.txt"

int main(int argc, char **argv)
{
    unsigned int m, k;
    char* file_name = NULL;
    FILE * output = NULL;

    if (argc != 3 && argc != 4) 
    {
        fprintf (stderr, "Usage: %s <m> <k> [file name]\n", argv[0]);
        return 1;
    }
    else
    {
        m = atoi(argv[1]);
        k = atoi(argv[2]);
        file_name = (char*)malloc(256*sizeof(char));
        if (argc == 3)
            strcpy(file_name, OUTOUT_FILE);
        else if (argc == 4)
            strcpy(file_name, argv[3]);
    }
    
    srand(0);
    
    int size_ab = m * k * sizeof(PRECISION_T);
    int size_c  = m * m * sizeof(PRECISION_T);

    PRECISION_T *a = (PRECISION_T*)malloc(size_ab);
    PRECISION_T *b = (PRECISION_T*)malloc(size_ab);
    PRECISION_T *c = (PRECISION_T*)malloc(size_c);

    RANDOM_ARRAY_2D(m, k, a);
    RANDOM_ARRAY_2D(k, m, b);

    KERNEL(m, k, a, b, c);

    output = fopen(file_name, "w");
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            fprintf(output, "%f ", c[i*m+j]);
        }
        fprintf(output, "\n");
    }
    fclose(output);

    free(file_name);
    free(a);
    free(b);
    free(c);
}
