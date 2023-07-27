#include <stdio.h>
#include <stdlib.h>
#include "tab.h"


void random_Darray_2D(unsigned int row, unsigned int col, double* array)
{
    for (unsigned int i = 0; i < row; i++)
    {
        for (unsigned int j = 0; j < col; j++)
        {
            array[i*col+j] = (double)rand() / (double)RAND_MAX;
        }
    }
}

void random_Sarray_2D(unsigned int row, unsigned int col, float* array)
{
    for (unsigned int i = 0; i < row; i++)
    {
        for (unsigned int j = 0; j < col; j++)
        {
            array[i*col+j] = (float)rand() / (float)RAND_MAX;
        }
    }
}

