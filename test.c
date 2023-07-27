#include <stdio.h>
#include <stdlib.h>

void random_Darray_2D(unsigned int row, unsigned int col, double** array)
{
    for (unsigned int i = 0; i < row; i++)
    {
        for (unsigned int j = 0; j < col; j++)
        {
            (*array)[i*col+j] = (double)rand() / (double)RAND_MAX;
        }
    }
}

int main() {
	unsigned int m = 3, k = 2;

    srand(0);
    
    int size_ab = m * k * sizeof(double);

    double *a = (double*)malloc(size_ab);

    random_Darray_2D(m, k, &a);
    
    printf("A :\n");
    for (int i = 0; i < m; i++){
        for (int j = 0; j < k; j++)
            printf( "%f ", a[i*k+j]);
        printf( "\n");
    }
        
    free(a);
}
