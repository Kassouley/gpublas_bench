#ifndef PRINT_MEASURE_H
#define PRINT_MEASURE_H
#ifndef NB_META
#define NB_META 31
#endif
#include <stdint.h>
void print_measure(unsigned int m, unsigned int k, unsigned int nrep, double tdiff[NB_META]);
static int cmp_uint64 (const void *a, const void *b);
#endif