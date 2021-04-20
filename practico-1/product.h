#include "aux.h"

int mult_simple   (const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n); 
int mult_fila     (const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n);
int mult_bl_simple(const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n, size_t nb); 
int mult_bl_fila(const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n, size_t nb);