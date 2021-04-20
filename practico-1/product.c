#include "aux.h"

// __restrict__ keyword hints to the compiler that only this pointer can point to the obcol_Aect that is pointing to.
int mult_simple     (const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n) {
    int cellA, cellB, cellC, k;
    for (int row_A = 0; row_A < n; ++row_A) {
        for (int col_B = 0; col_B < n; ++col_B) {
            for (int k = 0; k < n; ++k) {
                cellA = row_A * n + k;
                cellB = col_B + k * n;
                cellC = col_B + row_A * n;
                C[cellC]+= A[cellA] * B[cellB];
            }
        }
    }
}

int mult_fila     (const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n) {
    int cellA, cellB, cellC, k;
    for (int row_A = 0; row_A < n; ++row_A) {
        for (int col_A = 0; col_A < n; ++col_A) {
            for (int k = 0; k < n; ++k) {
                cellA = row_A * n + col_A;
                cellB = col_A * n + k;
                cellC = row_A * n + k;
                C[cellC]+= A[cellA] * B[cellB];
            }
        }
    }
}

int mult_bl_simple(const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n, size_t nb) {

    int cellA, cellB, cellC, numBlocks = n/nb;
    
    for (int numBlockRow = 0; numBlockRow < numBlocks; numBlockRow++) {
        for (int numBlockCol = 0; numBlockCol < numBlocks; numBlockCol++) {
                
            int rowBlockC = numBlockRow * nb;
            int colBlockC = numBlockCol * nb;
                
            for (int rowA = rowBlockC; rowA < nb * (numBlockRow+1) ; rowA++) {
                for (int colB = colBlockC; colB < nb * (numBlockCol+1); colB++) {
                
                    for (int k = 0; k < n; k++) {
                        cellA = n * rowA + k;
                        cellB = n * k + colB;
                        cellC = n * rowA + colB;
                        C[cellC]+= A[cellA] * B[cellB];
                    }
                }
            }
        }
    }
}

int mult_bl_fila(const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n, size_t nb) {

    int cellA, cellB, cellC, numBlocks = n/nb;
    
    for (int numBlockRow = 0; numBlockRow < numBlocks; numBlockRow++) {
        for (int numBlockCol = 0; numBlockCol < numBlocks; numBlockCol++) {
                
            int rowBlockC = numBlockRow * nb;
            int colBlockC = numBlockCol * nb;
                
            for (int rowA = rowBlockC; rowA < nb * (numBlockRow+1) ; rowA++) {
                for (int colA = colBlockC; colA< nb * (numBlockCol+1); colA++) {
                
                    for (int k = 0; k < n; k++) {
                        cellA = n * rowA + colA;
                        cellB = n * colA + k;
                        cellC = n * rowA + k;
                        C[cellC]+= A[cellA] * B[cellB];
                    }
                }
            }
        }
    }
}

