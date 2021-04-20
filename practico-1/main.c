// Directives - libraries
#include <stdio.h>
#include "bench.h"
#include "sum.h"
#include "product.h"
#include "aux.h"


// int mult_bl_fila  (const VALT * __restrict__ A, const VALT * __restrict__ B, VALT * __restrict__ C, size_t n, size_t bl_sz); 



// main program
// argc (argument count) is the number of strings pointed to by argv (argument vector) (1 + number of arguments)
int main(char argc, char * argv[]){

    // const char * fname;

    if (argc < 2) {
        printf("El programa recibe n, y nb: n es la dimensión de las matrices y nb es el tamaño de bloque\n");
        exit(1);
    }

    // Converts argv (strings) to int
    int n = atoi(argv[1]);
    int nb = atoi(argv[2]);
    int alignment = 64;

    // srand(0); // Inicializa la semilla aleatoria

    VALT * A = (VALT *) aligned_alloc( alignment, n*n*sizeof(VALT) );
    VALT * B = (VALT *) aligned_alloc( alignment, n*n*sizeof(VALT) );
    VALT * C = (VALT *) aligned_alloc( alignment, n*n*sizeof(VALT) );





    random_vector(A, n*n);
    random_vector(B, n*n);

    struct timeb t_ini;
    struct timeb t_fin;

    // declared the array
    VALT A_est[N][N];

    // assign random doubles to each array's element
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A_est[i][j]=1;


    BENCH_RUN(suma_est_fil (A_est)          , t_suma_est_fil , t_suma_est_fil_runs );
    BENCH_RUN(suma_est_col (A_est)          , t_suma_est_col , t_suma_est_col_runs );
    BENCH_RUN(suma_din_fil (A,n)            , t_sum_din_fil  , t_suma_din_fil_runs );
    // BENCH_RUN(suma_din_fil_simpleFor(A,n)   , t_sum_din_fil2 , t_suma_din_fil2_runs);
    BENCH_RUN(suma_din_col (A,n)            , t_suma_din_col , t_suma_din_col_runs );
    BENCH_RUN(suma_din_rand(A,n)            , t_suma_din_rand, t_suma_din_rand_runs);
    BENCH_RUN(mult_simple(A,B,C,n)          , t_mm_simple    , t_mm_simple_runs    );
    BENCH_RUN(mult_fila (A,B,C,n)           , t_mm_fila      , t_mm_fila_runs      );
    BENCH_RUN(mult_bl_simple(A,B,C,n,nb)    , t_mm_bl_simple    , t_mm_bl_simple_runs );
    BENCH_RUN(mult_bl_fila (A,B,C,n,nb)     , t_mm_bl_fila      , t_mm_bl_fila_runs   );

    printf("suma_est_fil         : %.2f ms\t              runs: %d\n" , t_suma_est_fil , t_suma_est_fil_runs );
    printf("suma_est_col         : %.2f ms\t              runs: %d\n" , t_suma_est_col , t_suma_est_col_runs );
    printf("suma_din_fil         : %.2f ms\t              runs: %d\n" , t_sum_din_fil  , t_suma_din_fil_runs );
    // printf("suma_din_filSimpleFor: %.2f ms\t              runs: %d\n" , t_sum_din_fil2 , t_suma_din_fil2_runs);
    printf("suma_din_col         : %.2f ms\t              runs: %d\n" , t_suma_din_col , t_suma_din_col_runs );
    printf("suma_din_rand        : %.2f  ms\t             runs: %d\n" , t_suma_din_rand, t_suma_din_rand_runs);
    printf("mult_simple          : %.2f  ms, %.2f GFlops, runs: %d\n" , t_mm_simple , ( ((double)n/t_mm_simple )*((double)n/ 1000.0)*((double)n/1000.0)) , t_mm_simple_runs  );
    printf("mult_fila            : %.2f  ms, %.2f GFlops, runs: %d\n"   , t_mm_fila   , ( ((double)n/t_mm_fila   )*((double)n/ 1000.0)*((double)n/1000.0)) , t_mm_fila_runs    );
    printf("mult_simple_block    : %.2f  ms, %.2f GFlops, runs: %d\n" , t_mm_bl_simple , ( ((double)n/ t_mm_bl_simple )*((double)n/ 1000.0)*((double)n/1000.0)) , t_mm_bl_simple_runs  );
    printf("mult_fila_block      : %.2f  ms, %.2f GFlops, runs: %d\n" , t_mm_bl_fila , ( ((double)n/t_mm_bl_fila )*((double)n/ 1000.0)*((double)n/1000.0)) , t_mm_simple_runs  );

	return 0;
}
// 4.a)sqrt(32768/8/3) = 36
// sqrt(32768*2/8/3) = 52
// sqrt(262144/8/3) = 104
// sqrt(6291456/8/3) = 512