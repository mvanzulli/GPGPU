#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"
/*#include <Windows.h>*/

	#define CLK_CUEVTS_INIT \
		cudaEvent_t evt_start, evt_stop; \
		float t_elap_cuda; \
		cudaEventCreate(&evt_start); \
		cudaEventCreate(&evt_stop) ;

	#define CLK_CUEVTS_START \
        cudaEventRecord(evt_start, 0);

	#define CLK_CUEVTS_STOP \
        cudaEventRecord(evt_stop, 0); \
        cudaEventSynchronize(evt_stop);

	#define CLK_CUEVTS_ELAPSED \
		t_elap_cuda = (cudaEventElapsedTime( &t_elap_cuda, evt_start, evt_stop ))?0:t_elap_cuda;


	#define CLK_POSIX_INIT \
		struct timeval t_i, t_f; \
		float t_elap_get

	#define CLK_POSIX_START \
		gettimeofday(&t_i,NULL)

	#define CLK_POSIX_STOP \
		gettimeofday(&t_f,NULL) 

	#define CLK_POSIX_ELAPSED \
		t_elap_get = ((double) t_f.tv_sec * 1000.0 + (double) t_f.tv_usec / 1000.0 - \
			 	 ((double) t_i.tv_sec * 1000.0 + (double) t_i.tv_usec / 1000.0))



// unsigned long long int sum_matrix(const unsigned long long int *M, int width);
// void print_matrix(const unsigned long long int *M, int width);
// void clean_matrix(unsigned long long int *M, int width);
// void clean_float_matrix(float *M, int width, int height);
// void copy_float_matrix(float *M, float *N, int width, int height);
// void init_matrix(unsigned long long int *M, int width);
// void clockStart(cudaEvent_t start);
// void clockStop(cudaEvent_t stop);
// float clockElapsed(cudaEvent_t start, cudaEvent_t stop);
// void clockInit(cudaEvent_t *start, cudaEvent_t *stop);
// void  mult_matrix(unsigned long long int *M1,unsigned long long int *M2, int width, unsigned long long int *M3);
