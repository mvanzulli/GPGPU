#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void read_file(const char*, int*);
int get_text_length(const char * fname);

#define A 15
#define B 27
#define M 256
#define A_MMI_M -17


__device__ int modulo(int a, int b){
	int r = a % b;
	r = (r < 0) ?  r+ b : r;
	return r;
}

__global__ void decrypt_kernel(int *d_message, int length, int parte) {

	if (parte == 1) {
		// Parte a:
		d_message[threadIdx.x] =  modulo(A_MMI_M * (d_message[threadIdx.x] - B), M);
	} else if (parte == 2) {
		// Parte b:
		d_message[blockIdx.x*blockDim.x + threadIdx.x] =  modulo(A_MMI_M * (d_message[blockIdx.x*blockDim.x + threadIdx.x] - B), M);
	} else if (parte == 3) {
			// Parte c:
			for (int i = 0; i < length / blockDim.x / gridDim.x + 1; ++i) {
				d_message[blockIdx.x*blockDim.x + gridDim.x*blockDim.x*i + threadIdx.x] =  modulo(A_MMI_M * (d_message[blockIdx.x*blockDim.x + gridDim.x*blockDim.x*i + threadIdx.x] - B), M);
			}
		}
}

int main(int argc, char *argv[])
{
	int *h_message;
	int *d_message;
	unsigned int size;
	int nb; // cuda grid size (# blocks per grid)

	const char * fname;

	if (argc < 4) 
		printf("Debes ingresar el nombre del archivo a desencriptar, el tamaño del bloque (1024 como máximo) y la parte del practico 2 a ejecutar: 1, 2 o 3.\n");
	else
		fname = argv[1];
		int parte = atoi(argv[3]); // cuda block size (# threads per block ) max 1024
		int n = atoi(argv[2]);

	if (parte < 1 || parte > 3) {
		printf("%d\n",parte);
		printf("Debe ingresar como segundo parámetro la parte del ejercicio a ejecutar: 1, 2 o 3.\n");
		exit(1);
	} else if (n > 1024) {
		printf("El tamaño máximo del bloque es 1024. La arquitectura actual a utilizar soporta un máximo de 1024 threads por bloque.\n");
		exit(1);
	}

	int length = get_text_length(fname);

	size = length * sizeof(int);

	// Parte a y c:
	if (parte == 1 || parte == 3) {
		nb = 16;
	// parte b:
	} else if (parte == 2) {
		if (length % n == 0 ){
			nb = length/n;
		}else
			nb = length / n + 1;
	}

	// reservar memoria para el mensaje
	h_message = (int *)malloc(size);

	// leo el archivo de la entrada
	read_file(fname, h_message);

	// reservo memoria en la GPU
	CUDA_CHK(cudaMalloc((void**)&d_message, size));

	// copio los datos a la memoria de la GPU
	CUDA_CHK(cudaMemcpy(d_message, h_message, size, cudaMemcpyHostToDevice));

	// configuro la grilla de threads
	dim3 gridSize(nb,1);
	dim3 blockSize(n, 1, 1);

	// ejecuto el kernel
	decrypt_kernel <<< gridSize, blockSize >>> (d_message, length, parte);

	// Obtengo los posibles errores en la llamada al kernel
	CUDA_CHK(cudaGetLastError());

	// Obligo al Kernel a llegar al final de su ejecucion y hacer obtener los posibles errores
	CUDA_CHK(cudaDeviceSynchronize());



	// copio los datos nuevamente a la memoria de la CPU
	CUDA_CHK(cudaMemcpy(h_message, d_message, size, cudaMemcpyDeviceToHost));

	// despliego el mensaje

	// Si estoy en la primera parte despliego solamente la parte desencriptada.
	if (parte == 1)
		length = n;
	
	for (int i = 0; i < length; i++) {
		printf("%c", (char)h_message[i]);
	}
	printf("\n");

	// libero la memoria en la GPU
	cudaFree(d_message);

	// libero la memoria en la CPU
	free(h_message);

	return 0;
}

	
int get_text_length(const char * fname)
{
	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags

	size_t pos = ftell(f);    
	fseek(f, 0, SEEK_END);    
	size_t length = ftell(f); 
	fseek(f, pos, SEEK_SET);  

	fclose(f);

	return length;
}

void read_file(const char * fname, int* input)
{
	// printf("leyendo archivo %s\n", fname );

	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find %s file \n", fname);
		exit(1);
	}

	//fread(input, 1, N, f);
	int c; 
	while ((c = getc(f)) != EOF) {
		*(input++) = c;
		printf("%d",c);
	}
	

	fclose(f);
}
