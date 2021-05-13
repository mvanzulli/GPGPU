#include "util.h"

#include "cuda.h"
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

using namespace std;

__global__ void ajustar_brillo_coalesced_kernel(float* d_img, int width, int height, float coef) {
    int threadId, blockId;

    blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    threadId = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (threadId <= width * height)
        d_img[threadId] = min(255.0f,max(0.0f,d_img[threadId]+coef));
}

__global__ void ajustar_brillo_no_coalesced_kernel(float* d_img, int width, int height, float coef) {

    int threadId, blockId;

    blockId = (gridDim.y * blockIdx.x) + blockIdx.y;
    threadId = (blockId * (blockDim.y * blockDim.x)) + (threadIdx.x * blockDim.y) + threadIdx.y;

    if (threadId <= width * height)
        d_img[threadId] = min(255.0f,max(0.0f,d_img[threadId]+coef));
}


void ajustar_brillo_gpu(float * img_in, int width, int height, float * img_out, float coef, int coalesced, int threadPerBlockx, int threadPerBlocky) {

    float *d_img;
    int nbx;
    int nby;
    unsigned int size_img = width * height * sizeof(float);

    width % threadPerBlockx == 0 ? nbx = width / threadPerBlockx : nbx = width / threadPerBlockx + 1;
    height % threadPerBlocky == 0 ? nby = height / threadPerBlocky : nby = height / threadPerBlocky + 1;

    printf("Image dimensions:\n");
    printf("width: %d px\n", width);
    printf("height: %d px\n", height);
    printf("\n");

    // Inicializo variables para medir tiempos
    CLK_CUEVTS_INIT;
    CLK_POSIX_INIT;
    
    // Reservar memoria en la GPU
    CLK_POSIX_START;
    CLK_CUEVTS_START;
    CUDA_CHK(cudaMalloc((void**)&d_img, size_img));
    CLK_CUEVTS_STOP;
    CLK_POSIX_STOP;
    CLK_CUEVTS_ELAPSED;
    CLK_POSIX_ELAPSED;
    float t_elap_cuda_malloc = t_elap_cuda;
    float t_elap_get_malloc = t_elap_get;

    // copiar imagen a la GPU
    CLK_POSIX_START;
    CLK_CUEVTS_START;
    CUDA_CHK(cudaMemcpy(d_img, img_in, size_img, cudaMemcpyHostToDevice));
    CLK_CUEVTS_STOP;
    CLK_POSIX_STOP;
    CLK_CUEVTS_ELAPSED;
    CLK_POSIX_ELAPSED;
    float t_elap_cuda_cpyHtoD = t_elap_cuda;
    float t_elap_get_cpyHtoD = t_elap_get;

    // configurar grilla y lanzar kernel
    dim3 grid(nbx,nby);
    dim3 block(threadPerBlockx,threadPerBlocky);

    CLK_POSIX_START;
    CLK_CUEVTS_START;
    if (coalesced == 1) {
        ajustar_brillo_coalesced_kernel <<< grid, block >>> (d_img, width, height, coef);
    } else {
        ajustar_brillo_no_coalesced_kernel <<< grid, block >>> (d_img, width, height, coef);
    }
    CLK_CUEVTS_STOP;

    // Obtengo los posibles errores en la llamada al kernel
	CUDA_CHK(cudaGetLastError());

	// Obligo al Kernel a llegar al final de su ejecucion y hacer obtener los posibles errores
	CUDA_CHK(cudaDeviceSynchronize());

    CLK_POSIX_STOP;
    CLK_CUEVTS_ELAPSED;
    CLK_POSIX_ELAPSED;
    float t_elap_cuda_kernel = t_elap_cuda;
    float t_elap_get_kernel = t_elap_get;

    // transferir resultado a la memoria principal
    CLK_POSIX_START;
    CLK_CUEVTS_START;
    CUDA_CHK(cudaMemcpy(img_out, d_img, size_img, cudaMemcpyDeviceToHost));
    CLK_CUEVTS_STOP;
    CLK_POSIX_STOP;
    CLK_CUEVTS_ELAPSED;
    CLK_POSIX_ELAPSED;
    float t_elap_cuda_cpyDtoH = t_elap_cuda;
    float t_elap_get_cpyDtoH = t_elap_get;

    // liberar la memoria
    CLK_POSIX_START;
    CLK_CUEVTS_START;
    cudaFree(d_img);
    CLK_CUEVTS_STOP;
    CLK_POSIX_STOP;
    CLK_CUEVTS_ELAPSED;
    CLK_POSIX_ELAPSED;
    float t_elap_cuda_free = t_elap_cuda;
    float t_elap_get_free = t_elap_get;

    printf("Bright adjustment timing:\n");
    printf("type:     | cudaEvents      | gettimeofday\n");
    printf("malloc:   | %06.3f ms       | %06.3f ms\n", t_elap_cuda_malloc, t_elap_get_malloc);
    printf("cpyHtoD:  | %06.3f ms       | %06.3f ms\n", t_elap_cuda_cpyHtoD, t_elap_get_cpyHtoD);
    printf("kernel:   | %06.3f ms       | %06.3f ms\n", t_elap_cuda_kernel, t_elap_get_kernel);
    printf("cpyDtoH:  | %06.3f ms       | %06.3f ms\n", t_elap_cuda_cpyDtoH, t_elap_get_cpyDtoH);
    printf("free:     | %06.3f ms       | %06.3f ms\n", t_elap_cuda_free, t_elap_get_free);
    printf("TOTAL:    | %06.3f ms       | %06.3f ms\n", t_elap_cuda_malloc + t_elap_cuda_cpyHtoD + t_elap_cuda_kernel + t_elap_cuda_cpyDtoH + t_elap_cuda_free + t_elap_cuda_malloc, t_elap_get_malloc + t_elap_get_cpyHtoD + t_elap_get_kernel + t_elap_get_cpyDtoH + t_elap_get_free + t_elap_get_malloc);
    printf("\n");
}

void ajustar_brillo_cpu(float * img_in, int width, int height, float * img_out, float coef){

    for(int imgx=0; imgx < width ; imgx++){
        for(int imgy=0; imgy < height; imgy++){
            img_out[imgy*width+imgx] = min(255.0f,max(0.0f,img_in[imgy*width+imgx]+coef));
        }
    }
}