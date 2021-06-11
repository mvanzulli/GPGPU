#include "util.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define THREAD_PER_BLOCK 32
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

__global__ void transpose_kernel_global(float* d_img_in, float* d_img_out, int width, int height) {
    
    int pixel_x, pixel_y,threadId_original,threadId_trans; //Declaro variables
    pixel_x = blockIdx.x * blockDim.x + threadIdx.x; //Indices imgx análogo a el CPU transpose
    pixel_y = blockIdx.y * blockDim.y + threadIdx.y; //Indices imgy análogo a el CPU transpose

    threadId_original = pixel_y*width+pixel_x; //Indice de acceso a la imagen original

    threadId_trans = (pixel_x*height+pixel_y);//Indice de acceso a la transpuesta
    
    if (threadId_original < width * height && threadId_trans < width * height)
        d_img_out[threadId_trans] = d_img_in[threadId_original];
}

__global__ void transpose_kernel_shared(float* d_img_in, float* d_img_out, int width, int height) {

    extern __shared__ float tile[]; //Defino el arrray tile en shared memory  
    
    //PASO 1: Leo variables en la imagen original por filas y copio al tile de forma coalseced por filas
    int original_pixel_x, original_pixel_y,threadId_original,threadId_tile_row;
    
    original_pixel_x = blockIdx.x  * blockDim.x + threadIdx.x;
    original_pixel_y = blockIdx.y  * blockDim.y + threadIdx.y;
    
    threadId_original = original_pixel_y * width + original_pixel_x ;//Indice de acceso a la imagen original
    threadId_tile_row = threadIdx.y * blockDim.x + threadIdx.x      ;//El block dim.x es el ancho del tile
    
    tile[threadId_tile_row] = d_img_in[threadId_original];
    __syncthreads(); // Me aseguro que se hayan copiado todos los datos al tile sino algunos threades impertientens se pueden encontrar con datos nulos
     //    Garantizado los datos en memoria compartida

    //PASO 2: Accedo por columnas al tile y calculo ese índice. 
    int threadId_tile_col;
    threadId_tile_col = threadIdx.x * blockDim.y + threadIdx.y;//El block dim.y es el height del tile

    // PASO 3: Pego en las filas de la imagen de salida de forma coalesced
    int transpose_pixel_x,transpose_pixel_y,threadId_trans;
    transpose_pixel_x = blockIdx.y * blockDim.y + threadIdx.x ;//Se accede por columnas
    transpose_pixel_y = blockIdx.x * blockDim.x + threadIdx.y ;
    threadId_trans    = transpose_pixel_x + transpose_pixel_y * height ;
    
    if (threadId_trans < width * height)
        d_img_out[threadId_trans] = tile[threadId_tile_col];
}

__global__ void transpose_kernel_shared_noBankConflicts(float* d_img_in, float* d_img_out, int width, int height) {

    __shared__ float tile_b[34][33]; //Defino el arrray tile_b en shared memory  

    //PASO 1: Leo variables en la imagen original por filas y copio al tile_b de forma coalseced por filas
    int original_pixel_x, original_pixel_y,threadId_original;

    
    original_pixel_x = blockIdx.x  * blockDim.x + threadIdx.x;
    original_pixel_y = blockIdx.y  * blockDim.y + threadIdx.y;
    
    threadId_original = original_pixel_y * width + original_pixel_x ;//Indice de acceso a la imagen original
    // int threadId_tile_b_row = threadIdx.y * blockDim.x + threadIdx.x      ;//El block dim.x es el ancho del tile_b
    
    tile_b[threadIdx.x][threadIdx.y]= d_img_in[threadId_original];

    __syncthreads(); 

    //PASO 2: Accedo por columnas al tile_b y calculo ese índice. 
    // int threadId_tile_b_col;
    // threadId_tile_b_col = threadIdx.x * blockDim.y + threadIdx.y;//El block dim.y es el height del tile_b

    // PASO 3: Pego en las filas de la imagen de salida de forma coalesced
    int transpose_pixel_x,transpose_pixel_y,threadId_trans;
    transpose_pixel_x = blockIdx.y * blockDim.y + threadIdx.x ;//Se accede por columnas
    transpose_pixel_y = blockIdx.x * blockDim.x + threadIdx.y ;
    threadId_trans    = transpose_pixel_x + transpose_pixel_y * height ;
    
    if (threadId_trans < width * height)
        d_img_out[threadId_trans] = tile_b[threadIdx.y][threadIdx.x];
}

void transpose_gpu(float * img_in, int width, int height, float * img_out, int threadPerBlockx, int threadPerBlocky) {

    float *d_img_in, *d_img_out;
    int nbx;
    int nby;
    unsigned int size_img, tile_size ;

    // Determino la cantidad de bloques a utilizar en función del tamaño de la imagen en pixels y del número de bloques pasado como parámetro por el usuario.
    width % threadPerBlockx == 0 ? nbx = width / threadPerBlockx : nbx = width / threadPerBlockx + 1;
    height % threadPerBlocky == 0 ? nby = height / threadPerBlocky : nby = height / threadPerBlocky + 1;
    
    // Determino el tamaño de la imagen en bytes
    size_img = width * height * sizeof(float);

    // Reservar memoria en la GPU
    CUDA_CHK(cudaMalloc((void**)&d_img_in, size_img));
    CUDA_CHK(cudaMalloc((void**)&d_img_out, size_img));

    // copiar imagen a la GPU
    CUDA_CHK(cudaMemcpy(d_img_in, img_in, size_img, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_img_out, img_out, size_img, cudaMemcpyHostToDevice));

    // configurar grilla y lanzar kernel
    dim3 grid(nbx,nby);
    dim3 block(threadPerBlockx,threadPerBlocky);

    // Defino el tamaño de la memoria compartida en bytes:
    tile_size = threadPerBlockx * threadPerBlocky * sizeof(float);

    // Utilizando global mem
    transpose_kernel_global <<< grid, block >>> (d_img_in, d_img_out, width, height);
    // Utilizando shared memory para transponer de a pequeños bloques
    transpose_kernel_shared <<< grid, block, tile_size >>> (d_img_in, d_img_out, width, height);
    // Utilizando shared memory e intentando solucionar conflictos de bancos (el tamaño del tile que no sea múlitplo del tamaño del bloque)
    transpose_kernel_shared_noBankConflicts <<< grid, block >>> (d_img_in, d_img_out, width, height);

    // Obtengo los posibles errores en la llamada al kernel
	CUDA_CHK(cudaGetLastError());

	// Obligo al Kernel a llegar al final de su ejecucion y así obtener los posibles errores
	CUDA_CHK(cudaDeviceSynchronize());



    // transferir resultado a RAM CPU:
    CUDA_CHK(cudaMemcpy(img_out, d_img_out, size_img, cudaMemcpyDeviceToHost));

    // liberar GPU global mem:
    cudaFree(d_img_in);
    cudaFree(d_img_out);
}
