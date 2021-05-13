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

__global__ void blur_kernel(float* d_input, int width, int height, float* d_output, float * d_msk,   int m_size){

    int threadIdPixel, blockId;
    int neighbourPixel;
    float val_pixel = 0;

    blockId         = (gridDim.x * blockIdx.y) + blockIdx.x;
    threadIdPixel   = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    for (int i = 0; i < m_size ; i++){
        for (int j = 0; j < m_size ; j++){
            neighbourPixel =threadIdPixel + (j- m_size/2) +(i-m_size/2)*width ;                 
            if(neighbourPixel >= 0 && neighbourPixel < width * height ){
                val_pixel = val_pixel +  d_input[neighbourPixel] * d_msk[i*m_size+j];
            }
        }
    }
    if (threadIdPixel <= width * height )
        d_output[threadIdPixel] = val_pixel;
}




void blur_gpu(float * image_in, int width, int height, float * image_out,  float mask[], int m_size, int threadPerBlockx, int threadPerBlocky){
    
    // Reservar memoria en la GPU
    float *d_img_in; float *d_img_out; float *d_mask;
    int nbx;//Número de blques x
    int nby;//Número de blques Y
    unsigned int size_img = width * height * sizeof(float);
    unsigned int size_msk = m_size * m_size * sizeof(int);

    width % threadPerBlockx == 0 ? nbx = width / threadPerBlockx : nbx = width / threadPerBlockx + 1;
    height % threadPerBlocky == 0 ? nby = height / threadPerBlocky : nby = height / threadPerBlocky + 1;

    // Inicializo variables para medir tiempos
    CLK_CUEVTS_INIT;
    CLK_POSIX_INIT;


    CLK_CUEVTS_START;
    CLK_POSIX_START;
    CUDA_CHK(cudaMalloc( (void**)&d_img_in   , size_img));//Reservo memoria en el device para la imagen original
    CUDA_CHK(cudaMalloc( (void**)&d_img_out  , size_img));//Reservo memoria en el device para la imagen de salida
    CUDA_CHK(cudaMalloc( (void**)&d_mask     , size_msk));//Reservo memoria para la mascada
    CLK_POSIX_STOP;
    CLK_CUEVTS_STOP;
    CLK_CUEVTS_ELAPSED;
    CLK_POSIX_ELAPSED;
    float t_elap_cuda_malloc = t_elap_cuda;
    float t_elap_get_malloc = t_elap_get;
    

    
    // copiar imagen y máscara a la GPU
    CLK_POSIX_START;
    CLK_CUEVTS_START;
    CUDA_CHK(cudaMemcpy(d_img_in  , image_in  , size_img, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_img_out , image_out , size_img, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_mask    , &mask[0]  , size_msk, cudaMemcpyHostToDevice));
    CLK_CUEVTS_STOP;
    CLK_POSIX_STOP;
    CLK_CUEVTS_ELAPSED;
    CLK_POSIX_ELAPSED;
    float t_elap_cuda_cpyHtoD = t_elap_cuda;
    float t_elap_get_cpyHtoD = t_elap_get;

    // configurar grilla y lanzar kernel
    dim3 grid(nbx,nby)  ;
    dim3 block(threadPerBlockx,threadPerBlocky) ;

    CLK_POSIX_START;
    CLK_CUEVTS_START;
    blur_kernel <<< grid, block >>> (d_img_in, width, height, d_img_out, d_mask,  m_size); 
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
    CUDA_CHK(cudaMemcpy(image_out  , d_img_out , size_img, cudaMemcpyDeviceToHost));
    CLK_CUEVTS_STOP;
    CLK_POSIX_STOP;
    CLK_CUEVTS_ELAPSED;
    CLK_POSIX_ELAPSED;
    float t_elap_cuda_cpyDtoH = t_elap_cuda;
    float t_elap_get_cpyDtoH = t_elap_get;
	
    // liberar la memoria
    CLK_POSIX_START;
    CLK_CUEVTS_START;
    cudaFree(d_img_in); cudaFree(d_img_out) ; cudaFree(d_mask);
    CLK_CUEVTS_STOP;
    CLK_POSIX_STOP;
    CLK_CUEVTS_ELAPSED;
    CLK_POSIX_ELAPSED;
    float t_elap_cuda_free = t_elap_cuda;
    float t_elap_get_free = t_elap_get;

    printf("blur filter timing GPU:\n");
    printf("type:     | cudaEvents      | gettimeofday\n");
    printf("malloc:   | %06.3f ms       | %06.3f ms\n", t_elap_cuda_malloc, t_elap_get_malloc);
    printf("cpyHtoD:  | %06.3f ms       | %06.3f ms\n", t_elap_cuda_cpyHtoD, t_elap_get_cpyHtoD);
    printf("kernel:   | %06.3f ms       | %06.3f ms\n", t_elap_cuda_kernel, t_elap_get_kernel);
    printf("cpyDtoH:  | %06.3f ms       | %06.3f ms\n", t_elap_cuda_cpyDtoH, t_elap_get_cpyDtoH);
    printf("free:     | %06.3f ms       | %06.3f ms\n", t_elap_cuda_free, t_elap_get_free);
    printf("TOTAL:    | %06.3f ms       | %06.3f ms\n", t_elap_cuda_malloc + t_elap_cuda_cpyHtoD + t_elap_cuda_kernel + t_elap_cuda_cpyDtoH + t_elap_cuda_free + t_elap_cuda_malloc, t_elap_get_malloc + t_elap_get_cpyHtoD + t_elap_get_kernel + t_elap_get_cpyDtoH + t_elap_get_free + t_elap_get_malloc);
    printf("\n");
}

void blur_cpu(float * img_in, int width, int height, float * img_out, float msk[], int m_size){

    float val_pixel=0;

    // Inicializo variables para medir tiempos
    CLK_POSIX_INIT;
    
    CLK_POSIX_START;
    //para cada pixel aplicamos el filtro
    for(int imgx=0; imgx < width ; imgx++){
        for(int imgy=0; imgy < height; imgy++){

            val_pixel = 0;

            // aca aplicamos la mascara
            for (int i = 0; i < m_size ; i++){
                for (int j = 0; j < m_size ; j++){
                    
                    int ix =imgx + i - m_size/2;
                    int iy =imgy + j - m_size/2;
                    
                    if(ix >= 0 && ix < width && iy>= 0 && iy < height )
                        val_pixel = val_pixel +  img_in[iy * width +ix] * msk[i*m_size+j];
                    }
            }      
            // guardo valor resultado
            img_out[imgy*width+imgx]= val_pixel;
        }
    }
    CLK_POSIX_STOP;
    CLK_POSIX_ELAPSED;

    float t_elap = t_elap_get;

    printf("blur filter timing CPU:\n");
    printf("type:                       | gettimeofday\n");
    printf("TOTAL:                      | %06.3f ms\n",t_elap);
    printf("\n");
}
