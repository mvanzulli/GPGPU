#include "util.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define THREAD_PER_BLOCK 32
#define MASK_SIZE 5

// Defino la máscara como memoria constante (para blur_b_iv)
__constant__ float d_mask_c[25];

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

  __global__ void blur_gl(float* d_input, int width, int height, float* d_output, float * d_msk,   int m_size) {

    int tid_x, tid_y, tid;
    tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    tid = tid_x + tid_y * width;
    
    float val_pixel = 0;
    int neighbourPixel,mask_pixel_x, mask_pixel_y, threadId_mask;

    for (int i = 0; i < m_size ; i++) {
        for (int j = 0; j < m_size ; j++) {
          mask_pixel_x    = tid_x + i - m_size/2;
          mask_pixel_y    = tid_y + j - m_size/2;
          neighbourPixel  = mask_pixel_x + mask_pixel_y * width;
          threadId_mask   = i*m_size+j;
            if(mask_pixel_x >=0 && mask_pixel_y >=0 && mask_pixel_x <width && mask_pixel_y <height) {
                val_pixel = val_pixel +  d_input[neighbourPixel] * d_msk[threadId_mask];
            }
        }
    }
    if (tid < width * height )
        d_output[tid] = val_pixel;
}

  __global__ void blur_a_i(float* d_input, int width, int height, float* d_output, float * d_msk, int m_size){
  
      // Defino el tile: (el array guardado en shared mem)
      extern __shared__ float tile[];
  
      // PARTE 1 - ESCRITURA AL TILE DE TODOS LOS PIXELS Y SUS VECINOS DE ESTE BLOQUE //

      // Defino indices de mapeo en el tile
      // Defino el tamaño del tile
      int width_tile   = blockDim.x + m_size - 1;

      // Defino los índices a leer en la imagen original y escribir en el tile (traslado el bloque a la esq. superior izquierda)
      int tid_moved_x = blockIdx.x  * blockDim.x + threadIdx.x - m_size/2;
      int tid_moved_y = blockIdx.y  * blockDim.y + threadIdx.y - m_size/2;
      int tid_tile  = threadIdx.x + threadIdx.y * width_tile;
      int tid_moved = (tid_moved_y) * width + (tid_moved_x) ;
      

      //PASO 1, Todos los threads de este bloque escriben en el tile
      if (tid_moved >= 0 && tid_moved_x >= 0 && tid_moved_y >= 0) {
        tile[tid_tile] = d_input[tid_moved];
      }
      

      //PASO 2, traslado el bloque a la derecha sumándole blockDim.x. Evito sobreescribir lo escrito en el paso 1 y no pasarme del tamaño del tile ni de la imagen.

      // Redefino tid_tile y tid_moved trasladándolas blockDim.x a la derechaÑ
      int tid_tile_2  = tid_tile + blockDim.x;
      int tid_moved_2 = tid_moved + blockDim.x;

      // Me aseguro de estar dentro de la imagen:
      if (tid_moved_2 >= 0 && tid_moved_2 < width * height && tid_moved_y >= 0) {
        // Me aseguro de estar dentro del tile:
        if (tid_tile_2 < width_tile * width_tile) {
          // Los threads en x mayores a m_size - 1 quedan ociosos (porque estos están fuera del tile dado que corrí el bloque hacia la derecha)
          if (threadIdx.x < m_size - 1) {
            // Me aseguro de no sobreescribir. (Se escribió en paso 1 hasta blockDim.x - 1 en la dirección x)
            if (tid_tile_2 > blockDim.x)
                tile[tid_tile_2] = d_input[tid_moved_2];
          }
        }
      }

      
      //PASO 3, traslado el bloque hacia abajo (sumo blockDim.y * width). Evito sobreescribir lo escrito anteriormente y no pasarme del tamaño del tile ni de la imagen.

      // Redefino tid_tile y tid_moved trasladándolas blockDim.y hacia abajo:
      int tid_tile_3  = tid_tile + blockDim.y * width_tile;
      int tid_moved_3 = tid_moved + blockDim.y * width; 
      
      // Me aseguro de estar dentro de la imagen:
      if (tid_moved_3 >= 0 && tid_moved_3 < width * height && tid_moved_x >= 0) {
        // Me aseguro de estar dentro del tile:
        if (tid_tile_3 < width_tile * width_tile) {
          // Los threads en y mayores a m_size - 1 quedan ociosos (porque estos están fuera del tile dado que corrí el bloque hacia abajo)
          if (threadIdx.y < m_size - 1) {
            // // Me aseguro de no sobreescribir. (Se escribió en paso 1 y 2 hasta width tile * blockDim.y)
            // if (tid_tile_3 > width_tile * (blockDim.y))
              tile[tid_tile_3] = d_input[tid_moved_3];
          }
        }
      }


      //PASO 4, traslado el bloque hacia la derecha y hacia abajo, sumando (blockDim + blockDim.y * width). Evito sobreescribir lo escrito anteriormente y no pasarme del tamaño del tile ni de la imagen.

      // Redefino tid_tile y tid_moved trasladándolas blockDim.y hacia abajo:
      int tid_tile_4  = tid_tile + blockDim.x + blockDim.y * width_tile;
      int tid_moved_4 = tid_moved + blockDim.x + blockDim.y * width; 

      // Me aseguro de estar dentro de la imagen:
      if (tid_moved_4 >= 0 && tid_moved_4 < width * height) {
        // Me aseguro de estar dentro del tile:
        if (tid_tile_4 < width_tile * width_tile) {
          // Los threads en x e y mayores a (m_size - 1) quedan ociosos (porque están fuera del tile dado que corrí el bloque hacia abajo y hacia la derecha)
          if (threadIdx.x < m_size - 1 && threadIdx.y < m_size - 1)
              tile[tid_tile_4] = d_input[tid_moved_4];
        }
      }




      // PARTE 2 - LECTURA DEL TILE, REDEFINICIÓN DEL PIXEL Y ESCRITURA EN IMAGEN DE SALIDA //

      // Me aseguro de que se haya completado la escritura en el tile.
      __syncthreads();
      
      // Redefino el pixel en cuestión en función de sus vecinos y la máscara:
      // Inicializo en 0 el valor del pixel:
      float val_pixel  = 0;
      int tid_x  = threadIdx.x + blockIdx.x * blockDim.x;
      int tid_y  = threadIdx.y + blockIdx.y * blockDim.y;
      int read_tile_x, read_tile_y, read_threadId;
      // Recorro entre todos los vecinos (m_size*m_size)
      for (int i = 0; i < m_size ; i++) {
        for (int j = 0; j < m_size ; j++) {
          
        // Defino los índices para acceder al pixel actual (este thread) y a sus vecinos
          read_tile_x = threadIdx.x + j - m_size/2;
          read_tile_y = threadIdx.y + i - m_size/2;
          // La lectura de la imagen y la escritura al tile realizadas al principio de este kernel están corridas m_size/2 por cada dirección. Para compensar le sumo m_size/2 en cada dirección al índice para obtener el valor de la imagen del vecino (o del propio pixel).
          read_threadId = read_tile_x + m_size/2 + width_tile * (read_tile_y + m_size/2);
  
          // Me aseguro de que los vecinos estén en la imagen. (Los pixel en los bordes tendrían vecinos fuera de la imagen)
          if(read_threadId >= 0 && tid_x + j - m_size/2 >= 0 && tid_y + i - m_size/2 >= 0 && read_threadId < width_tile * width_tile) {
            // Sumo ponderadamente el valor de los vecinos guardado en el tile por el valor de la máscara.
            val_pixel +=  tile[read_threadId] * d_msk[i*m_size + j];
          }
        }
      }

      // Escribo el valor con blur del pixel (este thread) en la imagen de salida (transfiero a global mem)
      int tid = tid_x + tid_y * width;
      if (tid < width * height)
        d_output[tid] = val_pixel;
      
    }
  
__global__ void blur_a_ii(float* d_input, int width, int height, float* d_output, float * d_msk,   int m_size){

    // Defino el tile: (el array guardado en shared mem)
    __shared__ float tile[THREAD_PER_BLOCK*THREAD_PER_BLOCK];
    
    //PASO 1: Leo variables en la imagen original por filas y copio al tile de forma coalseced por filas
    int tid_x = blockIdx.x  * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y  * blockDim.y + threadIdx.y;
    int tid = tid_y * width + tid_x; //Indice de acceso a la imagen original
    
    int tid_tile = threadIdx.y * blockDim.y + threadIdx.x; //El block dim.x es el ancho del tile
    
    tile[tid_tile] = d_input[tid];

    __syncthreads();

    // PASO 2: Aplicamos la mascara
    float val_pixel = 0;
    int mask_pixel_x, mask_pixel_y, tile_pixel_x, tile_pixel_y, threadId_tile, threadId_mask;
    for (int i = 0; i < m_size ; i++) { //Los indices se mueven en el tile que es igual al tamaño de la mascara
        for (int j = 0; j < m_size ; j++) {
          mask_pixel_x = tid_x + i - m_size/2;
          mask_pixel_y = tid_y + j - m_size/2;
          // Este índice representa a un pixel dentro de la imagen completa
          threadId_mask= mask_pixel_x + mask_pixel_y * width;
          

          tile_pixel_x = threadIdx.x + i - m_size/2;
          tile_pixel_y = threadIdx.y + j - m_size/2;
          threadId_tile= tile_pixel_x + tile_pixel_y * blockDim.x;

          
          // Me aseguro que el vecino esté dentro de la imagen.
          if(mask_pixel_x >=0 && mask_pixel_y >=0 && mask_pixel_x <width && mask_pixel_y <height) {
            // Si este vecino está dentro del bloque:
            if(tile_pixel_x >= 0 && tile_pixel_x < blockDim.x && tile_pixel_y >= 0 && tile_pixel_y < blockDim.y ) {
            val_pixel += tile[threadId_tile] * d_msk[i*m_size+j];
            // Sino, voy a buscar a memoria global el valor del pixel vecino
            } else {
              val_pixel += d_input[threadId_mask] * d_msk[i*m_size + j];
            }
          }
        }
      }
      
      // Escribo el valor con blur de este pixel en la imagen de salida (global mem):
      if (tid < width * height )
        d_output[tid] = val_pixel;
  }

  __global__ void blur_b_i(float* d_input, int width, int height, float* d_output, float * d_msk, int m_size){
  

      // Defino la shared mem que va a alojar a la máscara
      __shared__ float tile_mask[MASK_SIZE * MASK_SIZE];

      // Cada thread copia una parte de la máscara
      int tid_mask = threadIdx.x + threadIdx.y * m_size;
      if (tid_mask < m_size * m_size)
        tile_mask[tid_mask] = d_msk[tid_mask];
      

      // Defino el tile: (el array guardado en shared mem)
      __shared__ float tile[(THREAD_PER_BLOCK + MASK_SIZE - 1) * (THREAD_PER_BLOCK + MASK_SIZE - 1)];
  
      // PARTE 1 - ESCRITURA AL TILE DE TODOS LOS PIXELS Y SUS VECINOS DE ESTE BLOQUE //

      // Defino indices de mapeo en el tile
      // Defino el tamaño del tile
      int width_tile   = blockDim.x + m_size - 1;

      // Defino los índices a leer en la imagen original y escribir en el tile (traslado el bloque a la esq. superior izquierda)
      int tid_moved_x = blockIdx.x  * blockDim.x + threadIdx.x - m_size/2;
      int tid_moved_y = blockIdx.y  * blockDim.y + threadIdx.y - m_size/2;
      int tid_tile  = threadIdx.x + threadIdx.y * width_tile;
      int tid_moved = (tid_moved_y) * width + (tid_moved_x) ;
      

     //PASO 1, Todos los threads de este bloque escribe en el tile
      if (tid_moved >= 0 ) {
        tile[tid_tile] = d_input[tid_moved];
      }
      

      //PASO 2, traslado el bloque a la derecha sumándole blockDim.x. Evito sobreescribir lo escrito en el paso 1 y no pasarme del tamaño del tile ni de la imagen.

      // Redefino tid_tile y tid_moved trasladándolas blockDim.x a la derechaÑ
      int tid_tile_2  = tid_tile + blockDim.x;
      int tid_moved_2 = tid_moved + blockDim.x; 

      // Me aseguro de estar dentro de la imagen:
      if (tid_moved_2 >= 0 && tid_moved_2 < width * height) {
        // Me aseguro de estar dentro del tile:
        if (tid_tile_2 < width_tile * width_tile) {
          // Los threads en x mayores a m_size - 1 quedan ociosos (porque estos están fuera del tile dado que corrí el bloque hacia la derecha)
          if (threadIdx.x < m_size - 1) {
            // Me aseguro de no sobreescribir. (Se escribió en paso 1 hasta blockDim.x - 1 en la dirección x)
            if (tid_tile_2 > blockDim.x)
                tile[tid_tile_2] = d_input[tid_moved_2];
          }
        }
      }

      
      //PASO 3, traslado el bloque hacia abajo (sumo blockDim.y * width). Evito sobreescribir lo escrito anteriormente y no pasarme del tamaño del tile ni de la imagen.

      // Redefino tid_tile y tid_moved trasladándolas blockDim.y hacia abajo:
      int tid_tile_3  = tid_tile + blockDim.y * width_tile;
      int tid_moved_3 = tid_moved + blockDim.y * width; 
      
      // Me aseguro de estar dentro de la imagen:
      if (tid_moved_3 >= 0 && tid_moved_3 < width * height) {
        // Me aseguro de estar dentro del tile:
        if (tid_tile_3 < width_tile * width_tile) {
          // Los threads en y mayores a m_size - 1 quedan ociosos (porque estos están fuera del tile dado que corrí el bloque hacia abajo)
          if (threadIdx.y < m_size - 1) {
            // // Me aseguro de no sobreescribir. (Se escribió en paso 1 y 2 hasta width tile * blockDim.y)
            // if (tid_tile_3 > width_tile * (blockDim.y))
              tile[tid_tile_3] = d_input[tid_moved_3];
          }
        }
      }


      //PASO 4, traslado el bloque hacia la derecha y hacia abajo, sumando (blockDim + blockDim.y * width). Evito sobreescribir lo escrito anteriormente y no pasarme del tamaño del tile ni de la imagen.

      // Redefino tid_tile y tid_moved trasladándolas blockDim.y hacia abajo:
      int tid_tile_4  = tid_tile + blockDim.x + blockDim.y * width_tile;
      int tid_moved_4 = tid_moved + blockDim.x + blockDim.y * width; 

      // Me aseguro de estar dentro de la imagen:
      if (tid_moved_4 >= 0 && tid_moved_4 < width * height) {
        // Me aseguro de estar dentro del tile:
        if (tid_tile_4 < width_tile * width_tile) {
          // Los threads en x e y mayores a (m_size - 1) quedan ociosos (porque están fuera del tile dado que corrí el bloque hacia abajo y hacia la derecha)
          if (threadIdx.x < m_size - 1 && threadIdx.y < m_size - 1)
              tile[tid_tile_4] = d_input[tid_moved_4];
        }
      }




      // PARTE 2 - LECTURA DEL TILE, REDEFINICIÓN DEL PIXEL Y ESCRITURA EN IMAGEN DE SALIDA //

      // Me aseguro de que se haya completado la escritura en el tile.
      __syncthreads();
      
      // Redefino el pixel en cuestión en función de sus vecinos y la máscara:
      // Inicializo en 0 el valor del pixel:
      float val_pixel  = 0;
      int tid_x  = threadIdx.x + blockIdx.x * blockDim.x;
      int tid_y  = threadIdx.y + blockIdx.y * blockDim.y;
      int read_tile_x, read_tile_y, read_threadId;
      // Recorro entre todos los vecinos (m_size*m_size)
      for (int i = 0; i < m_size ; i++) {
        for (int j = 0; j < m_size ; j++) {
          
        // Defino los índices para acceder al pixel actual (este thread) y a sus vecinos
          read_tile_x = threadIdx.x + i - m_size/2;
          read_tile_y = threadIdx.y + j - m_size/2;
          // La lectura de la imagen y la escritura al tile realizadas al principio de este kernel están corridas m_size/2 por cada dirección. Para compensar le sumo m_size/2 en cada dirección al índice para obtener el valor de la imagen del vecino (o del propio pixel).
          read_threadId = read_tile_x + m_size/2 + width_tile * (read_tile_y + m_size/2);
  
          // Me aseguro de que los vecinos estén en la imagen. (Los pixel en los bordes tendrían vecinos fuera de la imagen)
          if(read_threadId >= 0 && tid_x + i - m_size/2 >= 0 && tid_y + j - m_size/2 >= 0 && read_threadId < width_tile * width_tile) {
            // Sumo ponderadamente el valor de los vecinos guardado en el tile por el valor de la máscara.
            val_pixel +=  tile[read_threadId] * tile_mask[i*m_size + j];
          }
        }
      }

      // Escribo el valor con blur del pixel (este thread) en la imagen de salida (transfiero a global mem)
      int tid = tid_x + tid_y * width;
      if (tid < width * height) {
        d_output[tid] = val_pixel;
      }
      
    }

      __global__ void blur_b_ii(float* d_input, int width, int height, float* d_output, float * d_msk, int m_size){
  

      // Defino shared mem:
      extern __shared__ float shared_mem_p[];

      // Defino el tile: (el array de pixels guardado en shared mem)
      float* tile = shared_mem_p;

      // Defino la máscara que estará en shared mem (a partir de la shared mem alocada dinámicamente):
      float* tile_mask = (float*)&shared_mem_p[(blockDim.x + m_size - 1) * (blockDim.x + m_size - 1)]; 

      // Cada thread copia una parte de la máscara
      int tid_mask = threadIdx.x + threadIdx.y * m_size;
      if (tid_mask < m_size * m_size)
        tile_mask[tid_mask] = d_msk[tid_mask];
      

  
      // PARTE 1 - ESCRITURA AL TILE DE TODOS LOS PIXELS Y SUS VECINOS DE ESTE BLOQUE //

      // Defino indices de mapeo en el tile
      // Defino el tamaño del tile
      int width_tile   = blockDim.x + m_size - 1;

      // Defino los índices a leer en la imagen original y escribir en el tile (traslado el bloque a la esq. superior izquierda)
      int tid_moved_x = blockIdx.x  * blockDim.x + threadIdx.x - m_size/2;
      int tid_moved_y = blockIdx.y  * blockDim.y + threadIdx.y - m_size/2;
      int tid_tile  = threadIdx.x + threadIdx.y * width_tile;
      int tid_moved = (tid_moved_y) * width + (tid_moved_x) ;
      

     //PASO 1, Todos los threads de este bloque escribe en el tile
      if (tid_moved >= 0 ) {
        tile[tid_tile] = d_input[tid_moved];
      }
      

      //PASO 2, traslado el bloque a la derecha sumándole blockDim.x. Evito sobreescribir lo escrito en el paso 1 y no pasarme del tamaño del tile ni de la imagen.

      // Redefino tid_tile y tid_moved trasladándolas blockDim.x a la derechaÑ
      int tid_tile_2  = tid_tile + blockDim.x;
      int tid_moved_2 = tid_moved + blockDim.x; 

      // Me aseguro de estar dentro de la imagen:
      if (tid_moved_2 >= 0 && tid_moved_2 < width * height) {
        // Me aseguro de estar dentro del tile:
        if (tid_tile_2 < width_tile * width_tile) {
          // Los threads en x mayores a m_size - 1 quedan ociosos (porque estos están fuera del tile dado que corrí el bloque hacia la derecha)
          if (threadIdx.x < m_size - 1) {
            // Me aseguro de no sobreescribir. (Se escribió en paso 1 hasta blockDim.x - 1 en la dirección x)
            if (tid_tile_2 > blockDim.x)
                tile[tid_tile_2] = d_input[tid_moved_2];
          }
        }
      }

      
      //PASO 3, traslado el bloque hacia abajo (sumo blockDim.y * width). Evito sobreescribir lo escrito anteriormente y no pasarme del tamaño del tile ni de la imagen.

      // Redefino tid_tile y tid_moved trasladándolas blockDim.y hacia abajo:
      int tid_tile_3  = tid_tile + blockDim.y * width_tile;
      int tid_moved_3 = tid_moved + blockDim.y * width; 
      
      // Me aseguro de estar dentro de la imagen:
      if (tid_moved_3 >= 0 && tid_moved_3 < width * height) {
        // Me aseguro de estar dentro del tile:
        if (tid_tile_3 < width_tile * width_tile) {
          // Los threads en y mayores a m_size - 1 quedan ociosos (porque estos están fuera del tile dado que corrí el bloque hacia abajo)
          if (threadIdx.y < m_size - 1) {
            // // Me aseguro de no sobreescribir. (Se escribió en paso 1 y 2 hasta width tile * blockDim.y)
            // if (tid_tile_3 > width_tile * (blockDim.y))
              tile[tid_tile_3] = d_input[tid_moved_3];
          }
        }
      }


      //PASO 4, traslado el bloque hacia la derecha y hacia abajo, sumando (blockDim + blockDim.y * width). Evito sobreescribir lo escrito anteriormente y no pasarme del tamaño del tile ni de la imagen.

      // Redefino tid_tile y tid_moved trasladándolas blockDim.y hacia abajo:
      int tid_tile_4  = tid_tile + blockDim.x + blockDim.y * width_tile;
      int tid_moved_4 = tid_moved + blockDim.x + blockDim.y * width; 

      // Me aseguro de estar dentro de la imagen:
      if (tid_moved_4 >= 0 && tid_moved_4 < width * height) {
        // Me aseguro de estar dentro del tile:
        if (tid_tile_4 < width_tile * width_tile) {
          // Los threads en x e y mayores a (m_size - 1) quedan ociosos (porque están fuera del tile dado que corrí el bloque hacia abajo y hacia la derecha)
          if (threadIdx.x < m_size - 1 && threadIdx.y < m_size - 1)
              tile[tid_tile_4] = d_input[tid_moved_4];
        }
      }




      // PARTE 2 - LECTURA DEL TILE, REDEFINICIÓN DEL PIXEL Y ESCRITURA EN IMAGEN DE SALIDA //

      // Me aseguro de que se haya completado la escritura en el tile.
      __syncthreads();
      
      // Redefino el pixel en cuestión en función de sus vecinos y la máscara:
      // Inicializo en 0 el valor del pixel:
      float val_pixel  = 0;
      int tid_x  = threadIdx.x + blockIdx.x * blockDim.x;
      int tid_y  = threadIdx.y + blockIdx.y * blockDim.y;
      int read_tile_x, read_tile_y, read_threadId;
      // Recorro entre todos los vecinos (m_size*m_size)
      for (int i = 0; i < m_size ; i++) {
        for (int j = 0; j < m_size ; j++) {
          
        // Defino los índices para acceder al pixel actual (este thread) y a sus vecinos
          read_tile_x = threadIdx.x + i - m_size/2;
          read_tile_y = threadIdx.y + j - m_size/2;
          // La lectura de la imagen y la escritura al tile realizadas al principio de este kernel están corridas m_size/2 por cada dirección. Para compensar le sumo m_size/2 en cada dirección al índice para obtener el valor de la imagen del vecino (o del propio pixel).
          read_threadId = read_tile_x + m_size/2 + width_tile * (read_tile_y + m_size/2);
  
          // Me aseguro de que los vecinos estén en la imagen. (Los pixel en los bordes tendrían vecinos fuera de la imagen)
          if(read_threadId >= 0 && tid_x + i - m_size/2 >= 0 && tid_y + j - m_size/2 >= 0 && read_threadId < width_tile * width_tile) {
            // Sumo ponderadamente el valor de los vecinos guardado en el tile por el valor de la máscara.
            val_pixel +=  tile[read_threadId] * tile_mask[i*m_size + j];
          }
        }
      }

      // Escribo el valor con blur del pixel (este thread) en la imagen de salida (transfiero a global mem)
      int tid = tid_x + tid_y * width;
      if (tid < width * height) {
        d_output[tid] = val_pixel;
      }
      
    }

      __global__ void blur_b_iii(float* d_input, int width, int height, float* d_output, const float* __restrict__ d_msk, int m_size){
      // Defino el tile: (el array guardado en shared mem)
      extern __shared__ float tile[];
  
      // PARTE 1 - ESCRITURA AL TILE DE TODOS LOS PIXELS Y SUS VECINOS DE ESTE BLOQUE //

      // Defino indices de mapeo en el tile
      // Defino el tamaño del tile
      int width_tile   = blockDim.x + m_size - 1;

      // Defino los índices a leer en la imagen original y escribir en el tile (traslado el bloque a la esq. superior izquierda)
      int tid_moved_x = blockIdx.x  * blockDim.x + threadIdx.x - m_size/2;
      int tid_moved_y = blockIdx.y  * blockDim.y + threadIdx.y - m_size/2;
      int tid_tile  = threadIdx.x + threadIdx.y * width_tile;
      int tid_moved = (tid_moved_y) * width + (tid_moved_x) ;
      

      //PASO 1, Todos los threads de este bloque escriben en el tile
      if (tid_moved >= 0 && tid_moved_x >= 0 && tid_moved_y >= 0) {
        tile[tid_tile] = d_input[tid_moved];
      }
      

      //PASO 2, traslado el bloque a la derecha sumándole blockDim.x. Evito sobreescribir lo escrito en el paso 1 y no pasarme del tamaño del tile ni de la imagen.

      // Redefino tid_tile y tid_moved trasladándolas blockDim.x a la derechaÑ
      int tid_tile_2  = tid_tile + blockDim.x;
      int tid_moved_2 = tid_moved + blockDim.x;

      // Me aseguro de estar dentro de la imagen:
      if (tid_moved_2 >= 0 && tid_moved_2 < width * height && tid_moved_y >= 0) {
        // Me aseguro de estar dentro del tile:
        if (tid_tile_2 < width_tile * width_tile) {
          // Los threads en x mayores a m_size - 1 quedan ociosos (porque estos están fuera del tile dado que corrí el bloque hacia la derecha)
          if (threadIdx.x < m_size - 1) {
            // Me aseguro de no sobreescribir. (Se escribió en paso 1 hasta blockDim.x - 1 en la dirección x)
            if (tid_tile_2 > blockDim.x)
                tile[tid_tile_2] = d_input[tid_moved_2];
          }
        }
      }

      
      //PASO 3, traslado el bloque hacia abajo (sumo blockDim.y * width). Evito sobreescribir lo escrito anteriormente y no pasarme del tamaño del tile ni de la imagen.

      // Redefino tid_tile y tid_moved trasladándolas blockDim.y hacia abajo:
      int tid_tile_3  = tid_tile + blockDim.y * width_tile;
      int tid_moved_3 = tid_moved + blockDim.y * width; 
      
      // Me aseguro de estar dentro de la imagen:
      if (tid_moved_3 >= 0 && tid_moved_3 < width * height && tid_moved_x >= 0) {
        // Me aseguro de estar dentro del tile:
        if (tid_tile_3 < width_tile * width_tile) {
          // Los threads en y mayores a m_size - 1 quedan ociosos (porque estos están fuera del tile dado que corrí el bloque hacia abajo)
          if (threadIdx.y < m_size - 1) {
            // // Me aseguro de no sobreescribir. (Se escribió en paso 1 y 2 hasta width tile * blockDim.y)
            // if (tid_tile_3 > width_tile * (blockDim.y))
              tile[tid_tile_3] = d_input[tid_moved_3];
          }
        }
      }


      //PASO 4, traslado el bloque hacia la derecha y hacia abajo, sumando (blockDim + blockDim.y * width). Evito sobreescribir lo escrito anteriormente y no pasarme del tamaño del tile ni de la imagen.

      // Redefino tid_tile y tid_moved trasladándolas blockDim.y hacia abajo:
      int tid_tile_4  = tid_tile + blockDim.x + blockDim.y * width_tile;
      int tid_moved_4 = tid_moved + blockDim.x + blockDim.y * width; 

      // Me aseguro de estar dentro de la imagen:
      if (tid_moved_4 >= 0 && tid_moved_4 < width * height) {
        // Me aseguro de estar dentro del tile:
        if (tid_tile_4 < width_tile * width_tile) {
          // Los threads en x e y mayores a (m_size - 1) quedan ociosos (porque están fuera del tile dado que corrí el bloque hacia abajo y hacia la derecha)
          if (threadIdx.x < m_size - 1 && threadIdx.y < m_size - 1)
              tile[tid_tile_4] = d_input[tid_moved_4];
        }
      }




      // PARTE 2 - LECTURA DEL TILE, REDEFINICIÓN DEL PIXEL Y ESCRITURA EN IMAGEN DE SALIDA //

      // Me aseguro de que se haya completado la escritura en el tile.
      __syncthreads();
      
      // Redefino el pixel en cuestión en función de sus vecinos y la máscara:
      // Inicializo en 0 el valor del pixel:
      float val_pixel  = 0;
      int tid_x  = threadIdx.x + blockIdx.x * blockDim.x;
      int tid_y  = threadIdx.y + blockIdx.y * blockDim.y;
      int read_tile_x, read_tile_y, read_threadId;
      // Recorro entre todos los vecinos (m_size*m_size)
      for (int i = 0; i < m_size ; i++) {
        for (int j = 0; j < m_size ; j++) {
          
        // Defino los índices para acceder al pixel actual (este thread) y a sus vecinos
          read_tile_x = threadIdx.x + j - m_size/2;
          read_tile_y = threadIdx.y + i - m_size/2;
          // La lectura de la imagen y la escritura al tile realizadas al principio de este kernel están corridas m_size/2 por cada dirección. Para compensar le sumo m_size/2 en cada dirección al índice para obtener el valor de la imagen del vecino (o del propio pixel).
          read_threadId = read_tile_x + m_size/2 + width_tile * (read_tile_y + m_size/2);
  
          // Me aseguro de que los vecinos estén en la imagen. (Los pixel en los bordes tendrían vecinos fuera de la imagen)
          if(read_threadId >= 0 && tid_x + j - m_size/2 >= 0 && tid_y + i - m_size/2 >= 0 && read_threadId < width_tile * width_tile) {
            // Sumo ponderadamente el valor de los vecinos guardado en el tile por el valor de la máscara.
            val_pixel +=  tile[read_threadId] * d_msk[i*m_size + j];
          }
        }
      }

      // Escribo el valor con blur del pixel (este thread) en la imagen de salida (transfiero a global mem)
      int tid = tid_x + tid_y * width;
      if (tid < width * height)
        d_output[tid] = val_pixel;
      
    }

    __global__ void blur_b_iv(float* d_input, int width, int height, float* d_output, int m_size){
  
      // Defino el tile: (el array guardado en shared mem)
      extern __shared__ float tile[];
  
      // PARTE 1 - ESCRITURA AL TILE DE TODOS LOS PIXELS Y SUS VECINOS DE ESTE BLOQUE //

      // Defino indices de mapeo en el tile
      // Defino el tamaño del tile
      int width_tile   = blockDim.x + m_size - 1;

      // Defino los índices a leer en la imagen original y escribir en el tile (traslado el bloque a la esq. superior izquierda)
      int tid_moved_x = blockIdx.x  * blockDim.x + threadIdx.x - m_size/2;
      int tid_moved_y = blockIdx.y  * blockDim.y + threadIdx.y - m_size/2;
      int tid_tile  = threadIdx.x + threadIdx.y * width_tile;
      int tid_moved = (tid_moved_y) * width + (tid_moved_x) ;
      

      //PASO 1, Todos los threads de este bloque escribe en el tile
      if (tid_moved >= 0 ) {
        tile[tid_tile] = d_input[tid_moved];
      }
      

      //PASO 2, traslado el bloque a la derecha sumándole blockDim.x. Evito sobreescribir lo escrito en el paso 1 y no pasarme del tamaño del tile ni de la imagen.

      // Redefino tid_tile y tid_moved trasladándolas blockDim.x a la derechaÑ
      int tid_tile_2  = tid_tile + blockDim.x;
      int tid_moved_2 = tid_moved + blockDim.x; 

      // Me aseguro de estar dentro de la imagen:
      if (tid_moved_2 >= 0 && tid_moved_2 < width * height) {
        // Me aseguro de estar dentro del tile:
        if (tid_tile_2 < width_tile * width_tile) {
          // Los threads en x mayores a m_size - 1 quedan ociosos (porque estos están fuera del tile dado que corrí el bloque hacia la derecha)
          if (threadIdx.x < m_size - 1) {
            // Me aseguro de no sobreescribir. (Se escribió en paso 1 hasta blockDim.x - 1 en la dirección x)
            if (tid_tile_2 > blockDim.x)
                tile[tid_tile_2] = d_input[tid_moved_2];
          }
        }
      }

      
      //PASO 3, traslado el bloque hacia abajo (sumo blockDim.y * width). Evito sobreescribir lo escrito anteriormente y no pasarme del tamaño del tile ni de la imagen.

      // Redefino tid_tile y tid_moved trasladándolas blockDim.y hacia abajo:
      int tid_tile_3  = tid_tile + blockDim.y * width_tile;
      int tid_moved_3 = tid_moved + blockDim.y * width; 
      
      // Me aseguro de estar dentro de la imagen:
      if (tid_moved_3 >= 0 && tid_moved_3 < width * height) {
        // Me aseguro de estar dentro del tile:
        if (tid_tile_3 < width_tile * width_tile) {
          // Los threads en y mayores a m_size - 1 quedan ociosos (porque estos están fuera del tile dado que corrí el bloque hacia abajo)
          if (threadIdx.y < m_size - 1) {
            // // Me aseguro de no sobreescribir. (Se escribió en paso 1 y 2 hasta width tile * blockDim.y)
            // if (tid_tile_3 > width_tile * (blockDim.y))
              tile[tid_tile_3] = d_input[tid_moved_3];
          }
        }
      }


      //PASO 4, traslado el bloque hacia la derecha y hacia abajo, sumando (blockDim + blockDim.y * width). Evito sobreescribir lo escrito anteriormente y no pasarme del tamaño del tile ni de la imagen.

      // Redefino tid_tile y tid_moved trasladándolas blockDim.y hacia abajo:
      int tid_tile_4  = tid_tile + blockDim.x + blockDim.y * width_tile;
      int tid_moved_4 = tid_moved + blockDim.x + blockDim.y * width; 

      // Me aseguro de estar dentro de la imagen:
      if (tid_moved_4 >= 0 && tid_moved_4 < width * height) {
        // Me aseguro de estar dentro del tile:
        if (tid_tile_4 < width_tile * width_tile) {
          // Los threads en x e y mayores a (m_size - 1) quedan ociosos (porque están fuera del tile dado que corrí el bloque hacia abajo y hacia la derecha)
          if (threadIdx.x < m_size - 1 && threadIdx.y < m_size - 1)
              tile[tid_tile_4] = d_input[tid_moved_4];
        }
      }




      // PARTE 2 - LECTURA DEL TILE, REDEFINICIÓN DEL PIXEL Y ESCRITURA EN IMAGEN DE SALIDA //

      // Me aseguro de que se haya completado la escritura en el tile.
      __syncthreads();
      
      // Redefino el pixel en cuestión en función de sus vecinos y la máscara:
      // Inicializo en 0 el valor del pixel:
      float val_pixel  = 0;
      int tid_x  = threadIdx.x + blockIdx.x * blockDim.x;
      int tid_y  = threadIdx.y + blockIdx.y * blockDim.y;
      int read_tile_x, read_tile_y, read_threadId;
      // Recorro entre todos los vecinos (m_size*m_size)
      for (int i = 0; i < m_size ; i++) {
        for (int j = 0; j < m_size ; j++) {
          
        // Defino los índices para acceder al pixel actual (este thread) y a sus vecinos
          read_tile_x = threadIdx.x + i - m_size/2;
          read_tile_y = threadIdx.y + j - m_size/2;
          // La lectura de la imagen y la escritura al tile realizadas al principio de este kernel están corridas m_size/2 por cada dirección. Para compensar le sumo m_size/2 en cada dirección al índice para obtener el valor de la imagen del vecino (o del propio pixel).
          read_threadId = read_tile_x + m_size/2 + width_tile * (read_tile_y + m_size/2);
  
          // Me aseguro de que los vecinos estén en la imagen. (Los pixel en los bordes tendrían vecinos fuera de la imagen)
          if(read_threadId >= 0 && tid_x + i - m_size/2 >= 0 && tid_y + j - m_size/2 >= 0 && read_threadId < width_tile * width_tile) {
            // Sumo ponderadamente el valor de los vecinos guardado en el tile por el valor de la máscara.
            val_pixel +=  tile[read_threadId] * d_mask_c[i*m_size + j];
          }
        }
      }

      // Escribo el valor con blur del pixel (este thread) en la imagen de salida (transfiero a global mem)
      int tid = tid_x + tid_y * width;
      if (tid < width * height)
        d_output[tid] = val_pixel;
      
    }

  void blur_gpu(float * image_in, int width, int height, float * image_out,  float mask[], int m_size, int threadPerBlockx, int threadPerBlocky){
    
    // Reservar memoria en la GPU
    float *d_img_in; float *d_img_out; float *d_mask;
    int nbx;//Número de blques x
    int nby;//Número de blques Y
    unsigned int size_img = width * height * sizeof(float);
    unsigned int size_msk = m_size * m_size * sizeof(float);
    int shared_memSize, num_sm, const_memSize;

    width % threadPerBlockx == 0 ? nbx = width / threadPerBlockx : nbx = width / threadPerBlockx + 1;
    height % threadPerBlocky == 0 ? nby = height / threadPerBlocky : nby = height / threadPerBlocky + 1;

    // Defino el tamaño shared memory para el tile que contendrá todos los pixel del block y sus vecinos.
    unsigned int size_tile_a_i = (threadPerBlockx+m_size-1) * (threadPerBlocky+m_size-1) * sizeof(float);
    // Defino el tamaño de shared memory a utilizar para el tile y la máscara
    unsigned int size_tile_b_ii = size_tile_a_i + (m_size) * (m_size) * sizeof(float);



    CUDA_CHK(cudaMalloc( (void**)&d_img_in   , size_img));//Reservo memoria en el device para la imagen original
    CUDA_CHK(cudaMalloc( (void**)&d_img_out  , size_img));//Reservo memoria en el device para la imagen de salida
    CUDA_CHK(cudaMalloc( (void**)&d_mask     , size_msk));//Reservo memoria para la máscara


    // copiar imagen y máscara a la GPU
    CUDA_CHK(cudaMemcpy(d_img_in  , image_in  , size_img, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_img_out , image_out , size_img, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_mask    , &mask[0]  , size_msk, cudaMemcpyHostToDevice));

    // Copio la máscara como constante (blur_b_iv)
    CUDA_CHK(cudaMemcpyToSymbol(d_mask_c, mask, size_msk));

    // configurar grilla y lanzar kernel
    dim3 grid(nbx,nby);
    dim3 block(threadPerBlockx,threadPerBlocky);

    // Kernels executions:

    // Utilizando exclusivamente global mem
    blur_gl <<< grid, block >>> (d_img_in, width, height, d_img_out, d_mask, m_size);
    
    // Utilizando shared memory (tile) del tamaño de (blockDim.x + m_size - 1) * (blockDim.x + m_size - 1). Es decir, todos los pixels vecinos están en el tile.
    // blur_a_i <<< grid, block, size_tile_a_i >>> (d_img_in, width, height, d_img_out, d_mask, m_size);
    
    // Utilizando combinación de global y shared mem. El tile tiene tamaño blockDim.x * blockDim.x, los vecinos que quedan afuera del tile se leen de la global mem. Se espera que el cache L1 ayude.
    blur_a_ii <<< grid, block >>> (d_img_in, width, height, d_img_out, d_mask, m_size);

    // Adapto blur_a_i para almacenarz la máscara en shared mem alocándola estáticamente.
    // blur_b_i <<< grid, block >>> (d_img_in, width, height, d_img_out, d_mask, m_size);

    // Adapto blur_a_i para almacenar la máscara en shared mem alocándola dinámicamente.
    // blur_b_ii <<< grid, block, size_tile_b_ii >>> (d_img_in, width, height, d_img_out, d_mask, m_size);

    // Adapto blur_a_i agregándola const __restrict__ al parámetro del kernel correspondiente a la máscara
    // blur_b_iii <<< grid, block, size_tile_a_i >>> (d_img_in, width, height, d_img_out, d_mask, m_size);
    
    // Adapto blur_a_i copiando la máscara como memoria constante (no preciso pasarla como parámetro al kernel)
    // blur_b_iv <<< grid, block, size_tile_a_i >>> (d_img_in, width, height, d_img_out, m_size);


    // Obtengo los posibles errores en la llamada al kernel
    CUDA_CHK(cudaGetLastError());

	  // Obligo al Kernel a llegar al final de su ejecucion y hacer obtener los posibles errores
    CUDA_CHK(cudaDeviceSynchronize());

    // transferir resultado a la memoria principal
    CUDA_CHK(cudaMemcpy(image_out, d_img_out , size_img, cudaMemcpyDeviceToHost));
	
  //  printf("sizeof(image_out): %d\n", sizeof(image_out));
  //  printf("sizeof(image_out[0]): %d\n", sizeof(image_out[0]));
  //  printf("d_output[256]_gpu_cpu: %f\n", image_out[256]);
    // liberar la memoria
    cudaFree(d_img_in); 
    cudaFree(d_img_out); 
    cudaFree(d_mask);

    cudaDeviceGetAttribute(&shared_memSize, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&const_memSize, cudaDevAttrTotalConstantMemory, 0);


    printf("shared_memSize: %dB\n", shared_memSize);
    printf("num_sm: %d\n", num_sm);
    printf("const_memSize: %dB\n", const_memSize);


    // // Destroy CUDA context (which means all device allocations are removed)
    // cudaDeviceReset();
}




