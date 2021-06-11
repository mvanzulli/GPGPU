#include "util.h"

void blur_cpu(float * img_in, int width, int height, float * img_out, float msk[], int m_size){

    float val_pixel=0;

    // Inicializo variables para medir tiempos
    CLK_POSIX_INIT;
    
    CLK_POSIX_START;
    //para cada pixel aplicamos el filtro
    // printf("0,0 pixel friends CPU:\n");
    // printf("img_in[65]: %f \n", img_in[65]);
    for(int imgx=0; imgx < width ; imgx++){
        for(int imgy=0; imgy < height; imgy++){

            val_pixel = 0;

            // aca aplicamos la mascara
            for (int i = 0; i < m_size ; i++){
                for (int j = 0; j < m_size ; j++){
                    
                    int ix =imgx + i - m_size/2;
                    int iy =imgy + j - m_size/2;
                    
                    if(ix >= 0 && ix < width && iy>= 0 && iy < height ) {
                        val_pixel = val_pixel +  img_in[iy * width +ix] * msk[i*m_size+j];

                    //     if (imgx == 0 && imgy == 255)
                    //         printf("%03.0f | %05d | %d | %d | %02.0f | %05.0f \n", img_in[iy * width +ix], iy * width +ix, i, j, msk[i*m_size+j], val_pixel);
                    }

                    }
            }     
            // guardo valor resultado
            img_out[imgy*width+imgx]= val_pixel;
        }
    }
    CLK_POSIX_STOP;
    CLK_POSIX_ELAPSED;


    // printf("img_out[256]_cpu: %f\n", img_out[256]);

    float t_elap = t_elap_get;
}